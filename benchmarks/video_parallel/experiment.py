"""Controlled replay and validation for video-guided Joint denoising."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHODS = "native_n10,native_n4,native_n6,native_n8,legacy_n10,fixed_k2_r1,fixed_k4_r1,residual_k2,residual_k4,adaptive"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def cases_for(args):
    from .runtime import method_spec

    cases = []
    names = args.methods.split(",")
    if not names or len(set(names)) != len(names) or "native_n10" not in names:
        raise ValueError("Select unique methods including native_n10")
    for name in names:
        _, config = method_spec(name)
        overrides = dict(args.solver_overrides) if config is not None and config != "legacy" else {}
        if args.mode == "sweep" and config is not None and config != "legacy":
            for tolerance in args.tolerances:
                variant = {**overrides, "video_tolerance": tolerance, "local_tolerance": tolerance * 3}
                cases.append({"name": f"{name}_v{tolerance:g}", "method": name, "overrides": variant})
        else:
            cases.append({"name": name, "method": name, "overrides": overrides})
    for case in cases:
        _, config = method_spec(case["method"])
        if config is not None and config != "legacy":
            from .solver import SolverConfig

            SolverConfig(**{**asdict(config), **case["overrides"]})
    return cases


def preflight(args):
    import numpy as np
    import yaml
    from benchmarks.libero.replay_joint import load_observations, resolve_checkpoint

    args.ckpt_dir = args.ckpt_dir.resolve()
    args.trace = args.trace.resolve()
    args.ckpt_name = resolve_checkpoint(args.ckpt_dir, args.ckpt_name)
    training = yaml.safe_load((args.ckpt_dir / "config.yaml").read_text())
    if training["model"]["architecture"].get("attention_mask_mode") != "mutual":
        raise ValueError("Expected a mutual checkpoint; changing its mask is prohibited")
    if not (args.ckpt_dir / "normalization_stats.npy").is_file():
        raise FileNotFoundError("Checkpoint normalization_stats.npy missing")
    observations = load_observations(args.trace)
    if len(observations[0]) < args.warmup + args.measure:
        raise ValueError("Trace too short for requested warmup + measure")
    disk_path = args.output.parent
    while not disk_path.exists():
        disk_path = disk_path.parent
    free_gib = shutil.disk_usage(disk_path).free / 1024**3
    if free_gib < args.min_free_gib:
        raise OSError(f"Insufficient free disk at {disk_path.resolve()}: {free_gib:.2f} GiB available, "
                      f"{args.min_free_gib:g} GiB required. Free space, choose another output filesystem, "
                      "or explicitly set MIN_FREE_GIB for the launcher.")
    metadata_path = args.trace.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    trace_hash = digest(args.trace)
    config_hash = digest(args.ckpt_dir / "config.yaml")
    if metadata and metadata.get("trace_sha256") not in (None, trace_hash):
        raise ValueError("Trace digest mismatch")
    if metadata and metadata.get("checkpoint_config") not in (None, config_hash):
        raise ValueError("Trace checkpoint configuration mismatch")
    verified = bool(metadata and metadata.get("trace_sha256") == trace_hash
                    and metadata.get("checkpoint_config") == config_hash)
    if not verified and not args.allow_unverified_trace:
        raise ValueError("Trace lacks hash-bound checkpoint provenance; explicitly pass --allow-unverified-trace for historical diagnostic traces")
    architecture = training["model"]["architecture"]
    loader = training["dataloader"]
    expected_state = architecture.get("state_dim")
    if loader.get("unify_action"):
        from openwam.dataloader.utils.unify_action import parse_unify_spec

        expected_state = len(parse_unify_spec(loader.get("unify_state_map") or loader["unify_action_map"]))
    if expected_state is not None and observations[2].shape[1] != expected_state:
        raise ValueError(f"Raw trace proprio dimensions do not match checkpoint: trace={args.trace} "
                         f"has {observations[2].shape[1]} dimensions; checkpoint={args.ckpt_dir} "
                         f"expects {expected_state} raw dimensions (before unification). "
                         "Select a matching trace/checkpoint pair; do not pad or truncate states.")
    if not np.isfinite(observations[2]).all():
        raise ValueError("Nonfinite states")
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((ROOT / "openwam/deploy").glob("*.py"))
    paths += list((ROOT / "openwam/model").rglob("*.py"))
    weight = args.ckpt_dir / args.ckpt_name
    if metadata:
        for key, expected in (("checkpoint", str(weight.resolve())), ("checkpoint_size", weight.stat().st_size),
                              ("checkpoint_mtime_ns", weight.stat().st_mtime_ns)):
            if key in metadata and metadata[key] != expected:
                raise ValueError(f"Trace weight identity mismatch: {key}")
    return {"trace": str(args.trace), "trace_sha256": trace_hash, "trace_provenance": "verified" if verified else "unverified_diagnostic",
            "checkpoint": str(weight), "weight_size": weight.stat().st_size, "weight_mtime_ns": weight.stat().st_mtime_ns,
            "checkpoint_config_sha256": config_hash, "benchmark": loader["type"], "requests": len(observations[0]),
            "sources": {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)},
            "cases": cases_for(args), "gpu_validation": "not_run", "closed_loop": "not_run"}


def build_engine(args):
    from omegaconf import OmegaConf
    from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

    config = _load_deploy_yaml()
    settings = {"inference.sampler": "native", "inference.denoise_steps": 10,
                "inference.inference_horizon": 8, "inference.inference_mode": "sync",
                "inference.denoise_mode": "sync", "inference.cfg_scale": 1.,
                "optimization.compile.enabled": False, "optimization.dit_cache.enabled": False,
                "optimization.decode_video": False}
    for key, value in settings.items():
        OmegaConf.update(config, key, value)
    return build_server_from_config(config, str(args.ckpt_dir), device=args.device, ckpt_name=args.ckpt_name).engine


def stats(values):
    import numpy as np

    return {"count": len(values), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))} if values else None


def report(output):
    output = Path(output)
    identity = json.loads((output / "identity.json").read_text())
    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()] if (output / "requests.jsonl").exists() else []
    validation = json.loads((output / "validation.json").read_text()) if (output / "validation.json").exists() else {"status": "not_run"}
    failure = json.loads((output / "failure.json").read_text()) if (output / "failure.json").exists() else None
    methods = {}
    for case in identity["cases"]:
        measured = [row for row in rows if row["method"] == case["name"] and row["measured"]]
        methods[case["name"]] = {"wall_ms": stats([row["wall_ms"] for row in measured]),
                                 "peak_allocated_gib": max((row["peak_allocated_gib"] for row in measured), default=None),
                                 "fallbacks": sum(row.get("sampler_stats", {}).get("fallbacks", 0) for row in measured),
                                 "errors": [row["errors"] for row in measured if "errors" in row]}
        methods[case["name"]]["cold_wall_ms"] = stats([row["wall_ms"] for row in measured if row.get("sampler_stats", {}).get("cold") is True])
        methods[case["name"]]["warm_wall_ms"] = stats([row["wall_ms"] for row in measured if row.get("sampler_stats", {}).get("cold") is False])
    expected = identity["measure"]
    complete = bool(methods) and all(value["wall_ms"] and value["wall_ms"]["count"] == expected for value in methods.values())
    result = {"status": "failed" if failure else "complete" if complete else "incomplete", "validation": validation,
              "failure": failure, "methods": methods, "closed_loop": "not_run", "decode_video": identity["decode_video"],
              "interpretation": "Approximate sampling study, not equivalence or closed-loop quality proof. Unverified provenance or partition failures imply diagnostic timings only."}
    result["diagnostic_only"] = identity["trace_provenance"] != "verified" or validation.get("status") != "passed"
    result["memory_boundary"] = "Single shared engine with retained comparison sampler states; peak totals are not isolated per-method capacity measurements. requests.jsonl also records incremental peaks."
    write_json(output / "comparison.json", result)
    lines = ["# Video-guided Joint sampling", "", f"Status: {result['status']}; validation: {validation['status']}; diagnostic only: {result['diagnostic_only']}",
             f"Trace provenance: {identity['trace_provenance']}; VAE decode: {identity['decode_video']}",
             "", "Paired fixed observations; not closed-loop task success. Native N10 is the only speed baseline.", "",
             "| Method | N | Wall P50 ms | Wall P95 ms | Speedup | Peak GiB | Fallbacks |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    reference = methods.get("native_n10", {}).get("wall_ms")
    for name, value in methods.items():
        wall = value["wall_ms"]
        if wall:
            speedup = f"{reference['p50'] / wall['p50']:.3f}x" if reference else "pending"
            lines.append(f"| {name} | {wall['count']} | {wall['p50']:.3f} | {wall['p95']:.3f} | {speedup} | {value['peak_allocated_gib']:.2f} | {value['fallbacks']} |")
        else:
            lines.append(f"| {name} | 0 | pending | pending | pending | pending | pending |")
    lines += ["", result["interpretation"], "", "Residual acceptance is not an equivalence proof. Errors use unchanged rtol=atol=0.05.",
              "Decode-off timings include full future-video latents, not playable video. Cold/warm legacy labels are in requests.jsonl.",
              "Inclusive profiler regions overlap and must not be summed.", ""]
    lines += [result["memory_boundary"], "", "## Output fidelity to Native N10", "",
              "| Method | Median latent relative RMS | Median output-action RMSE | Median E8 action RMSE |",
              "|---|---:|---:|---:|"]
    for name, value in methods.items():
        errors = value["errors"]
        if errors:
            video = stats([entry["future_video"]["relative_rms"] for entry in errors])
            action = stats([entry["actions"]["rmse"] for entry in errors])
            executed = stats([entry["executed_actions"]["rmse"] for entry in errors])
            lines.append(f"| {name} | {video['p50']:.6f} | {action['p50']:.6f} | {executed['p50']:.6f} |")
    (output / "comparison.md").write_text("\n".join(lines))
    with (output / "curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "wall_p50_ms", "wall_p95_ms", "video_relative_rms_p50", "action_rmse_p50", "fallbacks"])
        for name, value in methods.items():
            wall, errors = value["wall_ms"], value["errors"]
            if wall:
                writer.writerow([name, wall["p50"], wall["p95"],
                                 stats([entry["future_video"]["relative_rms"] for entry in errors])["p50"] if errors else "",
                                 stats([entry["actions"]["rmse"] for entry in errors])["p50"] if errors else "", value["fallbacks"]])
    return result


def run(args, identity):
    import numpy as np
    import torch
    from PIL import Image
    from benchmarks.libero.joint_experiment import hardware_snapshot, require_exclusive_device
    from benchmarks.libero.replay_joint import load_observations
    from .runtime import PartitionValidator, capture_video, configure_engine, tensor_error
    from .solver import SolverConfig, VideoGuidedSampler

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; real-model validation and timing not_run")
    require_exclusive_device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    identity.update({"warmup": args.warmup, "measure": args.measure, "decode_video": args.decode_video,
                     "device": args.device, "seed": args.seed, "torch": torch.__version__,
                     "created_at": datetime.now().astimezone().isoformat(), "hardware_before": hardware_snapshot()})
    write_json(args.output / "identity.json", identity)
    observations = load_observations(args.trace)
    engine = build_engine(args)
    cases = identity["cases"]

    def generate(index):
        return engine.generate({"first_frame_image": [Image.fromarray(observations[0][index])],
                                "prompt": str(observations[1][index]), "proprio": observations[2][index], "seed": args.seed})

    def new_sampler(case):
        return configure_engine(engine, case["method"], case["overrides"])

    validator = PartitionValidator()
    validation = {"status": "running", "partition": [], "outputs": [], "k1": []}
    write_json(args.output / "validation.json", validation)
    for index in range(args.validation_samples):
        configure_engine(engine, "native_n10")
        with capture_video(engine) as captured:
            reference = generate(index)
        reference_video = captured["latents"].clone()
        serial_sampler = VideoGuidedSampler(SolverConfig(policy="serial", window=1))
        engine._joint_sampler = serial_sampler
        with capture_video(engine) as captured:
            serial_result = generate(index)
        validation["k1"].append({"index": index, "video": tensor_error(captured["latents"], reference_video),
                                 "action": tensor_error(serial_result["actions"], reference["actions"])})
        for case in cases:
            sampler = new_sampler(case)
            if isinstance(sampler, VideoGuidedSampler):
                sampler.velocity = validator
            with capture_video(engine) as captured:
                result = generate(index)
            validation["outputs"].append({"index": index, "method": case["name"],
                "video": tensor_error(captured["latents"][:, :, 1:], reference_video[:, :, 1:]),
                "action": tensor_error(result["actions"], reference["actions"])})
    validation["partition"] = validator.checks
    partition_passed = all(row[key]["passed"] for row in validator.checks for key in ("video", "action"))
    k1_passed = all(row[key]["passed"] for row in validation["k1"] for key in ("video", "action"))
    finite = all(row[key]["finite"] for row in validation["outputs"] for key in ("video", "action"))
    validation.update({"status": "passed" if partition_passed and k1_passed and finite else "failed",
                       "partition_status": "passed" if validator.checks and partition_passed else "failed" if validator.checks else "not_exercised",
                       "k1_passed": k1_passed, "approximate_outputs_need_quality_evaluation": True})
    write_json(args.output / "validation.json", validation)
    if not k1_passed or not finite or (not partition_passed and not args.allow_parity_failure):
        raise RuntimeError("Validation gate failed; inspect validation.json. Only finite partition failures may be explicitly allowed for diagnostics.")
    samplers = {case["name"]: new_sampler(case) for case in cases}
    quality_samples = []
    scene = -1
    with (args.output / "requests.jsonl").open("x", buffering=1) as stream:
        for index in range(args.warmup + args.measure):
            require_exclusive_device(args.device)
            if observations[3][index] or index == 0 or observations[1][index] != observations[1][index - 1]:
                scene += 1
                for sampler in samplers.values():
                    if sampler is not None:
                        sampler.reset()
            order = list(cases)
            random.Random(args.seed + index).shuffle(order)
            request_rows, reference_video, reference_actions = [], None, None
            captured_outputs, decoded_outputs = {}, {}
            for case in order:
                engine.cfg.inference.denoise_steps = 10
                from .runtime import method_spec

                engine.cfg.inference.denoise_steps = method_spec(case["method"])[0]
                engine._joint_sampler = samplers[case["name"]]
                torch.cuda.synchronize(args.device)
                allocated_before = torch.cuda.memory_allocated(args.device) / 1024**3
                torch.cuda.reset_peak_memory_stats(args.device)
                started = time.perf_counter()
                with capture_video(engine, args.decode_video) as captured:
                    result = generate(index)
                torch.cuda.synchronize(args.device)
                elapsed = (time.perf_counter() - started) * 1000
                peak = torch.cuda.max_memory_allocated(args.device) / 1024**3
                captured_outputs[case["name"]] = (captured["latents"].float().cpu(), np.asarray(result["actions"]))
                if args.warmup <= index < args.warmup + args.video_samples:
                    if result["video"] is None:
                        raise RuntimeError("Requested decoded-video samples but VAE produced no pixels")
                    frames = np.stack([np.asarray(frame) for frame in result["video"]])
                    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 2:
                        raise ValueError("Expected decoded RGB frames including a clean first frame")
                    decoded_outputs[case["name"]] = frames[1:]
                if case["method"] == "native_n10":
                    reference_video, reference_actions = captured_outputs[case["name"]]
                request_rows.append({"index": index, "method": case["name"], "measured": index >= args.warmup,
                                     "wall_ms": elapsed, "peak_allocated_gib": peak,
                                     "incremental_peak_gib": max(0., peak - allocated_before),
                                     "episode_start": bool(observations[3][index]),
                                     "sampler_stats": result.get("sampler_stats", {})})
            for row in request_rows:
                video, actions = captured_outputs[row["method"]]
                row["errors"] = {"future_video": tensor_error(video[:, :, 1:], reference_video[:, :, 1:]),
                                  "actions": tensor_error(actions, reference_actions),
                                  "executed_actions": tensor_error(actions[:8], reference_actions[:8])}
                if not all(value["finite"] for value in row["errors"].values()):
                    raise FloatingPointError("Nonfinite final output")
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if row["method"] in decoded_outputs:
                    directory = args.output / "videos"
                    directory.mkdir(exist_ok=True)
                    relative = Path("videos") / f"{index:06d}_{row['method']}.npz"
                    np.savez_compressed(args.output / relative, candidate_rgb=decoded_outputs[row["method"]],
                                        reference_rgb=decoded_outputs["native_n10"])
                    quality_samples.append({"method": row["method"], "scene": f"episode_{scene}",
                                            "sample_id": str(index), "reference_kind": "native_fidelity", "path": str(relative)})
                    write_json(args.output / "quality_manifest.json", {"samples": quality_samples})
            require_exclusive_device(args.device)
            if index == args.warmup or index == args.warmup + args.measure - 1:
                print(f"[video-parallel] replay {index + 1}/{args.warmup + args.measure}", flush=True)
    if args.mode == "profile":
        from .profiling import boundaries

        profiles = []
        for case in cases:
            new_sampler(case)
            for index in range(args.profile_samples):
                with capture_video(engine, args.decode_video):
                    with boundaries(engine, args.device) as regions:
                        generate(index)
                profiles.append({"method": case["name"], "index": index, **regions})
        write_json(args.output / "profile.json", profiles)
    write_json(args.output / "hardware_after.json", hardware_snapshot())
    require_exclusive_device(args.device)
    if quality_samples:
        from .quality import evaluate_manifest

        evaluate_manifest(args.output / "quality_manifest.json", args.output / "video_quality.json")
    report(args.output)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("preflight", "profile", "sweep", "evaluate", "report"))
    result.add_argument("--trace", type=Path)
    result.add_argument("--ckpt-dir", type=Path, default=Path("/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO"))
    result.add_argument("--ckpt-name")
    result.add_argument("--output", type=Path, default=ROOT / "outputs" / ("video_parallel_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--methods", default=DEFAULT_METHODS)
    result.add_argument("--solver-overrides", type=json.loads, default={})
    result.add_argument("--tolerances", type=lambda value: [float(item) for item in value.split(",")], default=[.001, .003, .01])
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--measure", type=int, default=30)
    result.add_argument("--validation-samples", type=int, default=3)
    result.add_argument("--profile-samples", type=int, default=1)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--min-free-gib", type=float, default=5.)
    result.add_argument("--decode-video", action="store_true")
    result.add_argument("--video-samples", type=int, default=0, help="Save this many paired decoded samples after warmup, excluding first frame")
    result.add_argument("--allow-parity-failure", action="store_true")
    result.add_argument("--allow-unverified-trace", action="store_true")
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.mode == "report":
        report(args.output)
        return
    if (args.trace is None or args.warmup < 0 or args.measure < 1 or args.validation_samples < 1
            or args.validation_samples > args.warmup + args.measure or args.profile_samples < 1
            or args.min_free_gib < 0 or not isinstance(args.solver_overrides, dict)
            or args.video_samples < 0 or args.video_samples > args.measure or (args.video_samples and not args.decode_video)):
        arguments.error("Specify --trace, valid sample counts, nonnegative disk margin and an overrides object")
    identity = preflight(args)
    if args.mode == "preflight":
        print(json.dumps(identity, indent=2))
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    print(f"Output: {args.output}; device={args.device}; existing GPU jobs are never stopped", flush=True)
    try:
        run(args, identity)
    except BaseException as error:
        if (args.output / "identity.json").exists():
            try:
                write_json(args.output / "failure.json", {"error": str(error), "type": type(error).__name__})
                report(args.output)
            except OSError as artifact_error:
                print(f"Cannot persist failure artifacts: {artifact_error}; original error: {error}", file=sys.stderr)
        raise
    print(f"Report: {args.output / 'comparison.md'}")


if __name__ == "__main__":
    main()
