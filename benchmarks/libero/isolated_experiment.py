"""Current LIBERO weights: experimental isolated-mask quality and latency ablation."""

import argparse
import json
import time
from pathlib import Path

from openwam.deploy.isolated_reference import DEFAULT_PARAFLOW, DEFAULT_NATIVE, load_reference


METHODS = ("mutual_native_n10", "isolated_native_n10", "isolated_paraflow_n10")


def summary(output):
    from benchmarks.libero.joint_quality import summarize_quality

    report = summarize_quality(output, METHODS)
    report["experiment"] = "same_LIBERO_weights_experimental_isolated_not_original_isolated_checkpoint"
    report["cross_semantics_speedup_valid"] = False
    lines = ["# LIBERO isolated mask ablation", "",
             "Same LIBERO weights; isolated is an experimental mask override, not original isolated trained weights.",
             "Mutual Native denoises future video; both isolated methods use only current-frame visual prefill.",
             "No cross-semantics algorithm speedup claim. Closed-loop timings are not paired timings.",
             "", "| Method | Status | Successes / Trials | Success rate |", "|---|---|---:|---:|"]
    for name, row in report["methods"].items():
        overall = row.get("overall")
        count = f"{overall['successes']} / {overall['trials']}" if overall else "—"
        rate = f"{overall['success_rate']:.2%}" if overall and overall["trials"] else "—"
        lines.append(f"| {name} | {row['status']} | {count} | {rate} |")
    lines += ["", "## Per-trial results", "", "| Method | Suite | Task | Trial | Success | Policy steps |",
              "|---|---|---:|---:|---|---:|"]
    for name, row in report["methods"].items():
        for trial in row["trials"]:
            lines.append(f"| {name} | {trial['suite']} | {trial['task_id']} | {trial['trial']} | {trial['success']} | {trial['policy_steps']} |")
    lines += ["", "Interpret mask-switch quality separately from isolated serial-to-parallel quality.",
              "Read server logs for first-frame/full isolated parity and per-request call-budget evidence.",
              "Five trials are diagnostic only, not a benchmark-level success-rate estimate."]
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    return all(row["status"] == "complete" for row in report["methods"].values())


def replay(args):
    import hashlib
    import random
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from benchmarks.libero.joint_experiment import selected_indices, percentiles, require_exclusive_device, hardware_snapshot
    from benchmarks.libero.replay_joint import load_observations, resolve_checkpoint
    from openwam.deploy.server import _load_deploy_yaml, build_server_from_config
    from openwam.deploy.isolated_reference import IsolatedReferenceSampler

    if not torch.cuda.is_available():
        raise RuntimeError("GPU validation not_run: CUDA unavailable")
    require_exclusive_device(args.device)
    observations = load_observations(args.trace)
    selected = selected_indices(observations, args.warmup, args.measure)
    cfg = _load_deploy_yaml()
    settings = {"inference.sampler": "isolated-native", "inference.denoise_steps": 10,
                "inference.inference_horizon": 8, "inference.inference_mode": "sync", "inference.denoise_mode": "sync",
                "optimization.compile.enabled": False, "optimization.dit_cache.enabled": False, "optimization.decode_video": False,
                "inference.isolated_paraflow_root": args.paraflow_root, "inference.isolated_native_root": args.native_root,
                "inference.isolated_allow_parity_failure": args.allow_parity_failure}
    for key, value in settings.items():
        OmegaConf.update(cfg, key, value)
    checkpoint = resolve_checkpoint(args.ckpt_dir, args.ckpt_name)
    args.output.mkdir(parents=True, exist_ok=False)
    before = hardware_snapshot()
    engine = build_server_from_config(cfg, args.ckpt_dir, device=args.device, ckpt_name=checkpoint).engine
    states = {method: IsolatedReferenceSampler(method=method, paraflow_root=args.paraflow_root,
                                               native_root=args.native_root, allow_partition_failure=args.allow_parity_failure)
              for method in ("native", "paraflow")}
    rows = []
    previous_prompt = None
    try:
        with (args.output / "requests.jsonl").open("x") as stream:
            for index in range(selected[-1] + 1):
                prompt = str(observations[1][index])
                reset = bool(observations[3][index]) or prompt != previous_prompt
                order = list(states)
                random.Random(42 + index).shuffle(order)
                for method in order:
                    sampler = states[method]
                    if reset:
                        sampler.reset()
                    engine._joint_sampler = sampler
                    torch.cuda.synchronize(args.device)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    started = time.perf_counter()
                    begin.record(torch.cuda.current_stream(args.device))
                    result = engine.generate({"first_frame_image": [Image.fromarray(observations[0][index])],
                                              "prompt": prompt, "proprio": observations[2][index], "seed": 42})
                    end.record(torch.cuda.current_stream(args.device))
                    torch.cuda.synchronize(args.device)
                    elapsed = (time.perf_counter() - started) * 1000
                    if not bool(torch.isfinite(torch.as_tensor(result["actions"])).all()):
                        raise FloatingPointError("Nonfinite isolated output")
                    row = {"index": index, "method": method, "measured": index in selected,
                           "wall_ms": elapsed, "cuda_ms": begin.elapsed_time(end), "stats": sampler.last_stats}
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                previous_prompt = prompt
                if index % 10 == 0:
                    require_exclusive_device(args.device)
                    print(f"isolated replay {index + 1}/{selected[-1] + 1}", flush=True)
        require_exclusive_device(args.device)
    except BaseException as error:
        (args.output / "failure.json").write_text(json.dumps({"status": "failed", "error": str(error)}))
        raise
    methods = {method: {"wall_ms": percentiles([row["wall_ms"] for row in rows if row["method"] == method and row["measured"]]),
                         "cuda_ms": percentiles([row["cuda_ms"] for row in rows if row["method"] == method and row["measured"]])}
               for method in states}
    validations = {method: {"first_frame_operator": sampler.validation, "stage_batch": sampler.batch_validation}
                   for method, sampler in states.items()}
    passed = all(value["first_frame_operator"]["status"] == "passed" for value in validations.values())
    passed = passed and states["paraflow"].batch_validation is not None and states["paraflow"].batch_validation["status"] == "passed"
    report = {"status": "completed", "methods": methods, "validations": validations,
              "interpretation": "operator_checks_passed_not_policy_equivalence" if passed else "diagnostic_only_parity_failed",
              "isolated_p50_speedup": methods["native"]["wall_ms"]["p50"] / methods["paraflow"]["wall_ms"]["p50"],
              "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(), "checkpoint": str(Path(args.ckpt_dir) / checkpoint),
              "source_identity": states["native"].identity, "hardware_before": before, "hardware_after": hardware_snapshot(),
              "boundary": "same isolated model, shared request-local engine preprocessing caches; no future video; no mutual comparison",
              "warmup": args.warmup, "measure": args.measure, "closed_loop": "not_run"}
    (args.output / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Paired isolated replay", "", report["interpretation"], "",
             "| Method | P50 ms | P95 ms |", "|---|---:|---:|"]
    for method, result in methods.items():
        lines.append(f"| {method} | {result['wall_ms']['p50']:.3f} | {result['wall_ms']['p95']:.3f} |")
    lines += ["", f"Isolated P50 speedup: {report['isolated_p50_speedup']:.3f}x; not a mutual-to-isolated algorithm speedup."]
    (args.output / "timing.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "summary", "replay"))
    parser.add_argument("--paraflow-root", default=DEFAULT_PARAFLOW)
    parser.add_argument("--native-root", default=DEFAULT_NATIVE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--ckpt-dir", default="/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO")
    parser.add_argument("--ckpt-name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=48)
    parser.add_argument("--measure", type=int, default=120)
    parser.add_argument("--allow-parity-failure", action="store_true")
    args = parser.parse_args()
    if args.mode == "preflight":
        _, _, _, identity = load_reference(args.paraflow_root, args.native_root)
        print(json.dumps({"status": "imports_verified", "identity": identity,
                          "GPU_model_operator_validation": "not_run"}, indent=2))
    elif args.output is None:
        parser.error("--output is required")
    elif args.mode == "summary":
        if not summary(args.output):
            raise SystemExit(1)
    elif args.trace is None or args.warmup < 0 or args.measure < 1:
        parser.error("replay requires TRACE and valid WARMUP/MEASURE")
    else:
        replay(args)


if __name__ == "__main__":
    main()
