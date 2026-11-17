"""Validated fixed-observation replay for mutual Joint execution backends."""

import argparse
import copy
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

from benchmarks.robotwin import compare_joint as base


BACKENDS = ("legacy", "prepared", "prefix")


def validation_gates(errors, checks):
    return {
        "outputs_passed": all(row["passed"] for row in errors.values()),
        "operator_passed": all(row["video"]["passed"] and row["action"]["passed"] for row in checks.values()),
        "batch_partition_passed": all(row.get("partition", {"passed": True})["passed"] for row in checks.values()),
    }


def profile_indices_for(rows, selected, count, tail=False):
    if not tail:
        return set(sorted(selected)[:count])
    candidates = [row for row in rows if row["backend"] == "prefix" and row["measured"]]
    return {row["index"] for row in sorted(candidates, key=lambda row: row["wall_ms"], reverse=True)[:count]}


def validate_trace(trace, checkpoint, weight=None):
    metadata = base.read_json(trace.with_suffix(".json"))
    if (metadata.get("benchmark") != "robotwin" or metadata.get("source") != "actual_E8_requests"
            or metadata.get("trace_sha256") != base.digest(trace)
            or metadata.get("checkpoint_config") != base.digest(checkpoint / "config.yaml")):
        raise ValueError("Replay requires a verified RoboTwin E8 trace from this checkpoint configuration")
    if weight is not None and (metadata.get("checkpoint") != str(weight.resolve())
                               or metadata.get("checkpoint_size") != weight.stat().st_size
                               or metadata.get("checkpoint_mtime_ns") != weight.stat().st_mtime_ns):
        raise ValueError("Replay trace checkpoint identity changed; capture again from the requested immutable weights")
    return metadata


def validate_observations(observations, training):
    architecture = training["model"]["architecture"]
    loader = training["dataloader"]
    state_dim = architecture["state_dim"]
    if loader.get("unify_action"):
        from openwam.dataloader.utils.unify_action import parse_unify_spec

        state_dim = len(parse_unify_spec(loader.get("unify_state_map") or loader["unify_action_map"]))
    if (loader["type"] != "robotwin" or architecture["attention_mask_mode"] != "mutual"
            or loader.get("action_mode") != "eef" or observations[2].shape[1] != state_dim):
        raise ValueError("Expected mutual RoboTwin eef checkpoint and matching raw proprio dimensions")


def copy_history(source, target):
    import torch

    for name in ("history", "request", "signature"):
        setattr(target, name, getattr(source, name))
    target.video_phases = dict(source.video_phases)
    target._schedule_cache = dict(source._schedule_cache)
    for name in ("action_generator", "video_generator"):
        generator = getattr(source, name)
        cloned = None
        if generator is not None:
            cloned = torch.Generator(device=generator.device)
            cloned.set_state(generator.get_state())
        setattr(target, name, cloned)


@contextmanager
def profile_boundaries(engine, device, timings):
    import torch

    originals = []
    events = []
    targets = [(engine.architecture, "forward", "legacy_velocity"),
               (getattr(engine.architecture.video_backbone, "vae", None), "encode", "vae_encode")]
    try:
        for owner, name, label in targets:
            if owner is None or not callable(getattr(owner, name, None)):
                continue
            present = name in owner.__dict__
            previous = owner.__dict__.get(name)
            original = getattr(owner, name)

            def measured(*args, _original=original, _label=label, **kwargs):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(torch.cuda.current_stream(device))
                with torch.profiler.record_function("joint_replay." + _label):
                    result = _original(*args, **kwargs)
                end.record(torch.cuda.current_stream(device))
                events.append((_label, begin, end))
                return result

            originals.append((owner, name, present, previous))
            setattr(owner, name, measured)
        yield
        torch.cuda.synchronize(device)
        for label, begin, end in events:
            timings[label] = timings.get(label, 0.) + begin.elapsed_time(end)
    finally:
        for owner, name, present, previous in reversed(originals):
            if present:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)


def run(args):
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from omegaconf import OmegaConf
    from benchmarks.libero.joint_experiment import selected_indices, percentiles, require_exclusive_device, hardware_snapshot, tensor_error
    from benchmarks.libero.replay_joint import load_observations, resolve_checkpoint
    from openwam.deploy.server import _load_deploy_yaml, build_server_from_config
    from openwam.deploy.parallel_joint import JointParaFlowSampler, core_identity

    if args.warmup < 0 or args.measure < 1 or args.profile_samples < 0:
        raise ValueError("Invalid replay sample counts")
    checkpoint = resolve_checkpoint(args.ckpt_dir, args.ckpt_name)
    weight = args.ckpt_dir / checkpoint
    metadata = validate_trace(args.trace, args.ckpt_dir, weight)
    observations = load_observations(args.trace)
    training = yaml.safe_load((args.ckpt_dir / "config.yaml").read_text())
    validate_observations(observations, training)
    selected = set(selected_indices(observations, args.warmup, args.measure))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; real-model execution validation not_run")
    require_exclusive_device(args.device)
    before = hardware_snapshot()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    validation_rows = []
    cfg = _load_deploy_yaml()
    for key, value in {"inference.sampler": "joint-paraflow", "inference.denoise_steps": 10,
                       "inference.inference_horizon": 8, "inference.inference_mode": "sync", "inference.denoise_mode": "sync",
                       "inference.cfg_scale": 1., "optimization.compile.enabled": False,
                       "optimization.dit_cache.enabled": False, "optimization.decode_video": False}.items():
        OmegaConf.update(cfg, key, value)
    try:
        engine = build_server_from_config(cfg, str(args.ckpt_dir), device=args.device, ckpt_name=checkpoint).engine
    except BaseException as error:
        base.write_json(args.output / "failure.json", {"status": "model_load_failed", "error": str(error)})
        raise

    def new_samplers():
        return {backend: JointParaFlowSampler(execution_backend=backend,
                    allow_execution_parity_failure=args.allow_parity_failure) for backend in BACKENDS}

    def generate(sampler, index):
        engine._joint_sampler = sampler
        try:
            return engine.generate({"first_frame_image": [Image.fromarray(observations[0][index])],
                                    "prompt": str(observations[1][index]), "proprio": observations[2][index], "seed": 42})
        except Exception as error:
            base.write_json(args.output / "failed_request.json", {
                "index": index, "backend": sampler.execution_backend,
                "prompt": str(observations[1][index]), "error": str(error),
                "execution_diagnostic": getattr(error, "execution_diagnostic", None)})
            raise

    def reset_required(index):
        return bool(observations[3][index]) or index == 0 or str(observations[1][index]) != str(observations[1][index - 1])

    try:
        samplers = new_samplers()
        passed = True
        with (args.output / "validation.jsonl").open("x") as stream:
            for index in range(max(selected) + 1):
                if reset_required(index):
                    for sampler in samplers.values():
                        sampler.reset()
                if getattr(args, "validate_every_request", False):
                    for sampler in samplers.values():
                        sampler.execution_validation.clear()
                results = {backend: generate(sampler, index) for backend, sampler in samplers.items()}
                for backend in BACKENDS[1:]:
                    reference, candidate = samplers["legacy"], samplers[backend]
                    errors = {"action_trajectory": tensor_error(candidate.history, reference.history),
                              "executed_actions": tensor_error(torch.as_tensor(results[backend]["actions"][:8]),
                                                               torch.as_tensor(results["legacy"]["actions"][:8])),
                              "output_actions": tensor_error(torch.as_tensor(results[backend]["actions"]),
                                                             torch.as_tensor(results["legacy"]["actions"]))}
                    for phase in reference.video_phases:
                        errors[f"video_phase_{phase}"] = tensor_error(candidate.video_phases[phase], reference.video_phases[phase])
                    finite = all(row["finite"] for row in errors.values())
                    checks = candidate.execution_validation
                    valid = all(row["passed"] for row in errors.values()) and all(row["passed"] for row in checks.values())
                    row = {"index": index, "backend": backend, "passed": valid, "errors": errors,
                           **validation_gates(errors, checks),
                           "frozen_velocity_and_partition": checks}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    validation_rows.append({key: row[key] for key in ("index", "backend", "passed", "outputs_passed",
                                                                     "operator_passed", "batch_partition_passed")})
                    passed &= valid
                    if not finite or (not valid and not args.allow_parity_failure):
                        raise RuntimeError("Joint replay trajectory/output parity failed; inspect validation.jsonl")
                if index % 20 == 0:
                    require_exclusive_device(args.device)
                    print(f"execution validation {index + 1}/{max(selected) + 1}", flush=True)
        base.write_json(args.output / "validation.json", {"passed": passed, "requests": validation_rows,
                        "boundary": "independent multi-request histories plus frozen identical-candidate operator checks"})
        samplers = new_samplers()
        for sampler in samplers.values():
            sampler.validate_execution = False
        with (args.output / "requests.jsonl").open("x") as stream:
            for index in range(max(selected) + 1):
                if reset_required(index):
                    for sampler in samplers.values():
                        sampler.reset()
                for backend in BACKENDS[1:]:
                    copy_history(samplers["legacy"], samplers[backend])
                order = list(BACKENDS)
                random.Random(42 + index).shuffle(order)
                for backend in order:
                    torch.cuda.synchronize(args.device)
                    torch.cuda.reset_peak_memory_stats(args.device)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(torch.cuda.current_stream(args.device))
                    started = time.perf_counter()
                    result = generate(samplers[backend], index)
                    end.record(torch.cuda.current_stream(args.device))
                    torch.cuda.synchronize(args.device)
                    elapsed = (time.perf_counter() - started) * 1000
                    if not np.isfinite(result["actions"]).all():
                        raise FloatingPointError("Nonfinite replay output")
                    row = {"index": index, "backend": backend, "measured": index in selected,
                           "wall_ms": elapsed, "cuda_ms": begin.elapsed_time(end),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
                           "stats": copy.deepcopy(samplers[backend].last_stats)}
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                if index % 20 == 0:
                    require_exclusive_device(args.device)
                    print(f"execution timing {index + 1}/{max(selected) + 1}", flush=True)
        profile_rows = []
        if args.profile_samples:
            profile_indices = profile_indices_for(rows, selected, args.profile_samples, getattr(args, "profile_tail", False))
            profiled = new_samplers()
            for sampler in profiled.values():
                sampler.validate_execution = False
            for index in range(max(profile_indices) + 1):
                if index % 20 == 0:
                    require_exclusive_device(args.device)
                if reset_required(index):
                    for sampler in profiled.values():
                        sampler.reset()
                for backend in BACKENDS[1:]:
                    copy_history(profiled["legacy"], profiled[backend])
                for backend in BACKENDS:
                    if index not in profile_indices and backend != "legacy":
                        continue
                    sampler = profiled[backend]
                    sampler.execution_profile = index in profile_indices
                    if index not in profile_indices:
                        generate(sampler, index)
                        continue
                    timings = {}
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=True) as profiler:
                        with profile_boundaries(engine, args.device, timings):
                            generate(sampler, index)
                    if index == min(profile_indices):
                        profiler.export_chrome_trace(str(args.output / f"profile_{backend}.json"))
                    profile_rows.append({"index": index, "backend": backend, "boundaries_ms": timings, "stats": sampler.last_stats})
            component_summary = {}
            for backend in BACKENDS:
                values = [row for row in profile_rows if row["backend"] == backend]
                names = set().union(*(row["boundaries_ms"].keys() for row in values))
                names.update(key for row in values for key in row["stats"].get("execution", {}).get("components_ms", {}))
                component_summary[backend] = {name: percentiles([
                    row["boundaries_ms"].get(name, row["stats"].get("execution", {}).get("components_ms", {}).get(name, 0.))
                    for row in values]) for name in names}
            base.write_json(args.output / "profile.json", {"separate_from_timing": True, "requests": profile_rows,
                            "components_ms": component_summary})
            profile_lines = ["# Joint component profile", "", "Separate instrumented run; not steady-state latency.", "",
                             "| Backend | Component | P50 ms | P95 ms |", "|---|---|---:|---:|"]
            for backend, components in component_summary.items():
                for name, value in sorted(components.items()):
                    profile_lines.append(f"| {backend} | {name} | {value['p50']:.3f} | {value['p95']:.3f} |")
            (args.output / "profile.md").write_text("\n".join(profile_lines) + "\n")
        methods = {}
        for backend in BACKENDS:
            measured = [row for row in rows if row["backend"] == backend and row["measured"]]
            methods[backend] = {key: percentiles([row[key] for row in measured]) for key in
                                ("wall_ms", "cuda_ms", "peak_allocated_bytes", "peak_reserved_bytes")}
        reference = methods["legacy"]["wall_ms"]
        ratios = {backend: reference["p50"] / methods[backend]["wall_ms"]["p50"] for backend in BACKENDS[1:]}
        performance_gate = (methods["prefix"]["wall_ms"]["p50"] <= .9 * reference["p50"]
                            and methods["prefix"]["wall_ms"]["p95"] <= reference["p95"])
        require_exclusive_device(args.device)
        validation_summary = {}
        for backend in BACKENDS[1:]:
            values = [row for row in validation_rows if row["backend"] == backend]
            validation_summary[backend] = {"requests": len(values), **{
                key: sum(row[key] for row in values) for key in
                ("passed", "outputs_passed", "operator_passed", "batch_partition_passed")}}
        report = {"status": "complete", "interpretation": "execution_validation_passed" if passed else "diagnostic_only_parity_failed",
                  "validation_summary": validation_summary,
                  "validate_every_request": getattr(args, "validate_every_request", False),
                  "profile_selection": "slowest_prefix_requests" if getattr(args, "profile_tail", False) else "first_requests",
                  "methods": methods, "p50_speedups": ratios, "prefix_performance_gate": bool(performance_gate and passed),
                  "closed_loop_quality_gate": "not_evaluated_by_replay", "trace": metadata, "checkpoint": str(weight),
                  "checkpoint_size": weight.stat().st_size, "checkpoint_mtime_ns": weight.stat().st_mtime_ns,
                  "source_identity": core_identity(), "hardware_before": before, "hardware_after": hardware_snapshot(),
                  "warmup": args.warmup, "measure": args.measure, "profile_samples": args.profile_samples,
                  "boundary": "timing uses identical legacy-owned candidate histories and cloned RNG per observation; validation uses independent histories",
                  "memory_note": "one resident model; history buffers for three samplers coexist; reserved memory is allocator-order-dependent",
                  "default_backend_changed": False}
        base.write_json(args.output / "timing.json", report)
        lines = ["# Joint execution replay", "", report["interpretation"], "", report["boundary"], "",
                 "| Backend | P50 ms | P95 ms | P50 speedup |", "|---|---:|---:|---:|"]
        for backend, result in methods.items():
            lines.append(f"| {backend} | {result['wall_ms']['p50']:.3f} | {result['wall_ms']['p95']:.3f} | {ratios.get(backend, 1.):.3f}x |")
        lines += ["", "## Independent validation gates", "",
                  "Counts are passed requests; overall gate and thresholds are unchanged.", "",
                  "| Backend | Requests | Output pass | Same-input operator pass | Batch partition pass | Overall pass |",
                  "|---|---:|---:|---:|---:|---:|"]
        for backend, value in validation_summary.items():
            lines.append(f"| {backend} | {value['requests']} | {value['outputs_passed']} | {value['operator_passed']} | "
                         f"{value['batch_partition_passed']} | {value['passed']} |")
        lines += ["", "## Peak memory (coexisting replay histories)", "",
                  "| Backend | Allocated P50 GiB | Reserved P50 GiB |", "|---|---:|---:|"]
        for backend, result in methods.items():
            lines.append(f"| {backend} | {result['peak_allocated_bytes']['p50'] / 2**30:.3f} | {result['peak_reserved_bytes']['p50'] / 2**30:.3f} |")
        lines += ["", f"Prefix performance gate (>=10% P50 reduction, no P95 regression, parity passed): {report['prefix_performance_gate']}",
                  "Closed-loop quality must pass separately; this report does not change the default backend.",
                  "Profiler/event measurements are separate from steady-state timing; see profile.json and profile_*.json."]
        (args.output / "timing.md").write_text("\n".join(lines) + "\n")
        print("Replay report:", args.output / "timing.md")
    except BaseException as error:
        base.write_json(args.output / "failure.json", {"status": "failed", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/data1/wam-ckpts/OpenWAM-Alpha-Sim-RoboTwin-Full"))
    parser.add_argument("--ckpt-name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=48)
    parser.add_argument("--measure", type=int, default=120)
    parser.add_argument("--profile-samples", type=int, default=8)
    parser.add_argument("--allow-parity-failure", action="store_true")
    parser.add_argument("--validate-every-request", action="store_true")
    parser.add_argument("--profile-tail", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
