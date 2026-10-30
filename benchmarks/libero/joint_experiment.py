"""One-command, real-observation Joint performance experiments."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "native_n1": ("native", 1, False),
    "native_n4": ("native", 4, False),
    "native_n10": ("native", 10, False),
    "joint_n4": ("joint-paraflow", 4, False),
    "joint_cached_n4": ("joint-paraflow", 4, True),
}


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def selected_indices(observations, warmup, measure):
    previous = None
    episode_request = 0
    selected = []
    for index, (prompt, start) in enumerate(zip(observations[1], observations[3])):
        if start or str(prompt) != previous:
            episode_request = 0
        if index >= warmup and episode_request >= 2:
            selected.append(index)
        episode_request += 1
        previous = str(prompt)
        if len(selected) == measure:
            return selected
    raise ValueError(f"Trace has only {len(selected)} eligible requests, need {measure}; "
                     "collect more real observations or lower WARMUP/MEASURE explicitly")


def method_order(index, seed=42):
    names = list(METHODS)
    random.Random(seed + index).shuffle(names)
    return names


def percentiles(values):
    return {"count": len(values), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95))} if values else None


def hardware_snapshot():
    commands = {
        "gpus": ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv"],
        "processes": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"],
    }
    results = {}
    for label, command in commands.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            results[label] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            results[label] = {"status": "unavailable", "error": str(error)}
    return results


def require_exclusive_device(device):
    import torch

    uuid = str(torch.cuda.get_device_properties(device).uuid).removeprefix("GPU-")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=True)
    foreign = [line for line in result.stdout.splitlines() if line.split(",")[0].strip().removeprefix("GPU-") == uuid
               and int(line.split(",")[1]) != os.getpid()]
    if foreign:
        raise RuntimeError(f"Exclusive GPU timing rejected: {foreign}")


@contextmanager
def scoped_annotations(architecture):
    import torch

    originals = []
    targets = [(architecture, "forward", "joint.forward"),
               (architecture.video_backbone, "prepare", "video.prepare"),
               (architecture.video_backbone, "finalize", "video.finalize"),
               (getattr(architecture.video_backbone, "vae", None), "encode", "vae.encode"),
               (architecture.action_backbone, "prepare_state", "action.prepare"),
               (architecture.action_backbone, "extract_prediction", "action.finalize")]
    for module, name, label in targets:
        original = getattr(module, name, None)
        if not callable(original):
            continue

        def annotated(*args, _original=original, _label=label, **kwargs):
            with torch.profiler.record_function(_label):
                return _original(*args, **kwargs)

        originals.append((module, name, original))
        setattr(module, name, annotated)
    handles = []
    active_scopes = {}
    named_modules = getattr(architecture, "named_modules", lambda: [])
    for name, module in named_modules():
        if name.rsplit(".", 1)[-1] not in {"cross_attn", "ffn", "text_embedding", "time_embedding", "time_projection"}:
            continue

        def enter(current, inputs, _label=name):
            scope = torch.profiler.record_function(_label)
            scope.__enter__()
            active_scopes.setdefault(id(current), []).append(scope)

        def leave(current, inputs, output):
            scopes = active_scopes.get(id(current), [])
            if scopes:
                scopes.pop().__exit__(None, None, None)

        handles.extend([module.register_forward_pre_hook(enter),
                        module.register_forward_hook(leave, always_call=True)])
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module, name, original in originals:
            setattr(module, name, original)


def build_engine(args):
    import torch
    from omegaconf import OmegaConf
    from openwam.deploy.server import _load_deploy_yaml, build_server_from_config
    from benchmarks.libero.replay_joint import resolve_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("GPU experiment not_run: CUDA is unavailable")
    cfg = _load_deploy_yaml()
    settings = {
        "inference.sampler": "joint-paraflow", "inference.denoise_steps": 4,
        "inference.inference_horizon": 8, "inference.inference_mode": "sync",
        "inference.denoise_mode": "sync", "optimization.compile.enabled": False,
        "optimization.dit_cache.enabled": False, "optimization.decode_video": False,
    }
    for key, value in settings.items():
        OmegaConf.update(cfg, key, value)
    checkpoint = resolve_checkpoint(args.ckpt_dir, args.ckpt_name)
    server = build_server_from_config(cfg, args.ckpt_dir, device=args.device, ckpt_name=checkpoint)
    return server.engine, checkpoint


def tensor_error(actual, expected, *, rtol=0.05, atol=0.05):
    import torch

    actual, expected = actual.detach().float(), expected.detach().float()
    difference = (actual - expected).abs()
    finite = torch.isfinite(actual).all() & torch.isfinite(expected).all()
    mismatches = difference > atol + rtol * expected.abs()
    count = int(mismatches.sum())
    if not bool(finite):
        return {"passed": False, "finite": False, "elements": actual.numel(), "mismatches": count,
                "mismatch_fraction": count / actual.numel(), "max_absolute_error": None,
                "mean_absolute_error": None, "rmse": None, "relative_l2": None}
    return {"passed": bool(finite) and count == 0, "finite": bool(finite),
            "elements": actual.numel(), "mismatches": count,
            "mismatch_fraction": count / actual.numel(),
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "rmse": float(difference.square().mean().sqrt()),
            "relative_l2": float(torch.linalg.vector_norm(actual - expected) /
                                 torch.linalg.vector_norm(expected).clamp_min(1e-12))}


def check_stage_batch(engine, conditions, *, rtol=0.05, atol=0.05, micro_warmup=5, micro_measure=20):
    import torch

    original = engine.architecture.forward
    result = {"status": "not_run", "rtol": rtol, "atol": atol}

    def checked(actions, *positional, **keywords):
        prediction = original(actions, *positional, **keywords)
        batch = len(actions)
        if batch == 4 and result["status"] == "not_run":
            partitions = []
            for width in (1, 2):
                def sliced(value, index):
                    return value[index:index + width] if torch.is_tensor(value) and value.ndim and value.shape[0] == batch else value

                outputs = [original(actions[index:index + width],
                                    *(sliced(value, index) for value in positional),
                                    **{key: sliced(value, index) for key, value in keywords.items()})
                           for index in range(0, batch, width)]
                errors, modalities = [], {}
                for modality in (0, 1):
                    expected = torch.cat([output[modality] for output in outputs])
                    diagnostic = tensor_error(prediction[modality], expected, rtol=rtol, atol=atol)
                    modalities[("video", "action")[modality]] = diagnostic
                    errors.append(diagnostic["max_absolute_error"])
                partitions.append({"stage_batch": width, "max_absolute_error": errors,
                                   "modalities": modalities})
            passed = all(value["passed"] for partition in partitions for value in partition["modalities"].values())
            finite = all(value["finite"] for partition in partitions for value in partition["modalities"].values())
            if not finite:
                result.update(status="failed", partitions=partitions, microbenchmarks={},
                              reason="nonfinite_predictions")
                return prediction
            microbenchmarks = {}
            for width in (1, 2, 4):
                def sliced(value, index):
                    return value[index:index + width] if torch.is_tensor(value) and value.ndim and value.shape[0] == batch else value

                def run_partition():
                    for index in range(0, batch, width):
                        original(actions[index:index + width],
                                 *(sliced(value, index) for value in positional),
                                 **{key: sliced(value, index) for key, value in keywords.items()})

                for _ in range(micro_warmup):
                    run_partition()
                wall, cuda = [], []
                device = actions.device
                for _ in range(micro_measure):
                    torch.cuda.synchronize(device)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    started = time.perf_counter()
                    begin.record(torch.cuda.current_stream(device))
                    run_partition()
                    end.record(torch.cuda.current_stream(device))
                    torch.cuda.synchronize(device)
                    wall.append((time.perf_counter() - started) * 1000)
                    cuda.append(begin.elapsed_time(end))
                microbenchmarks[f"batch{width}"] = {"calls": batch // width,
                                                    "wall_ms": percentiles(wall),
                                                    "cuda_event_ms": percentiles(cuda)}
            result.update(status="passed" if passed else "failed", partitions=partitions, microbenchmarks=microbenchmarks,
                          micro_warmup=micro_warmup, micro_measure=micro_measure,
                          micro_boundary="same frozen warm stage candidates; forward only, not native trajectory rollout")
        return prediction

    engine.architecture.forward = checked
    try:
        engine.reset()
        for observation in conditions:
            engine.generate(observation)
            if result["status"] != "not_run":
                break
    finally:
        engine.architecture.forward = original
        engine.reset()
    if result["status"] == "not_run":
        raise RuntimeError("No warm batch4 was available for parity verification")
    return result


def replay(args):
    import torch
    from PIL import Image
    from openwam.deploy.engine import _BoundedPromptEmbedCache
    from openwam.deploy.parallel_joint import JointParaFlowSampler, core_identity
    from benchmarks.libero.replay_joint import load_observations

    observations = load_observations(args.trace)
    selected = selected_indices(observations, args.warmup, args.measure)
    chosen = set(selected)
    engine, checkpoint = build_engine(args)
    if args.require_exclusive:
        require_exclusive_device(args.device)
    hardware_before = hardware_snapshot()
    conditions = [{"first_frame_image": [Image.fromarray(image)], "prompt": str(prompt),
                   "proprio": state, "seed": 42}
                  for image, prompt, state in zip(*observations[:3])]
    parity = check_stage_batch(engine, conditions, micro_warmup=args.micro_warmup,
                               micro_measure=args.micro_measure)
    write_json(args.output / "parity.json", parity)
    if parity["status"] != "passed":
        finite = all(value["finite"] for partition in parity["partitions"]
                     for value in partition["modalities"].values())
        if not args.allow_parity_failure or not finite:
            raise RuntimeError("Stage-batch parity failed; full video/action diagnostics saved to parity.json. "
                               "Thresholds are unchanged. Use --allow-parity-failure only for a labeled "
                               "performance diagnostic, not an equivalence claim.")
        print("[joint-experiment] WARNING: stage-batch parity FAILED; continuing diagnostic timings only", flush=True)
    states = {}
    for name, (sampler, _, cached) in METHODS.items():
        states[name] = {
            "sampler": JointParaFlowSampler(cache_schedule=cached) if sampler != "native" else None,
            "prompt_cache": _BoundedPromptEmbedCache(), "vace_cache": {},
            "cpu_rng": torch.random.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(args.device),
        }
    rows = {name: [] for name in METHODS}
    partition_rows = []
    forward_shapes = []
    original_forward = engine.architecture.forward
    prep_peak = 0
    prep_reserved = 0

    def counted(actions, *positional, **keywords):
        video_shape = list(keywords["latents"].shape)
        patch = getattr(engine.architecture.video_backbone, "_dit_patch_size", None) if hasattr(engine.architecture, "video_backbone") else None
        video_tokens = int(np.prod([size // stride for size, stride in zip(video_shape[2:], patch)])) if patch else None
        forward_shapes.append({"action": list(actions.shape), "video": video_shape,
                               "context": list(keywords["context"].shape),
                               "video_patch_size": list(patch) if patch else None,
                               "video_tokens_per_stage": video_tokens,
                               "action_tokens_per_stage": actions.shape[1],
                               "joint_tokens_per_stage": video_tokens + actions.shape[1] if video_tokens else None})
        with torch.profiler.record_function("joint.forward"):
            return original_forward(actions, *positional, **keywords)

    engine.architecture.forward = counted
    previous_prompt = None
    original_samplers = []
    for state in states.values():
        sampler = state["sampler"]
        if sampler is not None:
            original_sample = sampler.sample

            def sample(*positional, _sample=original_sample, **keywords):
                nonlocal prep_peak, prep_reserved
                prep_peak = torch.cuda.max_memory_allocated(args.device)
                prep_reserved = torch.cuda.max_memory_reserved(args.device)
                with torch.profiler.record_function("joint.solver_and_forward"):
                    return _sample(*positional, **keywords)

            original_samplers.append((sampler, original_sample))
            sampler.sample = sample
    try:
        with (args.output / "requests.jsonl").open("x") as log:
            for index in range(selected[-1] + 1):
                if args.require_exclusive and index % 10 == 0:
                    require_exclusive_device(args.device)
                reset = bool(observations[3][index]) or conditions[index]["prompt"] != previous_prompt
                order = method_order(index)
                for position, name in enumerate(order):
                    state = states[name]
                    engine._joint_sampler = state["sampler"]
                    engine._prompt_embed_cache = state["prompt_cache"]
                    engine._vace_cache = state["vace_cache"]
                    engine.cfg.inference.denoise_steps = METHODS[name][1]
                    if reset:
                        engine.reset()
                    torch.random.set_rng_state(state["cpu_rng"])
                    torch.cuda.set_rng_state(state["cuda_rng"], args.device)
                    forward_shapes.clear()
                    prep_peak = 0
                    prep_reserved = 0
                    torch.cuda.synchronize(args.device)
                    torch.cuda.reset_peak_memory_stats(args.device)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    started = time.perf_counter()
                    begin.record(torch.cuda.current_stream(args.device))
                    result = engine.generate(conditions[index])
                    end.record(torch.cuda.current_stream(args.device))
                    torch.cuda.synchronize(args.device)
                    elapsed = (time.perf_counter() - started) * 1000
                    stats = result.get("sampler_stats")
                    if index in chosen and stats and stats["cold"]:
                        raise RuntimeError(f"Unexpected cold Joint request at paired index {index}")
                    state["cpu_rng"] = torch.random.get_rng_state()
                    state["cuda_rng"] = torch.cuda.get_rng_state(args.device)
                    row = {"index": index, "method": name, "order_position": position,
                           "measured": index in chosen, "reset": reset, "wall_ms": elapsed,
                           "cuda_event_ms": begin.elapsed_time(end), "forward_shapes": list(forward_shapes),
                           "peak_allocated_bytes": max(prep_peak, torch.cuda.max_memory_allocated(args.device)),
                           "peak_reserved_bytes": max(prep_reserved, torch.cuda.max_memory_reserved(args.device)),
                           "free_bytes": torch.cuda.mem_get_info(args.device)[0], "sampler_stats": stats}
                    rows[name].append(row)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                base, cached = states["joint_n4"]["sampler"], states["joint_cached_n4"]["sampler"]
                errors = []
                diagnostics = {}
                for label, actual, expected in [("action_history", cached.history, base.history),
                                                *[(f"video_phase_{key}", cached.video_phases[key], base.video_phases[key])
                                                  for key in base.video_phases]]:
                    diagnostic = tensor_error(actual, expected)
                    diagnostics[label] = diagnostic
                    if diagnostic["finite"]:
                        errors.append(diagnostic["max_absolute_error"])
                if not all(value["passed"] for value in diagnostics.values()):
                    schedule_grid = torch.tensor(base.signature[2], device=base.history.device, dtype=torch.float32)
                    expected_sigmas = (
                        schedule_grid[:, 0] / float(engine.architecture.video_scheduler.num_train_timesteps),
                        schedule_grid[:, 1] / float(engine.architecture.action_scheduler.num_train_timesteps),
                    )
                    cached_sigmas = next(iter(cached._schedule_cache.values()))[:2]
                    write_json(args.output / "cache_parity_failure.json",
                               {"status": "failed", "index": index, "rtol": 0.05, "atol": 0.05,
                                "trajectories": diagnostics,
                                "schedule_sigmas_equal": all(torch.equal(left, right)
                                                             for left, right in zip(cached_sigmas, expected_sigmas))})
                    raise RuntimeError("Cached trajectory parity failed; see cache_parity_failure.json. "
                                       "No thresholds were relaxed.")
                partition_rows.append({"index": index, "max_absolute_error": max(errors)})
                previous_prompt = conditions[index]["prompt"]
                if index % 10 == 0 or index == selected[-1]:
                    print(f"[joint-experiment] replay {index + 1}/{selected[-1] + 1}", flush=True)
    finally:
        engine.architecture.forward = original_forward
        for sampler, original in original_samplers:
            sampler.sample = original
    summary = {}
    for name, requests in rows.items():
        measured = [row for row in requests if row["measured"]]
        summary[name] = {"wall_ms": percentiles([row["wall_ms"] for row in measured]),
                         "cuda_event_ms": percentiles([row["cuda_event_ms"] for row in measured]),
                         "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in measured) / 2**30,
                         "cold_wall_ms": percentiles([row["wall_ms"] for row in requests
                                                      if row["sampler_stats"] and row["sampler_stats"]["cold"]])}
    paired = {}
    baseline = [rows["native_n10"][index]["wall_ms"] for index in selected]
    for name in METHODS:
        values = [rows[name][index]["wall_ms"] for index in selected]
        paired[name] = {"native_n10_speedup": percentiles([base / value for base, value in zip(baseline, values)]),
                        "native_n10_minus_method_ms": percentiles([base - value for base, value in zip(baseline, values)])}
    properties = torch.cuda.get_device_properties(args.device)
    if args.require_exclusive:
        require_exclusive_device(args.device)
    report = {"status": "completed", "measurement_status": "complete", "timing_valid": True,
              "numerical_equivalence_validated": parity["status"] == "passed",
              "interpretation": "validated_comparison" if parity["status"] == "passed" else "performance_diagnostic_only_parity_failed",
              "protocol": "single_loaded_model_per_observation_seeded_interleaving",
              "warmup": args.warmup, "measure": args.measure, "measured_indices": selected,
              "trace": str(args.trace.resolve()), "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
              "checkpoint": str(Path(args.ckpt_dir) / checkpoint), "core_identity": core_identity(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": str(properties), "device_uuid": str(getattr(properties, "uuid", "unavailable")),
              "torch": torch.__version__, "cuda": torch.version.cuda, "total_memory_bytes": properties.total_memory,
              "hardware_before": hardware_before, "hardware_after": hardware_snapshot(),
              "memory_boundary": "one resident model plus all methods' independent trajectory and condition caches",
              "methods": summary, "paired": paired, "stage_batch_parity": parity,
              "cache_trajectory_parity": {"status": "passed", "rtol": 0.05, "atol": 0.05,
                                          "requests": partition_rows},
              "timing_boundary": "composed RGB + proprio to engine action result; excludes camera composition/simulator",
              "closed_loop": "not_run", "architectural_fastpaths": "not_implemented"}
    write_json(args.output / "timing.json", report)
    return report


def capture(args):
    from openwam.deploy.request_trace import assemble_sessions
    from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource
    from benchmarks.libero.replay_joint import resolve_checkpoint

    checkpoint = resolve_checkpoint(args.ckpt_dir, args.ckpt_name)
    env = dict(os.environ)
    env.update(OPENWAM_REQUEST_TRACE_DIR=str(args.output / "capture_shards"),
               OPENWAM_TRACE_WARMUP=str(args.warmup), OPENWAM_TRACE_MEASURE=str(args.measure),
               CKPT_DIR=str(args.ckpt_dir), CKPT_NAME=checkpoint, GPUS=args.gpu,
               GPU=args.gpu, PORT=str(args.port), COMPARE_ROOT=str(args.output / "capture_eval"),
               REPLICAS_PER_GPU="1", BASE_PORT=str(args.port),
               OUTPUT_DIR=str(args.output / "capture_eval"), SERVER_PYTHON=sys.executable)
    command = ["bash", "scripts/eval_alpha_libero.sh", "eval", "--sampler", "native",
               "--denoise-steps", "4", "--inference-horizon", "8", "--inference-mode", "sync",
               "--denoise-mode", "sync", "--suites", args.suite, "--task-ids", str(args.task_id),
               "--num-trials", "50", "--extra-deploy-arg", "optimization.dit_cache.enabled=false",
               "--extra-deploy-arg", "optimization.decode_video=false"]
    if args.benchmark == "robotwin":
        command = ["bash", "scripts/compare_alpha_robotwin.sh", "diagnostic"]
        weights = sorted(Path(args.ckpt_dir).glob("checkpoint_step_*.safetensors"),
                         key=lambda path: int(path.stem.rsplit("_", 1)[1]))
        if not args.capture_command_json and (not weights or weights[-1].name != checkpoint):
            raise ValueError("Built-in RoboTwin collector selects the latest checkpoint; select that weight or provide a matching TRACE")
    if args.capture_command_json:
        command = json.loads(args.capture_command_json)
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            raise ValueError("capture-command-json must be a nonempty JSON argv list")
    registry = ProcessRegistry()
    try:
        handle = (args.output / "capture.log").open("x")
        process = spawn_resource(command, cwd=ROOT, env=env, stdout=handle)
        registry.add(process, handle)
        started = time.monotonic()
        progress_at = started - 30
        previous_count = -1
        print(f"[joint-experiment] collecting real E8 requests; log: {args.output / 'capture.log'}", flush=True)
        while True:
            count = sum(1 for _ in (args.output / "capture_shards").glob("session_*/request_*.npz"))
            metadata = None
            if count != previous_count:
                metadata = assemble_sessions(args.output / "capture_shards", args.trace,
                                             warmup=args.warmup, measure=args.measure)
                previous_count = count
            if metadata is not None:
                write_json(args.output / "capture.json", {"status": "completed", "benchmark": args.benchmark,
                                                         "command": command, "trace": str(args.trace), **metadata,
                                                         "checkpoint": str(Path(args.ckpt_dir) / checkpoint),
                                                         "closed_loop_accuracy": "not_measured; stopped after trace completion"})
                break
            if process.poll() is not None:
                raise RuntimeError(f"Capture exited {process.returncode} before trace completion; see capture.log")
            if time.monotonic() - started > args.capture_timeout:
                raise TimeoutError("Capture timeout; see capture.log and capture_eval/logs")
            if time.monotonic() - progress_at >= 30:
                print(f"[joint-experiment] capture: {count} requests written; waiting for "
                      f"{args.measure} eligible warm requests after warmup={args.warmup}", flush=True)
                progress_at = time.monotonic()
            time.sleep(1)
    finally:
        registry.terminate_all()


def report_markdown(output):
    report = json.loads((output / "timing.json").read_text())
    lines = ["# Joint performance experiment", "", f"Protocol: `{report['protocol']}`; same real observation indices.",
             "", "| Method | Requests | Wall P50 ms | Wall P95 ms | CUDA event P50 ms | Peak allocated GiB |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, result in report["methods"].items():
        wall = result["wall_ms"]
        lines.append(f"| {name} | {wall['count']} | {wall['p50']:.3f} | {wall['p95']:.3f} | "
                     f"{result['cuda_event_ms']['p50']:.3f} | {result['peak_allocated_gib']:.3f} |")
    lines += ["", f"Stage-batch parity: **{report['stage_batch_parity']['status']}** at rtol=atol=0.05. "
              "Cached trajectories: parity passed. If stage parity failed, times are diagnostic only, not proof of numerical equivalence.",
              "", "Timing excludes simulator/camera composition. CUDA event intervals may include GPU idle time;",
              "use the separate profiler for operator attribution. This replay does not measure success rate.",
              "Native N1 is a latency reference, not evidence of equivalent policy quality.",
              "Prepared conditioning, prefix prefill, CUDA Graph and multi-GPU are not enabled.",
              "", "See `timing.json` for paired speedups, cold starts, hashes and hardware metadata;",
              "`requests.jsonl` contains every request, order, shape and memory sample."]
    lines += ["", "## Frozen-stage microbenchmarks", "",
              "Same warm stage inputs, forward only; not sequential Native trajectory integration.", "",
              "| Stage batch | Calls | Wall P50 ms | CUDA event P50 ms |", "|---|---:|---:|---:|"]
    for name, values in report["stage_batch_parity"].get("microbenchmarks", {}).items():
        lines.append(f"| {name} | {values['calls']} | {values['wall_ms']['p50']:.3f} | {values['cuda_event_ms']['p50']:.3f} |")
    if (output / "profile.json").is_file():
        profile_data = json.loads((output / "profile.json").read_text())
        lines += ["", "## Separate Profiler", "", "Observed attention events:", "",
                  *[f"- `{name}`" for name in profile_data.get("attention_events", [])], "",
                  "Operator times aggregate the short cold + warm profile, not the main timing experiment.",
                  "See `operators.txt` and `joint_trace.json` for shapes and conditioning/FFN/cross-attention scopes."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def profile(args):
    import torch
    from PIL import Image
    from benchmarks.libero.replay_joint import load_observations

    observations = load_observations(args.trace)
    selected = selected_indices(observations, 0, min(3, args.measure))
    engine, _ = build_engine(args)
    with scoped_annotations(engine.architecture), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True) as profiler:
        for index in range(selected[-1] + 1):
            if observations[3][index]:
                engine.reset()
            with torch.profiler.record_function("engine.generate"):
                engine.generate({"first_frame_image": [Image.fromarray(observations[0][index])],
                                 "prompt": str(observations[1][index]), "proprio": observations[2][index], "seed": 42})
        torch.cuda.synchronize(args.device)
    profiler.export_chrome_trace(str(args.output / "joint_trace.json"))
    averages = profiler.key_averages(group_by_input_shape=True)
    table = averages.table(sort_by="self_device_time_total", row_limit=100)
    (args.output / "operators.txt").write_text(table)
    attention = sorted({event.key for event in averages if any(term in event.key.lower()
                        for term in ("scaled_dot_product", "flash", "fmha", "efficient_attention"))})
    operators = sorted(averages, key=lambda event: event.self_device_time_total, reverse=True)[:50]
    write_json(args.output / "profile.json", {"status": "completed", "timing_valid": False,
                                              "trace": "joint_trace.json", "operators": "operators.txt",
                                              "attention_events": attention,
                                              "top_operators": [{"name": event.key, "count": event.count,
                                                                 "self_device_time_us": event.self_device_time_total,
                                                                 "input_shapes": event.input_shapes}
                                                                for event in operators]})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("all", "capture", "timing", "profile", "summary", "plan", "replay-worker", "profile-worker"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--trace", type=Path)
    result.add_argument("--benchmark", choices=("libero", "robotwin"), default="libero")
    result.add_argument("--ckpt-dir", default="/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO")
    result.add_argument("--ckpt-name")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--gpu", default="0")
    result.add_argument("--port", type=int, default=8950)
    result.add_argument("--suite", default="spatial")
    result.add_argument("--task-id", type=int, default=0)
    result.add_argument("--warmup", type=int, default=48)
    result.add_argument("--measure", type=int, default=120)
    result.add_argument("--micro-warmup", type=int, default=5)
    result.add_argument("--micro-measure", type=int, default=20)
    result.add_argument("--capture-timeout", type=float, default=7200)
    result.add_argument("--capture-command-json")
    result.add_argument("--skip-profile", action="store_true")
    result.add_argument("--allow-parity-failure", action="store_true",
                        help="Explicitly continue finite batch mismatches for labeled performance diagnostics only")
    result.add_argument("--require-exclusive", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.warmup < 0 or args.measure < 1 or args.micro_warmup < 0 or args.micro_measure < 1
            or args.capture_timeout <= 0 or not 0 < args.port < 65536):
        raise ValueError("Invalid experiment limits")
    args.output = args.output.resolve()
    if args.trace:
        args.trace = args.trace.resolve()
    if args.mode == "plan":
        print("Collect real E8 requests if TRACE is absent -> stage partition parity -> single-model interleaved "
              "Native N1/N4/N10 + Joint N4/cached -> separate profiler -> JSON/Markdown report. "
              "No closed-loop accuracy campaign. Explicit GPU selection is your responsibility.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "summary":
        report_markdown(args.output)
        return
    if args.mode in ("replay-worker", "profile-worker"):
        if args.mode == "replay-worker":
            replay(args)
        else:
            profile(args)
        return
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "experiment.json").exists():
            raise FileExistsError("Experiment directory already used; choose a fresh PROFILE_ROOT")
        for name in ("timing.json", "requests.jsonl", "parity.json", "capture.log", "capture_eval",
                     "capture_shards", "profile.json", "joint_trace.json"):
            if (args.output / name).exists():
                raise FileExistsError(f"Refusing to overwrite existing experiment artifact: {name}")
        manifest = {"status": "running", "mode": args.mode, "benchmark": args.benchmark,
                    "trace_source": "provided" if args.trace else "automatic_closed_loop_capture",
                    "warmup": args.warmup, "measure": args.measure, "closed_loop_accuracy": "not_run"}
        write_json(args.output / "experiment.json", manifest)
        try:
            if args.mode in ("all", "capture") and args.trace is None:
                manifest["stage"] = "capture"
                write_json(args.output / "experiment.json", manifest)
                args.trace = args.output / "observations.npz"
                capture(args)
            if args.trace is None or not args.trace.is_file():
                raise ValueError("Supply an existing TRACE or use all/capture for automatic collection")
            if args.mode in ("all", "timing"):
                manifest["stage"] = "timing"
                write_json(args.output / "experiment.json", manifest)
                run_worker(args, "replay-worker")
                report_markdown(args.output)
            if args.mode == "profile" or (args.mode == "all" and not args.skip_profile):
                manifest["stage"] = "profile"
                write_json(args.output / "experiment.json", manifest)
                run_worker(args, "profile-worker")
                if (args.output / "timing.json").exists():
                    report_markdown(args.output)
            manifest.update(status="completed", stage="complete", trace=str(args.trace))
        except BaseException as error:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(args.output / "experiment.json", manifest)
        print(f"[joint-experiment] completed: {args.output}", flush=True)


def run_worker(args, mode):
    from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource

    command = [sys.executable, "-m", "benchmarks.libero.joint_experiment", mode,
               "--output", str(args.output), "--trace", str(args.trace), "--ckpt-dir", str(args.ckpt_dir),
               "--device", args.device, "--warmup", str(args.warmup), "--measure", str(args.measure)]
    command.extend(["--micro-warmup", str(args.micro_warmup), "--micro-measure", str(args.micro_measure)])
    if args.allow_parity_failure:
        command.append("--allow-parity-failure")
    if args.require_exclusive:
        command.append("--require-exclusive")
    if args.ckpt_name:
        command.extend(["--ckpt-name", args.ckpt_name])
    registry = ProcessRegistry()
    try:
        env = {key: value for key, value in os.environ.items() if key != "OPENWAM_REQUEST_TRACE_DIR"}
        print(f"[joint-experiment] starting {mode}", flush=True)
        process = spawn_resource(command, cwd=ROOT, env=env)
        registry.add(process)
        if process.wait() != 0:
            raise RuntimeError(f"{mode} failed; inspect console and experiment artifacts")
    finally:
        registry.terminate_all()


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    main()
