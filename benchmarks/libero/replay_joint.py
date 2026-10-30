"""Offline fixed-observation timing; never steps a simulator or executes actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from openwam.deploy.server import _load_deploy_yaml, build_server_from_config


def load_observations(path):
    with np.load(path, allow_pickle=False) as data:
        images = data["images"]
        prompts = data["prompts"]
        states = data["states"]
        starts = data["episode_starts"]
    count = len(images)
    if count < 3 or images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("images must be uint8 [N,H,W,3], N >= 3")
    if prompts.shape != (count,) or prompts.dtype.kind not in "US":
        raise ValueError("prompts must be a non-object string array [N]")
    if states.ndim != 2 or len(states) != count or not np.isfinite(states).all():
        raise ValueError("states must be finite [N,D]")
    if starts.shape != (count,) or starts.dtype != np.bool_ or not starts[0]:
        raise ValueError("episode_starts must be bool [N] with first entry true")
    return images, prompts, states, starts


def summarize(rows, key):
    values = [row[key] for row in rows]
    return None if not values else {
        "count": len(values), "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def select_warm_rows(rows, warmup):
    eligible = [row for row in rows if row["index"] >= warmup and
                not row["episode_start"] and
                not row.get("protocol_cold", False) and
                (not row["sampler_stats"] or not row["sampler_stats"]["cold"])]
    return eligible


def resolve_checkpoint(directory, name):
    if name:
        path = Path(directory) / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return name
    candidates = sorted(Path(directory).glob("checkpoint*.safetensors"))
    if len(candidates) != 1:
        raise ValueError("Specify --ckpt-name when checkpoint discovery is ambiguous or empty")
    return candidates[0].name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--ckpt-name")
    parser.add_argument("--sampler", choices=("native", "joint-paraflow"), required=True)
    parser.add_argument("--steps", type=int, choices=(1, 4, 10), default=4)
    parser.add_argument("--warmup", type=int, default=48)
    parser.add_argument("--measure", type=int, default=120)
    parser.add_argument("--profile", type=Path, help="Export a separate CUDA/CPU trace; not a timing run")
    parser.add_argument("--cache-schedule", action="store_true", help="Opt-in Joint adapter schedule cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-batch", action="store_true",
                        help="Compare the first warm stage batch with sequential forwards; timings include verification")
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=0.05)
    args = parser.parse_args()
    if args.warmup < 0 or args.measure < 1:
        parser.error("warmup must be nonnegative and measure positive")
    if args.sampler == "joint-paraflow" and args.steps == 1:
        parser.error("Joint requires at least two denoising stages")
    if args.profile and args.profile.exists():
        raise FileExistsError(args.profile)
    args.ckpt_name = resolve_checkpoint(args.ckpt_dir, args.ckpt_name)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    observations = load_observations(args.observations)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU replay not_run: CUDA unavailable")
    cfg = _load_deploy_yaml()
    for key, value in {
        "inference.sampler": args.sampler, "inference.denoise_steps": args.steps,
        "inference.inference_horizon": 8, "inference.inference_mode": "sync",
        "inference.denoise_mode": "sync", "optimization.compile.enabled": False,
        "optimization.dit_cache.enabled": False, "optimization.decode_video": False,
    }.items():
        OmegaConf.update(cfg, key, value)
    server = build_server_from_config(cfg, args.ckpt_dir, device=args.device, ckpt_name=args.ckpt_name)
    engine = server.engine
    architecture = engine.architecture
    original_forward = architecture.forward
    batches = []
    forward_shapes = []
    sampler = getattr(engine, "_joint_sampler", None)
    if sampler:
        sampler.cache_schedule = args.cache_schedule
    original_sample = sampler.sample if sampler else None
    preprocessing_peak = 0
    parity = {"status": "not_run", "rtol": args.rtol, "atol": args.atol}

    def measured_sample(*positional, **keywords):
        nonlocal preprocessing_peak
        preprocessing_peak = torch.cuda.max_memory_allocated(args.device)
        return original_sample(*positional, **keywords)

    def counted_forward(actions, *positional, **keywords):
        batches.append(len(actions))
        forward_shapes.append({
            "actions": list(actions.shape),
            "latents": list(keywords["latents"].shape) if "latents" in keywords else None,
            "context": list(keywords["context"].shape) if torch.is_tensor(keywords.get("context")) else None,
        })
        with torch.profiler.record_function("joint.forward"):
            predictions = original_forward(actions, *positional, **keywords)
        if args.verify_batch and len(actions) > 1 and parity["status"] == "not_run":
            batch = len(actions)

            def select(value, index, width):
                if torch.is_tensor(value) and value.ndim and value.shape[0] == batch:
                    return value[index:index + width]
                return value

            partitions = []
            for width in sorted({1, min(2, batch), batch}):
                sequential = []
                torch.cuda.synchronize(args.device)
                started = time.perf_counter()
                for index in range(0, batch, width):
                    batches.append(len(actions[index:index + width]))
                    sequential.append(original_forward(
                        actions[index:index + width],
                        *(select(value, index, width) for value in positional),
                        **{key: select(value, index, width) for key, value in keywords.items()},
                    ))
                torch.cuda.synchronize(args.device)
                elapsed = (time.perf_counter() - started) * 1000
                errors = []
                for modality in (0, 1):
                    expected = torch.cat([value[modality] for value in sequential])
                    torch.testing.assert_close(predictions[modality], expected, rtol=args.rtol, atol=args.atol)
                    errors.append(float((predictions[modality].float() - expected.float()).abs().max()))
                partitions.append({"stage_batch": width, "calls": len(sequential),
                                   "diagnostic_ms": elapsed, "max_absolute_error": errors})
            parity.update(status="passed", partitions=partitions,
                          max_absolute_error=partitions[0]["max_absolute_error"],
                          timing_note="single diagnostic sample; not a steady-state speed comparison")
        return predictions

    architecture.forward = counted_forward
    if sampler:
        sampler.sample = measured_sample
    rows = []
    episode_request = 0
    previous_prompt = None
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) if args.profile else None
    try:
        if profiler is not None:
            profiler.start()
        for index, (image, prompt, state, start_episode) in enumerate(zip(*observations)):
            if start_episode:
                engine.reset()
            if start_episode or str(prompt) != previous_prompt:
                episode_request = 0
            batches.clear()
            forward_shapes.clear()
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            result = engine.generate({
                "first_frame_image": [Image.fromarray(image)], "prompt": str(prompt),
                "proprio": state, "seed": 42,
            })
            torch.cuda.synchronize(args.device)
            rows.append({
                "index": index, "episode_start": bool(start_episode),
                "protocol_cold": episode_request < 2,
                "full_request_ms": (time.perf_counter() - started) * 1000,
                "cuda_peak_allocated_bytes": max(preprocessing_peak, torch.cuda.max_memory_allocated(args.device)),
                "forward_batches": list(batches), "sampler_stats": result.get("sampler_stats"),
                "forward_shapes": list(forward_shapes),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
                "cuda_free_bytes": torch.cuda.mem_get_info(args.device)[0],
            })
            episode_request += 1
            previous_prompt = str(prompt)
            if len(select_warm_rows(rows, args.warmup)) >= args.measure:
                break
    finally:
        if profiler is not None:
            profiler.stop()
        architecture.forward = original_forward
        if sampler:
            sampler.sample = original_sample
    if profiler is not None:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.profile))
    warm = select_warm_rows(rows, args.warmup)
    report = {
        "status": "completed", "sampler": args.sampler, "steps": args.steps,
        "horizon": 32, "advance": 8, "checkpoint": str(Path(args.ckpt_dir) / args.ckpt_name),
        "observations": str(args.observations.resolve()),
        "observations_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
        "paraflow_core": sampler.identity if sampler else None,
        "torch": torch.__version__, "device": torch.cuda.get_device_name(args.device),
        "all_requests": summarize(rows, "full_request_ms"),
        "warm_requests": summarize(warm, "full_request_ms"), "requests": rows,
        "closed_loop": "not_run",
        "measurement_status": "complete" if len(warm) == args.measure else "insufficient_warm_observations",
        "warmup_requests": args.warmup, "requested_measurements": args.measure,
        "timing_valid": not args.verify_batch and profiler is None,
        "profile": str(args.profile) if args.profile else None,
        "cache_schedule": args.cache_schedule,
        "timing_boundary": "composed RGB canvas + proprio to normalized action result; excludes simulator/camera composition",
        "stage_batch_parity": parity, "timings_include_parity_verification": args.verify_batch,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
