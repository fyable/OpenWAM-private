"""Development-only direct-reuse, replay and operator profiling CLI.

Examples: python -m benchmarks.action_cache.experiment prepare --e2-run ... --output ...
          python -m benchmarks.action_cache.experiment run --protocol ... --output ... --gpu 0 --deadline ...
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from openwam.research.protocol import digest, file_hash, read_sealed, seal, write_json
from scripts.research_e2_robustness import atomic_json
from .runtime import CacheConfig, cached_request

ROOT = Path(__file__).resolve().parents[2]


def code_identity():
    paths = sorted(Path(__file__).resolve().parent.glob("*.py"))
    paths += [ROOT / "scripts/action_cache_python.sh", ROOT / "scripts/research_action_cache.sh"]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def prepare(e2_run, output):
    e2_run, output = Path(e2_run).resolve(), Path(output).resolve()
    split = read_sealed(e2_run / "manifests/split.json")
    if {name: sum(t["split"] == name for t in split["tasks"]) for name in
            ("discovery", "calibration", "heldout")} != {"discovery": 8, "calibration": 8, "heldout": 24}:
        raise ValueError("Expected frozen 8/8/24 task split")
    selection = read_sealed(e2_run / "base/manifests/selection.json")
    environment = json.loads((e2_run / "environment.json").read_text())
    cases = [dict(name="native_n10", steps=10, cache=None),
             dict(name="refresh_all", steps=10, cache=asdict(CacheConfig(policy="refresh")))]
    for interval in (2, 3):
        cases.append(dict(name=f"fixed_{interval}", steps=10,
                          cache=asdict(CacheConfig(interval=interval, max_age=interval))))
        cases.append(dict(name=f"random_{interval}", steps=10,
                          cache=asdict(CacheConfig(policy="random", interval=interval, max_age=interval))))
    for threshold in (.03, .1, .3):
        cases.append(dict(name=f"change_{threshold:g}", steps=10,
                          cache=asdict(CacheConfig(policy="change", max_age=3, threshold=threshold))))
    for steps in (4, 6, 8):
        cases.append(dict(name=f"native_n{steps}", steps=steps, cache=None))
    # Coarse probes precede any sensitivity-driven scheduling. No extrapolation
    # from E2's two layers to unmeasured layers is permitted.
    probe_cases = []
    for layer in (0, 5, 11, 17, 23, 29):
        for step in (1, 4, 7):
            probe_cases.append(dict(name=f"probe_L{layer}_S{step}", steps=10,
                cache=asdict(CacheConfig(policy="probe", layers=(layer,), reuse_steps=(step,)))))
    output.mkdir(parents=True, exist_ok=False)
    protocol = seal({"version": "action_cache_feasibility_v1", "e2_run": str(e2_run),
        "task_split_sha256": split["sha256"], "selection_sha256": selection["sha256"],
        "checkpoint_dir": environment["checkpoint_dir"], "checkpoint_name": "checkpoint_step_10690.safetensors",
        "source_files": code_identity(), "research_source_sha256": environment["source_sha256"],
        "cases": cases, "probe_cases": probe_cases, "policy_seeds": [43, 44, 45],
        "primary_metric": "heldout closed-loop success versus total-request latency; replay is screening only",
        "heldout_locked": True, "bootstrap_seed": 20261119, "noninferiority_margin": .02,
        "pilot_speedup_target": 1.2, "paper_speedup_target": 1.5,
        "state_selection": "earliest saved selected state per task; 8 tasks, no outcome filtering",
        "invalidation": "request-local; one clean prefix frame; no cross-request state; original mask/positions retained",
        "validation": "refresh-all versus native N10; no diagnostic intervention; exactly ten forwards",
        "primary_hardware": "exclusive resident A100/H100; 3090 offload separately labelled",
        "closed_loop": {"tasks": 24, "trials_per_task": 50, "policy_seeds": [43, 44, 45],
                        "environment_seed": 42, "initial_state_protocol": "existing pinned LIBERO evaluator"},
        "research_status": "implementation prototype; real cache GPU validation and literature novelty review pending"})
    write_json(output / "protocol.json", protocol)
    audit_e2(e2_run, output / "e2_evidence.json")
    return protocol


def audit_e2(run, output):
    """CPU evidence audit; never infer unobserved robustness/control outcomes."""
    run = Path(run)
    result = {"created_at": datetime.now().astimezone().isoformat(), "baseline": {}, "t0": {},
              "robustness": "not_complete", "deployable_cache": "not_tested_by_E2"}
    for subset in ("discovery", "calibration"):
        root = run / f"base/diagnostics/e2_{subset}"
        directories = [root] if root.exists() else list((run / "parallel" / f"replay_{subset}").glob("worker_*"))
        rows = []
        for directory in directories:
            for path in directory.glob("*.json"):
                row = json.loads(path.read_text())
                if "method_id" in row:
                    rows.append(row)
        keys = {(row["state_id"], row["method_id"]) for row in rows}
        result["baseline"][subset] = {"completed": sum(row.get("status") == "completed" and row.get("valid_run", False) for row in rows),
            "expected": 2304, "duplicate_pairs": len(rows) - len(keys), "merged": root.exists()}
        certificate = run / f"base/diagnostics/t0_{subset}/t0_certificate.json"
        result["t0"][subset] = read_sealed(certificate)["status"] if certificate.exists() else "pending"
    for name in ("robustness", "extensions"):
        status = run / name / "status.json"
        result[name] = json.loads(status.read_text()) if status.exists() else {"status": "not_started"}
    result["decision"] = "Wait for direct-reuse and exclusive profiling before judging acceleration feasibility"
    atomic_json(output, result)
    return result


def selected_states(protocol, subset):
    from benchmarks.libero import research
    run = Path(protocol["e2_run"])
    split = read_sealed(run / "manifests/split.json")
    if split["sha256"] != protocol["task_split_sha256"] or subset not in {"discovery", "calibration"}:
        raise ValueError("Development-only replay; heldout access requires a frozen closed-loop manifest")
    bank, states = research.load_states(run / f"bank/state_bank/{subset}", split)
    selection = read_sealed(run / "base/manifests/selection.json")
    if selection["sha256"] != protocol["selection_sha256"]:
        raise ValueError("State selection changed")
    eligible = set(selection["state_ids_by_split"][subset])
    by_task = {}
    for entry, condition in sorted(states, key=lambda pair: (pair[0]["episode_id"], pair[0]["request_index"])):
        if entry["state_id"] in eligible:
            by_task.setdefault((entry["suite"], entry["task_id"]), (entry, condition))
    if len(by_task) != 8:
        raise ValueError("Screening must include all eight tasks")
    return bank, [by_task[key] for key in sorted(by_task)]


def fidelity(reference, actual, horizon):
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape or reference.ndim != 2 or reference.shape[1] != 10:
        raise ValueError("Expected matching native EEF10 action chunks")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise FloatingPointError("Nonfinite actions")
    before = np.stack([native_eef10_to_libero7d(row) for row in reference])
    after = np.stack([native_eef10_to_libero7d(row) for row in actual])
    return {"prefix_native_rmse": float(np.sqrt(np.mean((actual[:horizon] - reference[:horizon])**2))),
            "full_native_rmse": float(np.sqrt(np.mean((actual - reference)**2))),
            "prefix_native_max_abs": float(np.max(np.abs(actual[:horizon] - reference[:horizon]))),
            "prefix_gripper_command_disagreement": float(np.mean(before[:horizon, 6] != after[:horizon, 6]))}


def choose_gpu(index):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"], text=True)
    gpus = {int(parts[0].strip()): (parts[1].strip(), int(parts[2].strip()))
            for parts in (line.split(",") for line in rows.splitlines())}
    if index not in gpus:
        raise ValueError("GPU index unavailable")
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True).splitlines()
    if gpus[index][0] in busy:
        raise RuntimeError("GPU busy; refusing to share or displace existing experiments")
    return gpus[index]


def execute(args):
    # Set device visibility before importing Torch (this module's CacheConfig
    # imports torch but does not initialize CUDA).
    import fcntl
    protocol = read_sealed(args.protocol)
    if protocol["source_files"] != code_identity():
        raise ValueError("Experiment implementation changed since preparation; create a new protocol")
    gpu_uuid, total_mib = choose_gpu(args.gpu)
    lock_path = ROOT / f".cache/e2-reproduction-{gpu_uuid}.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid, MUJOCO_EGL_DEVICE_ID=str(args.gpu))
        import torch
        from benchmarks.libero import research
        from .profiling import operator_profile
        from benchmarks.video_parallel.runtime import capture_video
        from openwam.research.runner import paired_generate
        if research.source_identity()["source_sha256"] != protocol["research_source_sha256"]:
            raise ValueError("Underlying research/model source changed")
        bank, states = selected_states(protocol, args.subset)
        args.output.mkdir(parents=True, exist_ok=False)
        deadline = args.deadline.timestamp()
        def expired(signum, frame):
            raise TimeoutError("Feasibility experiment absolute deadline reached")
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(.01, deadline - time.time()))
        if args.memory_mode == "resident" and total_mib < 28000:
            raise ValueError("Resident primary profiling requires a larger-memory GPU; use explicitly labelled text-offload on 3090")
        os.environ["E2_MEMORY_MODE"] = args.memory_mode
        server_args = SimpleNamespace(ckpt_dir=protocol["checkpoint_dir"], ckpt_name=protocol["checkpoint_name"],
                                      device="cuda:0", steps=10, horizon=None)
        rows = []
        try:
            atomic_json(args.output / "status.json", {"status": "loading", "gpu_uuid": gpu_uuid, "deadline": args.deadline.isoformat()})
            server = research.build_server(server_args)
            identity = research.model_identity(server, server_args)
            saved = read_sealed(Path(protocol["e2_run"]) / f"bank/state_bank/{args.subset}/model.json")
            # Cross-hardware replay uses fixed observations, not the saved actions
            # as an identity-authorized baseline. Every reference is recomputed.
            if identity["assets"] != saved["assets"]:
                raise ValueError("Model assets differ from the source experiment")
            write_json(args.output / "identity.json", seal({"model": identity, "source_model_sha256": saved["sha256"],
                "protocol_sha256": protocol["sha256"], "task_split_sha256": protocol["task_split_sha256"],
                "bank_sha256": bank["sha256"], "subset": args.subset, "phase": args.phase,
                "source_files": code_identity(), "memory_mode": args.memory_mode,
                "gpu_uuid": gpu_uuid, "primary_performance_evidence": args.memory_mode == "resident",
                "closed_loop": "not_run", "policy_seeds": args.seeds, "warmup": args.warmup,
                "decode_video": args.decode_video,
                "request_state": "cold reset and prompt cache cleared for every paired execution"}))
            cases = protocol["probe_cases"] if args.phase == "probe" else protocol["cases"]
            sensitivities = None
            if args.sensitivity:
                sensitivity = read_sealed(args.sensitivity)
                if (sensitivity["fitted_split"] != "discovery" or sensitivity["protocol_sha256"] != protocol["sha256"]
                    or sensitivity["source_files"] != code_identity()):
                    raise ValueError("Sensitivity provenance mismatch")
                sensitivities = sensitivity["sensitivities"]
                layers = tuple(sorted({int(key.split(":")[0]) for key in sensitivities}))
                if args.phase != "pilot":
                    raise ValueError("Sensitivity policies belong to the pilot comparison")
                cases = list(cases)
                for policy in ("static", "adaptive"):
                    for threshold in (.03, .1, .3):
                        cases.append(dict(name=f"{policy}_{threshold:g}", steps=10,
                            cache=asdict(CacheConfig(policy=policy, layers=layers, threshold=threshold, max_age=3))))
                write_json(args.output / "sensitivity.json", sensitivity)
            if args.cases:
                names = set(args.cases.split(","))
                if names - {case["name"] for case in cases}:
                    raise ValueError("Unknown cases")
                cases = [case for case in cases if case["name"] in names]
            if args.phase == "probe" and args.subset != "discovery":
                raise ValueError("Direct sensitivity fitting uses Discovery only")
            # Always verify refresh-all, including for a narrowed pilot case list.
            entry, conditions = states[0]
            reference, _ = paired_generate(server.engine, conditions, seed=args.seeds[0], steps=10)
            with cached_request(server.engine, CacheConfig(policy="refresh")):
                check, _ = paired_generate(server.engine, conditions, seed=args.seeds[0], steps=10)
            validation = fidelity(reference["actions"], check["actions"], 8)
            passed = bool(np.allclose(reference["actions"], check["actions"], rtol=1e-4, atol=1e-5))
            write_json(args.output / "validation.json", {"status": "passed" if passed else "failed", **validation,
                "rtol": 1e-4, "atol": 1e-5, "state_id": entry["state_id"]})
            if not passed:
                raise RuntimeError("Refresh-all/native validation failed")
            # Instrumented native profile is separate from uninstrumented timing.
            if args.phase == "profile":
                for entry, conditions in states:
                    with operator_profile(server.engine, server.engine.architecture.device) as measured:
                        paired_generate(server.engine, conditions, seed=args.seeds[0], steps=10)
                    write_json(args.output / f"profile_{entry['state_id']}.json", measured)
            for entry, conditions in states:
                horizon = research.execution_horizon(server, conditions, server_args)
                for seed in args.seeds:
                    with capture_video(server.engine, decode=args.decode_video) as reference_capture:
                        reference, _ = paired_generate(server.engine, conditions, seed=seed, steps=10)
                    reference_video = reference_capture["latents"][:, :, 1:].detach().float().cpu()
                    for case in random.Random(seed + int(digest(entry["state_id"])[:8], 16)).sample(cases, len(cases)):
                        if time.time() >= deadline:
                            raise TimeoutError("Deadline reached between requests")
                        if args.phase == "profile" and case["name"] != "native_n10":
                            continue
                        for repetition in range(args.warmup + 1):
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            started = time.perf_counter()
                            stats = None
                            with capture_video(server.engine, decode=args.decode_video) as actual_capture:
                                if case["cache"] is None:
                                    actual, _ = paired_generate(server.engine, conditions, seed=seed, steps=case["steps"])
                                else:
                                    with cached_request(server.engine, CacheConfig(**case["cache"]), sensitivities=sensitivities) as reuse:
                                        actual, _ = paired_generate(server.engine, conditions, seed=seed, steps=10)
                                        stats = reuse.report()
                            torch.cuda.synchronize()
                            wall_ms = (time.perf_counter() - started) * 1000
                            peak = torch.cuda.max_memory_allocated()
                            if repetition < args.warmup:
                                continue
                            actual_video = actual_capture["latents"][:, :, 1:].detach().float().cpu()
                            if not torch.isfinite(actual_video).all():
                                raise FloatingPointError("Nonfinite cached future-video latents")
                            video_error = (actual_video - reference_video).square().mean().sqrt()
                            row = {"state_id": entry["state_id"], "episode_id": entry["episode_id"],
                                "suite": entry["suite"], "task_id": entry["task_id"], "subset": args.subset,
                                "seed": seed, "method": case["name"], "config": case,
                                "wall_ms": wall_ms, "peak_allocated_bytes": peak, "cache_stats": stats,
                                "metrics": fidelity(reference["actions"], actual["actions"], horizon),
                                "protocol_sha256": protocol["sha256"], "status": "completed"}
                            row["metrics"].update(future_video_rmse=float(video_error),
                                future_video_relative_rms=float(video_error / reference_video.square().mean().sqrt().clamp_min(1e-6)))
                            # Bound per-request timing includes resets and input preparation;
                            # do not call it steady-state server latency.
                            write_json(args.output / "results" / f"{digest([entry['state_id'], seed, case['name']])}.json", seal(row))
                            rows.append(row)
                            atomic_json(args.output / "status.json", {"status": "running", "completed": len(rows),
                                "expected": len(states) * len(args.seeds) * len(cases), "last_method": case["name"],
                                "updated_at": datetime.now().astimezone().isoformat()})
                            print(f"{len(rows)}: {entry['state_id']} seed={seed} {case['name']} {wall_ms:.1f} ms", flush=True)
            summarize(args.output, status="complete")
            atomic_json(args.output / "status.json", {"status": "complete", "completed": len(rows)})
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            status = "budget_exhausted" if isinstance(error, TimeoutError) else "failed"
            atomic_json(args.output / "status.json", {"status": status, "completed": len(rows), "error": str(error)})
            if (args.output / "identity.json").exists():
                summarize(args.output, status=status)
            if status == "failed":
                raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


def summarize(output, status=None):
    output = Path(output)
    identity = read_sealed(output / "identity.json")
    rows = [read_sealed(path) for path in (output / "results").glob("*.json")]
    by_method = defaultdict(list)
    references = {(r["state_id"], r["seed"]): r for r in rows if r["method"] == "native_n10"}
    for row in rows:
        by_method[row["method"]].append(row)
    methods = {}
    for name, group in by_method.items():
        paired = [(row, references[row["state_id"], row["seed"]]) for row in group
                  if (row["state_id"], row["seed"]) in references]
        methods[name] = {"requests": len(group), "tasks": len({(r["suite"], r["task_id"]) for r in group}),
            "wall_p50_ms": float(np.median([r["wall_ms"] for r in group])),
            "wall_p95_ms": float(np.percentile([r["wall_ms"] for r in group], 95)),
            "matched_requests": len(paired),
            "paired_speedup_p50": float(np.median([ref["wall_ms"] / row["wall_ms"] for row, ref in paired])) if paired else None,
            "prefix_native_rmse_p50": float(np.median([r["metrics"]["prefix_native_rmse"] for r in group])),
            "prefix_native_rmse_p95": float(np.percentile([r["metrics"]["prefix_native_rmse"] for r in group], 95)),
            "max_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in group)}
    result = {"status": status or json.loads((output / "status.json").read_text())["status"],
        "methods": methods, "primary_performance_evidence": identity["primary_performance_evidence"],
        "timing_boundary": "Cold paired request, including engine reset, condition copy and cache bookkeeping; profiling separate",
        "closed_loop": "not_run", "decision": "No success-rate or deployment-safety conclusion from replay alone"}
    atomic_json(output / "report.json", result)
    lines = ["# Action-aware video reuse feasibility", "", f"Status: {result['status']}",
             "", "Fixed-observation development screening; no heldout success or noninferiority claim.", "",
             "| Method | Requests | Tasks | P50 ms | P95 ms | Paired speedup | Prefix native RMSE P95 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in methods.items():
        speedup = values["paired_speedup_p50"]
        lines.append(f"| {name} | {values['requests']} | {values['tasks']} | {values['wall_p50_ms']:.2f} | {values['wall_p95_ms']:.2f} | {speedup if speedup is not None else 'unmatched'} | {values['prefix_native_rmse_p95']:.6g} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def fit(probe_run, output):
    root = Path(probe_run)
    identity = read_sealed(root / "identity.json")
    if identity["subset"] != "discovery" or identity["phase"] != "probe":
        raise ValueError("Sensitivity fitting requires a Discovery direct-reuse probe run")
    if json.loads((root / "status.json").read_text())["status"] != "complete":
        raise ValueError("Do not fit incomplete probe runs")
    rows = [read_sealed(path) for path in (root / "results").glob("*.json")]
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["protocol_sha256"] != identity["protocol_sha256"]:
            raise ValueError("Probe protocol changed")
        config = row["config"]["cache"]
        if config["policy"] != "probe" or len(config["layers"]) != 1 or len(config["reuse_steps"]) != 1:
            raise ValueError("Expected one-layer one-step direct probes")
        if not row["cache_stats"]["reused_layers"]:
            raise ValueError("Probe did not actually reuse a video block")
        cell = f"{config['layers'][0]}:{config['reuse_steps'][0] * 3 // 10}"
        cells[cell][row["suite"], row["task_id"]].append(row["metrics"]["prefix_native_rmse"])
    if not cells or any(len(tasks) != 8 for tasks in cells.values()):
        raise ValueError("Every sensitivity cell needs all eight Discovery tasks")
    scores = {cell: float(np.mean([np.mean(values) for values in tasks.values()])) for cell, tasks in cells.items()}
    positive = [v for v in scores.values() if v > 0]
    if not positive:
        raise ValueError("All effects are zero; risk ranking is not identifiable")
    scale = max(float(np.median(positive)), 1e-12)
    write_json(output, seal({"fitted_split": "discovery", "protocol_sha256": identity["protocol_sha256"],
        "task_split_sha256": identity["task_split_sha256"], "source_model_sha256": identity["source_model_sha256"],
        "measurement_model_sha256": identity["model"]["sha256"], "source_files": identity["source_files"],
        "sensitivities": {key: value / scale for key, value in scores.items()}, "raw_native_rmse": scores,
        "normalization": "median positive task-uniform direct reuse effect", "fitted_tasks": 8,
        "probe_files": {str(path.resolve()): file_hash(path) for path in sorted((root / "results").glob("*.json"))},
        "limitations": "Replay surrogate; coarse sampled layers only; no interpolation; not a safety guarantee"}))


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--e2-run", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("audit-e2")
    audit.add_argument("--e2-run", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--gpu", type=int, required=True)
    run.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    run.add_argument("--phase", choices=("pilot", "probe", "profile"), default="pilot")
    run.add_argument("--subset", choices=("discovery", "calibration"), default="discovery")
    run.add_argument("--memory-mode", choices=("resident", "text-offload"), default="resident")
    run.add_argument("--seeds", type=int, nargs="+", default=[43])
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--cases")
    run.add_argument("--sensitivity", type=Path, help="Sealed Discovery-only direct-reuse measurements")
    run.add_argument("--decode-video", action="store_true", help="Separate decode-on panel; VAE latency included")
    summary = sub.add_parser("summarize")
    summary.add_argument("--output", type=Path, required=True)
    fitting = sub.add_parser("fit")
    fitting.add_argument("--probe-run", type=Path, required=True)
    fitting.add_argument("--output", type=Path, required=True)
    next_stage = sub.add_parser("next", help="Run one stage of the five-hour experiment")
    next_stage.add_argument("--stage", choices=("warm", "overhead", "group"), required=True)
    next_stage.add_argument("--output", type=Path, required=True, help="Run root containing protocol/protocol.json")
    next_stage.add_argument("--gpu", type=int, required=True)
    next_stage.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    next_stage.add_argument("--memory-mode", choices=("resident", "text-offload"), default="text-offload")
    next_report = sub.add_parser("next-report", help="Summarize the available five-hour experiment stages")
    next_report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.e2_run, args.output)
    elif args.command == "audit-e2":
        audit_e2(args.e2_run, args.output)
    elif args.command == "run":
        if args.deadline.tzinfo is None or args.deadline.timestamp() <= time.time():
            parser.error("A future absolute deadline including timezone is required")
        if args.warmup < 0 or len(set(args.seeds)) != len(args.seeds) or any(seed not in (43, 44, 45) for seed in args.seeds):
            parser.error("Use nonnegative warmup and unique predeclared seeds 43/44/45")
        execute(args)
    elif args.command == "fit":
        fit(args.probe_run, args.output)
    elif args.command == "next":
        from .next_experiment import run_stage

        if args.deadline.tzinfo is None or args.deadline.timestamp() <= time.time():
            parser.error("A future absolute deadline including timezone is required")
        run_stage(args)
    elif args.command == "next-report":
        from .next_experiment import summarize_next

        summarize_next(args.output)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
