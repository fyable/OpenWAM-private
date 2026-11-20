"""Freeze development/heldout jobs and summarize paired closed-loop outcomes.

Only LIBERO is wired here. Second environments/models are explicit research
milestones, not fabricated equivalents of the present checkpoint.
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from openwam.research.protocol import file_hash, read_sealed, seal, write_json
from scripts.research_e2_robustness import atomic_json
from .experiment import ROOT, code_identity


def read_trials(manifest):
    """Require complete scheduler evidence and paired initial-state manifests."""
    records, missing = {}, []
    for job in manifest["jobs"]:
        summary_path = Path(job["output"]) / "summary.json"
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        summary = json.loads(summary_path.read_text())
        expected_tasks = set(job["task_ids"])
        actual_tasks = set()
        for task in summary["task_results"]:
            if task["suite"] != job["suite"] or task["task_id"] not in expected_tasks:
                raise ValueError("Unexpected task in closed-loop output")
            if task["task_id"] in actual_tasks:
                raise ValueError("Duplicate closed-loop task")
            actual_tasks.add(task["task_id"])
            if not task["complete"] or len(task["result_paths"]) != 1:
                missing.append(f"{job['id']} task={task['task_id']}")
                continue
            payload = json.loads(Path(task["result_paths"][0]).read_text())
            if (payload["suite"] != job["suite"] or payload["task_id"] != task["task_id"]
                or not payload.get("manifest_hash") or len(payload["trials"]) != manifest["trials_per_task"]):
                raise ValueError("Invalid closed-loop result provenance/coverage")
            found = set()
            for trial in payload["trials"]:
                index = trial["trial"]
                if index in found or index not in range(manifest["trials_per_task"]) or trial.get("init_state_index") != index:
                    raise ValueError("Duplicate/mismatched initial-state trial")
                if type(trial["success"]) is not bool:
                    raise ValueError("Success must be a boolean")
                found.add(index)
                key = (job["method"], job["suite"], task["task_id"], job["policy_seed"], index)
                if key in records:
                    raise ValueError("Duplicate paired episode")
                records[key] = dict(success=trial["success"], manifest_hash=payload["manifest_hash"])
        if actual_tasks != expected_tasks:
            missing.append(f"{job['id']} missing tasks")
    return records, missing


def paired_interval(differences, replicates=10000, seed=20261119):
    """Task-uniform mean; resample tasks then initial states, keeping seeds paired."""
    grouped = defaultdict(lambda: defaultdict(list))
    for (suite, task, policy_seed, initial), value in differences.items():
        grouped[suite, task][initial].append(value)
    if not grouped:
        return None
    clusters = [np.asarray([np.mean(values) for _, values in sorted(initials.items())])
                for _, initials in sorted(grouped.items())]
    generator = np.random.default_rng(seed)
    sampled_tasks = generator.integers(0, len(clusters), (replicates, len(clusters)))
    estimates = np.zeros(replicates)
    for column in range(len(clusters)):
        for index, values in enumerate(clusters):
            mask = sampled_tasks[:, column] == index
            number = int(mask.sum())
            if number:
                samples = generator.integers(0, len(values), (number, len(values)))
                estimates[mask] += values[samples].mean(axis=1) / len(clusters)
    return {"mean": float(np.mean([cluster.mean() for cluster in clusters])),
            "ci95": [float(v) for v in np.percentile(estimates, [2.5, 97.5])],
            "tasks": len(clusters), "episodes": len(differences), "replicates": replicates,
            "sampling": "task then initial state; policy seeds averaged within initial state"}


def summarize(manifest_path, output):
    manifest = read_sealed(manifest_path)
    records, missing = read_trials(manifest)
    expected = len(manifest["tasks"]) * len(manifest["seeds"]) * manifest["trials_per_task"]
    methods = {}
    for case in manifest["cases"]:
        name = case["name"]
        rows = {key[1:]: value for key, value in records.items() if key[0] == name}
        reference = {key[1:]: value for key, value in records.items() if key[0] == "native_n10"}
        common = set(rows) & set(reference)
        differences = {}
        for key in common:
            if rows[key]["manifest_hash"] != reference[key]["manifest_hash"]:
                raise ValueError("Candidate/reference initial-state manifest mismatch")
            differences[key] = int(rows[key]["success"]) - int(reference[key]["success"])
        interval = paired_interval(differences, seed=manifest["bootstrap_seed"])
        complete = len(rows) == expected and len(common) == expected
        methods[name] = {"complete": complete, "episodes": len(rows), "expected": expected,
            "successes": sum(row["success"] for row in rows.values()),
            "success_rate": float(np.mean([row["success"] for row in rows.values()])) if rows else None,
            "paired_difference": interval, "regressions": sum(value < 0 for value in differences.values()),
            "improvements": sum(value > 0 for value in differences.values()),
            "noninferior": bool(complete and interval and interval["ci95"][0] >= -manifest["noninferiority_margin"])
                if manifest["stage"] == "heldout" else None}
    result = {"manifest_sha256": manifest["sha256"], "stage": manifest["stage"],
        "status": "complete" if not missing and all(row["complete"] for row in methods.values()) else "incomplete",
        "missing": missing, "methods": methods, "noninferiority_margin": manifest["noninferiority_margin"],
        "timing": "Closed-loop SR evidence only; use separate exclusive same-hardware replay for speedups",
        "interval": "Two-sided 95% paired hierarchical bootstrap; no confirmatory claim from development screen"}
    atomic_json(output, result)
    return result


def plan(args):
    pilot = Path(args.pilot_run).resolve()
    identity = read_sealed(pilot / "identity.json")
    if identity["subset"] != "calibration" or identity["phase"] != "pilot":
        raise ValueError("Choose deploy configurations from a Calibration pilot, not Discovery or heldout")
    if json.loads((pilot / "status.json").read_text())["status"] != "complete":
        raise ValueError("Complete Calibration replay before selecting candidates")
    if identity["source_files"] != code_identity():
        raise ValueError("Implementation changed since pilot")
    rows = [read_sealed(p) for p in (pilot / "results").glob("*.json")]
    cases, coverage = {}, defaultdict(set)
    for row in rows:
        name = row["method"]
        if name in cases and cases[name] != row["config"]:
            raise ValueError("Method config changed within pilot")
        cases[name] = row["config"]
        coverage[name].add((row["suite"], row["task_id"], row["seed"]))
    requested = list(dict.fromkeys(["native_n10", *args.methods.split(",")]))
    if args.candidate not in requested or args.candidate == "native_n10":
        raise ValueError("Include the explicitly selected candidate in --methods")
    if any(name not in cases or len(coverage[name]) != 8 * 3 for name in requested):
        raise ValueError("Each selected method needs all 8 Calibration tasks and seeds 43/44/45")
    if not {"native_n4", "native_n6", "native_n8"} <= set(requested):
        raise ValueError("Keep N4/N6/N8 quality/latency baselines in the frozen comparison")
    if not any(name.startswith("fixed_") for name in requested) or not any(name.startswith("change_") for name in requested):
        raise ValueError("Include fixed-refresh and feature-change cache baselines")
    protocol = read_sealed(args.protocol)
    if identity["protocol_sha256"] != protocol["sha256"]:
        raise ValueError("Pilot/protocol mismatch")
    split = read_sealed(Path(protocol["e2_run"]) / "manifests/split.json")
    if split["sha256"] != protocol["task_split_sha256"]:
        raise ValueError("Task split changed")
    if args.stage == "heldout":
        if args.screen is None:
            raise ValueError("Heldout freeze requires a completed Calibration closed-loop screen")
        screen = read_sealed(args.screen)
        report = summarize(args.screen, Path(args.screen).with_name("quality_report.json"))
        if (screen["stage"] != "screen" or report["status"] != "complete"
            or screen["candidate"] != args.candidate or screen["cases"] != [cases[n] for n in requested]
            or screen["protocol_sha256"] != protocol["sha256"] or screen["source_files"] != code_identity()):
            raise ValueError("Heldout requires the same frozen methods and complete screen")
    tasks = [task for task in split["tasks"] if task["split"] == ("heldout" if args.stage == "heldout" else "calibration")]
    if len(tasks) != (24 if args.stage == "heldout" else 8):
        raise ValueError("Wrong task coverage")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(protocol["checkpoint_dir"])
    jobs = []
    by_suite = defaultdict(list)
    for task in tasks:
        by_suite[task["suite"]].append(task["task_id"])
    trials = 50 if args.stage == "heldout" else 5
    for name in requested:
        for seed in (43, 44, 45):
            for suite, ids in sorted(by_suite.items()):
                job_output = output / "runs" / name / str(seed) / suite
                job_id = f"{name}_{seed}_{suite}"
                command = ["bash", str(ROOT / "scripts/eval_alpha_libero.sh"), "eval", "--sampler", "native",
                    "--denoise-steps", str(cases[name]["steps"]), "--inference-horizon", "8",
                    "--inference-mode", "sync", "--denoise-mode", "sync", "--suites", suite,
                    "--task-ids", ",".join(map(str, sorted(ids))), "--num-trials", str(trials), "--seed", "42",
                    "--extra-deploy-arg", "inference.cfg_scale=1.0",
                    "--extra-deploy-arg", "optimization.dit_cache.enabled=false",
                    "--extra-deploy-arg", "optimization.decode_video=false",
                    "--extra-deploy-arg", "inference.request_timing=true"]
                if args.stage == "screen":
                    command.append("--diagnostic")
                jobs.append({"id": job_id, "method": name, "policy_seed": seed, "suite": suite,
                    "task_ids": sorted(ids), "output": str(job_output), "command": command,
                    "environment": {"SERVER_PYTHON": str(ROOT / "scripts/action_cache_python.sh"),
                        "ACTION_CACHE_PYTHON": sys.executable, "ACTION_CACHE_CASE": json.dumps(cases[name]),
                        "ACTION_CACHE_POLICY_SEED": str(seed), "CKPT_DIR": str(checkpoint),
                        "CKPT_NAME": protocol["checkpoint_name"], "GPUS": args.gpus,
                        "REPLICAS_PER_GPU": "1", "OUTPUT_DIR": str(job_output), "E2_MEMORY_MODE": args.memory_mode,
                        "LIBERO_PYTHON": str(args.libero_python.resolve()),
                        "ACTION_CACHE_SENSITIVITY": str(pilot / "sensitivity.json") if (pilot / "sensitivity.json").exists() else ""}})
    write_json(output / "manifest.json", seal({"version": "action_cache_closed_loop_v1", "stage": args.stage,
        "protocol_sha256": protocol["sha256"], "candidate": args.candidate, "cases": [cases[n] for n in requested],
        "source_files": code_identity(), "tasks": tasks, "seeds": [43, 44, 45], "trials_per_task": trials,
        "environment_seed": 42, "noninferiority_margin": .02, "bootstrap_seed": 20261119,
        "task_split_sha256": split["sha256"], "jobs": jobs,
        "pilot_identity_sha256": identity["sha256"], "memory_mode": args.memory_mode,
        "asset_hashes": identity["model"]["assets"],
        "screen_manifest_sha256": read_sealed(args.screen)["sha256"] if args.screen else None,
        "sensitivity_sha256": file_hash(pilot / "sensitivity.json") if (pilot / "sensitivity.json").exists() else None,
        "selection": "Explicit developer-chosen configuration; no heldout tuning permitted"}))


def execute_locked(args):
    manifest = read_sealed(args.manifest)
    if manifest["source_files"] != code_identity():
        raise ValueError("Code changed since closed-loop freeze")
    from benchmarks.libero.research import assets_identity
    checkpoint = manifest["jobs"][0]["environment"]["CKPT_DIR"]
    from types import SimpleNamespace
    if assets_identity(SimpleNamespace(ckpt_dir=checkpoint)) != manifest["asset_hashes"]:
        raise ValueError("Checkpoint assets changed since frozen pilot")
    root = Path(args.manifest).parent
    deadline = args.deadline.timestamp()
    def expired(signum, frame):
        raise TimeoutError("Closed-loop absolute deadline reached")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline - time.time() - 15))
    try:
        for job in manifest["jobs"]:
            environment = {**os.environ, **job["environment"]}
            if environment["ACTION_CACHE_SENSITIVITY"] and file_hash(environment["ACTION_CACHE_SENSITIVITY"]) != manifest["sensitivity_sha256"]:
                raise ValueError("Frozen sensitivity file changed")
            if time.time() >= deadline - 15:
                raise TimeoutError("Deadline reached before dispatch")
            atomic_json(root / "status.json", {"status": "running", "job": job["id"], "deadline": args.deadline.isoformat()})
            with (root / f"{job['id']}.log").open("a") as stream:
                process = subprocess.Popen(job["command"], cwd=ROOT, env=environment, start_new_session=True,
                                           stdout=stream, stderr=subprocess.STDOUT)
                process.wait(timeout=max(1, deadline - time.time() - 15))
                if process.returncode:
                    raise RuntimeError(f"Closed-loop job failed: {job['id']}; inspect log")
        result = summarize(args.manifest, root / "quality_report.json")
        atomic_json(root / "status.json", {"status": result["status"]})
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        from scripts.reproduce_e2_fresh import stop_descendants
        stop_descendants(os.getpid(), grace=5)
        atomic_json(root / "status.json", {"status": "budget_exhausted" if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else "failed", "error": str(error)})
        summarize(args.manifest, root / "quality_report.json")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def execute(args):
    import fcntl
    from contextlib import ExitStack
    from .experiment import choose_gpu
    manifest = read_sealed(args.manifest)
    requested = [int(index) for index in manifest["jobs"][0]["environment"]["GPUS"].split(",")]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Use unique physical GPU indices")
    with ExitStack() as stack:
        for index in requested:
            gpu_uuid, total = choose_gpu(index)
            if manifest["memory_mode"] == "resident" and total < 28000:
                raise ValueError("Resident evaluation requires larger-memory GPU")
            lock = stack.enter_context((ROOT / f".cache/e2-reproduction-{gpu_uuid}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute_locked(args)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("plan")
    prep.add_argument("--protocol", type=Path, required=True)
    prep.add_argument("--pilot-run", type=Path, required=True)
    prep.add_argument("--candidate", required=True)
    prep.add_argument("--methods", required=True)
    prep.add_argument("--stage", choices=("screen", "heldout"), default="screen")
    prep.add_argument("--screen", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--gpus", default="0")
    prep.add_argument("--memory-mode", choices=("resident", "text-offload"), default="resident")
    prep.add_argument("--libero-python", type=Path, default=ROOT / ".venv-openwam/bin/python")
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args)
    elif args.command == "summarize":
        summarize(args.manifest, args.output)
    else:
        if args.deadline.tzinfo is None or args.deadline.timestamp() <= time.time() + 30:
            parser.error("Future absolute deadline including timezone required")
        execute(args)


if __name__ == "__main__":
    main()
