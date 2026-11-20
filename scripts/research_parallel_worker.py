"""One GPU worker for the E2 replication coordinator; no shared output writes."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = os.environ.get("E2_CPU_THREADS_PER_WORKER", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.libero import research
from openwam.research.protocol import digest, read_sealed, seal, write_json
from scripts.research_e2_robustness import atomic_json, verify_result


def serialize_model_loading():
    # Each replica temporarily allocates much more host RAM while loading.
    # Serialize that peak while allowing inference on all GPUs concurrently.
    original = research.build_server

    def build(args):
        import torch

        gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        deadline = float(os.environ.get("E2_TOTAL_DEADLINE", "inf"))
        print(f"Waiting for serialized model loading: GPU {gpu}", flush=True)
        with (ROOT / ".cache/e2-model-loading.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            failures = 0
            while time.time() < deadline:
                free = int(subprocess.check_output([
                    "nvidia-smi", "-i", gpu, "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"], text=True).strip())
                minimum = int(os.environ.get("E2_MIN_FREE_MIB", "28000"))
                if free < minimum:
                    print(f"Waiting inside model-loading lock: GPU {gpu}, free={free} MiB", flush=True)
                    time.sleep(10)
                    continue
                # Retry only a resource allocation failure during loading. Numerical
                # checks and inference failures still stop the experiment.
                try:
                    print(f"Loading model: GPU {gpu}; memory_mode={os.environ.get('E2_MEMORY_MODE', 'resident')}", flush=True)
                    server = original(args)
                    print(f"Model ready: GPU {gpu}; allocated={torch.cuda.memory_allocated() // 1048576} MiB", flush=True)
                    return server
                except torch.OutOfMemoryError as error:
                    print(f"GPU memory changed while loading; waiting to retry: {error}", flush=True)
                    failures += 1
                    if failures >= 3:
                        raise
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(10)
            raise TimeoutError("Total experiment deadline reached while waiting for model memory")

    research.build_server = build


def validate_cross_gpu_checks(run, model_sha256):
    records = run / "parallel/cross_gpu_checks.json"
    if not records.exists():
        return
    bank = read_sealed(run / "bank/state_bank/discovery/manifest.json")
    for worker in read_sealed(records)["workers"]:
        certificate_path = Path(worker["output"]) / "t0_certificate.json"
        while not certificate_path.exists():
            os.kill(worker["pid"], 0)
            if Path(f"/proc/{worker['pid']}/stat").read_text().split()[2] == "Z":
                raise RuntimeError("Cross-GPU T0 exited without a certificate")
            time.sleep(5)
        certificate = read_sealed(certificate_path)
        if (certificate["status"] != "passed" or certificate["model_sha256"] != model_sha256
            or certificate["bank_sha256"] != bank["sha256"]
            or certificate["states"] < 20 or certificate["seeds"] != [43, 44, 45]):
            raise ValueError(f"Cross-GPU T0 failed on GPU {worker['gpu']}")


def replay_shard(args):
    from openwam.research.exploratory import validate_bank_identity, validate_coverage
    from openwam.research.runner import execute_method, infrastructure_retries

    args.device, args.steps, args.horizon = "cuda:0", 10, None
    args.source_review = None
    split = read_sealed(args.split)
    bank, states = research.load_states(args.bank, split)
    plan = read_sealed(args.plan)
    server = research.build_server(args)
    identity = research.model_identity(server, args)
    validate_cross_gpu_checks(args.output.parents[2], identity["sha256"])
    validate_bank_identity(args, bank, identity)
    research.validate_t0(read_sealed(args.t0), identity, bank, 10)
    if plan["task_split_sha256"] != split["sha256"]:
        raise ValueError("Replay plan/split mismatch")
    mapping = {entry["state_id"]: (entry, condition) for entry, condition in states}
    selected = [mapping[state] for state in plan["state_ids"]]
    validate_coverage([entry for entry, _ in selected], plan.get("sampling_protocol"))
    if any(entry["split"] != plan["selected_split"] for entry, _ in selected):
        raise ValueError("Replay plan contains states from another split")
    selected = selected[args.shard_index::args.shard_count]
    horizon = research.execution_horizon(server, selected[0][1], args)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", seal({
        "model": identity, "bank_sha256": bank["sha256"], "task_split_sha256": split["sha256"],
        "seeds": [43], "source_review": None,
    }))
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "worker.json", seal({
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "state_ids": [entry["state_id"] for entry, _ in selected],
        "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "plan_sha256": plan["sha256"],
    }))
    completed = 0
    for entry, conditions in selected:
        for method in plan["methods"]:
            row = {**entry, "method_id": method["method_id"], "action_seed": 43, "video_seed": 43,
                   "steps": 10, "prefix_horizon": horizon, "success": None,
                   "plan_sha256": plan["sha256"], "model_sha256": identity["sha256"]}
            result, attempts = infrastructure_retries(lambda: execute_method(
                server.engine, conditions, method, seed=43, steps=10, horizon=horizon))
            row.update(valid_run=True, status="completed", retry_attempts=attempts, **result)
            verify_result(row, entry["state_id"], method["method_id"], identity["sha256"])
            name = digest([entry["state_id"], 43, method["method_id"]]) + ".json"
            write_json(args.output / name, row)
            completed += 1
            if completed == 1 or completed % 48 == 0:
                atomic_json(args.output / "progress.json", {"completed_pairs": completed,
                            "total_pairs": len(selected) * len(plan["methods"])})
                print(f"replay shard {args.shard_index}: {completed}/{len(selected)*len(plan['methods'])}", flush=True)


def robustness_shard(args):
    from scripts.research_e2_robustness import run_study

    spec = read_sealed(args.plan)
    budget = Path(spec["base"]).parent / "budget.json"
    if budget.exists():
        args.deadline = min(args.deadline, read_sealed(budget)["deadline_unix"])
    validate_cross_gpu_checks(Path(spec["base"]).parent, spec["model_sha256_by_split"]["discovery"])
    args.run = args.output
    args.run.mkdir(parents=True, exist_ok=False)
    args.base, args.bank_run = Path(spec["base"]), Path(spec["bank_run"])
    args.ckpt_dir, args.ckpt_name = Path(spec["checkpoint_dir"]), spec["checkpoint_name"]
    args.hours = max(0, (args.deadline - time.time()) / 3600)
    write_json(args.run / "worker.json", seal({
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "plan_sha256": spec["sha256"],
        "deadline_unix": args.deadline,
    }))
    status = "failed"
    try:
        status = "budget_exhausted" if args.hours <= 0 else run_study(args, spec, time.monotonic())
    finally:
        atomic_json(args.run / "status.json", {"status": status})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("research", "replay", "robustness"))
    parser.add_argument("--ckpt-dir")
    parser.add_argument("--ckpt-name", default="checkpoint_step_10690.safetensors")
    parser.add_argument("--split")
    parser.add_argument("--bank")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--t0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--deadline", type=float)
    args, rest = parser.parse_known_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index/count")
    serialize_model_loading()
    if args.mode == "research":
        # All CLI arguments after '--' belong to the unchanged research command.
        arguments = sys.argv[sys.argv.index("--") + 1:]
        sys.argv = ["benchmarks.libero.research", *arguments]
        research.main()
    elif args.mode == "replay":
        replay_shard(args)
    else:
        robustness_shard(args)


if __name__ == "__main__":
    main()
