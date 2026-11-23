"""Wait for the existing campaign and its locks before a bounded GPU pilot.

This watcher never preempts the original experiment or extends its deadline.
"""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.research_e2_robustness import atomic_json
from .experiment import ROOT, choose_gpu, execute


def wait_and_run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output / "queue_status.json"
    deadline = args.deadline.timestamp()
    while time.time() < deadline - args.minimum_remaining_seconds:
        previous = args.after_run / "parallel_status.json"
        try:
            core = json.loads(previous.read_text())
        except (OSError, json.JSONDecodeError):
            core = {}
        if (args.after_run / "final_status.json").exists():
            for gpu in args.gpus:
                try:
                    uuid, _ = choose_gpu(gpu)
                    with (ROOT / f".cache/e2-reproduction-{uuid}.lock").open("a") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # execute acquires the same lock and checks idleness again.
                    atomic_json(status, {"status": "starting", "gpu": gpu, "deadline": args.deadline.isoformat()})
                    execute(SimpleNamespace(protocol=args.protocol, output=args.output / "pilot", gpu=gpu,
                        deadline=args.deadline, phase="pilot", subset="discovery", memory_mode="text-offload",
                        seeds=[43], warmup=1, cases="native_n10,refresh_all,fixed_2,fixed_3,change_0.1",
                        sensitivity=None, decode_video=False))
                    atomic_json(status, {"status": "finished", "pilot": str(args.output / "pilot")})
                    return
                except (BlockingIOError, RuntimeError) as error:
                    # Allocation/lock competition may change between checking
                    # and starting. Actual experiment failures must stay failed.
                    if (args.output / "pilot").exists():
                        atomic_json(status, {"status": "failed", "error": str(error)})
                        raise
        atomic_json(status, {"status": "waiting_existing_run", "core_stage": core.get("stage"),
            "core_status": core.get("status"), "deadline": args.deadline.isoformat(),
            "minimum_remaining_seconds": args.minimum_remaining_seconds,
            "updated_at": datetime.now().astimezone().isoformat(), "pid": os.getpid()})
        time.sleep(min(30, max(0, deadline - args.minimum_remaining_seconds - time.time())))
    atomic_json(status, {"status": "not_run_budget_reserved_for_existing_experiment",
        "reason": "No idle unlocked GPU after existing campaign with enough budget for a valid pilot",
        "deadline": args.deadline.isoformat()})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--after-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    parser.add_argument("--minimum-remaining-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.deadline.tzinfo is None or args.minimum_remaining_seconds < 60:
        parser.error("Timezone and at least 60 seconds remaining budget are required")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("Use unique nonnegative GPU indices")
    wait_and_run(args)


if __name__ == "__main__":
    main()
