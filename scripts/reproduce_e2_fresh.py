"""Fresh multi-GPU E2 reproduction with one absolute deadline for every stage."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.research.protocol import file_hash, read_sealed, seal, write_json
from scripts.reproduce_e2_parallel import ParallelPipeline
from scripts.reproduce_e2_robustness import (
    SPLIT_NAME, SUBSETS, check_assets, freeze_selection, prepare_representatives,
)
from scripts.research_e2_robustness import atomic_json


def merge_collections(run, subset, count):
    """Merge task-disjoint banks without authorizing an identity transition."""
    split = read_sealed(run / "manifests/split.json")
    tasks = [task for task in split["tasks"] if task["split"] == subset]
    destination = run / f"bank/state_bank/{subset}"
    destination.mkdir(parents=True, exist_ok=False)
    identity = environment = None
    entries, seen = [], set()
    for index in range(count):
        source = run / f"parallel/collection_{subset}/worker_{index}"
        bank = read_sealed(source / "manifest.json")
        model = read_sealed(source / "model.json")
        current_env = read_sealed(source / "environment.json")
        if identity is None:
            identity, environment = model, current_env
        if (model != identity or current_env != environment
            or bank["model_sha256"] != identity["sha256"]
            or bank["environment_sha256"] != environment["sha256"]
            or bank["task_split_sha256"] != split["sha256"]):
            raise ValueError("Collection shard identity/environment/split mismatch")
        owned = {(task["suite"], task["task_id"]) for task in tasks[index::count]}
        actual_tasks = set()
        for entry in bank["states"]:
            task = entry["suite"], entry["task_id"]
            if (task not in owned or entry["split"] != subset or entry["state_id"] in seen
                or entry["model_sha256"] != identity["sha256"]
                or entry["environment_sha256"] != environment["sha256"]
                or entry["task_split_sha256"] != split["sha256"] or not entry["valid_run"]):
                raise ValueError("Invalid/duplicate collection state or shard assignment")
            for field, hash_field in (("file", "sha256"), ("snapshot_file", "snapshot_sha256")):
                path = source / entry[field]
                if Path(entry[field]).name != entry[field] or file_hash(path) != entry[hash_field]:
                    raise ValueError("Collection state file/hash mismatch")
                shutil.copyfile(path, destination / entry[field])
            seen.add(entry["state_id"])
            actual_tasks.add(task)
            entries.append(entry)
        if actual_tasks != owned:
            raise ValueError("Collection shard has missing tasks")
        for task in tasks[index::count]:
            for episode in range(10):
                episode_path = source / "episodes" / f"{task['suite']}_{task['task_id']}_{episode}" / "episode.json"
                outcome = json.loads(episode_path.read_text())
                if not outcome["valid_run"] or not outcome["selected_state_ids"]:
                    raise ValueError("Missing or invalid collection episode")
    order = {(task["suite"], task["task_id"]): i for i, task in enumerate(tasks)}
    entries.sort(key=lambda row: (order[row["suite"], row["task_id"]], row["episode_id"], row["request_index"]))
    write_json(destination / "model.json", identity)
    write_json(destination / "environment.json", environment)
    write_json(destination / "manifest.json", seal({
        "schema_version": 1, "task_split_sha256": split["sha256"],
        "model_sha256": identity["sha256"], "states": entries,
        "environment_sha256": environment["sha256"],
        "protocol": "development_progress_proxy; not official full evaluation",
    }))


def descendants(pid):
    """Snapshot descendants across worker sessions, without inspecting arguments."""
    rows = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
    children = {}
    for row in rows.splitlines():
        child, parent = map(int, row.split())
        children.setdefault(parent, []).append(child)
    found = []
    def visit(parent):
        for child in children.get(parent, []):
            visit(child)
            found.append(child)
    visit(pid)
    return found


def stop_descendants(pid, grace=10):
    # Record /proc start times so a recycled PID can never be signalled.
    processes = {}
    for child in descendants(pid):
        try:
            processes[child] = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            pass
    def send(sig):
        for child, started in processes.items():
            try:
                current = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()
                if current[19] == started and current[0] != "Z":
                    os.kill(child, sig)
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass
    send(signal.SIGTERM)
    end = time.monotonic() + grace
    while time.monotonic() < end:
        if all(not Path(f"/proc/{child}").exists() for child in processes):
            break
        time.sleep(0.2)
    send(signal.SIGKILL)


class FreshPipeline(ParallelPipeline):
    def spawn(self, name, gpu, arguments):
        # Collectors write LIBERO's config before import. Separate roots avoid
        # concurrent truncation of that YAML when four collectors initialize.
        previous = {key: os.environ.get(key) for key in ("LIBERO_CONFIG_ROOT", "LIBERO_CONFIG_PATH")}
        config_root = self.run / "parallel/config" / name
        os.environ.update(LIBERO_CONFIG_ROOT=str(config_root), LIBERO_CONFIG_PATH=str(config_root))
        try:
            return super().spawn(name, gpu, arguments)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def stage(self, name, command=None, operation=None, env=None):
        if command:
            def execute():
                print("Command:", json.dumps(list(map(str, command))), flush=True)
                process = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                           start_new_session=True, stdin=subprocess.DEVNULL)
                self.children.append(process)
                self.wait([process])
            return super().stage(name, operation=execute)
        return super().stage(name, operation=operation)

    def execute(self):
        args, run = self.args, self.run
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.free",
                                        "--format=csv,noheader,nounits"], text=True)
        available = {}
        for row in rows.splitlines():
            index, uuid, name, free = [part.strip() for part in row.split(",")]
            available[int(index)] = dict(index=int(index), uuid=uuid, name=name, free_mib=int(free))
        gpus = [available[index] for index in args.gpus]
        if len({gpu["name"] for gpu in gpus}) != 1:
            raise ValueError("All workers must use the same GPU model")
        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
        if any(gpu["uuid"] in busy.splitlines() for gpu in gpus):
            raise RuntimeError("Requested GPUs are busy; refusing to displace existing jobs")
        with ExitStack() as stack:
            for gpu in gpus:
                lock = stack.enter_context((ROOT / f".cache/e2-reproduction-{gpu['uuid']}.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.environ.update(CUDA_VISIBLE_DEVICES=gpus[0]["uuid"], GPU=str(gpus[0]["index"]),
                              MUJOCO_EGL_DEVICE_ID=str(gpus[0]["index"]))
            write_json(run / "environment.json", {
                "python": sys.version, "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                "gpu_workers": gpus, "source_sha256": self.source, "checkpoint_dir": str(args.ckpt_dir),
                "minimum_free_mib": int(os.environ["E2_MIN_FREE_MIB"]), "deadline": args.deadline.isoformat(),
                "memory_mode": os.environ.get("E2_MEMORY_MODE", "resident"),
            })
            write_json(run / "parallel/execution.json", seal({
                "gpu_workers": gpus, "source_sha256": self.source,
                "assignment": "collection tasks modulo shard count; replay states modulo worker count",
                "worker_sha256": file_hash(ROOT / "scripts/research_parallel_worker.py"),
                "coordinator_sha256": file_hash(Path(__file__)),
            }))
            self.stage("verify_local_assets", operation=lambda: write_json(
                run / "checks/local_assets.json", check_assets(args.reference_root, args.ckpt_dir, args.ckpt_name)))
            split = read_sealed(args.reference_root / SPLIT_NAME / "manifests/split.json")
            split_file = run / "manifests/split.json"
            write_json(split_file, split)
            self.stage("egl_smoke", ["bash", "benchmarks/libero/run_smoke.sh", "env"],
                       env={**os.environ, "LIBERO_PYTHON": sys.executable})
            self.stage("model_smoke", [sys.executable, ROOT / "scripts/reproduce_e2_robustness.py",
                "--model-smoke", "--run", run, "--ckpt-dir", args.ckpt_dir,
                "--ckpt-name", args.ckpt_name, "--reference-root", args.reference_root])
            common = ["--ckpt-dir", args.ckpt_dir, "--ckpt-name", args.ckpt_name, "--split", split_file]
            # Half of the cards collect each split; every task keeps all its episodes.
            groups = [gpus[::2], gpus[1::2]] if len(gpus) > 1 else [gpus, gpus]
            jobs = []
            for subset, group in zip(SUBSETS, groups):
                for index, gpu in enumerate(group):
                    jobs.append((f"collect_{subset}_{index}", gpu, ["research", "--", "collect", *common,
                        "--selected-split", subset, "--seeds", 43, "--episodes", 10,
                        "--task-shard-index", index, "--task-shard-count", len(group),
                        "--output", run / f"parallel/collection_{subset}/worker_{index}"]))
            if len(gpus) == 1:
                for job in jobs:
                    self.parallel_stage(job[0], [job])
            else:
                self.parallel_stage("collect", jobs)
            for subset, group in zip(SUBSETS, groups):
                self.stage(f"merge_collection_{subset}", operation=lambda subset=subset, group=group:
                           merge_collections(run, subset, len(group)))
            self.parallel_stage("t0", [(f"t0_{subset}", gpus[index], ["research", "--", "t0", *common,
                "--bank", run / f"bank/state_bank/{subset}", "--seeds", 43, 44, 45,
                "--output", run / f"base/diagnostics/t0_{subset}"])
                for index, subset in enumerate(SUBSETS)] if len(gpus) > 1 else [])
            if len(gpus) == 1:
                for subset in SUBSETS:
                    self.parallel_stage(f"t0_{subset}", [(f"t0_{subset}", gpus[0], ["research", "--", "t0", *common,
                        "--bank", run / f"bank/state_bank/{subset}", "--seeds", 43, 44, 45,
                        "--output", run / f"base/diagnostics/t0_{subset}"])])
            for subset in SUBSETS:
                if read_sealed(run / f"base/diagnostics/t0_{subset}/t0_certificate.json")["status"] != "passed":
                    raise ValueError(f"T0 failed: {subset}")
            checks, jobs = [], []
            for gpu in gpus:
                output = run / f"parallel/cross_gpu_t0/gpu_{gpu['index']}"
                jobs.append((f"cross_gpu_t0_{gpu['index']}", gpu,
                    ["research", "--", "t0", *common, "--bank", run / "bank/state_bank/discovery",
                     "--seeds", 43, 44, 45, "--output", output]))
                checks.append({"gpu": gpu["index"], "output": str(output)})
            self.parallel_stage("cross_gpu_t0", jobs)
            if any(read_sealed(Path(check["output"]) / "t0_certificate.json")["status"] != "passed" for check in checks):
                raise ValueError("Cross-GPU T0 failed")
            write_json(run / "parallel/cross_gpu_checks.json", seal({"workers": checks}))
            self.stage("representative_plan", operation=lambda: prepare_representatives(run, split))
            for index, layer in enumerate((0, 29)):
                self.parallel_stage(f"representative_L{layer}", [(f"representative_L{layer}", gpus[index % len(gpus)],
                    ["research", "--", "smoke", *common, "--bank", run / "bank/state_bank/discovery",
                     "--seeds", 43, "--plan", run / "representatives/plan.json", "--method-id", f"representative_L{layer}",
                     "--output", run / f"representatives/L{layer}"])])
            self.stage("freeze_selection", operation=lambda: freeze_selection(run, split))
            for index, subset in enumerate(SUBSETS):
                self.parallel_stage(f"features_{subset}", [(f"features_{subset}", gpus[index % len(gpus)],
                    ["research", "--", "features", *common, "--bank", run / f"bank/state_bank/{subset}",
                     "--selected-split", subset, "--selection", run / "base/manifests/selection.json",
                     "--t0", run / f"base/diagnostics/t0_{subset}/t0_certificate.json", "--seeds", 43,
                     "--output", run / f"base/diagnostics/features_{subset}"])])
            log = (run / "extensions_supervisor.log").open("a")
            extension = subprocess.Popen([sys.executable, ROOT / "scripts/research_e2_extensions.py",
                "--run", run, "--core-pid", str(os.getpid()), "--deadline",
                datetime.fromtimestamp(float(os.environ["E2_TOTAL_DEADLINE"]), args.deadline.tzinfo).isoformat()],
                cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            log.close()
            self.children.append(extension)
            self.finish(gpus, common, split_file)
            self.wait([extension])
            extension_status = json.loads((run / "extensions/status.json").read_text())
            if extension_status["status"] == "failed":
                raise RuntimeError("Extension supervisor failed; inspect extensions_supervisor.log")
            atomic_json(run / "final_status.json", {"status": extension_status["status"], "deadline": args.deadline.isoformat(),
                "core": json.loads((run / "parallel_status.json").read_text()),
                "extensions": json.loads((run / "extensions/status.json").read_text())})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / ".cache/models/OpenWAM-Alpha-Sim-LIBERO")
    parser.add_argument("--ckpt-name", default="checkpoint_step_10690.safetensors")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "experiment/reference/experiment")
    parser.add_argument("--minimum-free-mib", type=int, default=18000)
    args = parser.parse_args()
    if args.deadline.tzinfo is None or args.deadline.timestamp() <= time.time() + 120:
        parser.error("Deadline must include timezone and leave at least two minutes")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("GPUs must be distinct nonnegative indices")
    if args.minimum_free_mib < 1:
        parser.error("Minimum free memory must be positive")
    args.hours = (args.deadline.timestamp() - time.time()) / 3600
    args.run, args.ckpt_dir, args.reference_root = args.run.resolve(), args.ckpt_dir.resolve(), args.reference_root.resolve()
    args.run.mkdir(parents=True, exist_ok=False)
    (ROOT / ".cache").mkdir(exist_ok=True)
    os.environ["E2_MIN_FREE_MIB"] = str(args.minimum_free_mib)
    # Workers stop at pair boundaries first; the coordinator has another minute
    # for graceful completion, then a final minute for forced cleanup/reporting.
    work_deadline = args.deadline.timestamp() - 120
    os.environ["E2_TOTAL_DEADLINE"] = str(work_deadline)
    write_json(args.run / "budget.json", seal({"deadline_unix": work_deadline,
        "hard_deadline_unix": args.deadline.timestamp(), "deadline_local": args.deadline.isoformat(),
        "scope": "all stages including collection, preflight, memory waits and extensions"}))
    pipeline = FreshPipeline(args)
    def interrupted(signum, frame):
        if signum == signal.SIGALRM:
            raise TimeoutError("Unified experiment budget exhausted; saving partial evidence")
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, args.deadline.timestamp() - 60 - time.time()))
    def hard_stop():
        time.sleep(max(0, args.deadline.timestamp() - time.time() - 12))
        stop_descendants(os.getpid(), grace=5)
        atomic_json(args.run / "hard_stop.json", {"status": "budget_exhausted", "at": datetime.now().isoformat()})
        os._exit(124)
    threading.Thread(target=hard_stop, daemon=True).start()
    try:
        pipeline.execute()
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop_descendants(os.getpid())
        status = "budget_exhausted" if isinstance(error, TimeoutError) or time.time() >= work_deadline else "failed"
        pipeline.status(status, error_type=type(error).__name__, error=str(error))
        counts, unreadable = {}, []
        for name, pattern in (("baseline_shards", "parallel/replay_*/worker_*/*.json"),
                              ("robustness", "robustness/workers/worker_*/results/*/*.json"),
                              ("extensions", "extensions/workers/worker_*/results/**/*.json")):
            counts[name] = 0
            for path in args.run.glob(pattern):
                try:
                    row = json.loads(path.read_text())
                    counts[name] += "method_id" in row and row.get("status") == "completed"
                except (json.JSONDecodeError, OSError):
                    unreadable.append(str(path.relative_to(args.run)))
        atomic_json(args.run / "final_status.json", {"status": status, "stage": pipeline.current,
            "error": str(error), "completed_pairs": counts, "unreadable_results": unreadable,
            "deadline": args.deadline.isoformat()})
        if status == "failed":
            raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
