"""Adopt the active fresh E2 run and distribute independent pairs across idle GPUs."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.research.protocol import file_hash, read_sealed, seal, write_json
from scripts.reproduce_e2_robustness import (
    Pipeline, SPLIT_NAME, SUBSETS, freeze_selection, prepare_representatives, validate_baseline,
)
from scripts.research_e2_robustness import (
    atomic_json, owns_state, pair_path, prepare, report, verify_result,
)


def merge_replay(run, subset, count):
    destination = run / f"base/diagnostics/e2_{subset}"
    destination.mkdir(parents=True, exist_ok=False)
    expected_plan = read_sealed(run / f"base/manifests/e2_{subset}.json")
    manifest = None
    found = set()
    for index in range(count):
        shard = run / f"parallel/replay_{subset}/worker_{index}"
        current = read_sealed(shard / "manifest.json")
        if manifest is None:
            manifest = current
        if current != manifest or read_sealed(shard / "plan.json") != expected_plan:
            raise ValueError("Replay shard identities/plans differ")
        assignment = read_sealed(shard / "worker.json")
        expected_states = expected_plan["state_ids"][index::count]
        if assignment["state_ids"] != expected_states or assignment["plan_sha256"] != expected_plan["sha256"]:
            raise ValueError("Replay shard assignment changed")
        expected = {(state, method["method_id"]) for state in expected_states for method in expected_plan["methods"]}
        local = set()
        for path in shard.glob("*.json"):
            row = json.loads(path.read_text())
            if "method_id" not in row:
                continue
            key = row["state_id"], row["method_id"]
            verify_result(row, *key, manifest["model"]["sha256"])
            if key not in expected or key in found or row["plan_sha256"] != expected_plan["sha256"]:
                raise ValueError("Unexpected/duplicate replay result")
            local.add(key)
            found.add(key)
            shutil.copyfile(path, destination / path.name)
        if local != expected:
            raise ValueError("Incomplete replay shard")
    if len(found) != 2304:
        raise ValueError("Expected exactly 2304 fresh pairs per split")
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "plan.json", expected_plan)


def merge_robustness(run, spec, count):
    root = run / "robustness"
    floors = {subset: [] for subset in SUBSETS}
    total = 0
    complete = True
    for index in range(count):
        shard = root / f"workers/worker_{index}"
        status = json.loads((shard / "status.json").read_text())["status"]
        if status not in {"complete", "budget_exhausted"}:
            raise ValueError(f"Robustness worker {index}: {status}")
        complete &= status == "complete"
        assignment = read_sealed(shard / "worker.json")
        if assignment["plan_sha256"] != spec["sha256"] or assignment["gpu_uuid"] != spec["gpu_uuids"][index]:
            raise ValueError("Robustness assignment changed")
        floor_path = shard / "noise_floor.json"
        if floor_path.exists():
            noise = read_sealed(floor_path)
            if noise["plan_sha256"] != spec["sha256"]:
                raise ValueError("Wrong noise floor plan")
            for subset in SUBSETS:
                floors[subset].extend(noise["rows"][subset])
        for subset_index, subset in enumerate(SUBSETS):
            ids = spec["selection"]["state_ids_by_split"][subset]
            owned = {state for state_index, state in enumerate(ids)
                     if owns_state(subset_index, state_index, index, count)}
            expected_methods = {method["method_id"] for method in spec["methods"]}
            for path in (shard / f"results/{subset}").glob("*.json"):
                row = read_sealed(path)
                if row["state_id"] not in owned or row["method_id"] not in expected_methods:
                    raise ValueError("Result outside worker assignment")
                verify_result(row, row["state_id"], row["method_id"], spec["model_sha256_by_split"][subset], spec["sha256"])
                target = pair_path(root, subset, row["state_id"], row["method_id"])
                if target.exists():
                    raise ValueError("Duplicate robustness result")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                total += 1
    atomic_json(root / "noise_floor.json", seal({"plan_sha256": spec["sha256"], "rows": floors}))
    if complete and total != 7680:
        raise ValueError(f"Complete workers produced only {total}/7680 pairs")
    return "complete" if complete else "budget_exhausted", total


class ParallelPipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        self.children = []

    def status(self, status, **extra):
        atomic_json(self.run / "parallel_status.json", {
            "status": status, "stage": self.current, "completed_stages": self.completed,
            "pid": os.getpid(), "updated_unix": time.time(), "gpus": self.args.gpus, **extra,
        })

    def spawn(self, name, gpu, arguments):
        # Shared server: leave room above the measured ~25 GB replica peak.
        while True:
            budget = self.run / "budget.json"
            if budget.exists() and time.time() >= read_sealed(budget)["deadline_unix"]:
                raise TimeoutError("Total experiment deadline reached before launching a worker")
            free = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu["index"]),
                       "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
            minimum = int(os.environ.get("E2_MIN_FREE_MIB", "28000"))
            if free >= minimum:
                break
            print(f"Waiting for GPU {gpu['index']}: {free} MiB free; need {minimum} MiB", flush=True)
            time.sleep(10)
        logfile = self.run / f"parallel/logs/{name}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu["uuid"], "GPU": str(gpu["index"]),
               "MUJOCO_EGL_DEVICE_ID": str(gpu["index"]), "PYTHONUNBUFFERED": "1"}
        if (self.run / "budget.json").exists():
            env["E2_TOTAL_DEADLINE"] = str(read_sealed(self.run / "budget.json")["deadline_unix"])
        with logfile.open("a") as stream:
            process = subprocess.Popen([sys.executable, ROOT / "scripts/research_parallel_worker.py",
                                        *map(str, arguments)], cwd=ROOT, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(process)
        print(f"Started {name}: GPU {gpu['index']}, PID {process.pid}, log={logfile}", flush=True)
        return process

    def wait(self, processes):
        while True:
            states = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in states):
                raise RuntimeError(f"Parallel stage failed; exit codes={states}; see parallel/logs")
            if all(code == 0 for code in states):
                return
            time.sleep(5)

    def parallel_stage(self, name, jobs):
        def execute():
            self.wait([self.spawn(label, gpu, arguments) for label, gpu, arguments in jobs])
        self.stage(name, operation=execute)

    def execute(self):
        if self.args.resume_verified:
            return self.resume_verified()
        args, run = self.args, self.run
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.free",
                                        "--format=csv,noheader,nounits"], text=True)
        available = {}
        for row in rows.splitlines():
            index, uuid, name, free = [part.strip() for part in row.split(",")]
            available[int(index)] = {"index": int(index), "uuid": uuid, "name": name, "free_mib": int(free)}
        gpus = [available[index] for index in args.gpus]
        if len({gpu["name"] for gpu in gpus}) != 1:
            raise ValueError("Fresh run requires the same GPU model on every worker")
        with ExitStack() as stack:
            # GPU 4 remains held by the stopped original coordinator until adoption.
            for gpu in gpus:
                if gpu["index"] == args.original_gpu:
                    continue
                lock = stack.enter_context((ROOT / f".cache/e2-reproduction-{gpu['uuid']}.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_json(run / "parallel/execution.json", seal({
                "gpu_workers": gpus, "source_sha256": self.source,
                "original_coordinator_pid": args.original_pid,
                "assignment": "state ordinal modulo worker count; both signs stay on the same GPU",
                "host_memory": "Serialize model-loading peaks; independent inference replicas",
                "robustness_hours": args.hours,
                "worker_sha256": file_hash(ROOT / "scripts/research_parallel_worker.py"),
                "coordinator_sha256": file_hash(__file__),
            }))
            self.current = "adopt_collections"
            self.status("running")
            original = json.loads((run / "pipeline_status.json").read_text())
            if original["stage"] != "collect_discovery":
                raise ValueError("Expected the original run to be collecting discovery")
            if not (run / "checks/model_smoke.json").exists():
                raise ValueError("Original model preflight did not pass")
            split_file = run / "manifests/split.json"
            split = read_sealed(split_file)
            common = ["--ckpt-dir", args.ckpt_dir, "--ckpt-name", args.ckpt_name, "--split", split_file]
            t0_processes = []
            # Start T0 as soon as each bank finishes; overlap it with the other collector.
            for subset in SUBSETS:
                pid = args.discovery_pid if subset == "discovery" else args.calibration_pid
                while not (run / f"bank/state_bank/{subset}/manifest.json").exists():
                    if any(process.poll() not in (None, 0) for process in t0_processes):
                        raise RuntimeError("An overlapping T0 worker failed; see parallel/logs")
                    os.kill(pid, 0)
                    if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                        raise RuntimeError(f"{subset} collector exited without a manifest")
                    time.sleep(5)
                bank = read_sealed(run / f"bank/state_bank/{subset}/manifest.json")
                model = read_sealed(run / f"bank/state_bank/{subset}/model.json")
                if bank["model_sha256"] != model["sha256"] or model["source_sha256"] != self.source:
                    raise ValueError("Adopted collection identity/source mismatch")
                gpu = available[args.original_gpu] if subset == "discovery" else gpus[0]
                t0_processes.append(self.spawn(f"t0_{subset}", gpu, ["research", "--", "t0", *common,
                    "--bank", run / f"bank/state_bank/{subset}", "--seeds", 43, 44, 45,
                    "--output", run / f"base/diagnostics/t0_{subset}"]))
            os.kill(args.original_pid, signal.SIGTERM)
            os.kill(args.original_pid, signal.SIGCONT)
            time.sleep(1)
            lock = stack.enter_context((ROOT / f".cache/e2-reproduction-{available[args.original_gpu]['uuid']}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.completed.append("adopt_collections")
            atomic_json(run / "pipeline_status.json", {"status": "superseded", "parallel_pid": os.getpid(),
                        "status_file": str(run / "parallel_status.json")})
            self.stage("t0", operation=lambda: self.wait(t0_processes))
            for subset in SUBSETS:
                if read_sealed(run / f"base/diagnostics/t0_{subset}/t0_certificate.json")["status"] != "passed":
                    raise ValueError(f"T0 failed for {subset}")
            self.stage("representative_plan", operation=lambda: prepare_representatives(run, split))
            self.parallel_stage("representatives", [(f"representative_L{layer}", gpus[index], ["research", "--", "smoke",
                *common, "--bank", run / "bank/state_bank/discovery", "--seeds", 43,
                "--plan", run / "representatives/plan.json", "--method-id", f"representative_L{layer}",
                "--output", run / f"representatives/L{layer}"]) for index, layer in enumerate((0, 29))])
            self.stage("freeze_selection", operation=lambda: freeze_selection(run, split))
            self.parallel_stage("features", [(f"features_{subset}", gpus[index], ["research", "--", "features", *common,
                "--bank", run / f"bank/state_bank/{subset}", "--selected-split", subset,
                "--selection", run / "base/manifests/selection.json",
                "--t0", run / f"base/diagnostics/t0_{subset}/t0_certificate.json", "--seeds", 43,
                "--output", run / f"base/diagnostics/features_{subset}"]) for index, subset in enumerate(SUBSETS)])
            self.finish(gpus, common, split_file)

    def finish(self, gpus, common, split_file, subsets=SUBSETS):
        args, run = self.args, self.run
        for subset in subsets:
            if not (run / f"base/manifests/e2_{subset}.json").exists():
                self.research(f"plan_{subset}", "plan", "--experiment", "e2", "--layers", 30,
                "--selected-split", subset, "--split", split_file, "--selection", run / "base/manifests/selection.json",
                "--calibration", run / "base/diagnostics/features_calibration/calibration.json",
                "--output", run / f"base/manifests/e2_{subset}.json")
            self.parallel_stage(f"smoke_{subset}", [(f"smoke_{subset}", gpus[0], ["research", "--", "smoke", *common,
                "--bank", run / f"bank/state_bank/{subset}", "--plan", run / f"base/manifests/e2_{subset}.json",
                "--method-id", "E2_L0_S1_G0_sign-1", "--seeds", 43,
                "--output", run / f"base/diagnostics/smoke_{subset}"])])
            self.parallel_stage(f"baseline_{subset}", [(f"replay_{subset}_{index}", gpu, ["replay", *common,
                "--bank", run / f"bank/state_bank/{subset}", "--plan", run / f"base/manifests/e2_{subset}.json",
                "--t0", run / f"base/diagnostics/t0_{subset}/t0_certificate.json",
                "--shard-index", index, "--shard-count", len(gpus),
                "--output", run / f"parallel/replay_{subset}/worker_{index}"]) for index, gpu in enumerate(gpus)])
            self.stage(f"merge_{subset}", operation=lambda subset=subset: merge_replay(run, subset, len(gpus)))
        self.stage("validate_baseline", operation=lambda: validate_baseline(run))
        robust_root = run / "robustness"
        robust_root.mkdir(exist_ok=False)
        robust_args = SimpleNamespace(run=robust_root, base=run / "base", bank_run=run / "bank", split=split_file,
            ckpt_dir=args.ckpt_dir, ckpt_name=args.ckpt_name, hours=args.hours,
            parallel_gpus=[gpu["uuid"] for gpu in gpus])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]["uuid"]
        self.stage("robustness_prepare", operation=lambda: prepare(robust_args))
        spec = read_sealed(robust_root / "plan.json")
        deadline = time.time() + args.hours * 3600
        self.parallel_stage("robustness", [(f"robustness_{index}", gpu, ["robustness", "--plan", robust_root / "plan.json",
            "--shard-index", index, "--shard-count", len(gpus), "--deadline", deadline,
            "--output", robust_root / f"workers/worker_{index}"]) for index, gpu in enumerate(gpus)])
        status, total = merge_robustness(run, spec, len(gpus))
        atomic_json(robust_root / "status.json", {"status": status, "completed_pairs": total,
                                                 "workers": len(gpus), "deadline_unix": deadline})
        self.stage("report", operation=lambda: report(robust_args, spec, status))
        self.status(status, completed_pairs=total)

    def resume_verified(self):
        args, run = self.args, self.run
        previous = json.loads((run / "parallel_status.json").read_text())
        if previous["status"] != "failed" or previous["stage"] != "smoke_calibration":
            raise ValueError("Verified recovery currently supports failure at calibration smoke only")
        selection = read_sealed(run / "base/manifests/selection.json")
        calibration = read_sealed(run / "base/diagnostics/features_calibration/calibration.json")
        if selection["sha256"] != calibration["selection_sha256"]:
            raise ValueError("Selection/calibration changed")
        old_execution = read_sealed(run / "parallel/execution.json")
        if old_execution["source_sha256"] != self.source:
            raise ValueError("Research source changed since the original run")
        for subset in SUBSETS:
            bank = read_sealed(run / f"bank/state_bank/{subset}/manifest.json")
            model = read_sealed(run / f"bank/state_bank/{subset}/model.json")
            certificate = read_sealed(run / f"base/diagnostics/t0_{subset}/t0_certificate.json")
            if model["source_sha256"] != self.source or bank["model_sha256"] != model["sha256"]:
                raise ValueError("State bank/model identity changed")
            from benchmarks.libero.research import validate_t0
            validate_t0(certificate, model, bank, 10)
        # Re-validate all completed Discovery pairs, preserving their files and hashes.
        directory = run / "base/diagnostics/e2_discovery"
        manifest, plan = read_sealed(directory / "manifest.json"), read_sealed(directory / "plan.json")
        expected = {(state, method["method_id"]) for state in plan["state_ids"] for method in plan["methods"]}
        seen = set()
        for path in directory.glob("*.json"):
            row = json.loads(path.read_text())
            if "method_id" not in row:
                continue
            key = row["state_id"], row["method_id"]
            verify_result(row, *key, manifest["model"]["sha256"])
            if key in seen or row["plan_sha256"] != plan["sha256"]:
                raise ValueError("Duplicate or incompatible completed Discovery result")
            seen.add(key)
        if seen != expected or len(seen) != 2304:
            raise ValueError("Completed Discovery baseline is not complete")
        smoke = run / "base/diagnostics/smoke_calibration"
        if smoke.exists() and any(smoke.iterdir()):
            raise ValueError("Calibration smoke contains partial artifacts; refusing implicit reuse")
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.free",
                                        "--format=csv,noheader,nounits"], text=True)
        available = {}
        for row in rows.splitlines():
            index, uuid, name, free = [part.strip() for part in row.split(",")]
            available[int(index)] = {"index": int(index), "uuid": uuid, "name": name, "free_mib": int(free)}
        gpus = [available[index] for index in args.gpus]
        if len({gpu["name"] for gpu in gpus}) != 1:
            raise ValueError("GPU models differ")
        with ExitStack() as stack:
            for gpu in gpus:
                lock = stack.enter_context((ROOT / f".cache/e2-reproduction-{gpu['uuid']}.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            history = run / "parallel/recovery_history" / str(int(time.time()))
            write_json(history / "status.json", previous)
            write_json(history / "execution.json", old_execution)
            if smoke.exists():
                smoke.rename(history / "empty_failed_smoke")
            if (run / "extensions/status.json").exists():
                write_json(history / "extensions_status.json", json.loads((run / "extensions/status.json").read_text()))
            write_json(run / "parallel/recovery.json", seal({
                "reason": "Other server process allocated GPU 0 memory while checkpoint was loading",
                "completed_discovery_pairs_verified": len(seen), "gpu_workers": gpus,
                "source_sha256": self.source, "worker_sha256": file_hash(ROOT / "scripts/research_parallel_worker.py"),
                "coordinator_sha256": file_hash(__file__), "original_execution_sha256": old_execution["sha256"],
            }))
            self.completed = previous["completed_stages"]
            self.current = "resume_calibration_smoke"
            self.status("running", recovery=str(run / "parallel/recovery.json"))
            split_file = run / "manifests/split.json"
            common = ["--ckpt-dir", args.ckpt_dir, "--ckpt-name", args.ckpt_name, "--split", split_file]
            self.finish(gpus, common, split_file, subsets=("calibration",))

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--original-pid", type=int)
    parser.add_argument("--original-gpu", type=int, default=4)
    parser.add_argument("--discovery-pid", type=int)
    parser.add_argument("--calibration-pid", type=int)
    parser.add_argument("--resume-verified", action="store_true")
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / ".cache/models/OpenWAM-Alpha-Sim-LIBERO")
    parser.add_argument("--ckpt-name", default="checkpoint_step_10690.safetensors")
    parser.add_argument("--hours", type=float, default=10)
    args = parser.parse_args()
    if not args.resume_verified and any(value is None for value in (args.original_pid, args.discovery_pid, args.calibration_pid)):
        parser.error("Initial adoption requires the original coordinator and collector PIDs")
    if len(set(args.gpus)) != len(args.gpus) or args.original_gpu not in args.gpus:
        parser.error("Unique GPUs including the original GPU are required")
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error("--hours must be positive and finite")
    args.run, args.ckpt_dir = args.run.resolve(), args.ckpt_dir.resolve()
    pipeline = ParallelPipeline(args)
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    with (args.run / ".parallel.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            pipeline.execute()
        except BaseException as error:
            for process in pipeline.children:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            pipeline.status("failed", error=str(error), error_type=type(error).__name__)
            raise


if __name__ == "__main__":
    main()
