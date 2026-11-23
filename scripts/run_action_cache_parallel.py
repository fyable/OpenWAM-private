"""Run the hot-request experiment with one persistent model replica per GPU."""

import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks.action_cache import next_experiment as nx
from benchmarks.action_cache.experiment import choose_gpu, code_identity, prepare
from openwam.research.protocol import file_hash, read_sealed, seal, write_json
from scripts.research_e2_robustness import atomic_json
from scripts.research_parallel_worker import serialize_model_loading

PANELS = {"warm": ("text", "main"), "overhead": ("profile", "main"),
          "discovery": ("discovery",), "calibration": ("calibration",)}
PHASE_MINUTES = {"warm": 75, "overhead": 150, "discovery": 285, "calibration": 285}


def partition(states, rank, workers):
    tasks = sorted({nx.task_key(entry) for entry, _ in states})
    assigned = set(tasks[rank::workers])
    return [(entry, value) for entry, value in states if nx.task_key(entry) in assigned]


def group_panel(stage, replay, states, choices, panel):
    groups = stage.plan["target"]["token_groups"]
    horizon = stage.plan["execution_horizon"]
    for index, (entry, conditions) in enumerate(states):
        replay.run(conditions, nx.SEEDS[0], text="prime")
        for seed in nx.SEEDS:
            reference, baseline = replay.run(conditions, seed)
            if index == 0 and seed == nx.SEEDS[0]:
                empty, details = replay.run(conditions, seed, groups=groups, chosen=(), horizon=horizon)
                nx.check_parity(reference, empty, "empty group intervention")
                nx.validate_group_rows(details, (), groups)
            measurements = {"native_n10": {**baseline, **nx.action_metrics(reference, reference, horizon)}}
            for name in nx.sample_order(tuple(choices), entry, seed, 0):
                chosen = choices[name]
                actual, details = replay.run(conditions, seed, groups=groups, chosen=chosen, horizon=horizon)
                nx.validate_group_rows(details, chosen, groups)
                measurements[name] = {**details, **nx.action_metrics(reference, actual, horizon)}
            stage.save(entry, seed, 0, measurements, panel=panel)


def load_plan(root):
    plan = read_sealed(root / "next_plan.json")
    protocol = read_sealed(root / "protocol/protocol.json")
    layout = read_sealed(root / "parallel.json")
    if (layout["plan_sha256"] != plan["sha256"] or plan["protocol_sha256"] != protocol["sha256"]
            or layout["launcher_sha256"] != file_hash(Path(__file__).resolve())
            or protocol["source_files"] != code_identity()):
        raise ValueError("Parallel experiment plan or code changed")
    return plan, protocol, layout


def worker(root, rank):
    from benchmarks.libero import research

    plan, protocol, layout = load_plan(root)
    output = root / "workers" / str(rank)
    gpu = layout["gpus"][rank]
    deadline = layout["deadlines"]["warm"]
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], MUJOCO_EGL_DEVICE_ID=str(gpu["index"]),
                      E2_MEMORY_MODE="text-offload", E2_TOTAL_DEADLINE=str(deadline), E2_MIN_FREE_MIB="18000",
                      LIBERO_CONFIG_ROOT=str(output / "libero"), LIBERO_CONFIG_PATH=str(output / "libero"))
    serialize_model_loading()
    args = SimpleNamespace(ckpt_dir=protocol["checkpoint_dir"], ckpt_name=protocol["checkpoint_name"],
                           device="cuda:0", steps=10, horizon=plan["execution_horizon"])
    source = Path(protocol["e2_run"])
    with nx.alarm(deadline):
        atomic_json(output / "status.json", {"status": "loading", "gpu": gpu["index"]})
        server = research.build_server(args)
        model = research.model_identity(server, args)
        split = read_sealed(source / "manifests/split.json")
        if split["sha256"] != protocol["task_split_sha256"]:
            raise ValueError("Task split changed")
        for subset in ("discovery", "calibration"):
            saved = read_sealed(source / "bank/state_bank" / subset / "model.json")
            if model["assets"] != saved["assets"] or model["source_sha256"] != protocol["research_source_sha256"]:
                raise ValueError("Loaded model does not match source or assets")
        discovery = nx.load_selected(source, split, plan, "discovery")
        calibration = nx.load_selected(source, split, plan, "calibration")
        replay = nx.Replay(server.engine)
        with replay.measure():
            reference, _ = replay.run(discovery[0][1], nx.SEEDS[0], text="prime")
        atomic_json(output / "identity.json", seal({"plan_sha256": plan["sha256"], "gpu": gpu,
                                                    "model": model, "reference": reference.tolist()}))
    discovery = partition(discovery, rank, len(layout["gpus"]))
    calibration = partition(calibration, rank, len(layout["gpus"]))
    with replay.measure():
        for phase in PHASE_MINUTES:
            deadline = layout["deadlines"][phase]
            atomic_json(output / "status.json", {"status": "waiting", "phase": phase, "gpu": gpu["index"]})
            with nx.alarm(deadline):
                trigger = root / f"start_{phase}.json"
                while not trigger.exists():
                    time.sleep(0.5)
                payload = read_sealed(trigger)
                if payload["plan_sha256"] != plan["sha256"]:
                    raise ValueError("Phase trigger does not match the plan")
                if payload.get("skip"):
                    break
                stage_name = "group" if phase in {"discovery", "calibration"} else phase
                phase_root = output / phase
                phase_root.mkdir()
                stage = nx.Stage(phase_root, plan, stage_name, deadline)
                stage.status("running", gpu=gpu["index"])
                if stage_name == "group":
                    group_panel(stage, replay, discovery if phase == "discovery" else calibration,
                                payload["choices"], phase)
                else:
                    nx.run_speed(stage, replay, server, discovery)
                stage.status("complete", gpu=gpu["index"])
                atomic_json(output / f"done_{phase}.json", seal({"rows": len(stage.rows),
                           "plan_sha256": plan["sha256"], "phase": phase, "gpu": gpu["index"]}))
    atomic_json(output / "status.json", {"status": "complete", "gpu": gpu["index"]})


def wait_for(paths, children, deadline):
    while not all(path.exists() for path in paths):
        if any(p.poll() not in (None, 0) for p in children):
            raise RuntimeError("A GPU worker failed. See workers/*/worker.log")
        if time.time() >= deadline:
            raise TimeoutError("Parallel stage reached its deadline")
        time.sleep(1)


def merge(root, plan, layout, phase, complete):
    stage = "group" if phase in {"discovery", "calibration"} else phase
    output = root / stage / "pairs"
    subset = "calibration" if phase == "calibration" else "discovery"
    state_order = plan["state_ids"][subset]
    seen = set()
    for rank, gpu in enumerate(layout["gpus"]):
        # The plan lists the two episodes for each task consecutively.
        assigned = {state for index, state in enumerate(state_order) if index // 2 % len(layout["gpus"]) == rank}
        source = root / "workers" / str(rank) / phase / stage / "pairs"
        for path in source.glob("*.json"):
            row = read_sealed(path)
            key = row["panel"], row["state"], row["seed"], row["repetition"]
            if (row["plan_sha256"] != plan["sha256"] or row["stage"] != stage or row["panel"] not in PANELS[phase]
                    or row["state"] not in assigned or key in seen):
                raise ValueError(f"Missing or mismatched worker provenance: {path}")
            seen.add(key)
            data = {k: v for k, v in row.items() if k != "sha256"}
            atomic_json(output / path.name, seal({**data, "gpu": gpu}))
    expected = set()
    for panel in PANELS[phase]:
        if panel in {"text", "profile"}:
            expected.update((panel, state, nx.SEEDS[0], 0) for state in state_order[::2])
        else:
            repetitions = nx.REPETITIONS if panel == "main" else 1
            expected.update((panel, state, seed, repeat) for state in state_order
                            for seed in nx.SEEDS for repeat in range(repetitions))
    if not seen <= expected or (complete and seen != expected):
        raise ValueError(f"{phase}: expected {len(expected)} paired rows, received {len(seen)}")
    return len(seen)


def coordinate(root, source, gpus, cutoff):
    root.mkdir()
    protocol = prepare(source, root / "protocol")
    children = []
    phase = "loading"
    with ExitStack() as locks:
        devices = []
        for index in gpus:
            uuid, _ = choose_gpu(index)
            lock = locks.enter_context((ROOT / f".cache/e2-reproduction-{uuid}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            devices.append({"index": index, "uuid": uuid})
        started = time.time()
        plan = nx.freeze_plan(protocol, root, [gpu["uuid"] for gpu in devices], "text-offload",
                              min(started, cutoff - 300 * 60))
        deadlines = {name: min(started + minutes * 60, cutoff - 120)
                     for name, minutes in PHASE_MINUTES.items()}
        layout = seal({"plan_sha256": plan["sha256"], "gpus": devices, "deadlines": deadlines,
                       "cutoff_unix": cutoff, "launcher_sha256": file_hash(Path(__file__).resolve())})
        write_json(root / "parallel.json", layout)
        workers = [root / "workers" / str(rank) for rank in range(len(devices))]
        def update(status, **extra):
            atomic_json(root / "status.json", {"status": status, "phase": phase, "gpus": gpus,
                        "updated_at": datetime.now().astimezone().isoformat(), **extra})

        def stop(_signum, _frame):
            raise KeyboardInterrupt("Parallel experiment stopped")

        signal.signal(signal.SIGTERM, stop)
        try:
            update("loading")
            for rank, directory in enumerate(workers):
                directory.mkdir(parents=True)
                with (directory / "worker.log").open("w") as log:
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                        "--output", str(root), "--worker", str(rank)], stdout=log, stderr=subprocess.STDOUT)
                    children.append(child)
            wait_for([d / "identity.json" for d in workers], children, deadlines["warm"])
            identities = [read_sealed(d / "identity.json") for d in workers]
            for rank, identity in enumerate(identities):
                if (identity["plan_sha256"] != plan["sha256"] or identity["gpu"] != devices[rank]
                        or identity["model"]["sha256"] != identities[0]["model"]["sha256"]):
                    raise ValueError("GPU workers disagree on model or run identity")
                nx.check_parity(np.asarray(identities[0]["reference"]), np.asarray(identity["reference"]), "GPU parity")

            choices = None
            for phase in PHASE_MINUTES:
                update("running")
                payload = {"plan_sha256": plan["sha256"]}
                if phase == "discovery":
                    if not read_sealed(root / "overhead/report.json")["group_gate"]:
                        atomic_json(root / "start_discovery.json", seal({**payload, "skip": True}))
                        break
                    payload["choices"] = {f"group_{i}": [i] for i in range(8)}
                if phase == "calibration":
                    payload["choices"] = choices
                atomic_json(root / f"start_{phase}.json", seal(payload))
                wait_for([d / f"done_{phase}.json" for d in workers], children, deadlines[phase])
                count = merge(root, plan, layout, phase, complete=True)
                if phase in {"warm", "overhead", "calibration"}:
                    nx.stage_report(root, "group" if phase == "calibration" else phase, plan, "complete")
                else:
                    rows = [read_sealed(p) for p in (root / "group/pairs").glob("*.json")]
                    scores = {name: nx.task_mean(rows, lambda row: row["cases"][name]["prefix_rmse"])
                              for name in payload["choices"]}
                    ranked = sorted(range(8), key=lambda i: (scores[f"group_{i}"], i))
                    write_json(root / "group/ranking.json", seal({"plan_sha256": plan["sha256"],
                              "ascending_groups": ranked, "scores": scores, "split": "discovery"}))
                    choices = {"real_reuse": ranked[:2], "e2_low": plan["e2_low_pair"], "attention": "attention"}
                    choices.update({f"random_{i+1}": pair for i, pair in enumerate(plan["random_pairs"])})
                print(f"{phase}: merged {count} paired rows", flush=True)
            for child in children:
                if child.wait(timeout=max(1, cutoff - time.time())):
                    raise RuntimeError("A GPU worker failed on exit")
            update("complete")
        except (Exception, KeyboardInterrupt) as error:
            failure = "budget_exhausted" if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else "failed"
            update(failure, error=str(error))
            if phase in PHASE_MINUTES and (root / f"start_{phase}.json").exists():
                merge(root, plan, layout, phase, complete=False)
                name = "group" if phase in {"discovery", "calibration"} else phase
                nx.stage_report(root, name, plan, failure)
            raise
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            nx.summarize_next(root)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--e2-run", type=Path)
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--deadline", type=datetime.fromisoformat)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.worker is not None:
        worker(root, args.worker)
        return
    if (args.e2_run is None or args.deadline is None or args.deadline.tzinfo is None
            or args.deadline.timestamp() <= time.time() + 120
            or len(args.gpus or []) != 4 or len(set(args.gpus)) != 4):
        parser.error("Provide an E2 run, four distinct GPUs and a future deadline with timezone")
    coordinate(root, args.e2_run.resolve(), args.gpus, args.deadline.timestamp())


if __name__ == "__main__":
    main()
