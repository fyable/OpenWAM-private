"""Predeclared development-task follow-ups under one wall-clock deadline.

Ranks are fitted only on Discovery. Token gating and stale-V replacement are
mechanistic probes, not implementations of token pruning or cache acceleration.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = os.environ.get("E2_CPU_THREADS_PER_WORKER", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks.libero import research
from openwam.research.analysis import conditional_association
from openwam.research.exploratory import PROTOCOL, select_states, validate_coverage
from openwam.research.protocol import digest, file_hash, read_sealed, seal, token_groups, write_json
from scripts.research_e2_robustness import atomic_json, bootstrap, verify_result
from scripts.research_parallel_worker import serialize_model_loading, validate_cross_gpu_checks

SUBSETS = ("calibration", "discovery")
STRATEGIES = ("low", "high", "uniform", "random0", "random1", "random2", "random3", "random4")


def phase_definitions():
    # Order is fixed before observing extension outcomes. Later phases are optional
    # if the common deadline interrupts the queue; incomplete phases stay labelled.
    return [
        {"id": "group_controls_43", "family": "group_controls", "seeds": [43], "states": "original"},
        {"id": "policy_seeds_44_45", "family": "noise", "seeds": [44, 45], "states": "original"},
        {"id": "group_controls_44_45", "family": "group_controls", "seeds": [44, 45], "states": "original"},
        {"id": "new_episodes_noise", "family": "noise", "seeds": [43], "states": "new_episodes"},
        {"id": "new_episodes_controls", "family": "group_controls", "seeds": [43], "states": "new_episodes"},
        {"id": "extra_directions_amplitudes", "family": "noise_sweep", "seeds": [43], "states": "original"},
    ]


def discovery_ranking(directory):
    plan = read_sealed(directory / "plan.json")
    model = read_sealed(directory / "manifest.json")["model"]
    methods = {method["method_id"]: method for method in plan["methods"]}
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen = set()
    for path in directory.glob("*.json"):
        row = json.loads(path.read_text())
        if "method_id" not in row:
            continue
        verify_result(row, row["state_id"], row["method_id"], model["sha256"])
        if row["plan_sha256"] != plan["sha256"]:
            raise ValueError("Discovery ranking plan changed")
        key = row["state_id"], row["method_id"]
        if key in seen:
            raise ValueError("Duplicate Discovery ranking result")
        seen.add(key)
        method = methods[row["method_id"]]
        config = method["intervention"]
        group = int(row["method_id"].split("_G")[1].split("_")[0])
        position = config["layer_ids"][0], config["step_ids"][0], group
        values[position][(row["suite"], row["task_id"])][row["episode_id"]].append(
            row["metrics"]["prefix_normalized_rmse"] / config["epsilon"])
    expected = {(state, method) for state in plan["state_ids"] for method in methods}
    if seen != expected or len(seen) != 2304:
        raise ValueError("Ranking requires the complete frozen Discovery E2 baseline")
    output = {}
    for layer in (0, 29):
        for step in (1, 5, 9):
            scores = []
            for group in range(8):
                tasks = values[(layer, step, group)]
                if len(tasks) != 8:
                    raise ValueError("Ranking requires all eight Discovery tasks")
                scores.append(float(np.mean([np.mean([np.mean(rows) for rows in episodes.values()])
                                             for episodes in tasks.values()])))
            output[f"{layer}:{step}"] = {"scores": scores,
                "ascending": sorted(range(8), key=lambda group: (scores[group], group))}
    return output


def group_methods(selection, ranking, floor):
    methods = []
    for layer in (0, 29):
        groups = token_groups(selection["layouts"][str(layer)])
        if len(groups) != 8:
            raise ValueError("Follow-ups require the frozen eight-group layout")
        for step in (1, 5, 9):
            order = ranking[f"{layer}:{step}"]["ascending"]
            random_orders = [np.random.default_rng(20261117 + layer * 1000 + step * 10 + draw).permutation(8).tolist()
                             for draw in range(5)]
            for count in (2, 4, 6):
                choices = {"low": order[:count], "high": order[-count:],
                           "uniform": np.linspace(0, 7, count).round().astype(int).tolist()}
                choices.update({f"random{draw}": values[:count] for draw, values in enumerate(random_orders)})
                for kind in ("gate", "value_stale"):
                    for strategy in STRATEGIES:
                        selected = sorted(choices[strategy])
                        tokens = sorted(token for group in selected for token in groups[group])
                        methods.append({"method_id": f"C_L{layer}_S{step}_K{count}_{kind}_{strategy}",
                            "strategy": strategy, "group_count": count, "group_ids": selected,
                            "intervention": {"layer_ids": [layer], "step_ids": [step], "kind": kind,
                                             "gate": 0.0, "token_ids": tokens, "floor": floor}})
    return methods


def noise_methods(selection, floor, sweep=False):
    result = []
    for layer in (0, 29):
        groups = token_groups(selection["layouts"][str(layer)])
        for step in (1, 5, 9):
            for epsilon in ((0.01, 0.03, 0.10) if sweep else (0.03,)):
                for noise_seed in ((20261118, 20261119) if sweep else (20261112,)):
                    for group, tokens in enumerate(groups):
                        for sign in (-1, 1):
                            result.append({"method_id": f"N_L{layer}_S{step}_G{group}_E{epsilon}_N{noise_seed}_sign{sign}",
                                "group_id": group,
                                "intervention": {"kind": "value_noise", "layer_ids": [layer], "step_ids": [step],
                                    "token_ids": tokens, "epsilon": epsilon, "noise_seed": noise_seed,
                                    "floor": floor, "sign": sign}})
    return result


def extra_states(bank, original_ids):
    selected = set(original_ids)
    episodes = {(state["suite"], state["task_id"], state["episode_id"])
                for state in bank["states"] if state["state_id"] in selected}
    candidates = [state for state in bank["states"]
                  if (state["suite"], state["task_id"], state["episode_id"]) not in episodes]
    ids, coverage = select_states(candidates)
    validate_coverage([state for state in candidates if state["state_id"] in ids], PROTOCOL)
    return ids, coverage


def make_plan(run, protocol):
    selection = read_sealed(run / "base/manifests/selection.json")
    calibration = read_sealed(run / "base/diagnostics/features_calibration/calibration.json")
    if selection["sha256"] != protocol["selection_sha256"] or calibration["sha256"] != protocol["calibration_sha256"]:
        raise ValueError("Frozen selection/calibration changed")
    ranking = discovery_ranking(run / "base/diagnostics/e2_discovery")
    original = selection["state_ids_by_split"]
    new, coverage, banks, model_hashes = {}, {}, {}, {}
    for subset in SUBSETS:
        bank = read_sealed(run / f"bank/state_bank/{subset}/manifest.json")
        new[subset], coverage[subset] = extra_states(bank, original[subset])
        banks[subset] = bank["sha256"]
        model = read_sealed(run / f"bank/state_bank/{subset}/model.json")
        model_hashes[subset] = model["sha256"]
    if len(set(model_hashes.values())) != 1:
        raise ValueError("Fresh model identities differ between splits")
    phases = []
    for definition in protocol["phases"]:
        methods = group_methods(selection, ranking, calibration["floor"]) if definition["family"] == "group_controls" else (
            noise_methods(selection, calibration["floor"], definition["family"] == "noise_sweep"))
        phase = {**definition, "methods": methods,
                 "state_ids_by_split": original if definition["states"] == "original" else new}
        phase["expected_pairs"] = sum(map(len, phase["state_ids_by_split"].values())) * len(methods) * len(phase["seeds"])
        phases.append(phase)
    return seal({"protocol_sha256": protocol["sha256"], "deadline_unix": protocol["deadline_unix"],
        "source_sha256": research.source_identity()["source_sha256"], "ranking": ranking,
        "ranking_fitted_split": "discovery", "ranking_aggregation": "states/signs within episode, episodes within task, equal task weights",
        "phases": phases, "new_episode_coverage": coverage, "bank_sha256_by_split": banks,
        "model_sha256_by_split": model_hashes, "gpus": protocol["gpus"],
        "checkpoint_dir": protocol["checkpoint_dir"], "checkpoint_name": protocol["checkpoint_name"],
        "selection_sha256": selection["sha256"], "calibration_sha256": calibration["sha256"],
        "tool_sha256": file_hash(__file__), "expected_pairs": sum(phase["expected_pairs"] for phase in phases)})


def result_path(root, phase, subset, state, seed, method):
    return root / "results" / phase / subset / (digest([state, seed, method]) + ".json")


def verify_extension(row, plan, phase, subset, state, seed, method):
    metric = row.get("metrics", {}).get("prefix_normalized_rmse")
    if (row.get("status") != "completed" or row.get("valid_run") is not True
        or row.get("extension_sha256") != plan["sha256"] or row.get("phase_id") != phase
        or row.get("state_id") != state or row.get("method_id") != method
        or row.get("action_seed") != seed or row.get("video_seed") != seed
        or row.get("steps") != 10 or row.get("split") != subset
        or row.get("model_sha256") != plan["model_sha256_by_split"][subset]
        or metric is None or not np.isfinite(metric) or metric < 0):
        raise ValueError("Invalid or incompatible extension result")


def worker(args):
    from openwam.research.interventions import Intervention
    from openwam.research.runner import infrastructure_retries, run_pair, run_t0

    run = args.run
    plan = read_sealed(run / "extensions/plan.json")
    gpu = plan["gpus"][args.worker_index]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu["uuid"]:
        raise ValueError("Extension GPU assignment changed")
    root = run / f"extensions/workers/worker_{args.worker_index}"
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "assignment.json", seal({"plan_sha256": plan["sha256"], "gpu": gpu,
        "worker_index": args.worker_index, "worker_count": len(plan["gpus"])}))
    completed, current_phase = 0, "preflight"
    def status(value, **extra):
        atomic_json(root / "status.json", {"status": value, "phase": current_phase, "completed_pairs": completed,
                                           "updated_unix": time.time(), **extra})
    status("running")
    try:
        os.environ["E2_TOTAL_DEADLINE"] = str(plan["deadline_unix"])
        serialize_model_loading()
        model_args = SimpleNamespace(ckpt_dir=plan["checkpoint_dir"], ckpt_name=plan["checkpoint_name"],
                                      steps=10, device="cuda:0", horizon=None)
        if time.time() >= plan["deadline_unix"]:
            status("budget_exhausted", completed_phases={})
            return
        server = research.build_server(model_args)
        identity = research.model_identity(server, model_args)
        if identity["sha256"] != plan["model_sha256_by_split"]["discovery"]:
            raise ValueError("Extension model identity differs from the fresh baseline")
        validate_cross_gpu_checks(run, identity["sha256"])
        split = read_sealed(run / "manifests/split.json")
        mappings, horizons = {}, {}
        for subset in SUBSETS:
            bank, states = research.load_states(run / f"bank/state_bank/{subset}", split)
            if bank["sha256"] != plan["bank_sha256_by_split"][subset]:
                raise ValueError("Extension state bank changed")
            research.validate_t0(read_sealed(run / f"base/diagnostics/t0_{subset}/t0_certificate.json"), identity, bank, 10)
            mappings[subset] = {entry["state_id"]: (entry, condition) for entry, condition in states}
            horizons[subset] = research.execution_horizon(server, states[0][1], model_args)
            chosen = states[::max(1, len(states)//3)][:3]
            check = run_t0(server.engine, [(entry["state_id"], condition) for entry, condition in chosen],
                seeds=[43, 44, 45], steps=10, horizon=horizons[subset], output=root / f"checks/{subset}/t0.json")
            if check["status"] != "passed":
                raise ValueError(f"Extension repeat/gate-one check failed: {subset}")
        phase_counts = {}
        for phase in plan["phases"]:
            current_phase = phase["id"]
            phase_completed = 0
            expected = sum(len(ids[args.worker_index::len(plan['gpus'])])
                           for ids in phase["state_ids_by_split"].values()) * len(phase["seeds"]) * len(phase["methods"])
            print(f"Phase {current_phase}: expected {expected} pairs", flush=True)
            for subset in SUBSETS:
                for seed in phase["seeds"]:
                    for state in phase["state_ids_by_split"][subset][args.worker_index::len(plan["gpus"])]:
                        entry, conditions = mappings[subset][state]
                        for method in phase["methods"]:
                            if time.time() >= plan["deadline_unix"]:
                                status("budget_exhausted", phase_pairs=phase_completed, expected_phase_pairs=expected,
                                       completed_phases=phase_counts)
                                return
                            result, attempts = infrastructure_retries(lambda: run_pair(server.engine, conditions,
                                Intervention.from_dict(method["intervention"]), seed=seed, steps=10, horizon=horizons[subset]))
                            row = {**entry, **result, "method_id": method["method_id"], "action_seed": seed,
                                   "video_seed": seed, "steps": 10, "prefix_horizon": horizons[subset],
                                   "status": "completed", "valid_run": True, "retry_attempts": attempts,
                                   "phase_id": current_phase, "extension_sha256": plan["sha256"],
                                   "gpu_uuid": gpu["uuid"], "model_sha256": identity["sha256"]}
                            verify_extension(row, plan, current_phase, subset, state, seed, method["method_id"])
                            write_json(result_path(root, current_phase, subset, state, seed, method["method_id"]), seal(row))
                            completed += 1
                            phase_completed += 1
                            if completed == 1 or completed % 48 == 0:
                                status("running", phase_pairs=phase_completed, expected_phase_pairs=expected,
                                       completed_phases=phase_counts)
                                print(f"{current_phase}: {phase_completed}/{expected}; total={completed}", flush=True)
            if phase_completed != expected:
                raise ValueError("Extension phase count mismatch")
            phase_counts[current_phase] = phase_completed
            status("running", completed_phases=phase_counts)
        status("complete", completed_phases=phase_counts)
    except BaseException as error:
        status("failed", error_type=type(error).__name__, error=str(error))
        raise


def summarize(run, plan, status):
    root = run / "extensions"
    phases = {phase["id"]: phase for phase in plan["phases"]}
    rows_by_phase = defaultdict(list)
    seen = set()
    for path in (root / "workers").glob("worker_*/results/*/*/*.json"):
        row = read_sealed(path)
        phase = phases[row["phase_id"]]
        key = row["phase_id"], row["split"], row["state_id"], row["action_seed"], row["method_id"]
        if key in seen:
            raise ValueError("Duplicate extension pair")
        if row["state_id"] not in phase["state_ids_by_split"][row["split"]] or row["action_seed"] not in phase["seeds"]:
            raise ValueError("Extension result outside the frozen plan")
        verify_extension(row, plan, *key)
        seen.add(key)
        rows_by_phase[row["phase_id"]].append(row)
    phase_summaries, control_sections, seed_sections, episode_sections = [], [], [], []
    base_vectors = baseline_vectors(run)
    bank_entries = {subset: {entry["state_id"]: entry for entry in read_sealed(
        run / f"bank/state_bank/{subset}/manifest.json")["states"]} for subset in SUBSETS}
    for phase in plan["phases"]:
        rows = rows_by_phase[phase["id"]]
        phase_summaries.append({"phase": phase["id"], "completed_pairs": len(rows),
            "expected_pairs": phase["expected_pairs"], "complete": len(rows) == phase["expected_pairs"]})
        methods = {method["method_id"]: method for method in phase["methods"]}
        if phase["family"] == "group_controls":
            paired = defaultdict(dict)
            for row in rows:
                method = methods[row["method_id"]]
                config = method["intervention"]
                key = (row["split"], row["state_id"], row["action_seed"], config["layer_ids"][0],
                       config["step_ids"][0], method["group_count"], config["kind"])
                paired[key][method["strategy"]] = row
            comparisons = defaultdict(list)
            for key, strategies in paired.items():
                if set(strategies) != set(STRATEGIES):
                    continue
                low = strategies["low"]
                metric = low["metrics"]["prefix_normalized_rmse"]
                random = np.mean([strategies[f"random{draw}"]["metrics"]["prefix_normalized_rmse"] for draw in range(5)])
                comparisons[(key[0], *key[2:])].append({**{name: low[name] for name in ("suite", "task_id", "episode_id", "state_id")},
                    "low_rmse": metric, "random_rmse": float(random), "low_minus_random": metric - float(random),
                    "low_minus_high": metric - strategies["high"]["metrics"]["prefix_normalized_rmse"],
                    "low_minus_uniform": metric - strategies["uniform"]["metrics"]["prefix_normalized_rmse"]})
            for condition, records in comparisons.items():
                metrics = {key: bootstrap(records, key, 10000) for key in (
                    "low_rmse", "random_rmse", "low_minus_random", "low_minus_high", "low_minus_uniform")}
                control_sections.append({"phase": phase["id"], "split": condition[0], "policy_seed": condition[1],
                    "layer": condition[2], "step": condition[3], "group_count": condition[4], "kind": condition[5],
                    "states": len(records), "complete": len(records) == 24,
                    "primary_endpoint": condition[0] == "calibration", "metrics": metrics})
        else:
            responses = defaultdict(dict)
            metadata = {}
            for row in rows:
                method = methods[row["method_id"]]
                config = method["intervention"]
                key = (row["split"], row["state_id"], row["action_seed"], config["layer_ids"][0],
                       config["step_ids"][0], config["epsilon"], config["noise_seed"], method["group_id"])
                responses[key][config["sign"]] = row["metrics"]["prefix_normalized_rmse"]
                metadata[(row["split"], row["state_id"])] = row
            groups = defaultdict(dict)
            for key, signs in responses.items():
                if set(signs) == {-1, 1}:
                    groups[key[:-1]][key[-1]] = float(np.mean(list(signs.values())) / key[5])
            comparisons = defaultdict(list)
            for key, values in groups.items():
                if set(values) != set(range(8)):
                    continue
                subset, state, seed, layer, step, epsilon, noise_seed = key
                reference = base_vectors.get((subset, state, layer, step))
                if reference is None:
                    continue
                association = conditional_association(reference, [values[group] for group in range(8)])
                row = metadata[(subset, state)]
                comparisons[(subset, seed, layer, step, epsilon, noise_seed)].append({
                    **{name: row[name] for name in ("suite", "task_id", "episode_id", "state_id")}, **association})
            for condition, records in comparisons.items():
                seed_sections.append({"phase": phase["id"], "condition": list(condition), "states": len(records),
                    "complete": len(records) == 24, "reference": "same state, policy seed 43, epsilon .03, noise seed 20261112",
                    "metrics": {key: bootstrap(records, key, 10000) for key in ("spearman", "top25_overlap")}})
            if phase["states"] == "new_episodes":
                # Compare task-balanced group rankings on independent saved episodes.
                conditions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
                references = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
                for key, values in groups.items():
                    if len(values) != 8:
                        continue
                    subset, state, seed, layer, step, epsilon, noise_seed = key
                    row = metadata[(subset, state)]
                    task = row["suite"], row["task_id"]
                    condition = subset, seed, layer, step, epsilon, noise_seed
                    conditions[condition][task][row["episode_id"]].append([values[group] for group in range(8)])
                for (subset, state, layer, step), vector in base_vectors.items():
                    entry = bank_entries[subset][state]
                    references[(subset, layer, step)][(entry["suite"], entry["task_id"])][entry["episode_id"]].append(vector)
                for condition, tasks in conditions.items():
                    subset, seed, layer, step, epsilon, noise_seed = condition
                    reference_tasks = references[(subset, layer, step)]
                    def average(values):
                        return np.mean([np.mean([np.mean(vectors, axis=0) for vectors in episodes.values()], axis=0)
                                        for episodes in values.values()], axis=0)
                    actual, reference = average(tasks), average(reference_tasks)
                    episode_sections.append({"phase": phase["id"], "condition": list(condition),
                        "tasks": len(tasks), "complete": len(tasks) == 8 and sum(
                            len(vectors) for episodes in tasks.values() for vectors in episodes.values()) == 24,
                        "new_group_sensitivity": actual.tolist(), "original_group_sensitivity": reference.tolist(),
                        **conditional_association(reference, actual)})
    result = {"status": status, "plan_sha256": plan["sha256"], "completed_pairs": len(seen),
        "expected_pairs": plan["expected_pairs"], "phases": phase_summaries,
        "group_control_comparisons": control_sections, "ranking_stability": seed_sections,
        "new_episode_group_rankings": episode_sections,
        "limitations": ["Development tasks and saved-observation interventions only; no closed-loop success or speedup evidence",
                        "Stale V uses the preceding baseline denoise step; sigma mismatch remains part of this probe",
                        "Bootstrap intervals are descriptive; no familywise multiplicity correction or deployment claim",
                        "Only fully matched control sets enter contrasts; incomplete phases remain incomplete",
                        "Additional states use different saved episode IDs, not unseen tasks"]}
    atomic_json(root / "report.json", result)
    lines = ["# E2 development follow-ups", "", f"Status: {status}; completed {len(seen)}/{plan['expected_pairs']} pairs.", "",
             "| Phase | Completed | Planned | Complete |", "|---|---:|---:|---|"]
    lines += [f"| {phase['phase']} | {phase['completed_pairs']} | {phase['expected_pairs']} | {phase['complete']} |"
              for phase in phase_summaries]
    lines += ["", "Detailed paired controls and episode/task bootstrap intervals are in report.json.", "",
              *["- " + limitation for limitation in result["limitations"]]]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def baseline_vectors(run):
    paired = defaultdict(dict)
    for subset in SUBSETS:
        directory = run / f"base/diagnostics/e2_{subset}"
        methods = {method["method_id"]: method for method in read_sealed(directory / "plan.json")["methods"]}
        for path in directory.glob("*.json"):
            row = json.loads(path.read_text())
            if "method_id" not in row:
                continue
            config = methods[row["method_id"]]["intervention"]
            group = int(row["method_id"].split("_G")[1].split("_")[0])
            key = subset, row["state_id"], config["layer_ids"][0], config["step_ids"][0], group
            paired[key][config["sign"]] = row["metrics"]["prefix_normalized_rmse"] / config["epsilon"]
    grouped = defaultdict(dict)
    for key, signs in paired.items():
        if set(signs) == {-1, 1}:
            grouped[key[:-1]][key[-1]] = float(np.mean(list(signs.values())))
    return {key: [values[group] for group in range(8)] for key, values in grouped.items() if len(values) == 8}


def supervisor(args):
    run = args.run
    root = run / "extensions"
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = read_sealed(run / "base/manifests/selection.json")
        calibration = read_sealed(run / "base/diagnostics/features_calibration/calibration.json")
        execution = read_sealed(run / "parallel/execution.json")
        recovery = run / "parallel/recovery.json"
        if recovery.exists():
            execution = read_sealed(recovery)
        protocol_payload = {"version": "e2_ten_hour_followups_v1", "deadline_unix": args.deadline,
            "deadline_local": datetime.fromtimestamp(args.deadline).isoformat(), "phases": phase_definitions(),
            "selection_sha256": selection["sha256"], "calibration_sha256": calibration["sha256"],
            "checkpoint_dir": json.loads((run / "environment.json").read_text())["checkpoint_dir"],
            "checkpoint_name": "checkpoint_step_10690.safetensors", "gpus": execution["gpu_workers"],
            "primary_endpoint": "Calibration paired RMSE: perturb low sensitivity groups versus five matched random controls",
            "group_counts": [2, 4, 6], "strategies": list(STRATEGIES), "source_sha256": research.source_identity()["source_sha256"]}
        if (root / "protocol.json").exists():
            if not args.revise_deadline or (root / "plan.json").exists():
                raise ValueError("A frozen extension protocol already exists; deadline revision is allowed only before extension outcomes")
            previous = read_sealed(root / "protocol.json")
            history = root / "protocol_history" / previous["sha256"]
            write_json(history / "protocol.json", previous)
            write_json(history / "budget.json", read_sealed(run / "budget.json"))
            protocol_payload["supersedes_sha256"] = previous["sha256"]
            protocol_payload["revision_reason"] = ("Recover core worker GPU assignments after loading allocation failure; deadline unchanged"
                if previous["deadline_unix"] == args.deadline else "User selected approximately ten additional hours from the current time")
            protocol = seal(protocol_payload)
            atomic_json(root / "protocol.json", protocol)
        else:
            protocol = seal(protocol_payload)
            write_json(root / "protocol.json", protocol)
        atomic_json(run / "budget.json", seal({"deadline_unix": args.deadline, "deadline_local": protocol["deadline_local"],
            "scope": "original run plus follow-ups", "protocol_sha256": protocol["sha256"]}))
        children = []
        try:
            while True:
                core = json.loads((run / "parallel_status.json").read_text())
                atomic_json(root / "status.json", {"status": "waiting_core", "core_stage": core["stage"],
                    "deadline_unix": args.deadline, "pid": os.getpid(), "updated_unix": time.time()})
                if core["status"] in {"complete", "budget_exhausted"}:
                    break
                if core["status"] == "failed":
                    raise RuntimeError("Core study failed; follow-ups must not run")
                os.kill(args.core_pid, 0)
                if time.time() >= args.deadline:
                    atomic_json(root / "status.json", {"status": "budget_exhausted", "reason": "Core consumed the total budget"})
                    return
                time.sleep(10)
            if time.time() >= args.deadline or core["status"] != "complete":
                atomic_json(root / "status.json", {"status": "budget_exhausted", "reason": "No complete core/remaining budget"})
                return
            if research.source_identity()["source_sha256"] != protocol["source_sha256"]:
                raise ValueError("Research source changed while awaiting the core study")
            plan = make_plan(run, protocol)
            write_json(root / "plan.json", plan)
            for index, gpu in enumerate(plan["gpus"]):
                while time.time() < args.deadline:
                    free = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu["index"]), "--query-gpu=memory.free",
                                                        "--format=csv,noheader,nounits"], text=True).strip())
                    if free >= int(os.environ.get("E2_MIN_FREE_MIB", "28000")):
                        break
                    time.sleep(10)
                else:
                    break
                log = root / f"worker_{index}.log"
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu["uuid"], "MUJOCO_EGL_DEVICE_ID": str(gpu["index"])}
                with log.open("a") as stream:
                    process = subprocess.Popen([sys.executable, __file__, "--run", str(run), "--worker-index", str(index)],
                        cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(process)
            while any(process.poll() is None for process in children):
                counts = []
                for index in range(len(children)):
                    path = root / f"workers/worker_{index}/status.json"
                    if path.exists():
                        counts.append(json.loads(path.read_text()))
                atomic_json(root / "status.json", {"status": "running", "pid": os.getpid(), "workers": counts,
                    "completed_pairs": sum(row["completed_pairs"] for row in counts), "expected_pairs": plan["expected_pairs"],
                    "deadline_unix": args.deadline, "updated_unix": time.time()})
                if any(process.poll() not in (None, 0) for process in children):
                    raise RuntimeError("Extension worker failed; see extensions/worker_*.log")
                time.sleep(10)
            if any(process.returncode != 0 for process in children):
                raise RuntimeError("Extension worker failed")
            statuses = [json.loads((root / f"workers/worker_{index}/status.json").read_text())["status"]
                        for index in range(len(children))]
            status = "complete" if len(children) == len(plan["gpus"]) and all(value == "complete" for value in statuses) else "budget_exhausted"
            summarize(run, plan, status)
            atomic_json(root / "status.json", {"status": status, "deadline_unix": args.deadline, "report": str(root / "report.md")})
        except BaseException as error:
            for process in children:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
            atomic_json(root / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
            if (root / "plan.json").exists():
                summarize(run, read_sealed(root / "plan.json"), "failed")
            raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--core-pid", type=int)
    parser.add_argument("--deadline", type=lambda value: datetime.fromisoformat(value).timestamp())
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--revise-deadline", action="store_true", help="Archive and revise the waiting protocol before extension outcomes")
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.worker_index is None and (args.core_pid is None or args.deadline is None):
        parser.error("Supervisor requires --core-pid and --deadline")
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    if args.worker_index is None:
        supervisor(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
