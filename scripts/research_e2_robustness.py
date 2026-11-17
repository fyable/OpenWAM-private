"""Bounded, resumable H2 robustness study; diagnostic timings are not speedups."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import time

import numpy as np

from benchmarks.libero.research import source_identity
from openwam.research.analysis import conditional_association
from openwam.research.protocol import digest, file_hash, read_sealed, seal, token_groups, write_json


SUBSETS = ("discovery", "calibration")
OLD_SEED = 20261112
NEW_SEED = 20261114


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    write_json(temporary, value)
    os.replace(temporary, path)


def methods(selection, floor):
    result = []
    for stage, seed, epsilons, steps in (
        ("A", NEW_SEED, (0.03,), (1, 5, 9)),
        ("B", OLD_SEED, (0.01, 0.10), (9,)),
    ):
        for layer in (0, 29):
            groups = token_groups(selection["layouts"][str(layer)])
            if len(groups) != 8:
                raise ValueError("This protocol requires eight future-token groups")
            for step in steps:
                for epsilon in epsilons:
                    for group, tokens in enumerate(groups):
                        for sign in (-1, 1):
                            result.append({
                                "stage": stage, "group_id": group,
                                "method_id": f"R_{stage}_L{layer}_S{step}_G{group}_N{seed}_E{epsilon}_sign{sign}",
                                "intervention": {"kind": "value_noise", "layer_ids": [layer],
                                                 "step_ids": [step], "token_ids": tokens,
                                                 "epsilon": epsilon, "noise_seed": seed,
                                                 "floor": floor, "sign": sign},
                            })
    return result


def verify_result(row, state, method, model_hash, plan_hash=None):
    if (row.get("status") != "completed" or not row.get("valid_run")
        or row.get("state_id") != state or row.get("method_id") != method
        or row.get("model_sha256") != model_hash or row.get("action_seed") != 43
        or row.get("video_seed") != 43 or row.get("steps") != 10
        or (plan_hash is not None and row.get("robustness_sha256") != plan_hash)):
        raise ValueError("Incomplete or incompatible result; refusing reuse")
    metric = row.get("metrics", {}).get("prefix_normalized_rmse")
    if metric is None or not np.isfinite(metric) or metric < 0:
        raise ValueError("Missing/non-finite result metric")


def prepare(args):
    base, bank_root = args.base.resolve(), args.bank_run.resolve()
    selection = read_sealed(base / "manifests/selection.json")
    calibration = read_sealed(base / "diagnostics/features_calibration/calibration.json")
    if (selection["layer_ids"] != [0, 29] or selection["step_ids"] != [1, 5, 9]
        or calibration["selection_sha256"] != selection["sha256"]
        or calibration["fitted_split"] != "calibration"):
        raise ValueError("Unexpected frozen H2 selection/calibration")
    inputs = {}
    def track(path):
        inputs[str(path.resolve())] = file_hash(path)
    track(base / "manifests/selection.json")
    track(base / "diagnostics/features_calibration/calibration.json")
    split = read_sealed(args.split)
    if split["sha256"] != selection["task_split_sha256"]:
        raise ValueError("Wrong task split")
    track(args.split)
    source = source_identity()
    identities, old_counts, old_noise = {}, {}, {}
    for subset in SUBSETS:
        directory = base / f"diagnostics/e2_{subset}"
        manifest = read_sealed(directory / "manifest.json")
        model = manifest["model"]
        if model["source_sha256"] != source["source_sha256"]:
            raise ValueError("Model source changed since original E2; no automatic source authorization")
        identities[subset] = model["sha256"]
        bank_path = bank_root / "state_bank" / subset
        bank = read_sealed(bank_path / "manifest.json")
        features = read_sealed(base / f"diagnostics/features_{subset}/features.json")
        plan = read_sealed(directory / "plan.json")
        ids = selection["state_ids_by_split"][subset]
        if (len(ids) != 24 or len(set(ids)) != 24 or plan["state_ids"] != ids
            or plan["selection_sha256"] != selection["sha256"]
            or plan["calibration_sha256"] != calibration["sha256"]
            or manifest["bank_sha256"] != bank["sha256"]
            or selection["bank_sha256_by_split"][subset] != bank["sha256"]
            or features["model_sha256"] != model["sha256"]
            or features["bank_sha256"] != bank["sha256"]
            or features["selection_sha256"] != selection["sha256"]
            or calibration["model_sha256"] != model["sha256"]
            or manifest["task_split_sha256"] != selection["task_split_sha256"]):
            raise ValueError("Original E2/bank/feature identity mismatch")
        for path in (directory / "manifest.json", directory / "plan.json",
                     base / f"diagnostics/features_{subset}/features.json",
                     base / f"diagnostics/t0_{subset}/t0.json",
                     base / f"manifests/source_review_{subset}.json",
                     bank_path / "manifest.json", bank_path / "model.json"):
            track(path)
        t0 = json.loads((base / f"diagnostics/t0_{subset}/t0.json").read_text())
        if t0["status"] != "passed":
            raise ValueError("Original T0 failed")
        old_noise[subset] = max(row["baseline_repeat_max_abs"] for row in t0["rows"])
        old_methods = {method["method_id"]: method for method in plan["methods"]}
        expected = {(state, method) for state in ids for method in old_methods}
        found = set()
        for path in sorted(directory.glob("*.json")):
            row = json.loads(path.read_text())
            if "method_id" not in row:
                continue
            key = row["state_id"], row["method_id"]
            if key not in expected or key in found or row["plan_sha256"] != plan["sha256"]:
                raise ValueError("Unexpected/duplicate original E2 result")
            verify_result(row, *key, model["sha256"])
            config = old_methods[key[1]]["intervention"]
            if config.get("noise_seed", OLD_SEED) != OLD_SEED or config["epsilon"] != 0.03:
                raise ValueError("Unexpected original perturbation")
            found.add(key)
            track(path)
        if found != expected or len(found) != 2304:
            raise ValueError("Original E2 must be complete: 2304 pairs per split")
        old_counts[subset] = len(found)
    spec = seal({"protocol": "e2_robustness_v1", "base": str(base), "bank_run": str(bank_root),
                 "source_sha256": source["source_sha256"], "inputs": inputs,
                 "model_sha256_by_split": identities, "selection": selection,
                 "calibration": calibration, "methods": methods(selection, calibration["floor"]),
                 "original_pair_counts": old_counts, "original_repeat_max_abs": old_noise,
                 "tool_sha256": file_hash(__file__), "bootstrap_seed": NEW_SEED,
                 "launcher_sha256": file_hash(Path(__file__).with_name("research_alpha_libero_e2_robustness.sh")),
                 "bootstrap_replicates": 10000, "threshold_rho": 0.5, "threshold_overlap": 0.5,
                 "hours": args.hours, "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "split_file": str(args.split.resolve()),
                 "checkpoint_dir": str(args.ckpt_dir.resolve()), "checkpoint_name": args.ckpt_name})
    path = args.run / "plan.json"
    if path.exists():
        if read_sealed(path) != spec:
            raise ValueError("Resume specification/input/tool/GPU changed")
    else:
        write_json(path, spec)
    print(f"Frozen {len(spec['methods']) * 48} new pairs; run={args.run}", flush=True)
    return spec


def pair_path(root, subset, state, method):
    return root / "results" / subset / f"{digest([state, method])}.json"


def due(start, hours, now=None):
    return (time.monotonic() if now is None else now) - start >= hours * 3600


def run_study(args, spec, started):
    from benchmarks.libero.research import build_server, execution_horizon, load_states, model_identity, validate_t0
    from openwam.research.exploratory import validate_bank_identity
    from openwam.research.interventions import Intervention, ResearchContext
    from openwam.research.runner import compare_contexts, infrastructure_retries, paired_generate, run_pair, run_t0

    base = Path(spec["base"])
    split = read_sealed(spec["split_file"])
    if split["sha256"] != spec["selection"]["task_split_sha256"]:
        raise ValueError("Wrong task split")
    args.steps, args.horizon, args.device = 10, None, "cuda:0"
    print("Loading model; no experiment pair has started yet", flush=True)
    server = build_server(args)
    identity = model_identity(server, args)
    states_by_split, horizons, floors = {}, {}, {}
    attempt = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for subset in SUBSETS:
        if due(started, args.hours):
            return "budget_exhausted"
        if identity["sha256"] != spec["model_sha256_by_split"][subset]:
            raise ValueError("Loaded model/backend identity differs from original E2")
        args.bank = str(args.bank_run / "state_bank" / subset)
        args.source_review = str(base / f"manifests/source_review_{subset}.json")
        bank, states = load_states(args.bank, split)
        validate_bank_identity(args, bank, identity)
        validate_t0(read_sealed(base / f"diagnostics/t0_{subset}/t0_certificate.json"), identity, bank, 10)
        ids = spec["selection"]["state_ids_by_split"][subset]
        mapping = {entry["state_id"]: (entry, conditions) for entry, conditions in states}
        selected = [mapping[state] for state in ids]
        states_by_split[subset] = selected
        horizon = execution_horizon(server, selected[0][1], args)
        horizons[subset] = horizon
        old_rows = [json.loads(path.read_text()) for path in (base / f"diagnostics/e2_{subset}").glob("*.json")]
        if any(row["prefix_horizon"] != horizon for row in old_rows if "method_id" in row):
            raise ValueError("Execution horizon changed")
        audit = args.run / "checks" / attempt / subset
        check_states = selected[::8][:3]
        recorded = {}
        for entry, _ in check_states:
            with np.load(Path(args.bank) / entry["file"], allow_pickle=False) as data:
                recorded[entry["state_id"]] = (43, data["actions"].copy())
        check = run_t0(server.engine, [(entry["state_id"], cond) for entry, cond in check_states],
                       seeds=[43], steps=10, horizon=horizon, output=audit / "t0.json", recorded_outputs=recorded)
        if check["status"] != "passed":
            raise ValueError("Fresh no-op/repeat check failed")
        floor_rows = []
        for entry, conditions in check_states:
            for layer in (0, 29):
                for step in (1, 5, 9):
                    if due(started, args.hours):
                        return "budget_exhausted"
                    config = Intervention(layer_ids=(layer,), step_ids=(step,), gate=1.0)
                    result = run_pair(server.engine, conditions, config, seed=43, steps=10, horizon=horizon)
                    reference, actual = ResearchContext(config), ResearchContext(config)
                    reference_result, _ = paired_generate(server.engine, conditions, seed=43, context=reference)
                    actual_result, _ = paired_generate(server.engine, conditions, seed=43, context=actual)
                    metrics, _, _ = compare_contexts(server.engine, reference, actual, reference_result, actual_result, horizon)
                    floor_rows.append({"state_id": entry["state_id"], "layer_id": layer, "step_id": step,
                                       "repeat_rmse": metrics["prefix_normalized_rmse"],
                                       "gate_one_rmse": result["metrics"]["prefix_normalized_rmse"]})
        write_json(audit / "normalized_floor.json", floor_rows)
        floors[subset] = floor_rows
        print(f"Fresh checks passed: {subset}, horizon={horizon}", flush=True)
    atomic_json(args.run / "noise_floor.json", seal({"plan_sha256": spec["sha256"], "rows": floors}))
    completed, measured, elapsed_pairs = 0, 0, 0.0
    total = len(spec["methods"]) * 48
    for stage in ("A", "B"):
        for subset in SUBSETS:
            for entry, conditions in states_by_split[subset]:
                for method in (value for value in spec["methods"] if value["stage"] == stage):
                    path = pair_path(args.run, subset, entry["state_id"], method["method_id"])
                    if path.exists():
                        saved = read_sealed(path)
                        verify_result(saved, entry["state_id"], method["method_id"], identity["sha256"], spec["sha256"])
                        completed += 1
                        continue
                    if due(started, args.hours):
                        return "budget_exhausted"
                    row = {**{key: value for key, value in entry.items() if key != "sha256"},
                           "state_file_sha256": entry["sha256"],
                           "method_id": method["method_id"], "action_seed": 43, "video_seed": 43,
                           "steps": 10, "prefix_horizon": horizons[subset], "model_sha256": identity["sha256"],
                           "robustness_sha256": spec["sha256"], "success": None}
                    try:
                        before = time.monotonic()
                        result, attempts = infrastructure_retries(lambda: run_pair(
                            server.engine, conditions, Intervention.from_dict(method["intervention"]),
                            seed=43, steps=10, horizon=horizons[subset]))
                        elapsed_pairs += time.monotonic() - before
                        measured += 1
                        row.update(result, status="completed", valid_run=True, retry_attempts=attempts)
                        verify_result(row, entry["state_id"], method["method_id"], identity["sha256"], spec["sha256"])
                        write_json(path, seal(row))
                    except Exception as error:
                        row.update(status="policy_failure" if isinstance(error, FloatingPointError) else "failed",
                                   valid_run=isinstance(error, FloatingPointError),
                                   error_type=type(error).__name__, error=str(error),
                                   retry_attempts=getattr(error, "research_attempts", []))
                        write_json(args.run / "errors" / attempt / f"{subset}_{path.name}", seal(row))
                        raise
                    completed += 1
                    if measured == 1 or measured % 100 == 0:
                        eta = (total - completed) * elapsed_pairs / measured / 3600
                        print(f"{stage}/{subset}: {completed}/{total} pairs; ETA {eta:.2f}h; "
                              f"budget left {max(0, args.hours - (time.monotonic()-started)/3600):.2f}h", flush=True)
                        atomic_json(args.run / "progress.json", {"completed_pairs": completed,
                                    "total_pairs": total, "eta_hours": eta, "stage": stage, "split": subset})
    return "complete"


def bootstrap(records, key, replicates=10000):
    episodes = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row[key] is not None:
            episodes[(row["suite"], row["task_id"])][row["episode_id"]].append(row[key])
    values = [np.array([np.mean(episodes[task][episode]) for episode in sorted(episodes[task])])
              for task in sorted(episodes)]
    if not values:
        return {"mean": None, "ci95": None, "tasks": 0, "episodes": 0}
    generator = np.random.default_rng(NEW_SEED)
    indices = generator.integers(len(values), size=(replicates, len(values)))
    samples = np.empty((replicates, len(values)))
    for position in range(len(values)):
        for task_index, task in enumerate(values):
            selected = indices[:, position] == task_index
            samples[selected, position] = task[generator.integers(
                len(task), size=(int(selected.sum()), len(task)))].mean(axis=1)
    draws = samples.mean(axis=1)
    return {"mean": float(np.mean([task.mean() for task in values])),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(), "tasks": len(values),
            "episodes": sum(len(task) for task in values)}


def vectors(rows):
    paired = defaultdict(dict)
    metadata = {}
    for row, config, group in rows:
        key = (row["state_id"], config["layer_ids"][0], config["step_ids"][0],
               config.get("noise_seed", OLD_SEED), config["epsilon"], group)
        if config["sign"] in paired[key]:
            raise ValueError("Duplicate signed response")
        paired[key][config["sign"]] = row["metrics"]["prefix_normalized_rmse"]
        metadata[row["state_id"]] = row
    groups = defaultdict(dict)
    asymmetry = {}
    for key, signs in paired.items():
        if set(signs) != {-1, 1}:
            continue
        groups[key[:-1]][key[-1]] = np.mean(list(signs.values())) / key[4]
        asymmetry[key] = abs(signs[1] - signs[-1]) / max(signs[1] + signs[-1], 1e-30)
    complete = {key: np.array([group[index] for index in range(8)])
                for key, group in groups.items() if set(group) == set(range(8))}
    return complete, metadata, asymmetry


def task_metrics(records):
    tasks = defaultdict(list)
    for row in records:
        tasks[(row["suite"], row["task_id"])].append(row)
    return [{"suite": suite, "task_id": task,
             **{key: bootstrap(values, key, 1)["mean"] for key in ("spearman", "top25_overlap")}}
            for (suite, task), values in sorted(tasks.items())]


def ranking_passes(records, metrics):
    return (len(records) == 24 and all(row["spearman"] is not None for row in records)
            and metrics["spearman"]["tasks"] == 8
            and metrics["spearman"]["mean"] >= 0.5
            and metrics["top25_overlap"]["mean"] >= 0.5)


def report(args, spec, status, replicates=10000):
    base = Path(spec["base"])
    sections, checks = [], []
    floors_path = args.run / "noise_floor.json"
    floors = read_sealed(floors_path)["rows"] if floors_path.exists() else {}
    total_complete = 0
    for subset in SUBSETS:
        old = read_sealed(base / f"diagnostics/e2_{subset}/plan.json")
        old_methods = {value["method_id"]: value for value in old["methods"]}
        rows = []
        for path in sorted((base / f"diagnostics/e2_{subset}").glob("*.json")):
            row = json.loads(path.read_text())
            if "method_id" in row:
                method = old_methods[row["method_id"]]
                group = int(row["method_id"].split("_G")[1].split("_")[0])
                rows.append((row, method["intervention"], group))
        for method in spec["methods"]:
            for state in spec["selection"]["state_ids_by_split"][subset]:
                path = pair_path(args.run, subset, state, method["method_id"])
                if path.exists():
                    row = read_sealed(path)
                    verify_result(row, state, method["method_id"], spec["model_sha256_by_split"][subset], spec["sha256"])
                    total_complete += 1
                    rows.append((row, method["intervention"], method["group_id"]))
        data, metadata, asymmetry = vectors(rows)
        features = read_sealed(base / f"diagnostics/features_{subset}/features.json")
        changes = defaultdict(dict)
        floor = spec["calibration"]["floor"]
        for feature in features["records"]:
            if feature["seed"] == 43 and feature["change_norm"] is not None:
                changes[(feature["state_id"], feature["layer_id"], feature["step_id"])][feature["group_id"]] = (
                    feature["change_norm"] / (feature["feature_norm"] + floor))
        records = defaultdict(list)
        for key, values in data.items():
            state, layer, step, seed, epsilon = key
            meta = {name: metadata[state][name] for name in ("suite", "task_id", "episode_id")}
            meta["state_id"] = state
            visual = changes.get(key[:3], {})
            if set(visual) == set(range(8)):
                association = conditional_association([visual[index] for index in range(8)], values)
                records[("visual_association", layer, step, seed, epsilon)].append({**meta, **association})
            reference = data.get((state, layer, step, OLD_SEED, 0.03))
            if reference is None or (seed == OLD_SEED and epsilon == 0.03):
                continue
            association = conditional_association(reference, values)
            label = "direction" if seed != OLD_SEED else "amplitude"
            numerical = [max(row["repeat_rmse"], row["gate_one_rmse"]) for row in floors.get(subset, [])
                         if row["layer_id"] == layer and row["step_id"] == step]
            threshold = max(numerical) if numerical else None
            record = {**meta, **association, "zero_fraction": float(np.mean(values == 0)),
                      "tied_fraction": 1 - len(np.unique(values)) / 8,
                      "mean_response_rmse": float(values.mean() * epsilon),
                      "above_noise_fraction": None if threshold is None or threshold == 0 else
                      float(np.mean(values * epsilon > 3 * threshold)),
                      "sign_asymmetry": float(np.mean([asymmetry[key + (group,)] for group in range(8)]))}
            records[(label, layer, step, seed, epsilon)].append(record)
        for condition, values in sorted(records.items()):
            label, layer, step, seed, epsilon = condition
            metrics = {key: bootstrap(values, key, replicates) for key in
                       ("spearman", "top25_overlap")}
            for key in ("zero_fraction", "tied_fraction", "mean_response_rmse", "above_noise_fraction", "sign_asymmetry"):
                if key in values[0]:
                    metrics[key] = bootstrap(values, key, replicates)
            complete = len(values) == 24 and metrics["top25_overlap"]["tasks"] == 8
            passed = None
            if label != "visual_association" and complete:
                passed = ranking_passes(values, metrics)
            sections.append({"split": subset, "comparison": label, "layer": layer, "step": step,
                             "noise_seed": seed, "epsilon": epsilon, "complete": complete,
                             "passes_ranking_gate": passed, "metrics": metrics, "state_records": values,
                             "task_metrics": task_metrics(values)})
        checks.append({"split": subset, "original_repeat_max_abs": spec["original_repeat_max_abs"][subset],
                       "fresh_normalized_floor": floors.get(subset, [])})
    decisions = []
    for layer in (0, 29):
        for step in (1, 5, 9):
            required = [section for section in sections if section["split"] == "calibration"
                        and section["layer"] == layer and section["step"] == step
                        and section["comparison"] in {"direction", "amplitude"}]
            expected = 3 if step == 9 else 1
            passed = len(required) == expected and all(section["passes_ranking_gate"] is True for section in required)
            decisions.append({"layer": layer, "step": step, "ranking_gate_passed": passed,
                              "amplitude_checked": step == 9,
                              "interpretation": "eligible_for_profile_not_deployment" if passed else "not_established"})
    output = {"status": status, "completed_pairs": total_complete, "expected_pairs": 7680,
              "plan_sha256": spec["sha256"], "report_tool_sha256": file_hash(__file__),
              "sections": sections, "noise_checks": checks,
              "decisions": decisions, "bootstrap_replicates": replicates,
              "limitations": ["Exploratory development tasks only; no closed-loop or speedup evidence",
                              "Progress proxies are not contact labels; original missing stages preserved",
                              "Existing gripper_disagreement is numeric inequality, not binary command flips",
                              "Ranking gates are engineering screens, not significance tests",
                              "Zero repeat noise does not exclude quantization effects",
                              "No additional policy noise seed was tested"]}
    atomic_json(args.run / "report.json", output)
    lines = ["# H2 robustness", "", f"Status: {status}; new pairs: {total_complete}/7680.", "",
             "| Split | Comparison | Layer/step | Direction | Epsilon | Rho [95% CI] | Top-2 overlap | Complete |",
             "|---|---|---|---|---|---|---|---|"]
    for section in sections:
        rho = section["metrics"]["spearman"]
        overlap = section["metrics"]["top25_overlap"]["mean"]
        lines.append(f"| {section['split']} | {section['comparison']} | {section['layer']}/{section['step']} | "
                     f"{section['noise_seed']} | {section['epsilon']} | {rho['mean']} {rho['ci95']} | "
                     f"{overlap} | {section['complete']} |")
    lines += ["", "## Limitations", *[f"- {value}" for value in output["limitations"]]]
    temporary = args.run / f"report.tmp.{os.getpid()}"
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, args.run / "report.md")
    atomic_json(args.run / "progress.json", {"completed_pairs": total_complete,
                "total_pairs": 7680, "status": status, "eta_hours": None})
    print(f"Status={status}; report: {args.run / 'report.md'} ({total_complete}/7680)", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("experiment/e2_exploratory_20261113_205159_787207"))
    parser.add_argument("--bank-run", type=Path, default=Path("experiment/h2_states_20261113_175408_491989"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO"))
    parser.add_argument("--ckpt-name", default="checkpoint_step_10690.safetensors")
    parser.add_argument("--split", type=Path,
                        default=Path("experiment/h1h2_20261112_223445_299984/manifests/split.json"))
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.hours) or args.hours <= 0:
        parser.error("--hours must be positive and finite")
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        if args.report_only:
            spec = read_sealed(args.run / "plan.json")
            for path, checksum in spec["inputs"].items():
                if file_hash(path) != checksum:
                    raise ValueError("Original inputs changed")
            status_path = args.run / "status.json"
            status = json.loads(status_path.read_text())["status"] if status_path.exists() else "not_started"
            report(args, spec, status)
            return
        spec = prepare(args)
        if args.prepare_only:
            return
        status = "failed"
        try:
            status = run_study(args, spec, started)
        finally:
            atomic_json(args.run / "status.json", {"status": status, "elapsed_hours": (time.monotonic()-started)/3600})
            report(args, spec, status)


if __name__ == "__main__":
    main()
