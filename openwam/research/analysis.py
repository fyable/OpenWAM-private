"""Descriptive episode-level summaries; never claims closed-loop success."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from openwam.research.protocol import read_sealed, write_json


def conditional_association(changes, sensitivities):
    from scipy.stats import rankdata

    changes, sensitivities = np.asarray(changes), np.asarray(sensitivities)
    if len(changes) < 2 or len(changes) != len(sensitivities):
        raise ValueError("Association needs at least two matched groups")
    change_rank, sensitivity_rank = rankdata(changes), rankdata(sensitivities)
    correlation = None
    if np.std(change_rank) > 0 and np.std(sensitivity_rank) > 0:
        correlation = float(np.corrcoef(change_rank, sensitivity_rank)[0, 1])
    count = max(1, int(np.ceil(len(changes) / 4)))
    change_top = set(np.argsort(-changes, kind="stable")[:count].tolist())
    sensitivity_top = set(np.argsort(-sensitivities, kind="stable")[:count].tolist())
    return {"spearman": correlation, "top25_overlap": len(change_top & sensitivity_top) / count,
            "groups": len(changes), "top_count": count, "tie_rule": "stable ascending group ID"}


def summarize(directory, output, *, features=None, calibration=None):
    root = Path(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = read_sealed(root / "plan.json")
    methods = {method["method_id"]: method for method in plan["methods"]}
    rows = []
    for path in sorted(root.glob("*.json")):
        value = json.loads(path.read_text())
        if "method_id" in value and "state_id" in value:
            rows.append(value)
    episodes = defaultdict(list)
    flat = []
    for row in rows:
        if row.get("status") != "completed" or "metrics" not in row:
            continue
        metrics = row["metrics"]
        flat.append({key: row.get(key) for key in ("method_id", "suite", "task_id", "episode_id", "state_id", "phase")}
                    | metrics)
        episodes[(row["method_id"], row["suite"], row["task_id"], row["episode_id"])].append(metrics)
    episode_rows = []
    for (method, suite, task, episode), values in episodes.items():
        episode_rows.append({"method_id": method, "suite": suite, "task_id": task, "episode_id": episode,
                             **{key: float(np.mean([value[key] for value in values])) for key in values[0]}})
    if flat:
        with (output / "state_metrics.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    write_json(output / "episode_metrics.json", episode_rows)
    aggregates = {}
    for method in methods:
        values = [row for row in episode_rows if row["method_id"] == method]
        if values:
            aggregates[method] = {"episodes": len(values),
                                  "prefix_normalized_rmse": float(np.mean([row["prefix_normalized_rmse"] for row in values]))}
    associations = []
    if features is not None:
        if calibration is None:
            raise ValueError("Feature comparisons require the frozen calibration scale")
        feature_data, calibration_data = read_sealed(features), read_sealed(calibration)
        if any(value["task_split_sha256"] != plan["task_split_sha256"] for value in (feature_data, calibration_data)):
            raise ValueError("Features/calibration belong to a different split")
        if calibration_data["fitted_split"] != "calibration":
            raise ValueError("Feature scale must be calibrated on Calibration")
        manifest = read_sealed(root / "manifest.json")
        model_hash = manifest["model"]["sha256"]
        if (feature_data["model_sha256"] != model_hash or calibration_data["model_sha256"] != model_hash
            or feature_data["bank_sha256"] != manifest["bank_sha256"]
            or feature_data["fitted_split"] != plan["selected_split"]
            or calibration_data["sha256"] != plan["calibration_sha256"]
            or any(value["selection_sha256"] != plan["selection_sha256"] for value in (feature_data, calibration_data))):
            raise ValueError("Feature/calibration model, bank, selection or protocol identity mismatch")
        if not np.isfinite(calibration_data["floor"]) or calibration_data["floor"] <= 0:
            raise ValueError("Calibration floor must be positive and finite")
        feature_map = {(row["state_id"], row["seed"], row["layer_id"], row["step_id"], row["group_id"]): row
                       for row in feature_data["records"]}
        responses = defaultdict(list)
        for row in rows:
            config = methods[row["method_id"]].get("intervention", {})
            if row.get("status") != "completed" or config.get("kind") != "value_noise":
                continue
            group_id = int(row["method_id"].split("_G")[1].split("_")[0])
            key = (row["state_id"], row["action_seed"], config["layer_ids"][0], config["step_ids"][0], group_id)
            feature = feature_map.get(key)
            if feature is None or feature["change_norm"] is None:
                continue
            change = feature["change_norm"] / (feature["feature_norm"] + calibration_data["floor"])
            responses[key].append((change, row["metrics"]["prefix_normalized_rmse"] / config["epsilon"], config["sign"]))
        conditions = defaultdict(list)
        for key, values in responses.items():
            if len(values) != 2 or {value[2] for value in values} != {-1, 1}:
                continue
            conditions[key[:-1]].append((key[-1], float(np.mean([value[0] for value in values])),
                                         float(np.mean([value[1] for value in values]))))
        for key, values in conditions.items():
            values.sort()
            if len(values) >= 2:
                associations.append({"state_id": key[0], "seed": key[1], "layer_id": key[2], "step_id": key[3],
                                     **conditional_association([value[1] for value in values], [value[2] for value in values])})
    write_json(output / "associations.json", associations)
    association_episodes = defaultdict(list)
    metadata = {row["state_id"]: row for row in rows}
    for row in associations:
        state = metadata[row["state_id"]]
        association_episodes[(state["suite"], state["task_id"], state["episode_id"], row["layer_id"], row["step_id"])].append(row)
    write_json(output / "association_episode_metrics.json", [
        {"suite": key[0], "task_id": key[1], "episode_id": key[2], "layer_id": key[3], "step_id": key[4],
         "mean_spearman": None if not any(row["spearman"] is not None for row in values) else float(np.mean(
             [row["spearman"] for row in values if row["spearman"] is not None])),
         "mean_top25_overlap": float(np.mean([row["top25_overlap"] for row in values]))}
        for key, values in association_episodes.items()])
    write_json(output / "summary.json", {"runs": len(rows), "completed": sum(row.get("status") == "completed" for row in rows),
                                          "invalid_runs": sum(not row.get("valid_run", False) for row in rows),
                                          "policy_failures": sum(row.get("status") == "policy_failure" for row in rows),
                                          "methods": aggregates, "closed_loop_success": "not_measured",
                                          "speedup": "not_measured"})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError:
        pyplot = None
    if pyplot is not None:
        for direction in ("action", "future_video"):
            matrix = np.full((4, 3), np.nan)
            for method, metrics in aggregates.items():
                if method.startswith("E1_") and method.endswith(f"_{direction}"):
                    layer, step = method.split("_")[1:3]
                    matrix[int(layer[1:]), int(step[1:])] = metrics["prefix_normalized_rmse"]
            if np.isfinite(matrix).any():
                figure, axis = pyplot.subplots()
                image = axis.imshow(matrix, aspect="auto")
                axis.set(xlabel="Denoise group", ylabel="Layer group", title=f"Query: {direction}; episode-mean error")
                figure.colorbar(image, ax=axis)
                figure.savefig(output / f"e1_{direction}.png", bbox_inches="tight")
                pyplot.close(figure)
        for field, name in (("keep_ratio", "retention"), ("deleted_count", "cumulative_deletion")):
            points = [(method[field], aggregates[method["method_id"]]["prefix_normalized_rmse"], method["method_id"])
                      for method in methods.values() if field in method and method["method_id"] in aggregates]
            if points:
                figure, axis = pyplot.subplots()
                for budget, error, label in sorted(points):
                    axis.scatter(budget, error, label=label)
                axis.set(xlabel=field, ylabel="Episode-mean prefix normalized RMSE")
                axis.legend(fontsize="xx-small")
                figure.savefig(output / f"{name}.png", bbox_inches="tight")
                pyplot.close(figure)
    write_json(output / "report.json", {"plots": "generated_when_data_available" if pyplot else "not_run: matplotlib unavailable",
                                         "warning": "Open-loop intervention errors are not task quality or acceleration evidence"})
