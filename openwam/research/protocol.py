"""Reproducible development splits, sealed artifacts, and action metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
REFERENCE_COMMIT = "1dcaeea76501f2431286b511284a671778645996"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def seal(value):
    result = dict(value)
    result["sha256"] = digest(value)
    return result


def read_sealed(path):
    value = json.loads(Path(path).read_text())
    checksum = value.pop("sha256")
    if digest(value) != checksum:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return {**value, "sha256": checksum}


def split_tasks(task_map, seed=20261112):
    generator = np.random.Generator(np.random.PCG64(seed))
    tasks = []
    for suite in SUITES:
        names = task_map[suite]
        if len(names) != 10 or len(set(names)) != 10 or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{suite} must contain 10 distinct task names in original ID order")
        order = generator.permutation(10).tolist()
        for split, ids in (("discovery", order[:2]), ("calibration", order[2:4]), ("heldout", order[4:])):
            tasks.extend({"suite": suite, "task_id": index, "task_name": names[index], "split": split}
                         for index in ids)
    return seal({"schema_version": 1, "seed": seed, "rng": "numpy.PCG64", "tasks": tasks})


def require_development(split_manifest, metadata):
    if metadata.get("split") not in {"discovery", "calibration"}:
        raise ValueError("Discovery/calibration tools refuse held-out or unlabelled data")
    if metadata.get("task_split_sha256") != split_manifest["sha256"]:
        raise ValueError("State references a different task split")
    matches = [task for task in split_manifest["tasks"]
               if all(task[key] == metadata.get(key) for key in ("suite", "task_id", "task_name", "split"))]
    if len(matches) != 1:
        raise ValueError("State task identity does not match frozen split")


def progress_indices(count):
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        return []
    return sorted(set(int(np.rint((count - 1) * fraction)) for fraction in (0.2, 0.5, 0.8)))


def consecutive_groups(count, groups):
    if count < groups or groups < 1:
        raise ValueError("Cannot make nonempty groups")
    return [part.tolist() for part in np.array_split(np.arange(count), groups)]


def e1_configs(layers, steps=10):
    result = []
    for layer_group, layer_ids in enumerate(consecutive_groups(layers, 4)):
        for step_group, step_ids in enumerate(consecutive_groups(steps, 3)):
            for query, source in (("action", "future_video"), ("future_video", "action")):
                result.append({"method_id": f"E1_L{layer_group}_S{step_group}_{query}",
                               "intervention": {"layer_ids": layer_ids, "step_ids": step_ids,
                                                "query_modality": query, "source_modality": source,
                                                "kind": "gate", "gate": 0.0}})
    return result


def retention_configs(layer_ids, step_ids, ranked_positions, seed=20261112):
    positions = [(layer, step) for layer in layer_ids for step in step_ids]
    ranking = [tuple(position) for position in ranked_positions]
    if len(ranking) != len(positions) or set(ranking) != set(positions):
        raise ValueError("Ranking must cover each layer/step position exactly once")
    generator = np.random.Generator(np.random.PCG64(seed))
    result = []
    random_orders = [generator.permutation(len(positions)).tolist() for _ in range(5)]
    for ratio in (0.25, 0.5, 0.75):
        count = max(1, round(len(positions) * ratio))
        choices = {"ranked": ranking[:count],
                   "uniform": [positions[index] for index in np.linspace(0, len(positions) - 1, count, dtype=int)]}
        choices.update({f"random_{index}": [positions[offset] for offset in order[:count]]
                        for index, order in enumerate(random_orders)})
        for name, kept in choices.items():
            result.append({"method_id": f"keep_{ratio}_{name}", "keep_ratio": ratio,
                           "kept_positions": kept, "removed_positions": [position for position in positions if position not in kept]})
    return result


def active_indices(architecture):
    inactive = architecture._resolve_inactive_action_dims(None, architecture.device)
    if inactive is None:
        return list(range(architecture.action_dim))
    return (~inactive).nonzero().flatten().cpu().tolist()


def cumulative_configs(ranked_positions):
    positions = [tuple(position) for position in ranked_positions]
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Cumulative ranking must be nonempty and unique")
    result = []
    seen = set()
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9):
        count = max(1, round(len(positions) * fraction))
        if count not in seen:
            result.append({"method_id": f"delete_lowest_{count}", "deleted_count": count,
                           "removed_positions": positions[-count:]})
            seen.add(count)
    return result


def action_metrics(reference_normalized, actual_normalized, reference_native, actual_native, *, active, horizon,
                   binary_dims=()):
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d

    reference_normalized, actual_normalized = np.asarray(reference_normalized), np.asarray(actual_normalized)
    reference_native, actual_native = np.array(reference_native, copy=True), np.array(actual_native, copy=True)
    if (reference_normalized.shape != actual_normalized.shape or reference_normalized.ndim != 2
        or reference_native.shape != actual_native.shape or reference_native.shape != (len(reference_normalized), 10)
        or not 0 < horizon <= len(reference_normalized) or not active
        or min(active) < 0 or max(active) >= reference_normalized.shape[-1]):
        raise ValueError("Invalid action shapes, active mapping, or execution horizon")
    if not all(np.isfinite(value).all() for value in
               (reference_normalized, actual_normalized, reference_native, actual_native)):
        raise FloatingPointError("Non-finite policy actions")
    for dimension in binary_dims:
        reference_native[:, dimension] = np.where(reference_native[:, dimension] > 0.5, 1.0, -1.0)
        actual_native[:, dimension] = np.where(actual_native[:, dimension] > 0.5, 1.0, -1.0)
    reference_runtime = np.stack([native_eef10_to_libero7d(row) for row in reference_native])
    actual_runtime = np.stack([native_eef10_to_libero7d(row) for row in actual_native])
    result = {}
    for label, stop in (("prefix", horizon), ("full", len(reference_normalized))):
        diff = actual_normalized[:stop, active] - reference_normalized[:stop, active]
        baseline = reference_normalized[:stop, active]
        native_diff = actual_native[:stop] - reference_native[:stop]
        runtime_diff = actual_runtime[:stop] - reference_runtime[:stop]
        result.update({f"{label}_normalized_rmse": float(np.sqrt(np.mean(diff ** 2))),
                       f"{label}_normalized_max_abs": float(np.max(np.abs(diff))),
                       f"{label}_normalized_relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(baseline), 1e-12)),
                       f"{label}_native_rmse": float(np.sqrt(np.mean(native_diff ** 2))),
                       f"{label}_translation_command_l2": float(np.linalg.norm(runtime_diff[:, :3], axis=1).mean()),
                       f"{label}_rotation_command_l2": float(np.linalg.norm(runtime_diff[:, 3:6], axis=1).mean()),
                       f"{label}_gripper_disagreement": float(np.mean(
                           actual_runtime[:stop, 6] != reference_runtime[:stop, 6]))})
    return result


def token_groups(layout, max_groups=8):
    if max_groups < 1:
        raise ValueError("max_groups must be positive")
    count = layout["tokens_per_frame"]
    start = layout["prefix_tokens"]
    stop = layout["video_tokens"]
    if not 0 < start < stop or start % count or stop % count:
        raise ValueError("Invalid future latent layout")
    frames = list(range(start // count, stop // count))
    time_count = max(divisor for divisor in range(1, min(len(frames), max_groups) + 1) if len(frames) % divisor == 0)
    time_groups = consecutive_groups(len(frames), time_count)
    space_count = max(1, max_groups // len(time_groups))
    height, width = layout["grid_height"], layout["grid_width"]
    if height * width != count:
        raise ValueError("Spatial token layout mismatch")
    row_count = max(divisor for divisor in range(1, min(height, space_count) + 1) if height % divisor == 0)
    row_groups = consecutive_groups(height, row_count)
    groups = []
    for times in time_groups:
        for rows in row_groups:
            groups.append([frames[offset] * count + row * width + column
                           for offset in times for row in rows for column in range(width)])
    return groups
