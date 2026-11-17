"""Paired fixed-state generation. Diagnostic timings are never benchmark results."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from openwam.research.interventions import Intervention, ResearchContext
from openwam.research.protocol import action_metrics, active_indices, digest, write_json


def paired_generate(engine, conditions, *, seed, context=None, steps=10, cache=None):
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    architecture = engine.architecture
    previous_cache = engine._dit_cache
    device = architecture.device
    cuda_devices = [] if torch.device(device).type != "cuda" else [torch.device(device).index or 0]
    try:
        engine.reset()
        for name in ("_vace_cache", "_prompt_embed_cache"):
            value = getattr(engine, name, None)
            if value is not None:
                value.clear()
        engine._dit_cache = cache
        if cache is not None:
            cache.reset()
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            request = copy.deepcopy(conditions)
            request.update(seed=seed, denoise_steps=steps, denoise_mode="sync")
            if context is not None:
                request["research_context"] = context
            started = time.perf_counter()
            result = engine.generate(request)
            if context is not None and (context.normalized_actions is None or not context.records):
                raise RuntimeError("Research context was not consumed by the model")
            if not np.isfinite(result["actions"]).all():
                raise FloatingPointError("Non-finite policy output")
            return result, (time.perf_counter() - started) * 1000
    finally:
        engine._dit_cache = previous_cache
        try:
            engine.reset()
            for name in ("_vace_cache", "_prompt_embed_cache"):
                value = getattr(engine, name, None)
                if value is not None:
                    value.clear()
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)


def infrastructure_retries(operation, *, max_retries=2):
    attempts = []
    for index in range(max_retries + 1):
        try:
            return operation(), attempts
        except (OSError, TimeoutError, ConnectionError) as error:
            attempts.append({"attempt": index, "valid_run": False, "error_type": type(error).__name__,
                             "error": str(error)})
            if index == max_retries:
                error.research_attempts = attempts
                raise


def baseline_context(config, steps, *, capture=False):
    return ResearchContext(replace(config, kind="gate", gate=1.0,
                                   step_ids=tuple(range(steps)) if capture else config.step_ids),
                           capture_values=capture)


def compare_contexts(engine, reference, actual, reference_result, actual_result, horizon):
    active = active_indices(engine.architecture)
    metrics = action_metrics(reference.normalized_actions.numpy(), actual.normalized_actions.numpy(),
                             reference_result["actions"], actual_result["actions"],
                             active=active, horizon=horizon,
                             binary_dims=getattr(engine.architecture, "binary_command_dims", ()) or ())
    local = []
    for index, prediction in actual.predictions.items():
        delta = (prediction[..., active] - reference.predictions[index][..., active]).float()
        local.append({"step_id": index, "active_velocity_rmse": delta.square().mean().sqrt().item(),
                      "active_velocity_max_abs": delta.abs().max().item()})
    normalizer = getattr(engine.architecture, "normalizer", None)
    mapping = getattr(normalizer, "_dst_index", None)
    return metrics, local, digest({"active_indices": active,
                                  "raw_to_unified": None if mapping is None else np.asarray(mapping).tolist()})


def run_pair(engine, conditions, config, *, seed, steps, horizon):
    reference = baseline_context(config, steps, capture=config.kind == "value_stale")
    reference_result, _ = paired_generate(engine, conditions, seed=seed, context=reference, steps=steps)
    actual = ResearchContext(config, stale_values=reference.values)
    actual_result, elapsed = paired_generate(engine, conditions, seed=seed, context=actual, steps=steps)
    if config.kind == "value_stale":
        for record in actual.records:
            previous = next(row for row in reference.records if row["layer_id"] == record["layer_id"]
                            and row["step_id"] == record["step_id"] - 1)
            record.update(reference_sigma_video=previous["sigma_video"], reference_sigma_action=previous["sigma_action"])
    metrics, local, mapping_hash = compare_contexts(engine, reference, actual, reference_result, actual_result, horizon)
    return {"metrics": metrics, "local_predictions": local, "diagnostics": actual.records,
            "layouts": actual.layouts, "active_action_mapping_hash": mapping_hash,
            "full_forward_count": len(actual.predictions),
            "diagnostic_server_ms": elapsed, "timing_kind": "diagnostic_not_benchmark"}


def run_combo(engine, conditions, removed_positions, *, seed, steps, horizon):
    positions = {tuple(position) for position in removed_positions}
    if not positions:
        raise ValueError("Combination must remove at least one position")
    config = Intervention(layer_ids=tuple(sorted({position[0] for position in positions})),
                          step_ids=tuple(sorted({position[1] for position in positions})), gate=0.0)
    reference = baseline_context(config, steps)
    reference_result, _ = paired_generate(engine, conditions, seed=seed, context=reference, steps=steps)

    class PositionContext(ResearchContext):
        def step(self, *args):
            original = super().step(*args)

            class PositionStep:
                def __getattr__(self, name):
                    return getattr(original, name)

                def selected(self, layer):
                    return (layer, original.index) in positions

            return PositionStep()

    actual = PositionContext(config)
    actual_result, elapsed = paired_generate(engine, conditions, seed=seed, context=actual, steps=steps)
    metrics, local, mapping_hash = compare_contexts(engine, reference, actual, reference_result, actual_result, horizon)
    return {"metrics": metrics, "local_predictions": local, "diagnostics": actual.records,
            "layouts": actual.layouts, "active_action_mapping_hash": mapping_hash,
            "full_forward_count": len(actual.predictions),
            "diagnostic_server_ms": elapsed, "timing_kind": "diagnostic_not_benchmark"}


def execute_method(engine, conditions, method, *, seed, steps, horizon):
    if "removed_positions" in method:
        return run_combo(engine, conditions, method["removed_positions"], seed=seed, steps=steps, horizon=horizon)
    if "intervention" in method:
        return run_pair(engine, conditions, Intervention.from_dict(method["intervention"]),
                        seed=seed, steps=steps, horizon=horizon)
    from openwam.deploy.optimizations.dit_cache import DiTVelocityCache
    cache = DiTVelocityCache(**method["cache"]) if method.get("cache") is not None else None
    reference, _ = paired_generate(engine, conditions, seed=seed, steps=steps)
    actual, elapsed = paired_generate(engine, conditions, seed=seed, steps=method.get("steps", steps), cache=cache)
    return {"native_max_abs": float(np.max(np.abs(actual["actions"] - reference["actions"]))),
            "diagnostic_server_ms": elapsed, "timing_kind": "diagnostic_not_benchmark",
            "cache_stats": cache.stats if cache is not None else None}


def run_t0(engine, states, *, seeds, steps, horizon, output, recorded_outputs=None):
    rows = []
    for state_id, conditions in states:
        for seed in seeds:
            raw, _ = paired_generate(engine, conditions, seed=seed, steps=steps)
            repeat, _ = paired_generate(engine, conditions, seed=seed, steps=steps)
            config = Intervention(layer_ids=tuple(range(engine.architecture.video_backbone.num_layers)),
                                  step_ids=tuple(range(steps)), gate=1.0)
            context = ResearchContext(config)
            diagnosed, _ = paired_generate(engine, conditions, seed=seed, steps=steps, context=context)
            baseline_delta = np.abs(repeat["actions"] - raw["actions"])
            delta = np.abs(diagnosed["actions"] - raw["actions"])
            scale = np.maximum(np.abs(raw["actions"]), 1e-12)
            floor = np.finfo(raw["actions"].dtype).eps * np.maximum(1, np.abs(raw["actions"]))
            recorded_error = None
            recorded_matches = True
            if recorded_outputs is not None and recorded_outputs[state_id][0] == seed:
                expected = recorded_outputs[state_id][1]
                if expected.shape != raw["actions"].shape or not np.isfinite(expected).all():
                    raise ValueError("Invalid recorded baseline action shape or values")
                recorded_delta = np.abs(expected - raw["actions"])
                recorded_error = float(recorded_delta.max())
                recorded_matches = bool(np.all(recorded_delta <= baseline_delta + floor))
            dimensions = getattr(engine.architecture, "binary_command_dims", ()) or ()
            def project(actions):
                values = np.array(actions, copy=True)
                for dimension in dimensions:
                    values[:, dimension] = np.where(values[:, dimension] > 0.5, 1.0, -1.0)
                from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
                return np.stack([native_eef10_to_libero7d(row) for row in values])
            original_runtime, diagnostic_runtime = project(raw["actions"]), project(diagnosed["actions"])
            gripper_equal = bool(np.array_equal(original_runtime[:, 6], diagnostic_runtime[:, 6]))
            rows.append({"state_id": state_id, "seed": seed, "baseline_repeat_max_abs": float(baseline_delta.max()),
                         "gate_one_max_abs": float(delta.max()), "gate_one_max_relative": float((delta / scale).max()),
                         "prefix_max_abs": float(delta[:horizon].max()), "gripper_equal": gripper_equal,
                         "recorded_baseline_max_abs": recorded_error,
                         "passed": bool(np.all(delta <= baseline_delta + floor) and gripper_equal and recorded_matches)})
    result = {"status": "passed" if rows and all(row["passed"] for row in rows) else "failed",
              "tolerance_rule": "elementwise repeat error + dtype epsilon * max(1, abs(reference)); no gripper flips",
              "states": len(states), "seeds": seeds, "steps": steps, "horizon": horizon, "rows": rows}
    write_json(Path(output), result)
    return result
