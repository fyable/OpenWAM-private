from copy import deepcopy

import pytest

from openwam.research.interventions import Intervention
from scripts.research_e2_extensions import (
    extra_states, group_methods, noise_methods, phase_definitions, summarize, verify_extension,
    make_plan,
)
from openwam.research.protocol import seal, write_json


def selection():
    return {"layouts": {str(layer): {"tokens_per_frame": 120, "prefix_tokens": 120, "video_tokens": 360,
                                   "grid_height": 12, "grid_width": 10} for layer in (0, 29)}}


def test_matched_group_controls_preserve_budget_and_protected_prefix():
    ranking = {f"{layer}:{step}": {"ascending": list(range(8))} for layer in (0, 29) for step in (1, 5, 9)}
    methods = group_methods(selection(), ranking, 1e-4)
    assert len(methods) == 288
    assert len({method["method_id"] for method in methods}) == 288
    for method in methods:
        config = Intervention.from_dict(method["intervention"])
        assert min(config.token_ids) >= 120
        assert len(config.token_ids) == method["group_count"] * 30
        if method["strategy"] == "low":
            assert method["group_ids"] == list(range(method["group_count"]))
        if method["strategy"] == "high":
            assert method["group_ids"] == list(range(8 - method["group_count"], 8))
    assert group_methods(selection(), ranking, 1e-4) == methods


def test_noise_followups_keep_sign_pairs_and_factor_labels():
    assert len(noise_methods(selection(), 1e-4)) == 96
    sweep = noise_methods(selection(), 1e-4, True)
    assert len(sweep) == 576
    for index in range(0, len(sweep), 2):
        negative, positive = [deepcopy(method["intervention"]) for method in sweep[index:index+2]]
        assert negative.pop("sign") == -1
        assert positive.pop("sign") == 1
        assert negative == positive
    assert phase_definitions()[0]["family"] == "group_controls"


def test_additional_states_exclude_original_episode_ids():
    states = []
    for task in range(8):
        for episode in range(4):
            for request, progress in enumerate((0.2, 0.5, 0.8)):
                states.append({"suite": "suite", "task_id": task, "episode_id": episode, "progress": progress,
                               "request_index": request, "state_id": f"t{task}_e{episode}_r{request}"})
    original = [row["state_id"] for row in states if row["episode_id"] == 0]
    ids, coverage = extra_states({"states": states}, original)
    assert len(ids) == len(set(ids)) == 24
    assert not set(ids) & set(original)
    assert all("_e0_" not in state for state in ids)
    assert len(coverage) == 8


def test_extension_seed_identity_is_independent_of_core_seed_43():
    plan = {"sha256": "plan", "model_sha256_by_split": {"calibration": "model"}}
    row = {"status": "completed", "valid_run": True, "extension_sha256": "plan", "phase_id": "phase",
           "state_id": "state", "method_id": "method", "action_seed": 44, "video_seed": 44,
           "steps": 10, "split": "calibration", "model_sha256": "model",
           "metrics": {"prefix_normalized_rmse": 0.01}}
    verify_extension(row, plan, "phase", "calibration", "state", 44, "method")
    with pytest.raises(ValueError):
        verify_extension(row, plan, "phase", "calibration", "state", 43, "method")


def test_paired_report_uses_only_fully_matched_controls(tmp_path, monkeypatch):
    import json
    from scripts import research_e2_extensions as extensions

    monkeypatch.setattr(extensions, "baseline_vectors", lambda run: {})
    states = [{"state_id": f"t{task}_s{state}", "suite": "suite", "task_id": task,
               "episode_id": state, "split": "calibration"} for task in range(8) for state in range(3)]
    for subset in ("discovery", "calibration"):
        write_json(tmp_path / f"bank/state_bank/{subset}/manifest.json", seal({"states": states}))
    ranking = {f"{layer}:{step}": {"ascending": list(range(8))} for layer in (0, 29) for step in (1, 5, 9)}
    methods = group_methods(selection(), ranking, 1e-4)[:8]
    phase = {"id": "controls", "family": "group_controls", "states": "original", "methods": methods,
             "seeds": [43], "state_ids_by_split": {"calibration": [state["state_id"] for state in states]},
             "expected_pairs": 24 * 8}
    plan = {"sha256": "plan", "phases": [phase], "expected_pairs": 192,
            "model_sha256_by_split": {"calibration": "model"}}
    missing = None
    for state in states:
        for method in methods:
            strategy = method["strategy"]
            value = {"low": 0.1, "high": 0.9, "uniform": 0.6}.get(strategy, 0.5)
            row = {**state, "status": "completed", "valid_run": True, "extension_sha256": "plan",
                   "phase_id": "controls", "method_id": method["method_id"], "action_seed": 43,
                   "video_seed": 43, "steps": 10, "model_sha256": "model",
                   "metrics": {"prefix_normalized_rmse": value}}
            path = tmp_path / f"extensions/workers/worker_0/results/controls/calibration/{state['state_id']}_{strategy}.json"
            write_json(path, seal(row))
            if missing is None and strategy == "random4":
                missing = path
    summarize(tmp_path, plan, "complete")
    report = json.loads((tmp_path / "extensions/report.json").read_text())
    comparison = report["group_control_comparisons"][0]
    assert comparison["states"] == 24 and comparison["complete"]
    assert comparison["metrics"]["low_minus_random"]["mean"] == pytest.approx(-0.4)
    missing.unlink()
    summarize(tmp_path, plan, "budget_exhausted")
    report = json.loads((tmp_path / "extensions/report.json").read_text())
    assert not report["phases"][0]["complete"]
    assert report["group_control_comparisons"][0]["states"] == 23
    assert not report["group_control_comparisons"][0]["complete"]


def test_plan_freezes_discovery_ranks_and_independent_episode_sets(tmp_path, monkeypatch):
    from scripts import research_e2_extensions as extensions
    from openwam.research.protocol import read_sealed

    frozen_selection = selection()
    frozen_selection["state_ids_by_split"] = {}
    original_ids = [f"t{task}_e0_r{request}" for task in range(8) for request in range(3)]
    model = seal({"source_sha256": "source"})
    for subset in ("discovery", "calibration"):
        states = [{"suite": "suite", "task_id": task, "episode_id": episode,
                   "state_id": f"t{task}_e{episode}_r{request}", "request_index": request, "progress": progress}
                  for task in range(8) for episode in range(4)
                  for request, progress in enumerate((0.2, 0.5, 0.8))]
        write_json(tmp_path / f"bank/state_bank/{subset}/manifest.json", seal({"states": states}))
        write_json(tmp_path / f"bank/state_bank/{subset}/model.json", model)
        frozen_selection["state_ids_by_split"][subset] = original_ids
    write_json(tmp_path / "base/manifests/selection.json", seal(frozen_selection))
    write_json(tmp_path / "base/diagnostics/features_calibration/calibration.json", seal({"floor": 1e-4}))
    protocol = {"sha256": "protocol", "selection_sha256": read_sealed(tmp_path / "base/manifests/selection.json")["sha256"],
                "calibration_sha256": read_sealed(tmp_path / "base/diagnostics/features_calibration/calibration.json")["sha256"],
                "deadline_unix": 1000, "gpus": [{"uuid": "GPU-a"}], "phases": phase_definitions(),
                "checkpoint_dir": "/local/model", "checkpoint_name": "checkpoint.safetensors"}
    def ranking(directory):
        assert directory == tmp_path / "base/diagnostics/e2_discovery"
        return {f"{layer}:{step}": {"ascending": list(range(8)), "scores": list(range(8))}
                for layer in (0, 29) for step in (1, 5, 9)}
    monkeypatch.setattr(extensions, "discovery_ranking", ranking)
    plan = make_plan(tmp_path, protocol)
    assert plan["expected_pairs"] == 96768
    assert plan["ranking_fitted_split"] == "discovery"
    for phase in plan["phases"]:
        if phase["states"] == "new_episodes":
            assert all("_e0_" not in state for ids in phase["state_ids_by_split"].values() for state in ids)
