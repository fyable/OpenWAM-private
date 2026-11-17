import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.libero.research import make_plan, validate_t0
from openwam.research.analysis import conditional_association, summarize
from openwam.research.protocol import SUITES, cumulative_configs, read_sealed, seal, split_tasks, write_json
from openwam.research.runner import infrastructure_retries


def test_conditional_rankings_and_ties():
    assert conditional_association([1, 2, 3, 4], [4, 3, 2, 1])["spearman"] == -1
    assert conditional_association([1, 2, 3, 4], [4, 3, 2, 1])["top25_overlap"] == 0
    assert conditional_association([1, 1, 1], [1, 2, 3])["spearman"] is None
    with pytest.raises(ValueError):
        conditional_association([1], [1])


def test_retry_records_only_infrastructure_and_never_numeric():
    calls = []
    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("temporary IO error")
        return 42
    result, attempts = infrastructure_retries(operation)
    assert result == 42 and len(calls) == 3 and len(attempts) == 2
    assert all(not attempt["valid_run"] for attempt in attempts)
    def failure():
        raise OSError("persistent")
    with pytest.raises(OSError) as error:
        infrastructure_retries(failure)
    assert len(error.value.research_attempts) == 3
    def numeric():
        raise FloatingPointError("policy NaN")
    with pytest.raises(FloatingPointError):
        infrastructure_retries(numeric)


def test_t0_certificate_binding():
    certificate = {"status": "passed", "states": 20, "seeds": [42, 43, 44],
                   "model_sha256": "model", "bank_sha256": "bank", "steps": 10}
    validate_t0(certificate, {"sha256": "model"}, {"sha256": "bank"}, 10)
    for key, value in (("status", "failed"), ("states", 19), ("seeds", [42, 42, 42]),
                       ("model_sha256", "different"), ("bank_sha256", "different"), ("steps", 4)):
        with pytest.raises(ValueError, match="passing T0"):
            validate_t0(certificate | {key: value}, {"sha256": "model"}, {"sha256": "bank"}, 10)


def test_e1_plan_deterministic_episode_budget(tmp_path):
    split = split_tasks({suite: [f"task{index}" for index in range(10)] for suite in SUITES})
    path = tmp_path / "split.json"
    write_json(path, split)
    args = SimpleNamespace(split=path, output=tmp_path / "plan.json", experiment="e1", layers=30, steps=10,
                           selected_split="discovery", selection=None, calibration=None)
    make_plan(args)
    plan = read_sealed(args.output)
    assert plan["episode_ids"] == [0, 1, 2, 3, 4]
    assert len(plan["methods"]) == 29
    assert plan["task_split_sha256"] == split["sha256"]


def test_cumulative_positions_are_nested():
    positions = [(layer, step) for layer in range(4) for step in range(10)]
    configs = cumulative_configs(positions)
    previous = set()
    for config in configs:
        current = set(config["removed_positions"])
        assert previous <= current
        previous = current


def test_summary_averages_states_within_episodes_and_keeps_failures(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_json(runs / "plan.json", seal({"methods": [{"method_id": "custom"}], "task_split_sha256": "split"}))
    for index, error in enumerate((0.0, 2.0, 9.0)):
        write_json(runs / f"row{index}.json", {"method_id": "custom", "state_id": str(index), "suite": "libero_goal",
                                              "task_id": 0, "episode_id": 0 if index < 2 else 1,
                                              "phase": "progress_proxy", "status": "completed", "valid_run": True,
                                              "metrics": {"prefix_normalized_rmse": error}})
    write_json(runs / "failure.json", {"method_id": "custom", "state_id": "failed", "valid_run": True,
                                       "status": "policy_failure"})
    summarize(runs, tmp_path / "summary")
    summary = json.loads((tmp_path / "summary/summary.json").read_text())
    assert summary["runs"] == 4 and summary["completed"] == 3
    assert summary["methods"]["custom"]["prefix_normalized_rmse"] == 5
    assert summary["closed_loop_success"] == "not_measured"


def test_equal_token_group_sizes():
    from openwam.research.protocol import token_groups
    groups = token_groups({"tokens_per_frame": 21, "grid_height": 7, "grid_width": 3,
                           "prefix_tokens": 21, "video_tokens": 126})
    assert len({len(group) for group in groups}) == 1
    assert len(np.unique(np.concatenate(groups))) == 105


def test_e2_analysis_pairs_signs_and_verifies_feature_provenance(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    model = seal({"architecture": "synthetic"})
    calibration = seal({"floor": 1.0, "model_sha256": model["sha256"], "selection_sha256": "selection",
                         "task_split_sha256": "split", "fitted_split": "calibration"})
    methods = []
    records = []
    for group in range(4):
        records.append({"state_id": "state", "seed": 42, "layer_id": 0, "step_id": 1, "group_id": group,
                        "feature_norm": 1.0, "change_norm": float(group + 1)})
        for sign in (-1, 1):
            name = f"E2_L0_S1_G{group}_sign{sign}"
            methods.append({"method_id": name, "intervention": {"layer_ids": [0], "step_ids": [1],
                                                                 "kind": "value_noise", "sign": sign, "epsilon": 0.03}})
            write_json(runs / f"group{group}_sign{sign}.json", {"state_id": "state", "action_seed": 42,
                       "method_id": name, "status": "completed", "valid_run": True, "suite": "libero_goal",
                       "task_id": 0, "episode_id": 0, "phase": "progress_proxy",
                       "metrics": {"prefix_normalized_rmse": float(4 - group) * 0.03}})
    plan = seal({"task_split_sha256": "split", "selected_split": "discovery", "selection_sha256": "selection",
                 "calibration_sha256": calibration["sha256"], "methods": methods})
    write_json(runs / "plan.json", plan)
    write_json(runs / "manifest.json", seal({"model": model, "bank_sha256": "bank"}))
    write_json(tmp_path / "calibration.json", calibration)
    features = seal({"task_split_sha256": "split", "fitted_split": "discovery", "selection_sha256": "selection",
                     "model_sha256": model["sha256"], "bank_sha256": "bank", "records": records})
    write_json(tmp_path / "features.json", features)
    summarize(runs, tmp_path / "summary", features=tmp_path / "features.json", calibration=tmp_path / "calibration.json")
    associations = json.loads((tmp_path / "summary/associations.json").read_text())
    assert len(associations) == 1
    assert associations[0]["spearman"] == -1
    assert associations[0]["top25_overlap"] == 0
