from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.research_e2_robustness import (
    NEW_SEED, OLD_SEED, atomic_json, bootstrap, due, methods, pair_path,
    ranking_passes, report, task_metrics, vectors, verify_result,
)
from openwam.research.protocol import read_sealed, seal, write_json


def selection():
    layout = {"tokens_per_frame": 120, "prefix_tokens": 120, "video_tokens": 360,
              "grid_height": 12, "grid_width": 10}
    return {"layouts": {"0": layout, "29": layout}}


def row(state="state", method="method"):
    return {"status": "completed", "valid_run": True, "state_id": state, "method_id": method,
            "model_sha256": "model", "action_seed": 43, "video_seed": 43, "steps": 10,
            "robustness_sha256": "plan", "metrics": {"prefix_normalized_rmse": 0.03},
            "suite": "suite", "task_id": 0, "episode_id": 0}


def test_protocol_count_and_paired_directions():
    configs = methods(selection(), 1e-4)
    assert len(configs) == 160
    assert len({item["method_id"] for item in configs}) == 160
    assert sum(item["stage"] == "A" for item in configs) * 48 == 4608
    assert sum(item["stage"] == "B" for item in configs) * 48 == 3072
    for item in configs:
        config = item["intervention"]
        assert min(config["token_ids"]) >= 120
        assert len(config["token_ids"]) == 30
        assert config["noise_seed"] == (NEW_SEED if item["stage"] == "A" else OLD_SEED)
    for offset in range(0, len(configs), 2):
        negative, positive = [deepcopy(item["intervention"]) for item in configs[offset:offset+2]]
        assert negative.pop("sign") == -1
        assert positive.pop("sign") == 1
        assert negative == positive


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("model_sha256", "other"), ("action_seed", 44),
    ("video_seed", 44), ("steps", 8), ("valid_run", False), ("robustness_sha256", "other"),
])
def test_resume_rejects_incomplete_or_changed_identity(field, value):
    saved = row()
    saved[field] = value
    with pytest.raises(ValueError):
        verify_result(saved, "state", "method", "model", "plan")


def test_resume_sealed_output_and_corruption(tmp_path):
    path = pair_path(tmp_path, "discovery", "state", "method")
    write_json(path, seal(row()))
    saved = read_sealed(path)
    verify_result(saved, "state", "method", "model", "plan")
    with pytest.raises(FileExistsError):
        write_json(path, seal(row()))
    saved["metrics"]["prefix_normalized_rmse"] = 123
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_sealed(path)


def test_budget_boundary_and_atomic_status(tmp_path):
    assert not due(100, 1, now=3699)
    assert due(100, 1, now=3700)
    path = tmp_path / "status.json"
    atomic_json(path, {"status": "running"})
    atomic_json(path, {"status": "budget_exhausted"})
    assert json.loads(path.read_text())["status"] == "budget_exhausted"
    assert not list(tmp_path.glob("*.tmp.*"))


def test_grouping_does_not_mix_amplitude_or_seed():
    results = []
    for seed in (OLD_SEED, NEW_SEED):
        for epsilon in (0.01, 0.03):
            for group in range(8):
                for sign in (-1, 1):
                    config = {"layer_ids": [0], "step_ids": [9], "noise_seed": seed,
                              "epsilon": epsilon, "sign": sign}
                    item = row()
                    item["metrics"]["prefix_normalized_rmse"] = (group + 1) * epsilon
                    results.append((item, config, group))
    grouped, _, _ = vectors(results)
    assert len(grouped) == 4
    for values in grouped.values():
        np.testing.assert_allclose(values, np.arange(1, 9))
    with pytest.raises(ValueError, match="Duplicate"):
        vectors(results + [results[0]])
    incomplete, _, _ = vectors(results[:-1])
    assert len(incomplete) == 3


def test_bootstrap_weights_tasks_then_episodes_not_states():
    records = [dict(suite="suite", task_id=0, episode_id=0, score=0.0)] * 20
    records += [dict(suite="suite", task_id=0, episode_id=1, score=1.0),
                dict(suite="suite", task_id=1, episode_id=0, score=1.0)]
    result = bootstrap(records, "score", 100)
    assert result["mean"] == 0.75
    assert result["tasks"] == 2 and result["episodes"] == 3
    assert bootstrap(records, "score", 100) == result
    assert bootstrap([{**records[0], "score": None}], "score", 100)["mean"] is None


def test_partial_report_has_no_false_gate_or_speedup(tmp_path):
    base = tmp_path / "base"
    for subset in ("discovery", "calibration"):
        write_json(base / f"diagnostics/e2_{subset}/plan.json", seal({"methods": []}))
        write_json(base / f"diagnostics/features_{subset}/features.json", seal({"records": []}))
    run = tmp_path / "run"
    spec = {"base": str(base), "calibration": {"floor": 1e-4}, "methods": [],
            "selection": {"state_ids_by_split": {subset: [] for subset in ("discovery", "calibration")}},
            "original_repeat_max_abs": {"discovery": 0, "calibration": 0}, "sha256": "plan"}
    report(SimpleNamespace(run=run), spec, "budget_exhausted", replicates=10)
    result = json.loads((run / "report.json").read_text())
    assert result["status"] == "budget_exhausted"
    assert result["completed_pairs"] == 0
    assert not any(item["ranking_gate_passed"] for item in result["decisions"])
    assert "no closed-loop or speedup evidence" in (run / "report.md").read_text()


def test_constant_responses_have_no_spurious_rho():
    from openwam.research.analysis import conditional_association
    association = conditional_association(np.ones(8), np.ones(8))
    assert association["spearman"] is None
    assert task_metrics([{**row(), **association}])[0]["spearman"] is None


def test_rank_gate_does_not_drop_constant_states_or_tasks():
    records = [{"spearman": 0.9}] * 24
    metrics = {"spearman": {"tasks": 8, "mean": 0.9}, "top25_overlap": {"mean": 0.75}}
    assert ranking_passes(records, metrics)
    assert not ranking_passes(records[:-1], metrics)
    assert not ranking_passes(records[:-1] + [{"spearman": None}], metrics)
    metrics["spearman"]["tasks"] = 7
    assert not ranking_passes(records, metrics)


def test_runtime_budget_resume_and_mismatched_model(tmp_path, monkeypatch):
    import benchmarks.libero.research as research
    import openwam.research.exploratory as exploratory
    import openwam.research.runner as runner
    import scripts.research_e2_robustness as study

    base, bank, output = tmp_path / "base", tmp_path / "bank", tmp_path / "run"
    split_path = tmp_path / "split.json"
    split = seal({"tasks": []})
    write_json(split_path, split)
    selected = {subset: ["state"] for subset in ("discovery", "calibration")}
    entries = {}
    for subset in ("discovery", "calibration"):
        directory = bank / "state_bank" / subset
        directory.mkdir(parents=True)
        np.savez(directory / "state.npz", actions=np.zeros((32, 10)))
        entry = {"state_id": "state", "file": "state.npz", "sha256": "filehash", "split": subset}
        entries[subset] = [(entry, {})]
        write_json(base / f"diagnostics/t0_{subset}/t0_certificate.json", seal({"status": "passed"}))
    identity = {"sha256": "model"}
    server = SimpleNamespace(engine=object())
    monkeypatch.setattr(research, "build_server", lambda args: server)
    monkeypatch.setattr(research, "model_identity", lambda server, args: identity)
    monkeypatch.setattr(research, "load_states", lambda path, split: ({}, entries[path.split("/")[-1]]))
    monkeypatch.setattr(research, "execution_horizon", lambda *args: 32)
    monkeypatch.setattr(research, "validate_t0", lambda *args: None)
    monkeypatch.setattr(exploratory, "validate_bank_identity", lambda *args: None)
    monkeypatch.setattr(runner, "run_t0", lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setattr(runner, "paired_generate", lambda *args, **kwargs: ({}, 0))
    monkeypatch.setattr(runner, "compare_contexts", lambda *args: ({"prefix_normalized_rmse": 0}, [], "mapping"))
    calls = []
    def execute(engine, conditions, config, **kwargs):
        if config.kind == "value_noise":
            calls.append(config)
        return {"metrics": {"prefix_normalized_rmse": 0}}
    monkeypatch.setattr(runner, "run_pair", execute)
    monkeypatch.setattr(study, "due", lambda *args: len(calls) >= 1)
    args = SimpleNamespace(run=output, hours=10, bank_run=bank)
    spec = {"base": str(base), "split_file": str(split_path), "sha256": "plan",
            "selection": {"task_split_sha256": split["sha256"], "state_ids_by_split": selected},
            "model_sha256_by_split": {subset: "model" for subset in selected},
            "methods": methods(selection(), 1e-4)[:1]}
    assert study.run_study(args, spec, 0) == "budget_exhausted"
    assert len(calls) == 1
    path = pair_path(output, "discovery", "state", spec["methods"][0]["method_id"])
    saved = read_sealed(path)
    assert saved["state_file_sha256"] == "filehash"
    before = path.read_bytes()
    monkeypatch.setattr(study, "due", lambda *args: False)
    assert study.run_study(args, spec, 0) == "complete"
    assert len(calls) == 2
    assert path.read_bytes() == before
    identity["sha256"] = "other"
    with pytest.raises(ValueError, match="Loaded model/backend"):
        study.run_study(args, spec, 0)
