import json

import pytest

from benchmarks.libero.compare_original_paraflow import default_options, main, mode_options, original_command, summarize


def test_plan_no_assets_required(capsys):
    main(["plan"])
    assert "no cross-semantics" in capsys.readouterr().out


def test_preflight_rejects_missing_assets(monkeypatch):
    for name in ("PF_PYTHON", "PF_NATIVE_ROOT", "PF_OPTIONS", "TRACE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="TRACE"):
        main(["preflight"])


def test_official_command_requires_exclusive_and_validation():
    command = original_command("python", "options.json", "out.json", warmup=48, samples=120, steps=[4, 10])
    assert "paraflow.infer.benchmark" in command
    assert "--require-validation" in command
    assert "--require-exclusive" in command


def test_report_only_computes_within_panel_ratios(tmp_path):
    for mode, latency in [("cuda_graph", 100), ("model_deep_tuned", 50)]:
        (tmp_path / f"paraflow_{mode}.json").write_text(json.dumps({"completed": True, "results": [
            {"steps": 4, "median_ms": latency, "p95_ms": latency, "samples_ms": [latency]}]}))
    summarize(tmp_path)
    text = (tmp_path / "comparison.md").read_text()
    assert "2.000" in text
    assert "NOT an algorithm-equivalent" in text
    assert "not_run or incomplete" in text


@pytest.mark.parametrize("mode", ["cuda_graph", "model_fused", "model_deep"])
def test_non_tuned_modes_disable_tuned_only_operators(mode):
    options = {"operators": {"plan": "accepted", "refine_graph": True, "gpu_pixels": True, "early_vae": True}}
    local = mode_options(options, mode)
    assert local["operators"] == {
        "plan": "portable", "refine_graph": False, "gpu_pixels": False, "early_vae": False,
    }
    assert options["operators"]["plan"] == "accepted"


def test_tuned_options_preserved_and_eager_rejected():
    options = {"operators": {"plan": "accepted", "refine_graph": True}}
    assert mode_options(options, "model_deep_tuned")["operators"] == options["operators"]
    with pytest.raises(ValueError, match="Unsupported"):
        mode_options(options, "eager")


def test_default_tuned_plan_portable_with_graph_and_pixels():
    options = mode_options(default_options(), "model_deep_tuned")
    assert options["operators"] == {
        "plan": "portable", "refine_graph": True, "gpu_pixels": True, "early_vae": True,
    }


def test_report_records_actual_operator_plan(tmp_path):
    (tmp_path / "options_model_deep_tuned.json").write_text(json.dumps(default_options()))
    (tmp_path / "paraflow_model_deep_tuned.json").write_text(json.dumps({"completed": True, "results": [
        {"steps": 4, "median_ms": 30, "p95_ms": 35, "samples_ms": [30]}]}))
    summarize(tmp_path)
    assert "| model_deep_tuned | portable | 4 |" in (tmp_path / "comparison.md").read_text()
