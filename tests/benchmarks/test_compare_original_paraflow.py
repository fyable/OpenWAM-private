import json

import pytest

from benchmarks.libero.compare_original_paraflow import main, original_command, summarize


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
    for mode, latency in [("eager", 100), ("cuda_graph", 50)]:
        (tmp_path / f"paraflow_{mode}.json").write_text(json.dumps({"completed": True, "results": [
            {"steps": 4, "median_ms": latency, "p95_ms": latency, "samples_ms": [latency]}]}))
    summarize(tmp_path)
    text = (tmp_path / "comparison.md").read_text()
    assert "2.000" in text
    assert "NOT an algorithm-equivalent" in text
    assert "not_run or incomplete" in text
