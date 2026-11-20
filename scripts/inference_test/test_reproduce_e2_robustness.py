import json
import subprocess
from types import SimpleNamespace

import pytest

from openwam.research.protocol import seal, write_json
from scripts.reproduce_e2_robustness import BASE_NAME, Pipeline, check_assets, select_gpu


def test_gpu_selection_excludes_occupied_devices_and_breaks_ties_by_index():
    rows = "0, GPU-zero, RTX 4090, 48000\n1, GPU-one, RTX 4090, 49000\n2, GPU-two, RTX 4090, 48000"
    assert select_gpu(rows, {"GPU-one"})["index"] == 0
    assert select_gpu(rows, set(), "2")["uuid"] == "GPU-two"
    with pytest.raises(RuntimeError, match="No idle GPU"):
        select_gpu(rows, {"GPU-one"}, "1")
    with pytest.raises(RuntimeError, match="No idle GPU"):
        select_gpu(rows, set(), "7")


def test_local_assets_must_match_reference_including_extra_files(tmp_path, monkeypatch):
    from benchmarks.libero import research

    reference = tmp_path / "reference"
    expected = {"checkpoint.safetensors": "weights", "config.yaml": "configuration"}
    write_json(reference / BASE_NAME / "diagnostics/e2_discovery/manifest.json", seal({
        "model": {"checkpoint_file": "checkpoint.safetensors", "sha256": "model", "assets": expected},
    }))
    monkeypatch.setattr(research, "assets_identity", lambda args: dict(expected))
    assert check_assets(reference, tmp_path, "checkpoint.safetensors")["assets"] == expected
    monkeypatch.setattr(research, "assets_identity", lambda args: {**expected, "extra.pt": "unknown"})
    with pytest.raises(ValueError, match="extra.pt"):
        check_assets(reference, tmp_path, "checkpoint.safetensors")
    with pytest.raises(ValueError, match="filename"):
        check_assets(reference, tmp_path, "wrong.safetensors")


def test_failed_stage_does_not_run_following_work(tmp_path, monkeypatch):
    from scripts import reproduce_e2_robustness as reproduction

    monkeypatch.setattr(reproduction, "source_identity", lambda: {"source_sha256": "frozen"})
    pipeline = Pipeline(SimpleNamespace(run=tmp_path))
    reached = []

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["failed-preflight"])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.stage("preflight", ["failed-preflight"], operation=lambda: reached.append("bad"))
        pipeline.stage("collection", operation=lambda: reached.append("collection"))
    assert reached == []
    assert pipeline.completed == []
    pipeline.status("failed")
    status = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert status["stage"] == "preflight"
    assert status["status"] == "failed"


def test_source_changes_prevent_new_stage(tmp_path, monkeypatch):
    from scripts import reproduce_e2_robustness as reproduction

    monkeypatch.setattr(reproduction, "source_identity", lambda: {"source_sha256": "before"})
    pipeline = Pipeline(SimpleNamespace(run=tmp_path))
    monkeypatch.setattr(reproduction, "source_identity", lambda: {"source_sha256": "after"})
    with pytest.raises(ValueError, match="source changed"):
        pipeline.stage("baseline", operation=lambda: pytest.fail("Should not run"))
