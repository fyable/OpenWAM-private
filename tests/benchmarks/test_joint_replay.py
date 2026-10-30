"""Fixed-observation replay input contracts, without loading weights."""

import numpy as np
import pytest

from benchmarks.libero.replay_joint import load_observations, summarize, select_warm_rows, resolve_checkpoint


def test_native_and_joint_warm_selection():
    rows = [dict(index=index, episode_start=index == 0, sampler_stats=None) for index in range(5)]
    assert len(select_warm_rows(rows, 2)) == 3
    rows[2]["sampler_stats"] = {"cold": True}
    assert len(select_warm_rows(rows, 2)) == 2
    rows[3]["protocol_cold"] = True
    assert len(select_warm_rows(rows, 2)) == 1


def test_checkpoint_discovery(tmp_path):
    with pytest.raises(ValueError):
        resolve_checkpoint(tmp_path, None)
    (tmp_path / "checkpoint_step_123.safetensors").touch()
    assert resolve_checkpoint(tmp_path, None) == "checkpoint_step_123.safetensors"
    (tmp_path / "checkpoint_step_456.safetensors").touch()
    with pytest.raises(ValueError):
        resolve_checkpoint(tmp_path, None)


def test_load_observations(tmp_path):
    path = tmp_path / "observations.npz"
    arrays = dict(images=np.zeros((3, 8, 8, 3), dtype=np.uint8),
                  prompts=np.array(["task"] * 3), states=np.zeros((3, 8)),
                  episode_starts=np.array([True, False, False]))
    np.savez(path, **arrays)
    assert len(load_observations(path)[0]) == 3
    arrays["episode_starts"][:] = False
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="episode_starts"):
        load_observations(path)


def test_timing_percentiles():
    assert summarize([], "ms") is None
    assert summarize([{"ms": 1}, {"ms": 3}], "ms")["p50_ms"] == 2


def test_diagnostic_preserves_official_protocol():
    from benchmarks.libero.scheduler import _build_parser, _validate_protocol

    args = _build_parser().parse_args(["--diagnostic", "--num-trials", "5"])
    _validate_protocol(args)
    args.diagnostic = False
    with pytest.raises(ValueError, match="trial range"):
        _validate_protocol(args)


def test_resume_defaults_and_core_mismatch(tmp_path):
    import json
    from benchmarks.libero.scheduler import _validate_resume_signature

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"resume_signature": {"steps": 10}}))
    _validate_resume_signature(tmp_path, {"steps": 10, "sampler": "native", "diagnostic": False})
    path.write_text(json.dumps({"resume_signature": {
        "sampler": "joint-paraflow", "diagnostic": True, "paraflow_core": {"core.py": "old"},
    }}))
    with pytest.raises(RuntimeError, match="paraflow_core"):
        _validate_resume_signature(tmp_path, {
            "sampler": "joint-paraflow", "diagnostic": True, "paraflow_core": {"core.py": "new"},
        })
