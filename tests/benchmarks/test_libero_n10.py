import json
import os
import subprocess
from pathlib import Path

import pytest

from benchmarks.libero.compare_joint import summarize_root


def test_n10_plan_is_non_mutating(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "results"
    result = subprocess.run(["bash", "scripts/compare_alpha_libero_n10.sh", "plan"], cwd=root,
                            env=dict(os.environ, COMPARE_ROOT=str(output)), capture_output=True, text=True, check=True)
    assert "4000 episodes" in result.stdout
    assert "shared conditions" in result.stdout
    assert not output.exists()


def test_n10_summary_selects_two_methods(tmp_path):
    methods = ("native_n10_e8", "joint_shared_n10_e8")
    for name in methods:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"diagnostic": True}))
        (directory / "summary.json").write_text(json.dumps({
            "overall": {"trials": 20, "successes": 18, "success_rate": .9,
                        "tasks_complete": 4, "tasks_expected": 4},
            "suite_results": [], "missing_runs": [],
        }))
    report = summarize_root(tmp_path, methods)
    assert set(report) == set(methods)
    assert all(row["status"] == "complete" for row in report.values())


def test_subset_summary_uses_manifest_episode_count(tmp_path):
    directory = tmp_path / "joint_shared_n10_e8"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({
        "diagnostic": True, "resume_signature": {"jobs_count": 2},
        "trial_run": {"num_trials": 5},
    }))
    (directory / "summary.json").write_text(json.dumps({
        "overall": {"trials": 10, "successes": 8, "success_rate": .8,
                    "tasks_complete": 2, "tasks_expected": 2},
        "suite_results": [], "missing_runs": [],
    }))
    report = summarize_root(tmp_path, [directory.name])
    assert report[directory.name]["status"] == "complete"


def test_resume_rejects_changed_joint_optimizations(tmp_path):
    from benchmarks.libero.scheduler import _validate_resume_signature

    original = {"sampler": "joint-paraflow", "diagnostic": True,
                "joint_optimization_overrides": ["inference.joint_share_conditions=true"]}
    (tmp_path / "manifest.json").write_text(json.dumps({"resume_signature": original}))
    _validate_resume_signature(tmp_path, original)
    with pytest.raises(RuntimeError, match="joint_optimization_overrides"):
        _validate_resume_signature(tmp_path, {**original, "joint_optimization_overrides": []})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("scope", ["phase_video_action", "action_only"])
def test_engine_enables_measured_joint_optimizations(monkeypatch, enabled, scope):
    from omegaconf import OmegaConf
    from openwam.deploy.engine import BaseInferenceEngine, JointInferenceEngine

    cfg = OmegaConf.create({
        "inference": {"sampler": "joint-paraflow", "inference_horizon": 8,
                      "joint_cache_schedule": enabled, "joint_share_conditions": enabled,
                      "joint_history_scope": scope,
                      "joint_action_tail_init": "endpoint_bridge" if enabled else "stage_residual",
                      "joint_video_tail_init": "endpoint_bridge" if enabled else "stage_residual"},
        "model": {"video_backbone": {"name": "wan22_ti2v_5b"},
                  "architecture": {"attention_mask_mode": "mutual"}},
    })
    architecture = type("DualSystemSelfAttnArchitecture", (), {})()

    def initialize(self, cfg, **kwargs):
        self.cfg = cfg
        self.architecture = kwargs["architecture"]

    monkeypatch.setattr(BaseInferenceEngine, "__init__", initialize)
    monkeypatch.setattr(JointInferenceEngine, "_init_optimizations", lambda self: setattr(self, "_decode_video", False))
    monkeypatch.setattr(JointInferenceEngine, "_init_cfg", lambda self: setattr(self, "_cfg_scale", 1.0))
    engine = JointInferenceEngine(cfg, architecture=architecture)
    assert engine._joint_sampler.cache_schedule is enabled
    assert engine._joint_sampler.share_conditions is enabled
    assert engine._joint_sampler.history_scope == scope
    assert engine._joint_sampler.action_tail_init == ("endpoint_bridge" if enabled else "stage_residual")
    assert engine._joint_sampler.video_tail_init == ("endpoint_bridge" if enabled else "stage_residual")
