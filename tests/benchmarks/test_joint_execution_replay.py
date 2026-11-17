import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.robotwin import joint_execution_replay as replay
from benchmarks.robotwin.compare_joint import digest
from openwam.deploy.parallel_joint import JointParaFlowSampler


def test_trace_origin_and_content_are_checked(tmp_path):
    checkpoint = tmp_path / "weights"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text("fixture")
    trace = tmp_path / "observations.npz"
    trace.write_bytes(b"fixture")
    metadata = {"benchmark": "robotwin", "source": "actual_E8_requests", "trace_sha256": digest(trace),
                "checkpoint_config": digest(checkpoint / "config.yaml")}
    trace.with_suffix(".json").write_text(json.dumps(metadata))
    assert replay.validate_trace(trace, checkpoint) == metadata
    weight = checkpoint / "checkpoint_step_1.safetensors"
    weight.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="checkpoint identity"):
        replay.validate_trace(trace, checkpoint, weight)
    metadata["benchmark"] = "libero"
    trace.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="RoboTwin"):
        replay.validate_trace(trace, checkpoint)
    metadata["benchmark"] = "robotwin"
    trace.with_suffix(".json").write_text(json.dumps(metadata))
    trace.write_bytes(b"changed")
    with pytest.raises(ValueError):
        replay.validate_trace(trace, checkpoint)


def test_raw_proprio_is_not_model_padded_dimension():
    training = {"model": {"architecture": {"state_dim": 80, "attention_mask_mode": "mutual"}},
                "dataloader": {"type": "robotwin", "action_mode": "eef", "unify_action": True,
                               "unify_action_map": ["0-9", "34-43"]}}
    replay.validate_observations((None, None, np.ones((3, 20))), training)
    with pytest.raises(ValueError, match="raw proprio"):
        replay.validate_observations((None, None, np.ones((3, 80))), training)


def test_identical_candidate_replay_clones_rng_and_does_not_share_state():
    source, target = JointParaFlowSampler(), JointParaFlowSampler(execution_backend="prefix")
    source.action_generator = torch.Generator().manual_seed(42)
    source.video_generator = torch.Generator().manual_seed(24)
    source.history = torch.ones(3, 1, 32, 8)
    source.video_phases = {0: torch.ones(3, 1, 4, 3, 2, 2)}
    source.request = 4
    source.signature = ("fixture",)
    replay.copy_history(source, target)
    assert target.request == 4 and target.signature == source.signature
    assert target.video_phases is not source.video_phases
    assert target.action_generator is not source.action_generator
    torch.testing.assert_close(torch.randn(4, generator=target.action_generator), torch.randn(4, generator=source.action_generator))
    target.video_phases[1] = torch.zeros(1)
    assert 1 not in source.video_phases


def test_replay_launcher_wires_paths_and_threshold_flag(tmp_path):
    import os
    import subprocess

    interpreter = tmp_path / "python"
    interpreter.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    interpreter.chmod(0o755)
    env = dict(os.environ, SERVER_PYTHON=str(interpreter), TRACE="/trace/observations.npz", GPU="6",
               REPLAY_ROOT=str(tmp_path / "replay"), ALLOW_PARITY_FAILURE="1")
    result = subprocess.run(["bash", "scripts/compare_alpha_robotwin_joint_isolated.sh", "replay"],
                            env=env, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=True)
    assert "benchmarks.robotwin.joint_execution_replay" in result.stdout
    assert "cuda:6" in result.stdout and "--allow-parity-failure" in result.stdout


def test_profile_boundary_restores_original_on_exception():
    from types import SimpleNamespace

    architecture = SimpleNamespace(video_backbone=SimpleNamespace(vae=None), forward=lambda: "fixture")
    previous = architecture.forward
    with pytest.raises(RuntimeError, match="injected"):
        with replay.profile_boundaries(SimpleNamespace(architecture=architecture), "cuda:0", {}):
            assert architecture.forward is not previous
            raise RuntimeError("injected")
    assert architecture.forward is previous
