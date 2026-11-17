import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from openwam.deploy.joint_solver import align_trajectory, align_video_trajectory, couple_noise


def test_tail_bridge_formula_overlap_and_noise():
    trajectory = torch.randn(5, 1, 32, 10)
    fresh = torch.randn(1, 32, 10)
    sigmas = torch.tensor([1., .8, .5, .2, 0.])
    residual = align_trajectory(trajectory, fresh, sigmas, 8, 32)
    actual = align_trajectory(trajectory, fresh, sigmas, 8, 32, tail_state_init="endpoint_bridge")
    torch.testing.assert_close(actual[..., :24, :], residual[..., :24, :], rtol=0, atol=0)
    noise = couple_noise(trajectory[0], fresh, 8, 32)
    torch.testing.assert_close(actual[0], noise, rtol=0, atol=0)
    scale = sigmas[:, None, None, None]
    expected = (1 - scale) * trajectory[-1, ..., -1:, :] + scale * noise[..., 24:, :]
    torch.testing.assert_close(actual[..., 24:, :], expected)
    torch.testing.assert_close(actual[-1, ..., 24:, :], trajectory[-1, ..., -1:, :].expand(1, 8, 10))
    with pytest.raises(ValueError, match="tail initialization"):
        align_trajectory(trajectory, fresh, sigmas, 8, 32, tail_state_init="bad")


def test_video_tail_preserves_overlap_and_current_prefix():
    trajectory = torch.randn(5, 1, 2, 3, 2, 2)
    fresh = torch.randn(1, 2, 3, 2, 2)
    prefix = torch.randn(1, 2, 1, 2, 2)
    grid = torch.tensor([1., .8, .5, .2, 0.])
    actual = align_video_trajectory(trajectory, fresh, grid, 1, prefix, tail_state_init="endpoint_bridge")
    torch.testing.assert_close(actual[:, :, :, :1], prefix[None].expand(5, -1, -1, -1, -1, -1))
    torch.testing.assert_close(actual[1:, :, :, 1:2], trajectory[1:, :, :, 2:3])
    torch.testing.assert_close(actual[-1, :, :, 2:], trajectory[-1, :, :, -1:])


def test_tail_script_plan_and_four_configurations(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, COMPARE_ROOT=str(tmp_path / "out"))
    plan = subprocess.run(["bash", "scripts/ablate_alpha_joint_tail.sh", "plan"], cwd=root,
                          env=env, capture_output=True, text=True, check=True)
    assert "20 episodes" in plan.stdout
    assert not (tmp_path / "out").exists()
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "bash"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
    stub.chmod(0o755)
    env.update(PATH=f"{binary}:{os.environ['PATH']}", SERVER_PYTHON="/bin/echo")
    run = subprocess.run(["/bin/bash", "scripts/ablate_alpha_joint_tail.sh"], cwd=root,
                         env=env, capture_output=True, text=True, check=True)
    commands = [line for line in run.stdout.splitlines() if line.startswith("scripts/eval_alpha_libero.sh")]
    assert len(commands) == 4
    for command, action, video in zip(commands, ["stage_residual", "endpoint_bridge", "stage_residual", "endpoint_bridge"],
                                      ["stage_residual", "stage_residual", "endpoint_bridge", "endpoint_bridge"]):
        assert f"inference.joint_action_tail_init={action}" in command
        assert f"inference.joint_video_tail_init={video}" in command
        assert "inference.joint_share_conditions=true" in command


def test_resume_rejects_tail_change(tmp_path):
    from benchmarks.libero.scheduler import _validate_resume_signature

    original = {"sampler": "joint-paraflow", "diagnostic": True,
                "joint_optimization_overrides": ["inference.joint_action_tail_init=stage_residual"]}
    (tmp_path / "manifest.json").write_text(json.dumps({"resume_signature": original}))
    with pytest.raises(RuntimeError, match="joint_optimization_overrides"):
        _validate_resume_signature(tmp_path, {**original, "joint_optimization_overrides": ["inference.joint_action_tail_init=endpoint_bridge"]})


def test_episode_selection_and_capture_assembly(tmp_path):
    from benchmarks.libero.joint_tail_diagnosis import assemble, episode_indices
    from openwam.deploy.request_trace import RequestTrace

    assert list(episode_indices(np.array([True, False, True, False, False]), 1)) == [2, 3, 4]
    with pytest.raises(ValueError, match="unavailable"):
        episode_indices(np.array([True]), 1)
    trace = RequestTrace(tmp_path / "shards/session_test", warmup=0, measure=100)
    for _ in range(3):
        trace.record({"first_frame_image": [np.zeros((8, 8, 3), dtype=np.uint8)],
                      "proprio": np.zeros(10), "prompt": "task"})
    output = tmp_path / "trace.npz"
    assemble(tmp_path / "shards", output)
    assert output.is_file()
    (tmp_path / "shards/session_retry").mkdir()
    with pytest.raises(ValueError, match="exactly one"):
        assemble(tmp_path / "shards", tmp_path / "other.npz")


@pytest.mark.parametrize("action_tail", ["stage_residual", "endpoint_bridge"])
@pytest.mark.parametrize("video_tail", ["stage_residual", "endpoint_bridge"])
def test_tail_modes_single_warm_forward_constraints_and_reset(action_tail, video_tail):
    import runpy
    from openwam.deploy.parallel_joint import JointParaFlowSampler

    fixture = runpy.run_path(str(Path(__file__).resolve().parents[1] / "test_parallel_joint.py"))
    architecture = fixture["FakeArchitecture"]()
    sampler = JointParaFlowSampler(action_tail_init=action_tail, video_tail_init=video_tail)
    for index in range(5):
        video, action = fixture["sample"](sampler, architecture, prefix=float(index))
        assert sampler.last_stats["joint_forward_calls"] == (4 if index < 2 else 1)
        assert sampler.last_stats["action_tail_init"] == action_tail
        assert sampler.last_stats["video_tail_init"] == video_tail
        assert torch.all(video[:, :, :1] == index)
        assert torch.all(action[..., 10:] == 0)
    sampler.reset()
    actual = fixture["sample"](sampler, architecture)
    reference = fixture["sample"](JointParaFlowSampler(action_tail_init=action_tail, video_tail_init=video_tail), architecture)
    for value, expected in zip(actual, reference):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_diagnosis_cpu_same_noise_and_no_state_contamination(tmp_path, monkeypatch):
    import runpy
    from types import SimpleNamespace
    from benchmarks.libero import joint_tail_diagnosis as diagnosis
    from openwam.deploy import parallel_joint

    fixture = runpy.run_path(str(Path(__file__).resolve().parents[1] / "test_parallel_joint.py"))
    architecture = fixture["FakeArchitecture"]()
    architecture.normalizer = SimpleNamespace(unnormalize=lambda value: value[..., :10])
    schedule = [(float(value), float(value)) for value in torch.linspace(1000, 0, 11)]
    expected_sampler = parallel_joint.JointParaFlowSampler(cache_schedule=True)

    class Engine:
        def __init__(self):
            self.architecture = architecture
            self.cfg = SimpleNamespace(inference=SimpleNamespace(denoise_steps=10))
            self._joint_sampler = None

        def generate(self, condition):
            if self._joint_sampler is None:
                return {"actions": np.zeros((32, 10), dtype=np.float32)}
            video, action = fixture["sample"](self._joint_sampler, architecture, schedule=schedule)
            original = parallel_joint.joint_velocity
            parallel_joint.joint_velocity = original_velocity
            try:
                expected = fixture["sample"](expected_sampler, fixture["FakeArchitecture"](), schedule=schedule)
            finally:
                parallel_joint.joint_velocity = original
            torch.testing.assert_close(video, expected[0], rtol=0, atol=0)
            torch.testing.assert_close(action, expected[1], rtol=0, atol=0)
            return {"actions": action[0, ..., :10].numpy()}

    original_velocity = parallel_joint.joint_velocity
    engine = Engine()
    monkeypatch.setattr(diagnosis, "build_engine", lambda args: (engine, "checkpoint.safetensors"))
    trace = tmp_path / "trace.npz"
    np.savez(trace, images=np.zeros((4, 8, 8, 3), dtype=np.uint8), states=np.zeros((4, 10)),
             prompts=np.array(["task"] * 4), episode_starts=np.array([True, False, False, False]))
    args = SimpleNamespace(output=tmp_path / "diagnosis", trace=trace, episode=0, ckpt_dir=tmp_path,
                           device="cpu", action_tail_init="stage_residual", video_tail_init="stage_residual")
    diagnosis.run(args)
    rows = [json.loads(line) for line in (args.output / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and rows[0]["cold"] and not rows[-1]["cold"]
    assert len(rows[-1]["stages"]) == 10
    assert rows[-1]["batch_partition"]["action"]["passed"]
    assert "same_noise_environment_first8" in rows[-1]
    assert parallel_joint.joint_velocity is original_velocity
