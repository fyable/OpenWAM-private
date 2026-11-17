import pytest

from benchmarks.robotwin import compare_joint as base
from benchmarks.robotwin import compare_joint_isolated as comparison


def test_four_arm_checkpoint_and_command_wiring(monkeypatch, tmp_path):
    monkeypatch.setenv("JOINT_CKPT_DIR", str(tmp_path / "joint"))
    monkeypatch.setenv("ISOLATED_CKPT_DIR", str(tmp_path / "isolated"))
    monkeypatch.delenv("TASKS", raising=False)
    monkeypatch.delenv("EPISODES", raising=False)
    cfg = comparison.configuration("diagnostic")
    assert cfg.tasks == ["adjust_bottle"] and cfg.episodes == 5
    assert len(cfg.methods) == 4
    for method, selected in cfg.method_configs.items():
        isolated = method.startswith("isolated_")
        assert selected.expected_mask == ("isolated" if isolated else "mutual")
        assert selected.checkpoint == tmp_path / ("isolated" if isolated else "joint")
        selected.weight = selected.checkpoint / "checkpoint_step_1.safetensors"
        command = base.server_command(cfg, method, "token")
        assert command[command.index("--ckpt-dir") + 1] == str(selected.checkpoint)
        assert command[command.index("--sampler") + 1] == comparison.METHODS[method][0]
        assert command[command.index("--denoise-steps") + 1] == "10"


def test_mask_preflight_rejects_wrong_checkpoint(monkeypatch, tmp_path):
    import sys

    robotwin = tmp_path / "robotwin/script"
    robotwin.mkdir(parents=True)
    (robotwin / "eval_policy.py").write_text("")
    checkpoint = tmp_path / "isolated"
    checkpoint.mkdir()
    (checkpoint / "normalization_stats.npy").write_bytes(b"fixture")
    (checkpoint / "checkpoint_step_1.safetensors").write_bytes(b"fixture")
    (checkpoint / "config.yaml").write_text(
        "model:\n  architecture:\n    framework: dual_system\n    variant: joint_self_attn\n"
        "    attention_mask_mode: isolated\ndataloader:\n  type: robotwin\n  num_frames: 33\n"
        "  video_stride: 4\n  action_mode: eef\n")
    monkeypatch.setenv("ISOLATED_CKPT_DIR", str(checkpoint))
    monkeypatch.setenv("ROBOTWIN_PATH", str(robotwin.parent))
    monkeypatch.setenv("SERVER_PYTHON", sys.executable)
    monkeypatch.setenv("ROBOTWIN_PYTHON", sys.executable)
    cfg = comparison.configuration("diagnostic").method_configs["isolated_native_n10"]
    base.preflight(cfg)
    cfg.expected_mask = "mutual"
    with pytest.raises(ValueError, match="matching attention mask"):
        base.preflight(cfg)


def test_four_arm_summary_reads_recorded_methods(tmp_path):
    base.write_json(tmp_path / "protocol.json", {
        "identity": "fixture", "tasks": ["adjust_bottle"], "episodes_per_task": 5,
        "diagnostic": True, "methods": comparison.METHODS})
    assert not base.summarize(tmp_path)
    report = base.read_json(tmp_path / "comparison.json")
    assert set(report["methods"]) == set(comparison.METHODS)


def test_execution_modes_and_overrides(monkeypatch):
    monkeypatch.delenv("COMPARE_ROOT", raising=False)
    monkeypatch.delenv("TASKS", raising=False)
    monkeypatch.delenv("EPISODES", raising=False)
    cfg = comparison.configuration("execution-ablation")
    assert cfg.tasks == base.DIAGNOSTIC_TASKS
    assert len(cfg.methods) * len(cfg.tasks) * cfg.episodes == 75
    selected = cfg.method_configs["joint_prefix_n10"]
    selected.weight = selected.checkpoint / "checkpoint_step_1.safetensors"
    command = base.server_command(cfg, "joint_prefix_n10", "token")
    assert "inference.joint_execution_backend=prefix" in command
    assert "inference.request_timing=true" in command
    assert cfg.root != comparison.configuration("diagnostic").root
    regression = comparison.configuration("execution-regression")
    assert len(regression.methods) == 2 and regression.episodes == 20
    capture = comparison.configuration("capture")
    assert len(capture.methods) == 1
    assert "OPENWAM_REQUEST_TRACE_DIR" in capture.method_configs["joint_parallel_n10"].server_environment
