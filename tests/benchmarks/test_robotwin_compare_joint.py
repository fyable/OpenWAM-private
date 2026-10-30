import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.robotwin import compare_joint as comparison
from benchmarks.utils.eval_manifest import seal_manifest


def free_port():
    try:
        with socket.socket() as connection:
            connection.bind(("127.0.0.1", 0))
            return connection.getsockname()[1]
    except PermissionError:
        pytest.skip("Sandbox disallows network sockets")


@pytest.fixture
def campaign_env(tmp_path):
    robotwin = tmp_path / "RoboTwin"
    (robotwin / "script").mkdir(parents=True)
    (robotwin / "script/eval_policy.py").write_text("")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_step_1.safetensors").write_bytes(b"fake")
    (checkpoint / "normalization_stats.npy").write_bytes(b"fake")
    (checkpoint / "config.yaml").write_text(
        "model:\n  architecture:\n    framework: dual_system\n    variant: joint_self_attn\n"
        "    attention_mask_mode: mutual\ndataloader:\n  type: robotwin\n  num_frames: 33\n"
        "  video_stride: 4\n  action_mode: eef\n"
    )
    interpreter = tmp_path / "fake-python"
    fake = Path(__file__).with_name("fake_robotwin_labtasker.py")
    interpreter.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{fake}" "$@"\n')
    interpreter.chmod(0o755)
    return dict(os.environ, ROBOTWIN_PATH=str(robotwin), ROBOTWIN_PYTHON=str(interpreter),
                SERVER_PYTHON=str(interpreter), CKPT_DIR=str(checkpoint),
                COMPARE_ROOT=str(tmp_path / "comparison"), PORT="8940",
                SERVER_START_TIMEOUT="5", GPU="2", SIM_GPU="3",
                PYTHONPATH=str(comparison.ROOT))


def invoke(env, mode="diagnostic"):
    return subprocess.run([sys.executable, "-m", "benchmarks.robotwin.compare_joint", mode],
                          cwd=comparison.ROOT, env=env, capture_output=True, text=True, timeout=40)


def test_shell_plan_does_not_create_outputs(tmp_path):
    root = tmp_path / "absent"
    env = dict(os.environ, SERVER_PYTHON=sys.executable, COMPARE_ROOT=str(root))
    result = subprocess.run(["bash", "scripts/compare_alpha_robotwin.sh", "plan"],
                            cwd=comparison.ROOT, env=env, capture_output=True, text=True, check=True)
    assert "15000 episodes" in result.stdout
    assert "45 episodes" in result.stdout
    assert not root.exists()


def test_configuration_keeps_diagnostic_separate(monkeypatch):
    monkeypatch.delenv("COMPARE_ROOT", raising=False)
    full = comparison.configuration("full")
    diagnostic = comparison.configuration("diagnostic")
    assert full.root != diagnostic.root
    assert len(full.tasks) == 50 and full.episodes == 100
    assert diagnostic.tasks == comparison.DIAGNOSTIC_TASKS and diagnostic.episodes == 5


def test_server_parameters_are_matched():
    cfg = SimpleNamespace(server_python="python", checkpoint=Path("/weights"),
                          weight=Path("/weights/checkpoint_step_1.safetensors"), port=8940)
    for method, (sampler, steps) in comparison.METHODS.items():
        command = comparison.server_command(cfg, method, "unique-token")
        assert command[command.index("--sampler") + 1] == sampler
        assert command[command.index("--denoise-steps") + 1] == str(steps)
        assert command[command.index("--inference-horizon") + 1] == "8"
        assert command[command.index("--compile-enabled") + 1] == "false"
        assert command[command.index("--readiness-token") + 1] == "unique-token"
        assert "optimization.decode_video=false" in command


def test_readiness_rejects_foreign_server(monkeypatch):
    monkeypatch.setattr(comparison, "probe_server", lambda port: {"type": "pong", "readiness_token": "foreign"})
    with pytest.raises(RuntimeError, match="another server"):
        comparison.wait_ready(SimpleNamespace(poll=lambda: None), 8940, "owned", 1)


def test_readiness_exit_and_timeout(monkeypatch):
    with pytest.raises(RuntimeError, match="exited"):
        comparison.wait_ready(SimpleNamespace(poll=lambda: 1, returncode=1), 8940, "token", 1)

    def refused(port):
        raise ConnectionRefusedError

    monkeypatch.setattr(comparison, "probe_server", refused)
    with pytest.raises(TimeoutError):
        comparison.wait_ready(SimpleNamespace(poll=lambda: None), 8940, "token", 0)


def test_full_fake_campaign_and_resume(campaign_env):
    campaign_env["PORT"] = str(free_port())
    result = invoke(campaign_env)
    assert result.returncode == 0, result.stdout + result.stderr
    root = Path(campaign_env["COMPARE_ROOT"])
    report = comparison.read_json(root / "comparison.json")
    assert report["diagnostic"] is True
    for method, row in report["methods"].items():
        assert row["status"] == "complete"
        assert row["successes"] == row["episodes"] == 15
        assert len(list((root / method / "servers").glob("*/server.log"))) == 1
        command = comparison.read_json(next((root / method / "servers").glob("*/command.json")))
        assert command[command.index("--sampler") + 1] == comparison.METHODS[method][0]
    manifests = list((root / "manifests").glob("*.json"))
    assert len(manifests) == 3
    before = {str(path): path.stat().st_mtime_ns for path in root.rglob("result.json")}
    repeated = invoke(campaign_env)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert repeated.stdout.count("Skipping complete method") == 3
    assert before == {str(path): path.stat().st_mtime_ns for path in root.rglob("result.json")}
    summary = invoke(campaign_env, "summary")
    assert summary.returncode == 0
    assert (root / "comparison.md").is_file()
    for port in {int(campaign_env["PORT"])}:
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection(("127.0.0.1", port), timeout=1)


def test_failure_stops_later_methods_and_can_resume(campaign_env):
    campaign_env["PORT"] = str(free_port())
    failed = invoke(dict(campaign_env, FAKE_FAIL_ONCE="1", FAKE_FAIL_OPERATION="run_eval"))
    assert failed.returncode != 0
    root = Path(campaign_env["COMPARE_ROOT"])
    assert not (root / "native_n4_e8/servers").exists()
    assert all(row["status"] == "incomplete" for row in comparison.read_json(root / "comparison.json")["methods"].values())
    resumed = invoke(campaign_env)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr


def test_server_failure_and_port_ownership(campaign_env):
    campaign_env["PORT"] = str(free_port())
    failed = invoke(dict(campaign_env, FAKE_SERVER_FAIL_GPU="2"))
    assert failed.returncode != 0 and "server exited" in failed.stderr
    root = Path(campaign_env["COMPARE_ROOT"])
    assert not (root / "native_n4_e8/servers").exists()
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", int(campaign_env["PORT"])))
        occupied.listen()
        result = invoke(campaign_env)
        assert result.returncode != 0
        assert occupied.fileno() >= 0


def test_protocol_mismatch_rejected(campaign_env):
    campaign_env["PORT"] = str(free_port())
    result = invoke(campaign_env)
    assert result.returncode == 0, result.stdout + result.stderr
    root = Path(campaign_env["COMPARE_ROOT"])
    protocol = comparison.read_json(root / "protocol.json")
    protocol["episodes_per_task"] = 99
    comparison.write_json(root / "protocol.json", protocol)
    changed = invoke(campaign_env)
    assert changed.returncode != 0 and "protocol changed" in changed.stderr


def test_result_validation_and_incomplete_summary(tmp_path):
    manifest = seal_manifest({"benchmark": "robotwin", "mode": "demo_clean", "task": "adjust_bottle",
                              "entries": [{"episode": 0, "seed": 7, "instructions": {"unseen": "move"}}]})
    comparison.write_json(tmp_path / "manifests/adjust_bottle.json", manifest)
    comparison.write_json(tmp_path / "protocol.json", {
        "identity": "example", "diagnostic": True, "tasks": ["adjust_bottle"], "episodes_per_task": 1,
    })
    result = dict(operation="run_eval", task="adjust_bottle", mode="demo_clean", instruction_type="unseen",
                  episode_start=0, num_episodes=1, total_episodes=1, manifest_hash=manifest["manifest_hash"],
                  successes=0, episodes=[dict(episode=0, seed=7, instruction="move", success=False)])
    path = tmp_path / "native_n10_e8/results/adjust_bottle.json"
    comparison.write_json(path, result)
    assert comparison.complete_result(path, manifest, 1) is not None
    assert comparison.summarize(tmp_path) is False
    report = comparison.read_json(tmp_path / "comparison.json")
    assert report["methods"]["native_n10_e8"]["status"] == "complete"
    assert report["methods"]["native_n4_e8"]["success_rate"] is None
    result["episodes"][0]["seed"] = 8
    comparison.write_json(path, result)
    assert comparison.complete_result(path, manifest, 1) is None
    result["episodes"][0]["seed"] = 7
    result["episodes"][0]["success"] = 1
    comparison.write_json(path, result)
    assert comparison.complete_result(path, manifest, 1) is None


def test_client_env_and_commands(campaign_env, monkeypatch, tmp_path):
    for key, value in campaign_env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("ROBOTWIN_TEST_NUM", "1")
    monkeypatch.setenv("ROBOTWIN_EPISODE_MANIFEST", "stale")
    cfg = comparison.configuration("diagnostic")
    comparison.preflight(cfg)
    env = comparison.client_environment(cfg, tmp_path)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert "ROBOTWIN_TEST_NUM" not in env and "ROBOTWIN_EPISODE_MANIFEST" not in env
    command = comparison.client_command(cfg, "adjust_bottle", tmp_path)
    assert command[command.index("--total-episodes") + 1] == "5"
    assert command[command.index("--operation") + 1] == "build_manifest"


def test_campaign_without_network(campaign_env, monkeypatch):
    for key, value in campaign_env.items():
        monkeypatch.setenv(key, value)
    opened = []
    closed = []

    @contextmanager
    def fake_server(cfg, method):
        opened.append(method)
        try:
            yield SimpleNamespace(poll=lambda: None)
        finally:
            closed.append(method)

    monkeypatch.setattr(comparison, "policy_server", fake_server)
    assert comparison.main(["diagnostic"]) == 0
    assert opened == closed == list(comparison.METHODS)
    root = Path(campaign_env["COMPARE_ROOT"])
    report = comparison.read_json(root / "comparison.json")
    assert all(row["successes"] == row["episodes"] == 15 for row in report["methods"].values())
    assert len(list((root / "manifests").glob("*.json"))) == 3
    manifests = {path.name: path.read_bytes() for path in (root / "manifests").glob("*.json")}
    opened.clear()
    assert comparison.main(["diagnostic"]) == 0
    assert opened == []
    result_path = root / "native_n4_e8/results/open_laptop.json"
    result_path.unlink()
    assert comparison.main(["summary"]) == 1
    assert comparison.main(["diagnostic"]) == 0
    assert opened == ["native_n4_e8"]
    assert manifests == {path.name: path.read_bytes() for path in (root / "manifests").glob("*.json")}
    protocol = comparison.read_json(root / "protocol.json")
    protocol["episodes_per_task"] = 99
    comparison.write_json(root / "protocol.json", protocol)
    with pytest.raises(ValueError, match="protocol changed"):
        comparison.main(["diagnostic"])


def test_campaign_client_failure_closes_server(campaign_env, monkeypatch):
    for key, value in campaign_env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("FAKE_FAIL_ONCE", "1")
    monkeypatch.setenv("FAKE_FAIL_OPERATION", "run_eval")
    opened = []
    closed = []

    @contextmanager
    def fake_server(cfg, method):
        opened.append(method)
        try:
            yield SimpleNamespace(poll=lambda: None)
        finally:
            closed.append(method)

    monkeypatch.setattr(comparison, "policy_server", fake_server)
    with pytest.raises(RuntimeError, match="failed"):
        comparison.main(["diagnostic"])
    assert opened == closed == ["native_n10_e8"]
    monkeypatch.delenv("FAKE_FAIL_ONCE")
    assert comparison.main(["diagnostic"]) == 0
    assert opened == closed == ["native_n10_e8", *comparison.METHODS]


def test_policy_cleanup_when_readiness_fails(tmp_path, monkeypatch):
    class Reservation:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, endpoint):
            pass

    cfg = SimpleNamespace(root=tmp_path, server_python="python", checkpoint=tmp_path,
                          weight=tmp_path / "checkpoint_step_1.safetensors", port=8940, gpu="0", timeout=1)
    monkeypatch.setattr(comparison.socket, "socket", Reservation)
    process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(comparison, "spawn_resource", lambda *args, **kwargs: process)
    cleaned = []

    class Registry:
        def add(self, process, log):
            self.log = log

        def terminate_all(self):
            cleaned.append(True)
            self.log.close()

    def failed(*args):
        raise RuntimeError("another server")

    monkeypatch.setattr(comparison, "ProcessRegistry", Registry)
    monkeypatch.setattr(comparison, "wait_ready", failed)
    with pytest.raises(RuntimeError, match="another server"):
        with comparison.policy_server(cfg, "joint_n4_e8"):
            pytest.fail("Should not enter evaluation")
    assert cleaned == [True]


def test_lock_rejects_second_writer(tmp_path):
    with comparison.output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another comparison"):
            with comparison.output_lock(tmp_path):
                pass


def test_interrupt_cleans_owned_client_tree(campaign_env):
    env = dict(campaign_env, FAKE_HANG="1")
    process = subprocess.Popen([sys.executable, "-m", "benchmarks.robotwin.compare_joint", "diagnostic"],
                               cwd=comparison.ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    child_path = None
    try:
        deadline = time.monotonic() + 10
        root = Path(env["COMPARE_ROOT"])
        while time.monotonic() < deadline:
            matches = list(root.rglob("grandchild.pid"))
            if matches:
                child_path = matches[0]
                break
            time.sleep(.1)
        assert child_path is not None
        child_pid = int(child_path.read_text())
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=20)
        assert process.returncode == 130, output
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert (root / "comparison.json").is_file()
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=20)
