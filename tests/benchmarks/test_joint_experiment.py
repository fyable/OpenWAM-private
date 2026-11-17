"""CPU validation of trace collection, experiment scheduling and failure reporting."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.libero.joint_experiment import METHODS, main, method_order, percentiles, selected_indices
from openwam.deploy.request_trace import RequestTrace, assemble_sessions, assemble_trace, attach_request_trace


def conditions(prompt="task"):
    return {"first_frame_image": [np.zeros((8, 8, 3), dtype=np.uint8)],
            "proprio": np.arange(10, dtype=np.float32), "prompt": prompt}


def test_capture_atomic_completion_and_assembly(tmp_path):
    trace = RequestTrace(tmp_path / "session", warmup=0, measure=3)
    for _ in range(3):
        trace.record(conditions())
    trace.reset()
    for _ in range(4):
        trace.record(conditions())
    assert trace.complete
    output = tmp_path / "observations.npz"
    metadata = assemble_trace(trace.directory, output)
    assert metadata["requests"] == 7
    from benchmarks.libero.replay_joint import load_observations

    observations = load_observations(output)
    assert selected_indices(observations, 0, 3) == [2, 5, 6]
    np.testing.assert_array_equal(observations[3], [True, False, False, True, False, False, False])
    with pytest.raises(FileExistsError):
        assemble_trace(trace.directory, output)
    assert not list(trace.directory.glob("*.tmp"))


def test_trace_wraps_successful_chunk_requests_only(tmp_path, monkeypatch):
    from openwam.deploy.executors.sync_executor import SyncInferenceExecutor

    class Engine:
        cfg = SimpleNamespace(inference=SimpleNamespace(inference_mode="sync", inference_horizon=8))

        def generate(self, data):
            return {"actions": np.zeros((32, 10))}

        def reset(self):
            pass

    monkeypatch.setenv("OPENWAM_REQUEST_TRACE_DIR", str(tmp_path / "traces"))
    engine = Engine()
    trace = attach_request_trace(engine)
    executor = SyncInferenceExecutor(engine, inference_horizon=8)
    for _ in range(17):
        executor.predict_action(conditions())
    assert trace.count == 3
    engine.reset()
    assert trace.pending_reset


def test_assembly_across_short_server_sessions(tmp_path):
    for index in range(2):
        trace = RequestTrace(tmp_path / f"session_{index}", warmup=48, measure=120)
        for _ in range(4):
            trace.record(conditions())
    output = tmp_path / "observations.npz"
    assert assemble_sessions(tmp_path, output, warmup=0, measure=5) is None
    assert not output.exists()
    metadata = assemble_sessions(tmp_path, output, warmup=0, measure=3)
    assert metadata["requests"] == 7
    from benchmarks.libero.replay_joint import load_observations

    observations = load_observations(output)
    assert selected_indices(observations, 0, 3) == [2, 3, 6]


def test_failed_generation_is_not_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENWAM_REQUEST_TRACE_DIR", str(tmp_path / "traces"))

    def fail(data):
        raise RuntimeError("injected")

    engine = SimpleNamespace(cfg=SimpleNamespace(inference=SimpleNamespace(inference_mode="sync", inference_horizon=8)),
                             generate=fail, reset=lambda: None)
    trace = attach_request_trace(engine)
    with pytest.raises(RuntimeError, match="injected"):
        engine.generate(conditions())
    assert trace.count == 0


def test_trace_rejects_invalid_input_and_async(tmp_path, monkeypatch):
    trace = RequestTrace(tmp_path / "trace")
    data = conditions()
    data["proprio"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        trace.record(data)
    assert trace.count == 0
    monkeypatch.setenv("OPENWAM_REQUEST_TRACE_DIR", str(tmp_path / "async"))
    engine = SimpleNamespace(cfg=SimpleNamespace(inference=SimpleNamespace(inference_mode="async")))
    with pytest.raises(ValueError, match="synchronous"):
        attach_request_trace(engine)


def test_selected_indices_prompt_resets_and_insufficient():
    observations = (None, np.array(["a"] * 4 + ["b"] * 4), None,
                    np.array([True] + [False] * 7))
    assert selected_indices(observations, 0, 4) == [2, 3, 6, 7]
    with pytest.raises(ValueError, match="only 4"):
        selected_indices(observations, 0, 5)
    assert selected_indices(observations, 3, 3) == [3, 6, 7]


def test_interleaving_deterministic_and_balanced():
    orders = [method_order(index) for index in range(20)]
    assert orders == [method_order(index) for index in range(20)]
    assert all(set(order) == set(METHODS) for order in orders)
    assert len({order[0] for order in orders}) > 1
    assert percentiles([1, 3])["p50"] == 2


def test_plan_does_not_create_outputs(tmp_path, capsys):
    output = tmp_path / "plan"
    main(["plan", "--output", str(output)])
    assert "interleaved" in capsys.readouterr().out
    assert not output.exists()


def test_failure_manifest_and_refuse_overwrite(tmp_path):
    output = tmp_path / "failure"
    with pytest.raises(ValueError, match="TRACE"):
        main(["timing", "--output", str(output)])
    manifest = json.loads((output / "experiment.json").read_text())
    assert manifest["status"] == "failed"
    with pytest.raises(FileExistsError):
        main(["timing", "--output", str(output)])


def test_scopes_restore_methods_on_failure():
    from benchmarks.libero.joint_experiment import scoped_annotations

    def prepare():
        return 5

    video = SimpleNamespace(prepare=prepare)
    architecture = SimpleNamespace(video_backbone=video, action_backbone=SimpleNamespace())
    with pytest.raises(RuntimeError):
        with scoped_annotations(architecture):
            assert video.prepare() == 5
            raise RuntimeError("injected")
    assert video.prepare is prepare


@pytest.mark.parametrize("preset,mismatch_width,corrupt_shared,history_scope", [
    ("default", 0, False, "phase_video_action"), ("default", 4, False, "phase_video_action"),
    ("native-parallel", 0, False, "phase_video_action"),
    ("native-parallel", 4, False, "phase_video_action"), ("native-parallel", 10, False, "phase_video_action"),
    ("native-parallel", 0, True, "phase_video_action"),
    ("native-parallel", 0, False, "action_only"), ("native-parallel", 10, False, "action_only"),
    ("native-parallel", 0, True, "action_only"),
])
def test_paired_replay_complete_with_cpu_fake_engine(tmp_path, monkeypatch, preset, mismatch_width, corrupt_shared, history_scope):
    import torch
    from benchmarks.libero import joint_experiment as experiment
    from openwam.deploy.parallel_joint import JointParaFlowSampler

    if corrupt_shared:
        original_sample = JointParaFlowSampler.sample

        def corrupt_sample(self, architecture, inputs, noise, schedule, *args, **kwargs):
            result = original_sample(self, architecture, inputs, noise, schedule, *args, **kwargs)
            if self.share_conditions and len(schedule) == 11:
                self.history = self.history + 1
            return result

        monkeypatch.setattr(JointParaFlowSampler, "sample", corrupt_sample)

    class Architecture:
        dtype = torch.float32
        video_scheduler = SimpleNamespace(num_train_timesteps=1000)
        action_scheduler = SimpleNamespace(num_train_timesteps=1000)

        def named_modules(self):
            return []

        def forward(self, actions, action_time, *, latents, timestep, context, proprio, first_frame_latents):
            offset = 0.5 if len(actions) == mismatch_width else 0
            return latents * 0.01 + timestep.reshape(-1, 1, 1, 1, 1) / 1000 + offset, actions * 0.02

    class Engine:
        def __init__(self):
            self.architecture = Architecture()
            self._joint_sampler = JointParaFlowSampler(history_scope=history_scope)
            self.cfg = SimpleNamespace(inference=SimpleNamespace(denoise_steps=4))

        def reset(self):
            if self._joint_sampler:
                self._joint_sampler.reset()

        def generate(self, data):
            inputs = {"latents": torch.ones(1, 2, 3, 2, 2),
                      "first_frame_latents": torch.ones(1, 2, 1, 2, 2),
                      "context": torch.ones(1, 3, 4), "proprio": torch.ones(1, 80)}
            noise = torch.ones(1, 32, 80)
            schedule = [(float(value), float(value)) for value in
                        torch.linspace(1000, 0, self.cfg.inference.denoise_steps + 1)]
            if self._joint_sampler:
                _, actions = self._joint_sampler.sample(self.architecture, inputs, noise, schedule,
                                                       None, prompt=data["prompt"], seed=42)
                return {"actions": actions[0], "sampler_stats": self._joint_sampler.last_stats}
            for video_time, action_time in schedule[:-1]:
                self.architecture.forward(noise, torch.tensor([action_time]),
                                          **inputs, timestep=torch.tensor([video_time]))
            return {"actions": noise[0]}

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            pass

        def elapsed_time(self, end):
            return 1.0

    engine = Engine()
    monkeypatch.setattr(experiment, "build_engine", lambda args: (engine, "checkpoint.safetensors"))
    monkeypatch.setattr(experiment, "hardware_snapshot", lambda: {"status": "fake"})
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device=None: torch.random.get_rng_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device=None: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 2048)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (4096, 8192))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(total_memory=8192))
    path = tmp_path / "observations.npz"
    np.savez(path, images=np.zeros((5, 8, 8, 3), dtype=np.uint8),
             states=np.zeros((5, 10)), prompts=np.array(["task"] * 5),
             episode_starts=np.array([True, False, False, False, False]))
    args = experiment.parser().parse_args(["replay-worker", "--trace", str(path), "--output", str(tmp_path),
                                          "--device", "cpu", "--warmup", "0", "--measure", "3", "--preset", preset,
                                          "--history-scope", history_scope])
    if mismatch_width:
        with pytest.raises(RuntimeError, match="parity failed"):
            experiment.replay(args)
        failure = json.loads((tmp_path / "parity.json").read_text())
        assert failure["status"] == "failed"
        failed_stage = failure["by_stages"][str(mismatch_width)]
        assert failed_stage["partitions"][0]["modalities"]["video"]["passed"] is False
        assert failed_stage["partitions"][0]["modalities"]["action"]["passed"] is True
        assert not (tmp_path / "requests.jsonl").exists()
        args.allow_parity_failure = True
    if corrupt_shared:
        args.allow_parity_failure = True
        with pytest.raises(RuntimeError, match="N10 condition sharing trajectory parity failed"):
            experiment.replay(args)
        failure = json.loads((tmp_path / "condition_parity_failure.json").read_text())
        assert failure["stages"] == 10
        assert not (tmp_path / "timing.json").exists()
        return
    report = experiment.replay(args)
    assert report["measured_indices"] == [2, 3, 4]
    assert report["history_scope"] == history_scope
    assert report["cache_trajectory_parity"]["status"] == "passed"
    assert report["numerical_equivalence_validated"] is (preset == "default" and not mismatch_width)
    assert set(report["stage_batch_parity"]["microbenchmarks"]) == {"batch1", "batch2", "batch4"}
    assert all(result["wall_ms"]["count"] == 3 for result in report["methods"].values())
    rows = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    methods = experiment.COMPARISON_METHODS if preset == "native-parallel" else experiment.METHODS
    assert len(rows) == 5 * len(methods)
    assert report["condition_trajectory_parity"]["status"] == "passed"
    assert len(next(row for row in rows if row["method"] == "native_n10")["forward_shapes"]) == 10
    experiment.report_markdown(tmp_path)
    markdown = (tmp_path / "comparison.md").read_text()
    if preset == "native-parallel":
        assert set(report["methods"]) == set(experiment.COMPARISON_REPORTED)
        assert report["stage_batch_parity"]["by_stages"]["10"]["status"] == ("failed" if mismatch_width == 10 else "passed")
        assert len(report["condition_trajectory_parity"]["requests"]) == 10
        assert all(row["reference_only"] == (row["method"] not in experiment.COMPARISON_REPORTED) for row in rows)
        assert "joint_shared_n10" in markdown and "P50 speedup" in markdown
        assert report["paired"]["native_n10"]["native_n10_speedup"]["p50"] == 1.0
        assert {row["forward_shapes"][-1]["action"][0] for row in rows
                if row["method"] == "joint_shared_n10" and row["measured"]} == {10}
    else:
        assert "joint_cached_n4" in markdown


def test_tensor_error_reports_mismatches_and_nonfinite():
    import torch
    from benchmarks.libero.joint_experiment import tensor_error

    result = tensor_error(torch.tensor([0.0, 0.25]), torch.zeros(2))
    assert not result["passed"]
    assert result["mismatches"] == 1
    assert result["max_absolute_error"] == 0.25
    result = tensor_error(torch.tensor([float("nan")]), torch.zeros(1))
    assert not result["finite"]
    json.dumps(result, allow_nan=False)


def test_comparison_launcher_forwards_settings(tmp_path):
    import os
    import subprocess
    from benchmarks.libero.joint_experiment import ROOT

    env = dict(os.environ, GPU="7", OUT=str(tmp_path / "out"), TRACE=str(tmp_path / "trace.npz"),
               SERVER_PYTHON="/bin/echo", WARMUP="3", MEASURE="5")
    result = subprocess.run(["bash", str(ROOT / "scripts/compare_joint_native_parallel.sh")],
                            env=env, capture_output=True, text=True, check=True)
    assert "--preset native-parallel --require-exclusive" in result.stdout
    assert "--skip-profile" in result.stdout
    assert "--warmup 3 --measure 5" in result.stdout
    assert str(tmp_path / "out") in result.stdout
    assert not (tmp_path / "out").exists()


def test_comparison_worker_forwards_preset(tmp_path, monkeypatch):
    from benchmarks.libero import joint_experiment as experiment
    from benchmarks.utils import owned_processes

    captured = []
    monkeypatch.setattr(owned_processes, "spawn_resource",
                        lambda command, **kwargs: captured.append(command) or SimpleNamespace(wait=lambda: 0))
    monkeypatch.setattr(owned_processes, "ProcessRegistry", lambda: SimpleNamespace(
        add=lambda process: None, terminate_all=lambda: None))
    args = experiment.parser().parse_args([
        "timing", "--output", str(tmp_path), "--trace", str(tmp_path / "trace.npz"),
        "--preset", "native-parallel", "--require-exclusive", "--allow-parity-failure",
        "--action-tail-init", "endpoint_bridge", "--video-tail-init", "endpoint_bridge"])
    experiment.run_worker(args, "replay-worker")
    command = captured[0]
    assert command[command.index("--preset") + 1] == "native-parallel"
    assert "--require-exclusive" in command and "--allow-parity-failure" in command
    assert command[command.index("--action-tail-init") + 1] == "endpoint_bridge"
    assert command[command.index("--video-tail-init") + 1] == "endpoint_bridge"


def test_exclusive_gpu_rejects_foreign_context(monkeypatch):
    import os
    from benchmarks.libero import joint_experiment as experiment

    monkeypatch.setattr(experiment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=f"GPU-target, {os.getpid()}\nGPU-other, 999999\n"))
    experiment.require_exclusive_uuid("GPU-target")
    monkeypatch.setattr(experiment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout="GPU-target, 999999\n"))
    with pytest.raises(RuntimeError, match="Exclusive GPU timing rejected"):
        experiment.require_exclusive_uuid("GPU-target")


def test_capture_child_assembly_and_cleanup(tmp_path):
    import sys
    from benchmarks.libero import joint_experiment as experiment

    checkpoint = tmp_path / "weights"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_step_1.safetensors").touch()
    source = (
        "import os; from pathlib import Path; import numpy as np; "
        "from openwam.deploy.request_trace import RequestTrace; "
        "trace=RequestTrace(Path(os.environ['OPENWAM_REQUEST_TRACE_DIR'])/'session_test',warmup=0,measure=1); "
        "data={'first_frame_image':[np.zeros((4,4,3),dtype=np.uint8)],'prompt':'task','proprio':np.zeros(10)}; "
        "[trace.record(data) for _ in range(3)]"
    )
    output = tmp_path / "capture"
    output.mkdir()
    args = experiment.parser().parse_args(["capture", "--output", str(output), "--ckpt-dir", str(checkpoint),
                                          "--capture-command-json", json.dumps([sys.executable, "-c", source]),
                                          "--warmup", "0", "--measure", "1", "--capture-timeout", "30"])
    args.trace = output / "observations.npz"
    experiment.capture(args)
    assert args.trace.is_file()
    assert json.loads((output / "capture.json").read_text())["requests"] == 3
