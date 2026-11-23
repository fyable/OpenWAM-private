import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.utils.eval_manifest import seal_manifest
from scripts import run_native_chunk_comparison as comparison


def test_formal_matrix_has_14000_episodes():
    assert len(comparison.matrix("full", "libero")) == 40
    assert len(comparison.matrix("full", "robotwin")) == 50
    assert sum(count * 2 for benchmark in ("libero", "robotwin")
               for suite, task, count in comparison.matrix("full", benchmark)) == 14000
    assert sum(count * 2 for benchmark in ("libero", "robotwin")
               for suite, task, count in comparison.matrix("preflight", benchmark)) == 70


def test_environment_isolates_candidate_flags_and_gpu_mapping(monkeypatch):
    monkeypatch.setenv("OPENWAM_CHUNK_CUDA_GRAPH", "1")
    monkeypatch.setenv("OPENWAM_CHUNK_SHADOW_PARITY", "1")
    monkeypatch.setenv("E2_MEMORY_MODE", "cpu")
    monkeypatch.setenv("ROBOTWIN_ENABLE_PLANNER_FALLBACK", "1")
    native = comparison.environment("native_n10")
    optimized = comparison.environment("chunk_optimized")
    assert native["CUDA_VISIBLE_DEVICES"] == "0"
    assert optimized["CUDA_VISIBLE_DEVICES"] == "0,2,4,6"
    assert "OPENWAM_CHUNK_CUDA_GRAPH" not in native
    assert optimized["OPENWAM_CHUNK_CUDA_GRAPH"] == "1"
    for env in (native, optimized):
        assert env["OPENWAM_REQUIRE_GPU_RESIDENT"] == "1"
        assert "E2_MEMORY_MODE" not in env
        assert "OPENWAM_CHUNK_SHADOW_PARITY" not in env
        assert "ROBOTWIN_ENABLE_PLANNER_FALLBACK" not in env
    assert comparison.environment("chunk_optimized", client=True)["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize("method,sampler", [("native_n10", "native"), ("chunk_optimized", "chunk-pipeline")])
def test_server_commands_are_n10_h32_e8(method, sampler):
    command = comparison.server_command("robotwin", method, "/weights", "step.safetensors", 9075, "token")
    assert command[command.index("--sampler") + 1] == sampler
    assert command[command.index("--denoise-steps") + 1] == "10"
    assert command[command.index("--inference-horizon") + 1] == "8"
    assert "optimization.dit_cache.enabled=false" in command
    if sampler == "chunk-pipeline":
        assert command[command.index("--chunk-devices") + 1] == "cuda:0,cuda:1,cuda:2,cuda:3"
        assert "inference.chunk_worker_backend=process" in command
    else:
        assert "--chunk-devices" not in command


def result_fixture(tmp_path):
    manifest = seal_manifest({"benchmark": "libero", "suite": "libero_goal", "task_id": 0,
                              "entries": [{"trial": 0, "init_state_index": 0}]})
    value = {"manifest_hash": manifest["manifest_hash"], "suite": "libero_goal", "task_id": 0,
             "trial_start": 0, "trial_stop": 1, "successes": 1, "trials": [
                 {"trial": 0, "init_state_index": 0, "init_state_sha256": "same", "success": True,
                  "episode_ms": 100.0, "rollout_ms": 90.0, "policy_ms": 60.0, "env_ms": 20.0, "policy_steps": 8}]}
    path = tmp_path / "result.json"
    comparison.save(path, value)
    return path, manifest, value


@pytest.mark.parametrize("defect", ["hash", "timing", "success", "index", "missing"])
def test_resume_rejects_invalid_results(tmp_path, defect):
    path, manifest, value = result_fixture(tmp_path)
    assert comparison.valid_result(path, manifest, "libero", 1)
    if defect == "hash":
        value["manifest_hash"] = "wrong"
    elif defect == "timing":
        value["trials"][0]["episode_ms"] = -1
    elif defect == "success":
        value["successes"] = 0
    elif defect == "index":
        value["trials"][0]["init_state_index"] = 1
    else:
        value["trials"] = []
    comparison.save(path, value)
    assert comparison.valid_result(path, manifest, "libero", 1) is None


def test_request_and_stage_parsing_keeps_cached_requests_separate(tmp_path):
    path = tmp_path / "client.log"
    path.write_text('noise\n[BENCH_REQUEST] {"step": 0, "kind": "inference", "phase": "first"}\n'
                    '[BENCH_REQUEST] {"step": 1, "kind": "cached", "phase": "warmup"}\n')
    rows = comparison.extract_requests(path)
    assert [row["kind"] for row in rows] == ["inference", "cached"]
    assert comparison.extract_stages("log request stats={'cold': True, 'background_wait_ms': 2.0}") == [
        {"cold": True, "background_wait_ms": 2.0}]


def test_summary_reports_regression_and_not_complete(tmp_path):
    path, manifest, value = result_fixture(tmp_path)
    comparison.save(tmp_path / "protocol.json", {"benchmarks": ["libero"]})
    for method in comparison.METHODS:
        directory = tmp_path / "full/libero/libero_goal-0" / method
        directory.mkdir(parents=True)
        if method == "chunk_optimized":
            value["trials"][0]["success"] = False
            value["successes"] = 0
        comparison.save(directory / "result.json", value)
        (directory / "requests.jsonl").write_text(json.dumps({"kind": "inference", "phase": "steady", "client_ms": 20}) + "\n")
        (directory / "stages.jsonl").write_text("")
    result = comparison.summarize(tmp_path, confidence=True)
    assert result["status"] == "incomplete"
    assert result["benchmarks"]["libero"]["comparison"]["regressions"] == [{"task": "libero_goal-0", "episode": 0}]
    assert (tmp_path / "episodes.csv").exists()
    assert (tmp_path / "REPORT.md").exists()


def test_robotwin_timing_disables_actual_video_path(tmp_path, monkeypatch):
    from benchmarks.robotwin import eval_policy_wrapper as wrapper
    manifest = seal_manifest({"benchmark": "robotwin", "task": "adjust_bottle", "mode": "demo_clean",
                              "entries": [{"episode": 0, "seed": 100000, "instructions": {"unseen": "instruction"}}]})
    comparison.save(tmp_path / "manifest.json", manifest)
    for key, value in {"OPENWAM_BENCH_TIMING": "1", "ROBOTWIN_MANIFEST_HASH": manifest["manifest_hash"],
                       "ROBOTWIN_EPISODE_MANIFEST": str(tmp_path / "manifest.json"), "ROBOTWIN_EPISODE_START": "0",
                       "ROBOTWIN_TOTAL_EPISODES": "1", "ROBOTWIN_EPISODE_COUNT": "1", "ROBOTWIN_INSTRUCTION_TYPE": "unseen",
                       "ROBOTWIN_PROGRESS_FILE": str(tmp_path / "progress.json"),
                       "ROBOTWIN_LABTASKER_RESULT": str(tmp_path / "result.json")}.items():
        monkeypatch.setenv(key, value)
    policy = ModuleType("fake_comparison_policy")
    monkeypatch.setitem(sys.modules, policy.__name__, policy)
    monkeypatch.setattr(wrapper, "_canonicalize_torch_cuda", lambda seed: None)
    monkeypatch.setattr(wrapper, "_initialize_eval_task_state", lambda task, env: None)

    class Environment:
        render_freq = 0

        def setup_demo(self, **kwargs):
            assert kwargs["eval_video_log"] is False
            assert "eval_video_save_dir" not in kwargs
            self.eval_video_path = kwargs.get("eval_video_save_dir")
            self.take_action_cnt = 0
            self.step_lim = 1
            self.eval_success = False

        def set_instruction(self, instruction):
            pass

        def get_obs(self):
            return {}

        def close_env(self, **kwargs):
            pass

    def evaluate(env, model, observation):
        env.take_action_cnt += 1
        env.eval_success = True

    module = SimpleNamespace(eval_function_decorator=lambda policy_name, name:
                             evaluate if name == "eval" else lambda model: None)
    wrapper._install_eval(module)
    module.eval_policy("adjust_bottle", Environment(),
                       {"task_config": "demo_clean", "policy_name": policy.__name__, "clear_cache_freq": 1,
                        "eval_video_log": True, "eval_video_save_dir": "must-not-record"},
                       SimpleNamespace(), 0, test_num=1)
    value = json.loads((tmp_path / "result.json").read_text())
    assert value["episodes"][0]["policy_steps"] == 1
    assert value["episodes"][0]["success"] is True
    assert value["episodes"][0]["episode_ms"] >= value["episodes"][0]["rollout_ms"]
