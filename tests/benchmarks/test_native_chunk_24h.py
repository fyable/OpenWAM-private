import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_native_chunk_24h as campaign
from scripts import run_native_chunk_comparison as base


def test_real_server_parser_accepts_both_commands():
    source = ast.parse((base.ROOT / "openwam/deploy/server.py").read_text())
    definition = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_build_argparser")
    namespace = {"argparse": argparse}
    normalizer = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_normalize_compile_enabled_arg")
    exec(compile(ast.Module(body=[normalizer, definition], type_ignores=[]), "server_parser", "exec"), namespace)
    for method in base.METHODS:
        command = base.server_command("libero", method, "/weights", "test.safetensors", 9075, "token")
        namespace["_build_argparser"]().parse_args(command[2:])


def test_schedule_target_and_interleave():
    assert len(campaign.tasks("libero")) == 40
    assert len(campaign.tasks("robotwin")) == 50
    assert campaign.tasks("libero")[:4] == [(suite, 0) for suite in base.SUITES]
    assert campaign.tasks("robotwin") == campaign.tasks("robotwin")
    assert sum(len(campaign.tasks(benchmark)) * 5 * 2 for benchmark in ("libero", "robotwin")) == 900


def test_deadline_persists_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.time, "time", lambda: 10000)
    args = SimpleNamespace(run=tmp_path, budget_hours=24)
    first = campaign.BudgetComparison(args)
    monkeypatch.setattr(campaign.time, "time", lambda: 20000)
    second = campaign.BudgetComparison(args)
    assert first.deadline == second.deadline == 10000 + 24 * 3600
    monkeypatch.setattr(campaign.time, "time", lambda: second.work_deadline)
    with pytest.raises(campaign.BudgetExpired):
        second.admission()


def test_unpaired_excluded_and_preflight_not_counted(tmp_path):
    row = {"success": True, "episode_ms": 100, "policy_ms": 20, "env_ms": 30, "rollout_ms": 90}
    record = {"benchmark": "libero", "task": "libero_goal-0", "index": 0,
              "method": base.METHODS[0], "row": row, "requests": []}
    base.save(tmp_path / "formal/libero/libero_goal-0/native_n10/episodes/000/episode.json", record)
    base.save(tmp_path / "preflight/libero/libero_goal-0/native_n10/episodes/000/episode.json", record)
    summary = campaign.report(tmp_path)
    assert summary["completed_episodes"] == 1
    assert summary["benchmarks"]["libero"]["complete_pairs"] == 0
    record["method"] = base.METHODS[1]
    base.save(tmp_path / "formal/libero/libero_goal-0/chunk_optimized/episodes/000/episode.json", record)
    assert campaign.report(tmp_path)["benchmarks"]["libero"]["complete_pairs"] == 1


def test_slice_checks_absolute_index_and_manifest(tmp_path):
    manifest = {"manifest_hash": "sealed", "suite": "libero_goal", "task_id": 0,
                "entries": [{"init_state_index": index} for index in range(5)]}
    value = {"manifest_hash": "sealed", "suite": "libero_goal", "task_id": 0, "trial_start": 3,
             "trial_stop": 4, "successes": 1, "trials": [{"trial": 3, "init_state_index": 3,
             "init_state_sha256": "hash", "success": True, "episode_ms": 100, "policy_ms": 20,
             "env_ms": 30, "rollout_ms": 90, "policy_steps": 8}]}
    path = tmp_path / "results.json"
    base.save(path, value)
    assert campaign.sliced_result(path, manifest, "libero", 3)
    assert campaign.sliced_result(path, manifest, "libero", 0) is None
    manifest["manifest_hash"] = "other"
    assert campaign.sliced_result(path, manifest, "libero", 3) is None


def test_gpu_override_maps_native_client_and_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "GPUS", "0,2,4,6")
    base.Comparison(SimpleNamespace(run=tmp_path, gpus="1,2,4,6"))
    assert base.environment("native_n10")["CUDA_VISIBLE_DEVICES"] == "1"
    assert base.environment("chunk_optimized")["CUDA_VISIBLE_DEVICES"] == "1,2,4,6"
    assert base.environment("chunk_optimized", client=True)["CUDA_VISIBLE_DEVICES"] == "1"


@pytest.mark.parametrize("gpus", ["1,2,4", "1,1,4,6", "a,2,4,6"])
def test_gpu_override_rejects_invalid_sets(tmp_path, gpus):
    with pytest.raises(ValueError):
        base.Comparison(SimpleNamespace(run=tmp_path, gpus=gpus))
