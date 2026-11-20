import pytest

from openwam.research.protocol import read_sealed, seal, write_json
from scripts.reproduce_e2_parallel import merge_replay, merge_robustness
from scripts.research_e2_robustness import owns_state, pair_path


@pytest.mark.parametrize("workers", [1, 4, 6])
def test_state_shards_cover_all_pairs_once_and_keep_signed_methods_together(workers):
    counts = [0] * workers
    for subset in range(2):
        for state in range(24):
            owners = [worker for worker in range(workers) if owns_state(subset, state, worker, workers)]
            assert len(owners) == 1
            counts[owners[0]] += 160
    assert sum(counts) == 7680
    assert max(counts) - min(counts) <= 160
    with pytest.raises(ValueError):
        owns_state(0, 0, workers, workers)


def partial_workers(tmp_path):
    spec = {"sha256": "plan", "gpu_uuids": ["GPU-a", "GPU-b"],
            "model_sha256_by_split": {"discovery": "model", "calibration": "model"},
            "selection": {"state_ids_by_split": {s: [f"s{i}" for i in range(24)]
                                                  for s in ("discovery", "calibration")}},
            "methods": [{"method_id": "method"}]}
    for index, gpu in enumerate(spec["gpu_uuids"]):
        root = tmp_path / f"robustness/workers/worker_{index}"
        write_json(root / "status.json", {"status": "budget_exhausted"})
        write_json(root / "worker.json", seal({"gpu_uuid": gpu, "plan_sha256": "plan"}))
        write_json(root / "noise_floor.json", seal({"plan_sha256": "plan", "rows": {
            "discovery": [{"repeat_rmse": index + 1}], "calibration": [],
        }}))
        row = {"state_id": f"s{index}", "method_id": "method", "model_sha256": "model",
               "action_seed": 43, "video_seed": 43, "steps": 10, "valid_run": True,
               "status": "completed", "robustness_sha256": "plan",
               "metrics": {"prefix_normalized_rmse": 0.01}}
        write_json(pair_path(root, "discovery", row["state_id"], "method"), seal(row))
    return spec


def test_merge_partial_workers_preserves_plan_and_noise_checks(tmp_path):
    spec = partial_workers(tmp_path)
    assert merge_robustness(tmp_path, spec, 2) == ("budget_exhausted", 2)
    rows = read_sealed(tmp_path / "robustness/noise_floor.json")["rows"]["discovery"]
    assert max(row["repeat_rmse"] for row in rows) == 2
    assert len(list((tmp_path / "robustness/results/discovery").glob("*.json"))) == 2


def test_merge_rejects_result_from_another_workers_state(tmp_path):
    spec = partial_workers(tmp_path)
    root = tmp_path / "robustness/workers/worker_0"
    row = read_sealed(pair_path(root, "discovery", "s0", "method"))
    row.pop("sha256")
    row["state_id"] = "s1"
    write_json(pair_path(root, "discovery", "s1", "method"), seal(row))
    with pytest.raises(ValueError, match="outside worker assignment"):
        merge_robustness(tmp_path, spec, 2)


def test_merge_baseline_shards_verifies_complete_plan(tmp_path):
    from openwam.research.protocol import digest

    states = [f"state{i}" for i in range(24)]
    methods = [{"method_id": f"method{i}"} for i in range(96)]
    plan = seal({"state_ids": states, "methods": methods})
    manifest = seal({"model": {"sha256": "model"}})
    write_json(tmp_path / "base/manifests/e2_discovery.json", plan)
    for index in range(4):
        root = tmp_path / f"parallel/replay_discovery/worker_{index}"
        write_json(root / "manifest.json", manifest)
        write_json(root / "plan.json", plan)
        write_json(root / "worker.json", seal({"state_ids": states[index::4], "plan_sha256": plan["sha256"]}))
        for state in states[index::4]:
            for method in methods:
                row = {"state_id": state, "method_id": method["method_id"], "model_sha256": "model",
                       "action_seed": 43, "video_seed": 43, "steps": 10, "valid_run": True,
                       "status": "completed", "plan_sha256": plan["sha256"],
                       "metrics": {"prefix_normalized_rmse": 0.01}}
                name = digest([state, 43, method["method_id"]]) + ".json"
                write_json(root / name, row)
    merge_replay(tmp_path, "discovery", 4)
    target = tmp_path / "base/diagnostics/e2_discovery"
    assert len(list(target.glob("*.json"))) == 2306
    assert read_sealed(target / "plan.json") == plan


def test_failed_cross_gpu_certificate_blocks_pair_execution(tmp_path):
    from scripts.research_parallel_worker import validate_cross_gpu_checks

    write_json(tmp_path / "bank/state_bank/discovery/manifest.json", seal({"states": []}))
    bank = read_sealed(tmp_path / "bank/state_bank/discovery/manifest.json")
    output = tmp_path / "cross_check"
    write_json(tmp_path / "parallel/cross_gpu_checks.json", seal({"workers": [{"output": str(output), "gpu": 3}]}))
    certificate = {"status": "failed", "model_sha256": "model", "bank_sha256": bank["sha256"],
                   "states": 20, "seeds": [43, 44, 45]}
    write_json(output / "t0_certificate.json", seal(certificate))
    with pytest.raises(ValueError, match="Cross-GPU T0 failed"):
        validate_cross_gpu_checks(tmp_path, "model")


def test_total_budget_caps_core_worker_deadline(tmp_path, monkeypatch):
    import json
    import time
    from types import SimpleNamespace
    from scripts import research_e2_robustness
    from scripts.research_parallel_worker import robustness_shard

    deadline = time.time() + 60
    write_json(tmp_path / "budget.json", seal({"deadline_unix": deadline}))
    plan = {"base": str(tmp_path / "base"), "bank_run": str(tmp_path / "bank"),
            "checkpoint_dir": str(tmp_path / "model"), "checkpoint_name": "checkpoint.safetensors",
            "model_sha256_by_split": {"discovery": "model"}}
    write_json(tmp_path / "robustness/plan.json", seal(plan))
    args = SimpleNamespace(plan=tmp_path / "robustness/plan.json", output=tmp_path / "worker",
                           deadline=time.time()+3600, shard_index=0, shard_count=1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    def run(worker_args, spec, started):
        assert worker_args.deadline == deadline
        assert 0 < worker_args.hours <= 60/3600
        return "complete"
    monkeypatch.setattr(research_e2_robustness, "run_study", run)
    robustness_shard(args)
    assert read_sealed(tmp_path / "worker/worker.json")["deadline_unix"] == deadline
    assert json.loads((tmp_path / "worker/status.json").read_text())["status"] == "complete"


def test_model_loading_rechecks_free_memory_and_retries_only_allocation_failure(tmp_path, monkeypatch):
    import subprocess
    import time
    import torch
    from benchmarks.libero import research
    from scripts import research_parallel_worker
    from scripts.research_parallel_worker import serialize_model_loading

    monkeypatch.setattr(research_parallel_worker, "ROOT", tmp_path)
    (tmp_path / ".cache").mkdir()
    memory = iter(["17000", "33000", "33000"])
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: next(memory))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.setenv("E2_TOTAL_DEADLINE", str(time.time() + 60))
    calls = []
    expected = object()
    def build(args):
        calls.append(args)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("memory changed after free-memory check")
        return expected
    monkeypatch.setattr(research, "build_server", build)
    serialize_model_loading()
    assert research.build_server("arguments") is expected
    assert calls == ["arguments", "arguments"]
