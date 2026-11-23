import json
import sys

import pytest

from scripts import summarize_chunk_pipeline_smoke as summarize


def write_method(directory, sampler):
    directory.mkdir(parents=True, exist_ok=True)
    client_logs = directory / "logs" / "clients"
    server_logs = directory / "logs" / "servers"
    client_logs.mkdir(parents=True)
    server_logs.mkdir(parents=True)
    tasks = []
    stats = []
    for suite in ("libero_spatial", "libero_goal", "libero_object", "libero_10"):
        result_path = directory / f"{suite}.json"
        result_path.write_text(json.dumps({
            "suite": suite, "task_id": 0,
            "trials": [{"trial": trial, "success": True, "policy_steps": 40} for trial in range(5)],
        }))
        tasks.append({"complete": True, "result_paths": [str(result_path)]})
        lines = []
        for episode in range(5):
            for chunk_index in range(5):
                lines.append("[CLIENT_TIMING] " + json.dumps({
                    "episode": episode, "step": chunk_index * 8, "observation_to_action_ms": 1000.0,
                }))
                if sampler == "chunk-pipeline":
                    stats.append({
                        "cold": chunk_index < 3, "online_steps": 3,
                        "ready_stage_at_takeover": 7, "background_wait_ms": 200.0,
                        "head_worker": chunk_index % 4, "action_ready_ms": 900.0,
                        "worker_peak_memory": [{"allocated": 100, "reserved": 200}] * 4,
                    })
        (client_logs / f"{suite}_attempt01.log").write_text("\n".join(lines))
    (server_logs / "server.log").write_text("\n".join("chunk-pipeline request stats=" + repr(row) for row in stats))
    (directory / "summary.json").write_text(json.dumps({"task_results": tasks, "overall": {"successes": 20, "trials": 20}}))
    (directory / "manifest.json").write_text(json.dumps({"resume_signature": {
        "sampler": sampler, "chunk_pipeline_gpus": 4, "chunk_speculative_steps": 3,
    }}))


def test_paired_report_uses_common_client_latency(tmp_path, monkeypatch):
    for sampler in ("native", "chunk-pipeline"):
        write_method(tmp_path / "smoke" / sampler, sampler)
    monkeypatch.setattr(sys, "argv", ["summarize", str(tmp_path)])
    summarize.main()
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert len(report["paired_trials"]) == 20
    for method in report["methods"].values():
        assert method["client_chunk_requests_including_cold"]["count"] == 100
        assert method["client_chunk_requests_excluding_first_three"] == {"count": 40, "p50_ms": 1000.0, "p95_ms": 1000.0}
    pipeline = report["methods"]["chunk-pipeline"]
    assert pipeline["head_workers"] == [0, 1, 2, 3]
    assert pipeline["steady_online_steps"] == {"3": 40}
    assert pipeline["steady_takeover_stages"] == {"7": 40}
    assert pipeline["server_pipeline_background_wait_ms"] == {"count": 40, "p50_ms": 200.0, "p95_ms": 200.0}


def test_missing_client_timings_fail_validation(tmp_path):
    write_method(tmp_path, "native")
    for path in (tmp_path / "logs" / "clients").glob("*.log"):
        path.write_text("")
    with pytest.raises(ValueError, match="timed chunk requests"):
        summarize.summarize_method(tmp_path)


def test_retry_logs_require_explicit_filtering(tmp_path):
    write_method(tmp_path, "native")
    (tmp_path / "logs" / "clients" / "retry_attempt02.log").write_text("")
    with pytest.raises(ValueError, match="retries"):
        summarize.summarize_method(tmp_path)
