import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np


def percentiles(values):
    return {"count": len(values), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95))} if values else None


def summarize_method(directory):
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    trials = {}
    expected_requests = 0
    for task in summary["task_results"]:
        if not task["complete"] or len(task["result_paths"]) != 1:
            raise ValueError(f"Incomplete task: {task}")
        result = json.loads(Path(task["result_paths"][0]).read_text())
        for trial in result["trials"]:
            key = (result["suite"], result["task_id"], trial["trial"])
            if key in trials:
                raise ValueError(f"Duplicate trial: {key}")
            trials[key] = bool(trial["success"])
            expected_requests += (int(trial["policy_steps"]) + 7) // 8
    timings = []
    for path in sorted((directory / "logs" / "clients").glob("*.log")):
        if "attempt01" not in path.name:
            raise ValueError("Client retries require manual latency filtering")
        for line in path.read_text().splitlines():
            if line.startswith("[CLIENT_TIMING] "):
                timings.append(json.loads(line.split(" ", 1)[1]))
    stats = []
    for path in sorted((directory / "logs" / "servers").glob("*.log")):
        for line in path.read_text().splitlines():
            marker = "chunk-pipeline request stats="
            if marker in line:
                stats.append(ast.literal_eval(line.split(marker, 1)[1]))
    requests = [row for row in timings if row["step"] % 8 == 0]
    if not expected_requests or len(requests) != expected_requests:
        raise ValueError(f"Expected {expected_requests} timed chunk requests, got {len(requests)}")
    if manifest["resume_signature"].get("sampler") == "chunk-pipeline":
        if len(stats) != expected_requests or {row["head_worker"] for row in stats} != {0, 1, 2, 3}:
            raise ValueError("Pipeline server stats are missing or four-worker rotation was not observed")
    steady = [row for row in requests if row["step"] >= 24]
    stable_stats = [row for row in stats if not row["cold"]]
    peak = []
    for row in stats:
        for worker, memory in enumerate(row["worker_peak_memory"]):
            if memory is not None:
                while len(peak) <= worker:
                    peak.append({"allocated": 0, "reserved": 0})
                for name in peak[worker]:
                    peak[worker][name] = max(peak[worker][name], memory[name])
    return trials, {
        "overall": summary["overall"],
        "resume_signature": manifest["resume_signature"],
        "client_chunk_requests_including_cold": percentiles([row["observation_to_action_ms"] for row in requests]),
        "client_chunk_requests_excluding_first_three": percentiles([row["observation_to_action_ms"] for row in steady]),
        "steady_online_steps": dict(Counter(row["online_steps"] for row in stable_stats)),
        "steady_takeover_stages": dict(Counter(row["ready_stage_at_takeover"] for row in stable_stats
                                             if "ready_stage_at_takeover" in row)),
        "server_pipeline_background_wait_ms": percentiles([row["background_wait_ms"] for row in stable_stats
                                                           if "background_wait_ms" in row]),
        "head_workers": sorted({row["head_worker"] for row in stats}),
        "worker_peak_memory_bytes": peak,
        "server_pipeline_action_ready_ms": percentiles([row["action_ready_ms"] for row in stable_stats]),
        "server_pipeline_online_finish_ms": percentiles([row['online_finish_ms'] for row in stable_stats
                                                         if 'online_finish_ms' in row]),
        "server_pipeline_online_step_host_ms": percentiles([value for row in stable_stats
                                                            for value in row.get('online_step_host_ms', [])]),
        "server_pipeline_prior_decode_ms": percentiles([row['prior_decode_ms'] for row in stable_stats
                                                        if row.get('prior_decode_ms') is not None]),
        "server_pipeline_prior_worker_background_ms": percentiles([
            row['prior_worker_background_ms'] for row in stable_stats
            if row.get('prior_worker_background_ms') is not None]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    methods = {}
    paired = {}
    for name in ("native", "chunk-pipeline"):
        paired[name], methods[name] = summarize_method(args.run / "smoke" / name)
    if len(paired["native"]) != 20 or paired["native"].keys() != paired["chunk-pipeline"].keys():
        raise ValueError("Expected the same 20 trial keys in both methods")
    signature = methods["chunk-pipeline"]["resume_signature"]
    if signature["chunk_pipeline_gpus"] != 4 or signature["chunk_speculative_steps"] != 3:
        raise ValueError("Unexpected pipeline configuration")
    report = {
        "memory_policy": "gpu-resident-no-offload",
        "latency_scope": "Client observation preprocessing through returned action conversion, every eighth action; steady excludes first three chunks per episode",
        "methods": methods,
        "paired_trials": [
            {"suite": key[0], "task_id": key[1], "trial": key[2],
             "native": paired["native"][key], "chunk_pipeline": paired["chunk-pipeline"][key]}
            for key in sorted(paired["native"])
        ],
    }
    output = json.dumps(report, indent=2)
    (args.run / "comparison.json").write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
