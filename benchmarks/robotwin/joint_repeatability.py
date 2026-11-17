"""Manifest-exact, repeated Joint diagnostics with resumable GPU shards."""

import argparse
import copy
import fcntl
import itertools
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.robotwin import compare_joint as base
from benchmarks.utils.eval_manifest import content_hash, seal_manifest
from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource


METHODS = {"legacy": "joint_parallel_n10", "prepared": "joint_prepared_n10", "prefix": "joint_prefix_n10"}
SOURCE = base.ROOT / "outputs/robotwin/joint_three_backend_50tasks_e5_4gpu_v1"
OUTPUT = base.ROOT / "outputs/robotwin/joint_repeatability_v1"


def immutable_json(path, value):
    if path.exists():
        if base.read_json(path) != value:
            raise ValueError(f"Experiment identity changed: {path}; use a new output directory")
    else:
        base.write_json(path, value)


def select_cases(source, controls):
    candidates = []
    source_files = {}
    checkpoint = None
    seen_tasks = set()
    shards = sorted(source.glob("shard_*/protocol.json"))
    if not shards:
        raise ValueError("No source shard protocols found")
    for protocol_path in shards:
        shard = protocol_path.parent
        protocol = base.read_json(protocol_path)
        source_files[str(protocol_path.resolve())] = base.digest(protocol_path)
        identity = {key: protocol[key] for key in ("checkpoint", "checkpoint_size", "checkpoint_mtime_ns",
                                                  "checkpoint_config", "normalization_stats")}
        if checkpoint is not None and checkpoint != identity:
            raise ValueError("Source shards use different checkpoint identities")
        checkpoint = identity
        for method in METHODS.values():
            selected = protocol.get("method_protocols", {}).get(method, protocol)
            if any(selected.get(key) != value for key, value in identity.items()):
                raise ValueError("Source methods must share the Joint checkpoint")
        for task in protocol["tasks"]:
            if task not in base.all_tasks() or task in seen_tasks:
                raise ValueError(f"Invalid or duplicate source task: {task}")
            seen_tasks.add(task)
            manifest_path = shard / "manifests" / f"{task}.json"
            manifest = base.check_manifest(manifest_path, task, protocol["episodes_per_task"])
            source_files[str(manifest_path.resolve())] = base.digest(manifest_path)
            results = {}
            for backend, method in METHODS.items():
                path = shard / method / "results" / f"{task}.json"
                result = base.complete_result(path, manifest, len(manifest["entries"]))
                if result is None:
                    raise ValueError(f"Source result incomplete or invalid: {path}")
                results[backend] = result
                source_files[str(path.resolve())] = base.digest(path)
            for episode, entry in enumerate(manifest["entries"]):
                outcomes = {name: value["episodes"][episode]["success"] for name, value in results.items()}
                kind = "difference" if len(set(outcomes.values())) > 1 else "control" if all(outcomes.values()) else None
                if kind is None:
                    continue
                rebased = copy.deepcopy(entry)
                rebased["episode"] = 0
                single = seal_manifest({key: value for key, value in manifest.items()
                                        if key not in ("entries", "manifest_hash")} | {"entries": [rebased]})
                case_id = f"{task}_e{episode}_{content_hash(entry).split(':')[1][:12]}"
                candidates.append({"id": case_id, "task": task, "kind": kind,
                                   "source_manifest": str(manifest_path.resolve()),
                                   "source_manifest_hash": manifest["manifest_hash"],
                                   "source_episode": episode, "seed": entry["seed"],
                                   "instruction": entry["instructions"]["unseen"],
                                   "original_entry": entry, "historical_outcomes": outcomes, "manifest": single})
    candidates.sort(key=lambda row: (row["task"], row["source_episode"]))
    differences = [row for row in candidates if row["kind"] == "difference"]
    if not differences:
        raise ValueError("No paired differences found")
    selected_controls = []
    control_tasks = set()
    for row in candidates:
        if row["kind"] == "control" and row["task"] not in control_tasks and len(selected_controls) < controls:
            selected_controls.append(row)
            control_tasks.add(row["task"])
    if len(selected_controls) != controls:
        raise ValueError("Not enough distinct all-success control tasks")
    return differences + selected_controls, checkpoint, source_files


def make_suite(args):
    gpus = args.gpus.split(",")
    if (not gpus or any(not value.isdigit() for value in gpus) or len(set(gpus)) != len(gpus)
            or args.repeats < 2 or args.controls < 0 or not 0 < args.port <= 65536 - len(gpus)):
        raise ValueError("Require unique numeric GPU indices, repeats >=2, controls >=0 and valid ports")
    cases, checkpoint, source_files = select_cases(args.source.resolve(), args.controls)
    for index, case in enumerate(cases):
        case["shard"] = index % len(gpus)
    body = {"version": 1, "source": str(args.source.resolve()), "source_files": source_files,
            "checkpoint": checkpoint, "cases": cases, "gpus": gpus, "port": args.port,
            "repeats": args.repeats, "order_seed": 42, "allow_parity_failure": True,
            "capture": "all request inputs and actions in repeat 0; bounded diagnostic, not timed replay",
            "trace_limit": args.trace_limit, "methods": METHODS, "model_seed": 42,
            "decode_video": False, "stages": 10, "horizon": 32, "advance": 8}
    if args.trace_limit < 3:
        raise ValueError("Trace limit must be >=3")
    return body | {"identity": content_hash(body)}


def method_order(suite, shard, repeat):
    order = list(METHODS)
    random.Random(suite["order_seed"] + 1000 * shard + repeat).shuffle(order)
    return order


def run_path(root, case, backend, repeat):
    return root / f"shard_{case['shard']}" / "runs" / case["id"] / f"repeat_{repeat}" / backend


def manifest_path(root, case):
    return root / "cases" / case["id"] / "manifest.json"


def valid_run(root, suite, case, backend, repeat):
    directory = run_path(root, case, backend, repeat)
    result = base.complete_result(directory / "result.json", case["manifest"], 1)
    marker = directory / "complete.json"
    if result is None or not marker.exists():
        return None
    try:
        metadata = base.read_json(marker)
    except (OSError, ValueError):
        return None
    if (not isinstance(metadata, dict) or metadata.get("suite_identity") != suite["identity"] or metadata.get("case") != case["id"]
            or metadata.get("backend") != backend or metadata.get("repeat") != repeat
            or metadata.get("result_sha256") != base.digest(directory / "result.json")):
        return None
    return result


def gpu_snapshot():
    result = {}
    for label, query in (("gpus", "--query-gpu=index,uuid,name,utilization.gpu,memory.used"),
                         ("contexts", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory")):
        try:
            process = subprocess.run(["nvidia-smi", query, "--format=csv"], capture_output=True, text=True, timeout=10)
            result[label] = {"code": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[label] = {"error": str(error)}
    return result


def configured(suite, root, shard):
    cfg = base.configuration("diagnostic")
    cfg.root = root / f"shard_{shard}"
    cfg.checkpoint = Path(suite["checkpoint"]["checkpoint"]).parent
    cfg.episodes = 1
    cfg.tasks = [row["task"] for row in suite["cases"] if row["shard"] == shard]
    cfg.gpu = cfg.sim_gpu = suite["gpus"][shard]
    cfg.port = suite["port"] + shard
    cfg.expected_mask = "mutual"
    cfg.methods = {name: ("joint-paraflow", 10) for name in METHODS.values()}
    base.preflight(cfg)
    current = {"checkpoint": str(cfg.weight), "checkpoint_size": cfg.weight.stat().st_size,
               "checkpoint_mtime_ns": cfg.weight.stat().st_mtime_ns,
               "checkpoint_config": base.digest(cfg.checkpoint / "config.yaml"),
               "normalization_stats": base.digest(cfg.checkpoint / "normalization_stats.npy")}
    if current != suite["checkpoint"]:
        raise ValueError("Checkpoint changed since source experiment")
    cfg.method_configs = {}
    for backend, method in METHODS.items():
        selected = copy.copy(cfg)
        selected.deploy_overrides = [f"inference.joint_execution_backend={backend}",
                                     "inference.request_timing=true",
                                     "inference.joint_execution_allow_parity_failure=true"]
        selected.server_environment = {}
        cfg.method_configs[method] = selected
    return cfg


def prepare(args, suite):
    root = args.output.resolve()
    immutable_json(root / "suite.json", suite)
    for case in suite["cases"]:
        immutable_json(manifest_path(root, case), case["manifest"])
        immutable_json(manifest_path(root, case).with_name("source.json"),
                       {key: value for key, value in case.items() if key != "manifest"})
    for shard in range(len(suite["gpus"])):
        cfg = configured(suite, root, shard)
        cfg.root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base.HERE / "policy_config.yml", cfg.root / "policy_config.yml")
        shutil.copyfile(base.HERE / "step_limits.yml", cfg.root / "step_limits.yml")
        identity = base.protocol(cfg)
        identity["backends"] = {name: selected.deploy_overrides for name, selected in cfg.method_configs.items()}
        identity["launcher_sha256"] = base.digest(base.ROOT / "scripts/validate_joint_repeatability.sh")
        immutable_json(cfg.root / "execution_protocol.json", identity)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", cfg.port))
    return root


def request_paths(directory):
    return sorted(directory.glob("session_*/request_*.npz"))


def episode_diagnostics(path):
    steps = None
    if path.exists():
        with path.open(errors="replace") as stream:
            for line in stream:
                plain = re.sub(r"\x1b\[[0-9;]*m", "", line)
                matched = re.search(r"step:\s*(\d+)\s*/\s*(\d+)", plain)
                if matched:
                    steps = {"step": int(matched[1]), "limit": int(matched[2])}
    return {"last_simulator_step": steps, "termination_reason": "not_exposed_by_evaluator"}


def shard_run(root, suite, shard):
    cfg = configured(suite, root, shard)
    cases = [case for case in suite["cases"] if case["shard"] == shard]
    with base.output_lock(cfg.root):
        base.write_json(cfg.root / "hardware_before.json", gpu_snapshot())
        try:
            for repeat in range(suite["repeats"]):
                for backend in method_order(suite, shard, repeat):
                    pending = [case for case in cases if valid_run(root, suite, case, backend, repeat) is None]
                    if not pending:
                        continue
                    method = METHODS[backend]
                    capture_root = cfg.root / "captures" / f"repeat_{repeat}" / backend / str(time.time_ns())
                    environment = {"OPENWAM_TRACE_ALL": "0", "OPENWAM_TRACE_OUTPUTS": "0"}
                    if repeat == 0:
                        environment.update(OPENWAM_REQUEST_TRACE_DIR=str(capture_root), OPENWAM_TRACE_ALL="1",
                                           OPENWAM_TRACE_OUTPUTS="1", OPENWAM_TRACE_MAX_REQUESTS=str(suite["trace_limit"]))
                    else:
                        environment["OPENWAM_REQUEST_TRACE_DIR"] = ""
                    cfg.method_configs[method].server_environment = environment
                    with base.policy_server(cfg, method) as server:
                        for case in pending:
                            directory = run_path(root, case, backend, repeat)
                            attempt = directory / "attempts" / str(time.time_ns())
                            before = set(request_paths(capture_root)) if repeat == 0 else set()
                            print(f"shard={shard} repeat={repeat} backend={backend} case={case['id']}", flush=True)
                            base.run_client(cfg, case["task"], attempt, manifest_path(root, case), server)
                            result = base.complete_result(attempt / "result.json", case["manifest"], 1)
                            if result is None:
                                raise RuntimeError(f"Incomplete result excluded: {attempt}")
                            base.write_json(directory / "result.json", result)
                            records = sorted(set(request_paths(capture_root)) - before) if repeat == 0 else []
                            metadata = {"suite_identity": suite["identity"], "case": case["id"], "repeat": repeat,
                                        "backend": backend, "attempt": str(attempt),
                                        "result_sha256": base.digest(directory / "result.json"),
                                        "request_records": [str(path) for path in records],
                                        "capture_capped": bool(list(capture_root.glob("session_*/capped.json"))),
                                        "diagnostic_only": True, **episode_diagnostics(attempt / "client.log")}
                            base.write_json(directory / "complete.json", metadata)
                            summarize(root)
        finally:
            base.write_json(cfg.root / "hardware_after.json", gpu_snapshot())
            base.write_json(cfg.root / "closed_loop_timing.json", {
                "diagnostic_only": True, "includes_trace_and_validation_overhead": True,
                "methods": {backend: base.joint_timings(cfg.root / method) for backend, method in METHODS.items()}})
            summarize(root)


def summarize(root):
    suite = base.read_json(root / "suite.json")
    rows = []
    totals = {name: {"completed": 0, "successes": 0} for name in METHODS}
    paired = {name: {"pairs": 0, "regressions": 0, "improvements": 0} for name in ("prepared", "prefix")}
    for case in suite["cases"]:
        outcomes = {}
        for backend in METHODS:
            outcomes[backend] = []
            for repeat in range(suite["repeats"]):
                result = valid_run(root, suite, case, backend, repeat)
                value = result["episodes"][0]["success"] if result else None
                outcomes[backend].append(value)
                if value is not None:
                    totals[backend]["completed"] += 1
                    totals[backend]["successes"] += value
        legacy = [value for value in outcomes["legacy"] if value is not None]
        self_pairs = list(itertools.combinations(legacy, 2))
        baseline = {"pairs": len(self_pairs), "disagreements": sum(left != right for left, right in self_pairs)}
        classification = {}
        for backend in paired:
            pairs = [(left, right) for left, right in zip(outcomes["legacy"], outcomes[backend])
                     if left is not None and right is not None]
            losses = sum(left and not right for left, right in pairs)
            gains = sum(not left and right for left, right in pairs)
            paired[backend]["pairs"] += len(pairs)
            paired[backend]["regressions"] += losses
            paired[backend]["improvements"] += gains
            if len(pairs) != suite["repeats"]:
                label = "incomplete"
            elif len(set(legacy)) > 1:
                label = "baseline_variable"
            elif all(left and not right for left, right in pairs):
                label = "repeatable_candidate_regression"
            elif losses:
                label = "candidate_difference_unexplained"
            else:
                label = "no_paired_regression_observed"
            classification[backend] = {"label": label, "regressions": losses, "improvements": gains}
        rows.append({"case": case["id"], "task": case["task"], "kind": case["kind"], "seed": case["seed"],
                     "source_episode": case["source_episode"], "instruction": case["instruction"],
                     "outcomes": outcomes, "legacy_self_comparison": baseline, "classification": classification})
    expected = len(rows) * suite["repeats"]
    report = {"suite_identity": suite["identity"], "expected_per_backend": expected, "methods": totals,
              "paired": paired, "cases": rows, "diagnostic_only": True,
              "interpretation": "Selected-case diagnostics, not benchmark success rates or proof of equivalence",
              "status": "complete" if all(value["completed"] == expected for value in totals.values()) else "incomplete"}
    lines = ["# Joint repeatability diagnostics", "", report["interpretation"], "",
             "Known parity failures are allowed and thresholds unchanged. Default backend unchanged.", "",
             "| Backend | Complete / Expected | Successes | Paired regressions | Paired improvements |",
             "|---|---:|---:|---:|---:|"]
    for name, value in totals.items():
        lines.append(f"| {name} | {value['completed']}/{expected} | {value['successes']} | "
                     f"{paired.get(name, {}).get('regressions', '—')} | {paired.get(name, {}).get('improvements', '—')} |")
    lines += ["", "## Per-case outcomes", "", "T=success, F=failure, -=pending/invalid; repeat order is fixed.", "",
              "Legacy self disagreement compares all within-case repeat pairs, not cross-backend pairs.", "",
              "| Case | Kind | Legacy | Prepared | Prefix | Legacy self disagreement | Prepared / Prefix diagnosis |",
              "|---|---|---|---|---|---|---|"]
    for row in rows:
        outcomes = ["".join("-" if value is None else "T" if value else "F" for value in row["outcomes"][name])
                    for name in METHODS]
        own = row["legacy_self_comparison"]
        labels = " / ".join(row["classification"][name]["label"] for name in paired)
        lines.append(f"| {row['case']} | {row['kind']} | {' | '.join(outcomes)} | "
                     f"{own['disagreements']}/{own['pairs']} | {labels} |")
    lines += ["", "Closed-loop timings include tracing/validation and are diagnostic only. Use isolated fixed-input replay for speed.",
              "No statistical equivalence is claimed. Baseline variability does not automatically explain candidate differences."]
    captures = root / "capture_diagnostics.json"
    if captures.exists():
        report["capture_diagnostics"] = base.read_json(captures)
        lines += ["", "Capture/action divergence: see capture_diagnostics.json. Closed-loop observations and histories may differ."]
    for shard in range(len(suite["gpus"])):
        replay = root / f"shard_{shard}" / "replay" / "timing.json"
        if replay.exists():
            measured = base.read_json(replay)
            report.setdefault("replay", {})[str(shard)] = measured
            lines += ["", f"Replay shard {shard}: see shard_{shard}/replay/timing.md and validation.jsonl."]
            lines += ["", "| Backend | P50 ms | P95 ms |", "|---|---:|---:|"]
            for backend, value in measured["methods"].items():
                lines.append(f"| {backend} | {value['wall_ms']['p50']:.3f} | {value['wall_ms']['p95']:.3f} |")
    lock = root / "summary.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        base.write_json(root / "report.json", report)
        (root / "report.md").write_text("\n".join(lines) + "\n")
    return report


def replay_shard(root, suite, shard, warmup, measure, profile_samples):
    from openwam.deploy.request_trace import _write_records

    cfg = configured(suite, root, shard)
    with base.output_lock(cfg.root):
        records = []
        attribution = []
        for case in suite["cases"]:
            if case["shard"] != shard:
                continue
            if valid_run(root, suite, case, "legacy", 0) is None:
                raise ValueError("Replay requires complete legacy repeat-0 captures")
            metadata = base.read_json(run_path(root, case, "legacy", 0) / "complete.json")
            if metadata["capture_capped"] or not metadata["request_records"]:
                raise ValueError("Capture capped or empty; refusing incomplete performance trace")
            paths = [Path(value) for value in metadata["request_records"]]
            attribution.append({"case": case["id"], "begin": len(records), "end": len(records) + len(paths)})
            records.extend(paths)
        trace = cfg.root / "observations.npz"
        if not trace.exists():
            temporary = trace.with_suffix(".tmp")
            if temporary.exists():
                temporary.rename(cfg.root / f"observations_interrupted_{time.time_ns()}.tmp")
            _write_records(records, trace)
        immutable_json(trace.with_suffix(".json"), {
            "benchmark": "robotwin", "source": "actual_E8_requests", "trace_sha256": base.digest(trace),
            **{key: value for key, value in suite["checkpoint"].items() if key != "normalization_stats"},
            "tasks": sorted(set(cfg.tasks)), "suite_identity": suite["identity"], "attribution": attribution})
        output = cfg.root / "replay"
        options = {"warmup": warmup, "measure": measure, "profile_samples": profile_samples,
                   "validate_every_request": True,
                   "profile_tail": True,
                   "trace_sha256": base.digest(trace), "suite_identity": suite["identity"]}
        immutable_json(cfg.root / "replay_options.json", options)
        if (output / "timing.json").exists() and base.read_json(output / "timing.json").get("status") == "complete":
            print(f"Skipping complete replay shard={shard}", flush=True)
            return
        if output.exists():
            output.rename(cfg.root / f"replay_failed_{time.time_ns()}")
        command = [cfg.server_python, "-m", "benchmarks.robotwin.joint_execution_replay", "--trace", str(trace),
                   "--output", str(output), "--ckpt-dir", str(cfg.checkpoint), "--ckpt-name", cfg.weight.name,
                   "--device", f"cuda:{cfg.gpu}", "--warmup", str(warmup), "--measure", str(measure),
                   "--profile-samples", str(profile_samples), "--allow-parity-failure",
                   "--validate-every-request", "--profile-tail"]
        owned = ProcessRegistry()
        try:
            log = (cfg.root / f"replay_{time.time_ns()}.log").open("w")
            environment = dict(os.environ)
            environment.pop("CUDA_VISIBLE_DEVICES", None)
            process = spawn_resource(command, cwd=base.ROOT, env=environment, stdout=log)
            owned.add(process, log)
            if process.wait():
                raise RuntimeError(f"Replay failed on shard={shard}; see {log.name}")
        finally:
            owned.terminate_all()


def launch(args, suite, mode):
    root = args.output.resolve()
    with base.output_lock(root):
        prepare(args, suite)
        owned = ProcessRegistry()
        processes = []
        try:
            for shard in range(len(suite["gpus"])):
                command = [sys.executable, "-m", "benchmarks.robotwin.joint_repeatability", "worker",
                           "--output", str(root), "--shard", str(shard), "--phase", mode,
                           "--warmup", str(args.warmup), "--measure", str(args.measure),
                           "--profile-samples", str(args.profile_samples)]
                log = (root / f"{mode}_shard_{shard}_{time.time_ns()}.log").open("w")
                process = spawn_resource(command, cwd=base.ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"), stdout=log)
                owned.add(process, log)
                processes.append(process)
                print(f"Started {mode}: shard={shard} GPU={suite['gpus'][shard]} log={log.name}", flush=True)
            codes = [process.wait() for process in processes]
            if any(codes):
                raise RuntimeError(f"Some shards failed: {codes}; completed runs retained for resume")
            if mode == "run":
                analyze_captures(root, suite)
        finally:
            owned.terminate_all()
            summarize(root)


def analyze_captures(root, suite):
    import numpy as np

    rows = []
    for case in suite["cases"]:
        metadata = {}
        for backend in METHODS:
            if valid_run(root, suite, case, backend, 0) is not None:
                metadata[backend] = base.read_json(run_path(root, case, backend, 0) / "complete.json")
        for backend in ("prepared", "prefix"):
            if "legacy" not in metadata or backend not in metadata:
                continue
            reference = metadata["legacy"]["request_records"]
            candidate = metadata[backend]["request_records"]
            row = {"case": case["id"], "backend": backend, "reference_requests": len(reference),
                   "candidate_requests": len(candidate), "first_input_difference": None,
                   "first_action_difference": None, "first_executed_action_threshold_failure": None,
                   "finite": True, "capture_capped": metadata["legacy"]["capture_capped"] or metadata[backend]["capture_capped"]}
            for index, (left, right) in enumerate(zip(reference, candidate)):
                with np.load(left, allow_pickle=False) as expected, np.load(right, allow_pickle=False) as actual:
                    same_input = all(np.array_equal(expected[key], actual[key])
                                     for key in ("image", "state", "prompt", "episode_start"))
                    if not same_input and row["first_input_difference"] is None:
                        row["first_input_difference"] = index
                    actions, target = actual["actions"], expected["actions"]
                    finite = bool(np.isfinite(actions).all() and np.isfinite(target).all())
                    row["finite"] &= finite
                    if actions.shape != target.shape:
                        row["shape_mismatch"] = index
                        break
                    if not np.array_equal(actions, target) and row["first_action_difference"] is None:
                        row["first_action_difference"] = {"request": index, "identical_observation": same_input,
                                                          "max_absolute_error": float(np.abs(actions - target).max()) if finite else None}
                    if (not finite or not np.allclose(actions[:8], target[:8], rtol=.05, atol=.05)) and row["first_executed_action_threshold_failure"] is None:
                        row["first_executed_action_threshold_failure"] = {"request": index, "identical_observation": same_input}
            rows.append(row)
    base.write_json(root / "capture_diagnostics.json", {
        "suite_identity": suite["identity"], "rows": rows,
        "boundary": "Closed-loop paths diverge; matching current inputs does not imply matching sampler histories. Not an equivalence test."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "all", "run", "summary", "analyze", "replay", "worker"), nargs="?", default="plan")
    parser.add_argument("--source", type=Path, default=Path(os.environ.get("SOURCE_ROOT", SOURCE)))
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("REPEAT_ROOT", OUTPUT)))
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0,1,2,3"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BASE_PORT", "8950")))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "5")))
    parser.add_argument("--controls", type=int, default=int(os.environ.get("CONTROL_CASES", "10")))
    parser.add_argument("--trace-limit", type=int, default=10000)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--phase", choices=("run", "replay"))
    parser.add_argument("--warmup", type=int, default=48)
    parser.add_argument("--measure", type=int, default=120)
    parser.add_argument("--profile-samples", type=int, default=8)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.warmup < 0 or args.measure < 1 or args.profile_samples < 0:
        parser.error("Invalid replay sample counts")
    if args.mode in ("summary", "analyze"):
        if args.mode == "analyze":
            analyze_captures(root, base.read_json(root / "suite.json"))
        report = summarize(root)
        print(root / "report.md")
        return 0 if report["status"] == "complete" else 1
    if args.mode == "worker":
        suite = base.read_json(root / "suite.json")
        if args.shard is None or not 0 <= args.shard < len(suite["gpus"]) or args.phase is None:
            parser.error("Worker requires a valid shard and phase")
        if args.phase == "run":
            shard_run(root, suite, args.shard)
        else:
            replay_shard(root, suite, args.shard, args.warmup, args.measure, args.profile_samples)
        return 0
    suite = make_suite(args)
    if args.mode == "plan":
        print(json.dumps({"cases": len(suite["cases"]),
                          "differences": sum(row["kind"] == "difference" for row in suite["cases"]),
                          "difference_tasks": len({row["task"] for row in suite["cases"] if row["kind"] == "difference"}),
                          "controls": args.controls, "repeats": args.repeats, "gpus": suite["gpus"],
                          "ports": [suite["port"] + shard for shard in range(len(suite["gpus"]))],
                          "episodes": len(suite["cases"]) * args.repeats * len(METHODS),
                          "output": str(root), "gpu_validation": "not_run",
                          "method_order": [method_order(suite, 0, repeat) for repeat in range(args.repeats)]}, indent=2))
        return 0
    with base.output_lock(root / "campaign_lock"):
        for phase in ("run", "replay") if args.mode == "all" else (args.mode,):
            launch(args, suite, phase)
    return 0


def interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; owned processes cleaned up. Rerun unchanged command to resume.", file=sys.stderr)
        sys.exit(130)
