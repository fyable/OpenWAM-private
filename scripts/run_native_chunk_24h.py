"""Deadline-bounded, coverage-first paired evaluation using sealed five-episode manifests."""

import json
import signal
import time
from pathlib import Path
from threading import Thread

import numpy as np

from scripts import run_native_chunk_comparison as base


class BudgetExpired(Exception):
    pass


def tasks(benchmark):
    if benchmark == "libero":
        return [(suite, task) for task in range(10) for suite in base.SUITES]
    names = base.robotwin.all_tasks()
    return [("demo_clean", name) for name in np.random.default_rng(42).permutation(names).tolist()]


def sliced_result(path, manifest, benchmark, index):
    try:
        value = json.loads(Path(path).read_text())
        if value.get("manifest_hash") != manifest["manifest_hash"]:
            return None
        entry = manifest["entries"][index]
        if benchmark == "libero":
            if any(value.get(key) != expected for key, expected in {
                "suite": manifest["suite"], "task_id": manifest["task_id"],
                "trial_start": index, "trial_stop": index + 1}.items()):
                return None
            rows = value["trials"]
            identity = {"trial": index, "init_state_index": entry["init_state_index"]}
        else:
            if any(value.get(key) != expected for key, expected in {
                "operation": "run_eval", "task": manifest["task"], "mode": "demo_clean",
                "instruction_type": "unseen", "episode_start": index, "num_episodes": 1,
                "total_episodes": 5}.items()):
                return None
            rows = value["episodes"]
            identity = {"episode": index, "seed": entry["seed"], "instruction": entry["instructions"]["unseen"]}
        if len(rows) != 1:
            return None
        row = rows[0]
        if any(row.get(key) != expected for key, expected in identity.items()):
            return None
        if benchmark == "libero" and not isinstance(row.get("init_state_sha256"), str):
            return None
        if type(row.get("success")) is not bool or value.get("successes") != int(row["success"]):
            return None
        if not all(type(row.get(key)) in (int, float) and np.isfinite(row[key]) and row[key] >= 0
                   for key in ("episode_ms", "rollout_ms", "policy_ms", "env_ms", "policy_steps")):
            return None
        return row
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def report(run):
    records = []
    for path in run.glob("formal/*/*/*/episodes/*/episode.json"):
        records.append(json.loads(path.read_text()))
    summary = {"target_episodes": 900, "completed_episodes": len(records), "benchmarks": {}}
    rng = np.random.default_rng(42)
    for benchmark in ("libero", "robotwin"):
        selected = [row for row in records if row["benchmark"] == benchmark]
        paired = {}
        for row in selected:
            paired.setdefault((row["task"], row["index"]), {})[row["method"]] = row
        pairs = [pair for pair in paired.values() if all(method in pair for method in base.METHODS)]
        native = [pair[base.METHODS[0]]["row"] for pair in pairs]
        optimized = [pair[base.METHODS[1]]["row"] for pair in pairs]
        delta = np.array([int(after["success"]) - int(before["success"]) for before, after in zip(native, optimized)])
        task_deltas = {}
        for pair in pairs:
            task_deltas.setdefault(pair[base.METHODS[0]]["task"], []).append(
                int(pair[base.METHODS[1]]["row"]["success"]) - int(pair[base.METHODS[0]]["row"]["success"]))
        groups = list(task_deltas.values())
        bootstrap = [float(np.mean([item for chosen in rng.integers(len(groups), size=len(groups))
                                    for item in groups[chosen]])) for _ in range(2000)] if groups else []
        metrics = {"complete_pairs": len(pairs), "unpaired_episodes": len(selected) - 2 * len(pairs),
                   "paired_task_coverage": len(task_deltas), "improvements": int(sum(delta > 0)),
                   "regressions": int(sum(delta < 0)),
                   "success_delta": float(delta.mean()) if len(delta) else None,
                   "task_cluster_bootstrap_95ci": np.percentile(bootstrap, [2.5, 97.5]).tolist() if bootstrap else None}
        for method, rows in zip(base.METHODS, (native, optimized)):
            metrics[method] = {"success_rate": float(np.mean([row["success"] for row in rows])) if rows else None,
                               **{key: base.statistics([row[key] for row in rows])
                                  for key in ("episode_ms", "policy_ms", "env_ms", "rollout_ms")}}
            requests = [request for pair in pairs for request in pair[method]["requests"]]
            metrics[method]["requests"] = {phase: base.statistics([row["client_ms"] for row in requests
                if (row["kind"] == "cached" if phase == "cached" else row["kind"] == "inference" and row["phase"] == phase)])
                for phase in ("first", "warmup", "steady", "cached")}
        both = [pair for pair in pairs if all(pair[method]["row"]["success"] for method in base.METHODS)]
        metrics["both_success_episode_ms"] = {method: base.statistics([pair[method]["row"]["episode_ms"] for pair in both])
                                               for method in base.METHODS}
        summary["benchmarks"][benchmark] = metrics
    startups = [json.loads(path.read_text())["ready_ms"] for path in run.glob("**/startup.json")]
    summary["startup"] = base.statistics(startups)
    samples = []
    monitor = run / "gpu-monitor.jsonl"
    if monitor.exists():
        for line in monitor.read_text().splitlines():
            try:
                samples.append(json.loads(line))
            except ValueError:
                continue
    energy = 0.0
    utilization = 0.0
    measured = 0.0
    for before, after in zip(samples, samples[1:]):
        seconds = after["time"] - before["time"]
        if not 0 < seconds <= 30:
            continue
        try:
            power = sum(float(row[4]) for row in before["gpus"])
            busy = sum(float(row[2]) / 100 for row in before["gpus"])
        except ValueError:
            continue
        energy += power * seconds / 3600000
        utilization += busy * seconds / 3600
        measured += seconds
    summary["resources"] = {"selected_gpu_energy_kwh": energy, "utilization_equivalent_gpu_hours": utilization,
                            "sampled_seconds": measured, "selected_four_gpu_hours": measured * 4 / 3600}
    summary["resource_note"] = "Native model uses one GPU; optimized uses four. Simulator shares the first selected GPU. Episode lifecycle differs between benchmarks. No videos or CPU offload."
    base.save(run / "summary.json", summary)
    (run / "REPORT.md").write_text("# 24小时配对评测\n\n目标900回合，实际完成：" + str(len(records)) +
        "。仅完整配对纳入方法比较；预检不计入正式结果。\n\n详细统计见 summary.json；原始请求、阶段、GPU功耗/占用见各回合及 gpu-monitor.jsonl。\n\n" + summary["resource_note"] + "\n")
    return summary


class BudgetComparison(base.Comparison):
    def __init__(self, args):
        super().__init__(args)
        if args.budget_hours <= 1:
            raise ValueError("Budget must exceed one hour (reserved for finalization)")
        if "budget_hours" in self.state and self.state["budget_hours"] != args.budget_hours:
            raise ValueError("Cannot change persisted campaign budget")
        self.state.setdefault("budget_hours", args.budget_hours)
        self.state.setdefault("deadline", self.state["started_at"] + args.budget_hours * 3600)
        self.deadline = self.state["deadline"]
        self.work_deadline = self.deadline - 3600
        self.state.setdefault("preflight", {})
        self.heartbeat("initializing")

    def admission(self, seconds=0):
        if time.time() + seconds >= self.work_deadline:
            raise BudgetExpired("No new work inside final reporting hour")

    def episode_path(self, root, method, index):
        return root / method / "episodes" / f"{index:03d}" / "episode.json"

    def evaluate_episode(self, stage, benchmark, suite, task, index, method, root, manifest_path, manifest):
        output = self.episode_path(root, method, index)
        if output.exists():
            value = json.loads(output.read_text())
            if value["manifest_hash"] != manifest["manifest_hash"]:
                raise ValueError("Checkpoint manifest mismatch")
            return
        for attempt in range(3):
            self.admission(900)
            directory = output.parent / "attempts" / str(time.time_ns())
            directory.mkdir(parents=True)
            try:
                self.ensure_server(benchmark, method, root / method)
                self.contaminated.clear()
                self.active = f"{stage}/{benchmark}/{suite}-{task}/{method}/{index}"
                self.heartbeat("running")
                if benchmark == "libero":
                    command = [str(base.ROOT / ".venv-libero/bin/python"), str(base.ROOT / "benchmarks/libero/single_eval.py"),
                        "--config", str(base.ROOT / "benchmarks/libero/policy_config.yml"), "--host", "127.0.0.1",
                        "--port", str(self.args.port), "--suite", suite, "--task-id", str(task), "--num-trials", "1",
                        "--trial-start", str(index), "--total-trials", "5", "--manifest-file", str(manifest_path),
                        "--manifest-hash", manifest["manifest_hash"], "--result-dir", str(directory),
                        "--progress-file", str(directory / "progress.json")]
                    result = directory / "results.json"
                else:
                    command = base.robotwin.client_command(self.robo_config(5, root), task, directory, manifest_path)
                    command[command.index("--episode-start") + 1] = str(index)
                    command[command.index("--num-episodes") + 1] = "1"
                    result = directory / "result.json"
                offset = self.server_log.stat().st_size
                started = time.time()
                self.child(command, base.environment(benchmark=benchmark, client=True), directory,
                           timeout=min(1800, max(1, self.work_deadline - time.time())))
                row = sliced_result(result, manifest, benchmark, index)
                if row is None or self.contaminated.is_set():
                    raise RuntimeError("Invalid or contaminated episode")
                requests = base.extract_requests(directory / "client.log")
                with self.server_log.open() as handle:
                    handle.seek(offset)
                    stages = base.extract_stages(handle.read())
                inference = [request for request in requests if request["kind"] == "inference"]
                if len(requests) != row["policy_steps"] or len(stages) != len(inference) or not inference:
                    raise RuntimeError("Request/stage accounting mismatch")
                for stats, request in zip(stages, inference):
                    stats.update(phase=request["phase"], step=request["step"])
                if base.source_fingerprints() != self.protocol["sources"]:
                    raise RuntimeError("Source changed; use a new run directory")
                base.save(output, {"benchmark": benchmark, "task": f"{suite}-{task}", "method": method,
                    "index": index, "manifest_hash": manifest["manifest_hash"], "row": row,
                    "wall_seconds": time.time() - started, "requests": requests, "stages": stages,
                    "raw_result": str(result)})
                self.state["completed"] += 1
                self.heartbeat()
                return
            except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
                base.save(directory / "failure.json", {"error": repr(exc), "attempt": attempt + 1})
                self.close_server()
                if attempt == 2:
                    raise

    def run_block(self, stage, benchmark, block, indices, number):
        prepared = []
        for suite, task in block:
            self.admission(1800)
            root = self.run / stage / benchmark / f"{suite}-{task}"
            for attempt in range(3):
                try:
                    path, manifest = self.manifest(stage, benchmark, suite, task, 5, root)
                    prepared.append((suite, task, root, path, manifest))
                    break
                except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
                    base.save(root / f"manifest_failure_{time.time_ns()}.json", {"error": repr(exc)})
                    self.close_server()
                    if stage == "preflight" and attempt == 2:
                        raise
        order = base.METHODS if number % 2 == 0 else base.METHODS[::-1]
        pending = [(suite, task, root, path, manifest, index)
                   for suite, task, root, path, manifest in prepared for index in indices
                   if not all(self.episode_path(root, method, index).exists() for method in base.METHODS)]
        pending.sort(key=lambda item: -sum(self.episode_path(item[2], method, item[5]).exists() for method in base.METHODS))
        estimate = max([json.loads(path.read_text())["wall_seconds"]
                        for path in self.run.glob(f"**/{benchmark}/*/*/episodes/*/episode.json")] or [180])
        self.admission(1200 + len(pending) * estimate * 2)
        for method in order:
            for suite, task, root, path, manifest, index in pending:
                try:
                    self.evaluate_episode(stage, benchmark, suite, task, index, method, root, path, manifest)
                except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
                    base.save(root / method / f"episode_{index}_failed.json", {"error": repr(exc)})
                    if stage == "preflight":
                        raise
        report(self.run)

    def execute(self):
        def expired(signum, frame):
            raise BudgetExpired("Hard campaign deadline reached")
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(0.01, self.deadline - time.time()))
        thread = None
        try:
            self.admission()
            self.initialize()
            thread = Thread(target=self.monitor, daemon=True)
            thread.start()
            for benchmark in self.args.benchmarks:
                if benchmark in self.state["preflight"]:
                    continue
                try:
                    self.run_block("preflight", benchmark, tasks(benchmark)[:1], [0], 0)
                    self.state["preflight"][benchmark] = {"passed": True}
                except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
                    self.state["preflight"][benchmark] = {"passed": False, "error": repr(exc)}
                self.heartbeat()
                self.close_server()
            if not self.args.preflight_only:
                for indices in ([0], [1, 2, 3, 4]):
                    for benchmark in self.args.benchmarks:
                        if not self.state["preflight"][benchmark]["passed"]:
                            continue
                        selected = tasks(benchmark)
                        for offset in range(0, len(selected), 10):
                            self.run_block("formal", benchmark, selected[offset:offset + 10], indices, offset // 10)
                        self.close_server()
            self.heartbeat("complete")
        except BudgetExpired as exc:
            self.state["stop_reason"] = str(exc)
            self.heartbeat("budget_complete")
        except BaseException as exc:
            self.state["error"] = repr(exc)
            self.heartbeat("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            raise
        finally:
            self.stop.set()
            self.close_server()
            if thread:
                thread.join(timeout=10)
            report(self.run)
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
