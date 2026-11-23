"""Resumable paired native N10 versus optimized chunk evaluation on GPUs 0/2/4/6."""

import argparse
import ast
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
from threading import Event, Thread
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin import compare_joint as robotwin
from benchmarks.utils.eval_manifest import load_manifest
from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource

GPUS = "0,2,4,6"
METHODS = ("native_n10", "chunk_optimized")
FLAGS = ("SELECTED_FRAMES", "REUSE_NOISE", "TIMESTEP_CACHE", "CUDA_GRAPH", "VAE_GPU_ACCUM", "VAE_GRAPH")
SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_10")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_fingerprints():
    return {str(path.relative_to(ROOT)): digest(path)
            for folder in ("openwam", "benchmarks", "configs", "scripts")
            for path in sorted((ROOT / folder).rglob("*"))
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml", ".sh")}


def environment(method=None, benchmark="libero", client=False):
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(("OPENWAM_CHUNK_", "ROBOTWIN_")) or key in ("E2_MEMORY_MODE", "CUDA_VISIBLE_DEVICES"):
            env.pop(key)
    env.update(PYTHONUNBUFFERED="1", OPENWAM_BENCH_TIMING="1", OPENWAM_REQUIRE_GPU_RESIDENT="1",
               PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    env["CUDA_VISIBLE_DEVICES"] = GPUS.split(",")[0] if client or method != METHODS[1] else GPUS
    paths = [str(ROOT), str(ROOT / "third_party/LIBERO")]
    if benchmark == "robotwin":
        paths = [str(ROOT / "third_party/RoboTwin"), str(ROOT / "benchmarks/robotwin"), str(ROOT)]
        env.update(ROBOTWIN_PATH=str(ROOT / "third_party/RoboTwin"),
                   TORCH_EXTENSIONS_DIR=str(ROOT / ".cache/robotwin/torch_extensions"),
                   WARP_CACHE_PATH=str(ROOT / ".cache/robotwin/warp"),
                   XDG_CACHE_HOME=str(ROOT / ".cache/robotwin"),
                   MPLCONFIGDIR=str(ROOT / ".cache/robotwin/matplotlib"))
    env["PYTHONPATH"] = ":".join(paths + [env.get("PYTHONPATH", "")])
    if method == METHODS[1] and not client:
        env.update({"OPENWAM_CHUNK_" + flag: "1" for flag in FLAGS})
        env.update(OPENWAM_CHUNK_INTEROP_THREADS="1", OPENWAM_CHUNK_CPU_GROUPS="0-7;8-15;16-23;24-31")
    if benchmark == "libero":
        env.update(LIBERO_PATH=str(ROOT / "third_party/LIBERO"), OPENWAM_LIBERO_STATE_HASH="1")
    return env


def matrix(stage, benchmark):
    if benchmark == "libero":
        return [(suite, task, 5 if stage == "preflight" else 50)
                for suite in SUITES for task in ([0] if stage == "preflight" else range(10))]
    tasks = robotwin.DIAGNOSTIC_TASKS if stage == "preflight" else robotwin.all_tasks()
    return [("demo_clean", task, 5 if stage == "preflight" else 100) for task in tasks]


def server_command(benchmark, method, checkpoint, weight, port, token):
    command = [str(ROOT / ".venv-openwam/bin/python"), str(ROOT / "scripts/deploy.py"),
               "--ckpt-dir", str(checkpoint), "--ckpt-name", weight, "--device", "cuda:0",
               "--host", "127.0.0.1", "--port", str(port), "--readiness-token", token,
               "--sampler", "native" if method == METHODS[0] else "chunk-pipeline",
               "--denoise-steps", "10", "--inference-horizon", "8", "--inference-mode", "sync",
               "--denoise-mode", "sync", "--compile-enabled", "false"]
    if method == METHODS[1]:
        command += ["--chunk-devices", "cuda:0,cuda:1,cuda:2,cuda:3", "--chunk-speculative-steps", "3"]
    command += ["inference.cfg_scale=1.0", "optimization.dit_cache.enabled=false",
                "optimization.decode_video=false", "inference.request_timing=true"]
    if method == METHODS[1]:
        command.append("inference.chunk_worker_backend=process")
    return command


def valid_result(path, manifest, benchmark, count):
    try:
        value = json.loads(Path(path).read_text())
        if value.get("manifest_hash") != manifest["manifest_hash"]:
            return None
        if benchmark == "robotwin":
            if robotwin.complete_result(path, manifest, count) is None:
                return None
            rows = value["episodes"]
        else:
            if (value.get("suite") != manifest["suite"] or value.get("task_id") != manifest["task_id"]
                    or value.get("trial_start") != 0 or value.get("trial_stop") != count):
                return None
            rows = value["trials"]
            for index, row in enumerate(rows):
                if (row.get("trial") != index or row.get("init_state_index") != manifest["entries"][index]["init_state_index"]
                        or not isinstance(row.get("init_state_sha256"), str)):
                    return None
        if len(rows) != count or value.get("successes") != sum(row["success"] for row in rows):
            return None
        for row in rows:
            if type(row.get("success")) is not bool or not all(
                    isinstance(row.get(key), (float, int)) and np.isfinite(row[key]) and row[key] >= 0
                    for key in ("episode_ms", "rollout_ms", "policy_ms", "env_ms", "policy_steps")):
                return None
        return value
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def statistics(values):
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "total_ms": float(np.sum(values))}


def extract_requests(log):
    rows = []
    for line in Path(log).read_text(errors="replace").splitlines():
        if "[BENCH_REQUEST] " in line:
            rows.append(json.loads(line.split("[BENCH_REQUEST] ", 1)[1]))
    return rows


def extract_stages(text):
    rows = []
    for line in text.splitlines():
        if "request stats=" in line:
            value = ast.literal_eval(line.split("request stats=", 1)[1])
            if isinstance(value, dict):
                rows.append(value)
    return rows


def summarize(run, confidence=False):
    protocol = json.loads((run / "protocol.json").read_text())
    report = {"status": "incomplete", "benchmarks": {}, "formal_expected_episodes": 14000,
              "resource_note": "Native: one GPU/request; optimized: four GPUs/request; no concurrent trials.",
              "timing_note": "First, requests 2-3 (warmup), steady requests and cached actions are separate."
                             " Episode includes reset; rollout excludes reset. Simulator conventions differ across benchmarks."}
    episode_export = []
    task_export = []
    for benchmark in protocol["benchmarks"]:
        methods = {}
        by_method = {}
        for method in METHODS:
            episodes = {}
            requests = []
            stages = []
            groups = {}
            tasks = []
            for result in sorted((run / "full" / benchmark).glob(f"*/{method}/result.json")):
                value = json.loads(result.read_text())
                rows = value["trials" if benchmark == "libero" else "episodes"]
                task = result.parent.parent.name
                tasks.append(task)
                for index, row in enumerate(rows):
                    episodes[(task, index)] = row
                    episode_export.append(dict(benchmark=benchmark, method=method, task=task, index=index, **row))
                request_rows = [json.loads(line) for line in result.with_name("requests.jsonl").read_text().splitlines()]
                requests.extend(request_rows)
                stages.extend(json.loads(line) for line in result.with_name("stages.jsonl").read_text().splitlines())
                task_row = dict(benchmark=benchmark, method=method, task=task, episodes=len(rows),
                                successes=sum(row["success"] for row in rows))
                task_row.update({"steady_client_" + key: val for key, val in statistics(
                    [row["client_ms"] for row in request_rows if row["kind"] == "inference" and row["phase"] == "steady"]).items()})
                task_export.append(task_row)
                suite = task.rsplit("-", 1)[0] if benchmark == "libero" else "demo_clean"
                group = groups.setdefault(suite, {"episodes": 0, "successes": 0})
                group["episodes"] += len(rows)
                group["successes"] += task_row["successes"]
            by_method[method] = episodes
            metrics = {}
            for kind, phase in (("cached", None), ("inference", "first"), ("inference", "warmup"), ("inference", "steady")):
                selected = [row for row in requests if row["kind"] == kind and (phase is None or row["phase"] == phase)]
                metrics[kind + ("_" + phase if phase else "")] = {
                    field: statistics([row[field] for row in selected if row.get(field) is not None])
                    for field in ("client_ms", "transport_ms", "server_ms")}
            methods[method] = {"tasks_completed": len(tasks), "episodes": len(episodes),
                               "successes": sum(row["success"] for row in episodes.values()), "groups": groups,
                               "requests": metrics, "episode": {
                                   field: statistics([row[field] for row in episodes.values()])
                                   for field in ("episode_ms", "rollout_ms", "policy_ms", "env_ms")},
                               "server_stages_steady": {
                                   field: statistics([row[field] for row in stages if row.get("phase") == "steady"
                                                      and isinstance(row.get(field), (int, float))])
                                   for field in sorted({key for row in stages for key in row if key.endswith("_ms")})}}
        baseline, candidate = (by_method[method] for method in METHODS)
        paired = sorted(baseline.keys() & candidate.keys())
        regressions, improvements = [], []
        for key in paired:
            reference, actual = baseline[key], candidate[key]
            if benchmark == "libero" and reference["init_state_sha256"] != actual["init_state_sha256"]:
                raise ValueError(f"Initial state mismatch: {key}")
            if reference["success"] != actual["success"]:
                (regressions if reference["success"] else improvements).append({"task": key[0], "episode": key[1]})
        common_success = [key for key in paired if baseline[key]["success"] and candidate[key]["success"]]
        comparison = {"paired_episodes": len(paired), "regressions": regressions, "improvements": improvements,
                      "paired_success_delta_pp": 100 * (len(improvements) - len(regressions)) / len(paired) if paired else None,
                      "paired_both_success": {method: {
                          field: statistics([by_method[method][key][field] for key in common_success])
                          for field in ("episode_ms", "rollout_ms", "policy_ms", "env_ms")} for method in METHODS}}
        comparison["speedups"] = {}
        for phase in ("first", "warmup", "steady"):
            reference = methods[METHODS[0]]["requests"]["inference_" + phase]["client_ms"]
            actual = methods[METHODS[1]]["requests"]["inference_" + phase]["client_ms"]
            if reference["count"] and actual["count"]:
                comparison["speedups"]["request_" + phase + "_p50"] = reference["p50_ms"] / actual["p50_ms"]
        for field in ("episode_ms", "rollout_ms", "policy_ms"):
            reference, actual = (comparison["paired_both_success"][method][field] for method in METHODS)
            if reference["count"] and actual["count"] and actual["p50_ms"] > 0:
                comparison["speedups"]["both_success_" + field + "_p50"] = reference["p50_ms"] / actual["p50_ms"]
        if paired and confidence:
            generator = np.random.default_rng(42)
            task_keys = sorted({key[0] for key in paired})
            grouped = [[key for key in paired if key[0] == task] for task in task_keys]
            deltas = []
            for repeat in range(1000):
                selected = [keys[index] for keys in grouped for index in generator.integers(0, len(keys), len(keys))]
                deltas.append(100 * np.mean([int(candidate[key]["success"]) - int(baseline[key]["success"]) for key in selected]))
            comparison["success_delta_pp_95ci"] = np.percentile(deltas, [2.5, 97.5]).tolist()
            comparison["bootstrap"] = "1000 resamples of paired episodes within each fixed task; seed42"
            if common_success:
                ratios = []
                for repeat in range(1000):
                    selected = [common_success[index] for index in generator.integers(0, len(common_success), len(common_success))]
                    ratios.append(np.median([baseline[key]["episode_ms"] for key in selected]) /
                                  np.median([candidate[key]["episode_ms"] for key in selected]))
                comparison["both_success_episode_p50_speedup_95ci"] = np.percentile(ratios, [2.5, 97.5]).tolist()
        report["benchmarks"][benchmark] = {"methods": methods, "comparison": comparison}
    expected = {"libero": 2000, "robotwin": 5000}
    report["status"] = "complete" if all(
        row["methods"][method]["episodes"] == expected[benchmark]
        for benchmark, row in report["benchmarks"].items() for method in METHODS) else "incomplete"
    save(run / "summary.json", report)
    if (run / "gpu-monitor.jsonl").exists():
        resources = {}
        previous = None
        with (run / "gpu-monitor.jsonl").open() as handle:
            for line in handle:
                sample = json.loads(line)
                active = sample.get("active") or "idle"
                key = "/".join(active.split("/")[:5])
                group = resources.setdefault(key, {"gpu_memory_peak_mib": {}, "sampled_gpu_seconds": 0.0,
                                                   "utilization_sum": 0.0, "gpu_samples": 0})
                for row in sample["gpus"]:
                    index = row[0].strip()
                    group["gpu_memory_peak_mib"][index] = max(group["gpu_memory_peak_mib"].get(index, 0), float(row[3]))
                    group["utilization_sum"] += float(row[2])
                    group["gpu_samples"] += 1
                if previous and previous.get("active") == sample.get("active"):
                    group["sampled_gpu_seconds"] += min(10, sample["time"] - previous["time"]) * sum(float(row[3]) > 1000 for row in sample["gpus"])
                previous = sample
        save(run / "resources.json", resources)
    for filename, rows in (("episodes.csv", episode_export), ("tasks.csv", task_export)):
        if rows:
            with (run / filename).open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
                writer.writeheader()
                writer.writerows(rows)
    lines = ["# 原生N10与优化chunk完整对比", "", f"状态：{report['status']}；正式目标14,000回合，预检另计。", "",
             "| 基准 | 方法 | 完成任务 | 成功/回合 | 成功率 | 稳态请求P50/P95 ms | 回合P50 ms |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for benchmark, entry in report["benchmarks"].items():
        for method, row in entry["methods"].items():
            request = row["requests"]["inference_steady"]["client_ms"]
            latency = f"{request['p50_ms']:.2f}/{request['p95_ms']:.2f}" if request["count"] else "—"
            rate = f"{100 * row['successes'] / row['episodes']:.2f}%" if row["episodes"] else "—"
            episode = row["episode"]["episode_ms"].get("p50_ms")
            lines.append(f"| {benchmark} | {method} | {row['tasks_completed']} | {row['successes']}/{row['episodes']} | {rate} | {latency} | {episode or '—'} |")
        paired = entry["comparison"]
        lines += ["", f"{benchmark}：配对{paired['paired_episodes']}回合，退化{len(paired['regressions'])}，改善{len(paired['improvements'])}。"]
    lines += ["", "原生单请求一卡，优化单请求四卡；请求加速不等于四卡吞吐提升。",
              "不跨基准比较绝对耗时。缓存动作请求不计入推理请求P50；冷启动、前3次与稳态分开。",
              "成功率有退化时不宣称无损；默认部署配置不变。详见summary.json、tasks.csv、episodes.csv。"]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


class Comparison:
    def __init__(self, args):
        global GPUS
        selected_gpus = getattr(args, "gpus", GPUS).split(",")
        if len(selected_gpus) != 4 or len(set(selected_gpus)) != 4 or not all(value.isdigit() for value in selected_gpus):
            raise ValueError("--gpus requires four distinct physical GPU indices")
        GPUS = ",".join(selected_gpus)
        self.args = args
        self.run = args.run.resolve()
        self.registry = ProcessRegistry()
        self.server = None
        self.server_key = None
        self.stop = Event()
        self.contaminated = Event()
        self.active = None
        self.state = {"status": "initializing", "pid": os.getpid(), "started_at": time.time(), "completed": 0}
        if (self.run / "state.json").exists():
            self.state.update(json.loads((self.run / "state.json").read_text()), pid=os.getpid())

    def heartbeat(self, status=None):
        if status:
            self.state["status"] = status
        self.state.update(updated_at=time.time(), active=self.active)
        if self.active:
            progress_files = sorted(self.run.glob("**/attempts/*/progress.json"), key=lambda path: path.stat().st_mtime)
            if progress_files:
                self.state["latest_progress"] = json.loads(progress_files[-1].read_text())
        save(self.run / "state.json", self.state)
        (self.run / "STATUS.md").write_text(f"# 评测进度\n\n状态：{self.state['status']}\n\n当前：{self.active}\n\n管理器PID：{os.getpid()}\n\n已完成任务/方法单元：{self.state['completed']}\n\n预计剩余小时：{self.state.get('estimated_remaining_hours', '等待正式样本')}\n")

    def owned_pids(self):
        output = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
        parents = {int(parts[0]): int(parts[1]) for line in output.splitlines() if (parts := line.split())}
        owned = {os.getpid()}
        while True:
            expanded = owned | {pid for pid, parent in parents.items() if parent in owned}
            if expanded == owned:
                return owned
            owned = expanded

    def gpu_sample(self):
        query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"], text=True)
        gpu_rows = [row for row in csv.reader(query.splitlines()) if row[0].strip() in GPUS.split(",")]
        uuids = {row[1].strip() for row in gpu_rows}
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
        owned = self.owned_pids()
        external = [int(row[1]) for row in csv.reader(apps.splitlines()) if len(row) == 2 and row[0].strip() in uuids and int(row[1]) not in owned]
        return {"time": time.time(), "gpus": gpu_rows, "external_pids": external, "active": self.active}

    def monitor(self):
        while not self.stop.is_set():
            try:
                sample = self.gpu_sample()
                if sample["external_pids"] and self.active:
                    self.contaminated.set()
                with (self.run / "gpu-monitor.jsonl").open("a") as handle:
                    handle.write(json.dumps(sample) + "\n")
                self.heartbeat()
            except Exception as exc:
                self.contaminated.set()
                print(f"Monitor error: {exc}", flush=True)
            self.stop.wait(5)

    def wait_free(self):
        self.active = None
        self.heartbeat("waiting_for_exclusive_gpus")
        while self.gpu_sample()["external_pids"]:
            time.sleep(30)

    def close_server(self):
        self.registry.terminate_all()
        self.server = self.server_key = None
        time.sleep(2)

    def ensure_server(self, benchmark, method, directory):
        if self.server_key == (benchmark, method) and self.server.poll() is None:
            return
        self.close_server()
        self.wait_free()
        selected = self.protocol["checkpoints"][benchmark]
        logdir = directory / "servers" / str(time.time_ns())
        logdir.mkdir(parents=True)
        token = secrets.token_hex(24)
        command = server_command(benchmark, method, selected["path"], selected["weight"], self.args.port, token)
        save(logdir / "command.json", command)
        handle = (logdir / "server.log").open("w")
        self.server_log = logdir / "server.log"
        started = time.monotonic()
        self.server = spawn_resource(command, cwd=ROOT, env=environment(method, benchmark), stdout=handle)
        self.registry.add(self.server, handle)
        self.active = f"loading {benchmark}/{method}"
        robotwin.wait_ready(self.server, self.args.port, token, 1800)
        save(logdir / "startup.json", {"ready_ms": (time.monotonic() - started) * 1000,
                                      "visible_gpus": environment(method)["CUDA_VISIBLE_DEVICES"]})
        self.server_key = (benchmark, method)

    def child(self, command, env, directory, timeout=86400):
        if shutil.disk_usage(self.run).free < 5 * 1024 ** 3:
            raise RuntimeError("Disk free below 5 GiB; not deleting any existing files")
        directory.mkdir(parents=True, exist_ok=True)
        if "ROBOTWIN_PATH" in env:
            env["ROBOTWIN_RUNTIME_ROOT"] = str(directory / "runtime")
            egl = ROOT / ".venv-robotwin/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
            if egl.exists():
                env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl)
        env["LIBERO_CONFIG_PATH"] = str(self.run / "libero-config")
        env["LIBERO_CONFIG_ROOT"] = str(self.run / "libero-config")
        save(directory / "command.json", command)
        owned = ProcessRegistry()
        with (directory / "client.log").open("w") as handle:
            process = spawn_resource(command, cwd=ROOT, env=env, stdout=handle)
            owned.add(process)
            started = time.monotonic()
            changed_at = started
            size = -1
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    current_size = (directory / "client.log").stat().st_size
                    if size != current_size:
                        size, changed_at = current_size, now
                    if now - started > timeout or now - changed_at > 1800:
                        raise TimeoutError(f"Client timeout: {directory}")
                    if self.server is not None and self.server.poll() is not None:
                        raise RuntimeError(f"Server exited: {self.server_key}")
                    if self.contaminated.is_set():
                        raise RuntimeError("External GPU activity or monitoring failure; attempt invalid")
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError(f"Client exit {process.returncode}: {directory}")
            finally:
                owned.terminate_all()

    def robo_config(self, count, root):
        cfg = robotwin.configuration("full")
        cfg.root = root
        cfg.robotwin = ROOT / "third_party/RoboTwin"
        cfg.client_python = str(ROOT / ".venv-robotwin/bin/python")
        cfg.server_python = str(ROOT / ".venv-openwam/bin/python")
        cfg.episodes = count
        cfg.port = self.args.port
        cfg.sim_gpu = GPUS.split(",")[0]
        root.mkdir(parents=True, exist_ok=True)
        for name in ("policy_config.yml", "step_limits.yml"):
            shutil.copyfile(ROOT / "benchmarks/robotwin" / name, root / name)
        return cfg

    def manifest(self, stage, benchmark, suite, task, count, root):
        path = root / "manifest.json"
        if not path.exists():
            self.wait_free()
            self.contaminated.clear()
            self.active = f"manifest {stage}/{benchmark}/{suite}/{task}"
            directory = root / "manifest_attempts" / str(time.time_ns())
            if benchmark == "libero":
                command = [str(ROOT / ".venv-libero/bin/python"), str(ROOT / "benchmarks/libero/eval_manifest.py"),
                           "--config", str(ROOT / "benchmarks/libero/policy_config.yml"), "--suite", suite,
                           "--task-id", str(task), "--total-trials", str(count), "--result-file", str(directory / "manifest.json")]
            else:
                cfg = self.robo_config(count, root)
                command = robotwin.client_command(cfg, task, directory)
            self.child(command, environment(benchmark=benchmark, client=True), directory)
            built = directory / ("manifest.json" if benchmark == "libero" else "result.json")
            value = load_manifest(built)
            save(path, value)
        value = load_manifest(path)
        if len(value["entries"]) != count or value.get("benchmark") != benchmark:
            raise ValueError(f"Invalid manifest: {path}")
        if benchmark == "robotwin":
            robotwin.check_manifest(path, task, count)
        elif value.get("suite") != suite or value.get("task_id") != task:
            raise ValueError(f"Manifest task mismatch: {path}")
        return path, value

    def evaluate(self, stage, benchmark, suite, task, count, method, root, manifest_path, manifest):
        directory = root / method
        if valid_result(directory / "result.json", manifest, benchmark, count):
            if not (directory / "requests.jsonl").exists():
                raise ValueError(f"Missing request artifact: {directory}")
            return
        for attempt in range(3):
            self.wait_free()
            self.contaminated.clear()
            attempt_dir = directory / "attempts" / str(time.time_ns())
            attempt_dir.mkdir(parents=True)
            try:
                self.ensure_server(benchmark, method, directory)
                self.active = f"{stage}/{benchmark}/{suite}/{task}/{method}/attempt{attempt + 1}"
                self.heartbeat("running")
                if benchmark == "libero":
                    command = [str(ROOT / ".venv-libero/bin/python"), str(ROOT / "benchmarks/libero/single_eval.py"),
                               "--config", str(ROOT / "benchmarks/libero/policy_config.yml"), "--host", "127.0.0.1",
                               "--port", str(self.args.port), "--suite", suite, "--task-id", str(task),
                               "--num-trials", str(count), "--trial-start", "0", "--total-trials", str(count),
                               "--manifest-file", str(manifest_path), "--manifest-hash", manifest["manifest_hash"],
                               "--result-dir", str(attempt_dir), "--progress-file", str(attempt_dir / "progress.json")]
                    result_path = attempt_dir / "results.json"
                else:
                    command = robotwin.client_command(self.robo_config(count, root), task, attempt_dir, manifest_path)
                    result_path = attempt_dir / "result.json"
                started = time.monotonic()
                log_offset = self.server_log.stat().st_size
                self.child(command, environment(benchmark=benchmark, client=True), attempt_dir)
                if source_fingerprints() != self.protocol["sources"]:
                    raise RuntimeError("Source changed during task; results not accepted")
                value = valid_result(result_path, manifest, benchmark, count)
                if value is None or self.contaminated.is_set():
                    raise RuntimeError(f"Invalid or contaminated results: {attempt_dir}")
                requests = extract_requests(attempt_dir / "client.log")
                with self.server_log.open() as handle:
                    handle.seek(log_offset)
                    stage_rows = extract_stages(handle.read())
                inference_count = sum(row["kind"] == "inference" for row in requests)
                if len(stage_rows) != inference_count:
                    raise RuntimeError(f"Server/request accounting mismatch: {len(stage_rows)} vs {inference_count}")
                for stage_row, request in zip(stage_rows, (row for row in requests if row["kind"] == "inference")):
                    stage_row.update(phase=request["phase"], episode=request.get("episode"), step=request["step"])
                expected = sum(row["policy_steps"] for row in value["trials" if benchmark == "libero" else "episodes"])
                first_count = sum(row["kind"] == "inference" and row["phase"] == "first" for row in requests)
                if not requests or len(requests) != expected or first_count != count:
                    raise RuntimeError(f"Request accounting failed: {attempt_dir}")
                value["comparison"] = {"method": method, "attempt": str(attempt_dir.relative_to(self.run)),
                                       "wall_seconds": time.monotonic() - started, "request_count": len(requests)}
                (directory / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in requests))
                (directory / "stages.jsonl").write_text("".join(json.dumps(row) + "\n" for row in stage_rows))
                save(directory / "result.json", value)
                self.state["completed"] += 1
                if stage == "full":
                    self.state["formal_episodes"] = self.state.get("formal_episodes", 0) + count
                    self.state["formal_wall_seconds"] = self.state.get("formal_wall_seconds", 0) + time.monotonic() - started
                    total = sum(4000 if name == "libero" else 10000 for name in self.args.benchmarks)
                    self.state["estimated_remaining_hours"] = ((total - self.state["formal_episodes"]) *
                        self.state["formal_wall_seconds"] / self.state["formal_episodes"] / 3600)
                self.heartbeat("running")
                return
            except (RuntimeError, OSError, TimeoutError, ValueError) as exc:
                save(attempt_dir / "failure.json", {"error": str(exc), "time": time.time()})
                print(f"Attempt failed: {self.active}: {exc}", flush=True)
                self.active = None
                self.close_server()
                if attempt == 2:
                    raise

    def initialize(self):
        libero_root = ROOT / "third_party/LIBERO/libero/libero"
        save(self.run / "libero-config/config.yaml", {
            "benchmark_root": str(libero_root), "bddl_files": str(libero_root / "bddl_files"),
            "init_states": str(libero_root / "init_files"), "assets": str(libero_root / "assets"),
            "datasets": str(ROOT / "third_party/LIBERO/datasets")})
        checkpoints = {}
        for benchmark in self.args.benchmarks:
            checkpoint = Path("/data1/wam-ckpts") / ("OpenWAM-Alpha-Sim-LIBERO" if benchmark == "libero" else "OpenWAM-Alpha-Sim-RoboTwin-Full")
            weight = max(checkpoint.glob("checkpoint_step_*.safetensors"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
            print(f"Hashing checkpoint: {weight}", flush=True)
            checkpoints[benchmark] = {"path": str(checkpoint), "weight": weight.name,
                                      "sha256": digest(weight), "config_sha256": digest(checkpoint / "config.yaml"),
                                      "normalization_sha256": digest(checkpoint / "normalization_stats.npy")}
        protocol = {"version": 1, "benchmarks": self.args.benchmarks, "checkpoints": checkpoints,
                    "sources": source_fingerprints(), "gpus": GPUS, "methods": list(METHODS),
                    "horizon": 32, "execute": 8, "denoise_steps": 10, "seed_libero": 42,
                    "seed_robotwin": 0, "cpu_offload": False, "port": self.args.port,
                    "flags": list(FLAGS), "threads": 1, "cpu_groups": "0-7;8-15;16-23;24-31"}
        protocol["simulator_sources"] = {
            str(path.relative_to(ROOT)): digest(path)
            for folder in ("third_party/RoboTwin/script", "third_party/RoboTwin/envs", "third_party/RoboTwin/task_config",
                           "third_party/RoboTwin/description", "third_party/LIBERO/libero")
            for path in sorted((ROOT / folder).rglob("*"))
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml", ".json", ".bddl")}
        path = self.run / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("Protocol/source/weights changed; use a fresh run directory")
        save(path, protocol)
        self.protocol = protocol
        self.heartbeat("initialized")

    def execute(self):
        self.initialize()
        thread = Thread(target=self.monitor, daemon=True)
        thread.start()
        try:
            stages = ("preflight",) if self.args.preflight_only else ("preflight", "full")
            for stage in stages:
                for benchmark in self.args.benchmarks:
                    for index, (suite, task, count) in enumerate(matrix(stage, benchmark)):
                        if source_fingerprints() != self.protocol["sources"]:
                            raise RuntimeError("Source changed during campaign; stopping to preserve provenance")
                        root = self.run / stage / benchmark / f"{suite}-{task}"
                        root.mkdir(parents=True, exist_ok=True)
                        for attempt in range(3):
                            try:
                                manifest_path, manifest = self.manifest(stage, benchmark, suite, task, count, root)
                                break
                            except (RuntimeError, OSError, TimeoutError, ValueError) as exc:
                                save(root / f"manifest_failure_{time.time_ns()}.json", {"error": str(exc)})
                                self.active = None
                                self.close_server()
                                if attempt == 2:
                                    raise
                        for method in METHODS if index % 2 == 0 else METHODS[::-1]:
                            self.evaluate(stage, benchmark, suite, task, count, method, root, manifest_path, manifest)
                        summarize(self.run)
                    self.active = None
                    self.close_server()
                if stage == "preflight":
                    save(self.run / "preflight.json", {"passed": True, "note": "Execution/data integrity, not a quality non-regression gate"})
            summarize(self.run, confidence=not self.args.preflight_only)
            self.heartbeat("preflight_complete" if self.args.preflight_only else "complete")
        except BaseException as exc:
            self.state["error"] = repr(exc)
            self.heartbeat("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            summarize(self.run)
            raise
        finally:
            self.stop.set()
            thread.join(timeout=15)
            self.active = None
            self.close_server()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "experiment/native_chunk_full_20261121")
    parser.add_argument("--benchmarks", nargs="+", choices=("libero", "robotwin"), default=["libero", "robotwin"])
    parser.add_argument("--port", type=int, default=9075)
    parser.add_argument("--gpus", default=GPUS, help="Four physical GPU indices; first also runs native and simulator")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--detach", action="store_true", help="Start an independent session and write manager.log")
    parser.add_argument("--budget-hours", type=float, help="Coverage-first campaign; includes initialization and cleanup")
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (args.run / "manager.log").open("a") as handle:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                        *[item for item in sys.argv[1:] if item != "--detach"]],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError(f"Detached manager exited {process.returncode}; see {args.run / 'manager.log'}")
        print(f"Manager PID: {process.pid}; log: {args.run / 'manager.log'}", flush=True)
        return
    with (args.run / "manager.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.budget_hours is not None:
            from scripts.run_native_chunk_24h import BudgetComparison, report
            if args.summary:
                report(args.run)
            else:
                def budget_interrupted(signum, frame):
                    raise KeyboardInterrupt
                signal.signal(signal.SIGTERM, budget_interrupted)
                BudgetComparison(args).execute()
            return
        if args.summary:
            summarize(args.run, confidence=True)
            return
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        Comparison(args).execute()


if __name__ == "__main__":
    main()
