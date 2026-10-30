"""Sequential, manifest-paired RoboTwin comparison without a Labtasker service."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from benchmarks.utils.eval_manifest import content_hash, load_manifest
from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = {"native_n10_e8": ("native", 10), "native_n4_e8": ("native", 4),
           "joint_n4_e8": ("joint-paraflow", 4)}
DIAGNOSTIC_TASKS = ["adjust_bottle", "open_laptop", "stack_blocks_two"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def all_tasks():
    source = (HERE / "multi_eval.sh").read_text()
    matched = re.search(r"ROBOTWIN_ALL_TASKS=\((.*?)\)", source, re.S)
    if matched is None:
        raise ValueError("Cannot resolve RoboTwin task list")
    tasks = matched[1].split()
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError("Expected 50 unique RoboTwin tasks")
    return tasks


def configuration(mode):
    diagnostic = mode == "diagnostic"
    return argparse.Namespace(
        mode=mode, diagnostic=diagnostic,
        tasks=DIAGNOSTIC_TASKS if diagnostic else all_tasks(),
        episodes=5 if diagnostic else 100,
        root=Path(os.environ.get("COMPARE_ROOT", str(ROOT / "outputs/robotwin" /
                  ("compare_joint_diagnostic" if diagnostic else "compare_joint_clean")))).expanduser().resolve(),
        checkpoint=Path(os.environ.get("CKPT_DIR", "/data1/wam-ckpts/OpenWAM-Alpha-Sim-RoboTwin-Full")).expanduser().resolve(),
        robotwin=Path(os.environ.get("ROBOTWIN_PATH", "/nonexistent/ROBOTWIN_PATH")).expanduser().resolve(),
        client_python=os.environ.get("ROBOTWIN_PYTHON", ""),
        server_python=os.environ.get("SERVER_PYTHON", sys.executable),
        gpu=os.environ.get("GPU", "0"), sim_gpu=os.environ.get("SIM_GPU", os.environ.get("GPU", "0")),
        port=int(os.environ.get("PORT", "8940")),
        timeout=float(os.environ.get("SERVER_START_TIMEOUT", "900")),
    )


def executable(value):
    resolved = shutil.which(value) if value else None
    if resolved is None:
        raise ValueError(f"Python interpreter is not executable: {value!r}")
    return str(Path(resolved).absolute())


def preflight(cfg):
    if not 0 < cfg.port < 65536 or cfg.timeout <= 0:
        raise ValueError("PORT must be 1..65535 and SERVER_START_TIMEOUT must be positive")
    cfg.client_python = executable(cfg.client_python)
    cfg.server_python = executable(cfg.server_python)
    if not (cfg.robotwin / "script/eval_policy.py").is_file():
        raise ValueError("Set ROBOTWIN_PATH to an installed RoboTwin checkout with task assets")
    for name in ("config.yaml", "normalization_stats.npy"):
        if not (cfg.checkpoint / name).is_file():
            raise FileNotFoundError(cfg.checkpoint / name)
    checkpoints = list(cfg.checkpoint.glob("checkpoint_step_*.safetensors"))
    if not checkpoints:
        raise ValueError(f"No checkpoint found in {cfg.checkpoint}")
    cfg.weight = max(checkpoints, key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    import yaml

    training = yaml.safe_load((cfg.checkpoint / "config.yaml").read_text())
    architecture = training["model"]["architecture"]
    loader = training["dataloader"]
    if (architecture.get("framework") != "dual_system" or architecture.get("variant") != "joint_self_attn"
            or architecture.get("attention_mask_mode") != "mutual" or loader.get("type") != "robotwin"
            or loader.get("num_frames") != 33 or loader.get("video_stride") != 4
            or loader.get("action_mode") != "eef"):
        raise ValueError("Expected RoboTwin eef joint_self_attn/mutual checkpoint with H32/video_stride=4")


def protocol(cfg):
    sources = set((ROOT / "openwam/deploy").glob("*.py"))
    sources.update((ROOT / "openwam/model/architectures").glob("*.py"))
    sources.update(HERE.glob("*.py"))
    sources.update(HERE.glob("*.yml"))
    sources.update((ROOT / "openwam/dataloader").rglob("*robotwin*.py"))
    sources.update([ROOT / "configs/deploy.yaml", ROOT / "scripts/deploy.py",
                    HERE / "multi_eval.sh", ROOT / "scripts/compare_alpha_robotwin.sh"])
    watched = [cfg.robotwin / name for name in ("script", "envs", "task_config", "policy", "description")]
    upstream = {str(path.relative_to(cfg.robotwin)): digest(path)
                for directory in watched for path in sorted(directory.rglob("*"))
                if path.is_file() and path.suffix in (".py", ".yml", ".yaml", ".json")}
    stat = cfg.weight.stat()
    body = {
        "version": 1, "diagnostic": cfg.diagnostic, "mode": "demo_clean", "tasks": cfg.tasks,
        "episodes_per_task": cfg.episodes, "seed": 0, "instruction_type": "unseen",
        "methods": {name: list(spec) for name, spec in METHODS.items()},
        "inference_horizon": 8, "inference_mode": "sync", "denoise_mode": "sync",
        "compile": False, "dit_cache": False, "decode_video": False, "cfg_scale": 1.0,
        "checkpoint": str(cfg.weight), "checkpoint_size": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
        "checkpoint_config": digest(cfg.checkpoint / "config.yaml"),
        "normalization_stats": digest(cfg.checkpoint / "normalization_stats.npy"),
        "robotwin_path": str(cfg.robotwin), "robotwin_sources": upstream,
        "server_python": cfg.server_python, "client_python": cfg.client_python,
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in sorted(sources)},
    }
    return body | {"identity": content_hash(body)}


def check_protocol(root, current):
    path = root / "protocol.json"
    if path.exists():
        if read_json(path) != current:
            raise ValueError("Comparison protocol changed; use a new COMPARE_ROOT")
    else:
        if any(root.iterdir()):
            if {item.name for item in root.iterdir()} != {".lock"}:
                raise ValueError("Nonempty COMPARE_ROOT has no protocol.json; choose a fresh directory")
        write_json(path, current)


@contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another comparison is using {root}") from exc
        yield


def check_manifest(path, task, total):
    manifest = load_manifest(path)
    if (manifest.get("benchmark") != "robotwin" or manifest.get("task") != task
            or manifest.get("mode") != "demo_clean" or len(manifest["entries"]) != total):
        raise ValueError(f"Manifest protocol mismatch: {path}")
    for episode, entry in enumerate(manifest["entries"]):
        if (entry.get("episode") != episode or type(entry.get("seed")) is not int
                or not isinstance(entry.get("instructions", {}).get("unseen"), str)):
            raise ValueError(f"Invalid manifest entry: {path}, episode {episode}")
    return manifest


def complete_result(path, manifest, total):
    try:
        result = read_json(path)
        if any(result.get(key) != value for key, value in {
            "operation": "run_eval", "task": manifest["task"], "mode": "demo_clean",
            "instruction_type": "unseen", "episode_start": 0, "num_episodes": total,
            "total_episodes": total, "manifest_hash": manifest["manifest_hash"],
        }.items()):
            return None
        rows = result["episodes"]
        if len(rows) != total:
            return None
        for episode, (row, entry) in enumerate(zip(rows, manifest["entries"])):
            if (row.get("episode") != episode or row.get("seed") != entry["seed"]
                    or row.get("instruction") != entry["instructions"]["unseen"]
                    or type(row.get("success")) is not bool):
                return None
        if result.get("successes") != sum(row["success"] for row in rows):
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def client_environment(cfg, directory):
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("ROBOTWIN_"):
            env.pop(key)
    env.update(ROBOTWIN_PATH=str(cfg.robotwin), ROBOTWIN_RUNTIME_ROOT=str(directory / "runtime"),
               CUDA_VISIBLE_DEVICES=cfg.sim_gpu, PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = f"{cfg.robotwin}:{HERE}:{ROOT}:{env.get('PYTHONPATH', '')}"
    env["MPLCONFIGDIR"] = str(directory / "matplotlib")
    egl = Path(cfg.client_python).parent.parent / "lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    if not env.get("__EGL_VENDOR_LIBRARY_FILENAMES") and not env.get("__EGL_VENDOR_LIBRARY_DIRS") and egl.is_file():
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl)
    return env


def client_command(cfg, task, directory, manifest_path=None):
    command = [cfg.client_python, str(HERE / "eval_policy_wrapper.py"), "labtasker",
               "--task", task, "--mode", "demo_clean", "--policy-config", str(cfg.root / "policy_config.yml"),
               "--operation", "run_eval" if manifest_path else "build_manifest",
               "--result-file", str(directory / "result.json"), "--progress-file", str(directory / "progress.json"),
               "--total-episodes", str(cfg.episodes), "--instruction-type", "unseen"]
    if manifest_path:
        manifest = check_manifest(manifest_path, task, cfg.episodes)
        command += ["--host", "127.0.0.1", "--port", str(cfg.port), "--episode-start", "0",
                    "--num-episodes", str(cfg.episodes), "--manifest-file", str(manifest_path),
                    "--manifest-hash", manifest["manifest_hash"]]
    return command


def run_client(cfg, task, directory, manifest_path=None, server=None):
    directory.mkdir(parents=True, exist_ok=True)
    command = client_command(cfg, task, directory, manifest_path)
    write_json(directory / "command.json", command)
    owned = ProcessRegistry()
    try:
        log = (directory / "client.log").open("w")
        process = spawn_resource(command, cwd=ROOT, env=client_environment(cfg, directory), stdout=log)
        owned.add(process, log)
        while process.poll() is None:
            if server is not None and server.poll() is not None:
                raise RuntimeError(f"Policy server exited during {task}; see server log")
            time.sleep(0.2)
        if process.returncode:
            raise RuntimeError(f"RoboTwin {task} failed ({process.returncode}); see {directory / 'client.log'}")
    finally:
        owned.terminate_all()


def server_command(cfg, method, token):
    sampler, steps = METHODS[method]
    return [cfg.server_python, str(ROOT / "scripts/deploy.py"), "--ckpt-dir", str(cfg.checkpoint),
            "--ckpt-name", cfg.weight.name, "--device", "cuda:0", "--host", "127.0.0.1",
            "--port", str(cfg.port), "--readiness-token", token, "--sampler", sampler,
            "--denoise-steps", str(steps), "--inference-horizon", "8", "--inference-mode", "sync",
            "--denoise-mode", "sync", "--compile-enabled", "false", "inference.cfg_scale=1.0",
            "optimization.dit_cache.enabled=false", "optimization.decode_video=false"]


def probe_server(port):
    from websockets.sync.client import connect

    with connect(f"ws://127.0.0.1:{port}", open_timeout=1, close_timeout=1, proxy=None) as connection:
        connection.send(json.dumps({"type": "ping"}))
        return json.loads(connection.recv(timeout=1))


def wait_ready(process, port, token, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited ({process.returncode}); see server.log")
        try:
            response = probe_server(port)
        except ConnectionRefusedError:
            response = None
        except Exception as exc:
            raise RuntimeError(f"Port {port} is occupied, incompatible or unreachable") from exc
        if response is not None:
            if not isinstance(response, dict) or response.get("type") != "pong" or response.get("readiness_token") != token:
                raise RuntimeError(f"Port {port} belongs to another server")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Policy startup timed out; see server.log")
        time.sleep(0.2)


@contextmanager
def policy_server(cfg, method):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", cfg.port))
    directory = cfg.root / method / "servers" / str(time.time_ns())
    directory.mkdir(parents=True)
    token = secrets.token_hex(24)
    command = server_command(cfg, method, token)
    write_json(directory / "command.json", command)
    owned = ProcessRegistry()
    try:
        log = (directory / "server.log").open("w")
        process = spawn_resource(command, cwd=ROOT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=cfg.gpu, PYTHONUNBUFFERED="1"), stdout=log)
        owned.add(process, log)
        print(f"Starting {method}; log={directory / 'server.log'}", flush=True)
        wait_ready(process, cfg.port, token, cfg.timeout)
        yield process
    finally:
        owned.terminate_all()


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def joint_timings(root):
    groups = {"cold": [], "warm": []}
    for path in sorted((root / "servers").glob("*/server.log")):
        with path.open(errors="replace") as handle:
            for line in handle:
                if "joint-paraflow request stats=" not in line:
                    continue
                try:
                    stats = ast.literal_eval(line.split("joint-paraflow request stats=", 1)[1].strip())
                    value = float(stats["full_request_ms"])
                    if value >= 0 and value < float("inf"):
                        groups["cold" if stats["cold"] else "warm"].append(value)
                except (ValueError, SyntaxError, TypeError, KeyError):
                    continue
    return {name: {"count": len(values), "p50_ms": percentile(values, .5), "p95_ms": percentile(values, .95)}
            for name, values in groups.items() if values}


def summarize(root):
    recorded = read_json(root / "protocol.json")
    tasks, total = recorded["tasks"], recorded["episodes_per_task"]
    report = {"protocol_identity": recorded["identity"], "diagnostic": recorded["diagnostic"],
              "timing_note": "Joint timings include all attempts, including interrupted/retried tasks; not paired Native speedups.",
              "methods": {}}
    for method in METHODS:
        successes = 0
        episodes = 0
        rows = {}
        for task in tasks:
            try:
                manifest = check_manifest(root / "manifests" / f"{task}.json", task, total)
                result = complete_result(root / method / "results" / f"{task}.json", manifest, total)
            except (OSError, ValueError, KeyError, TypeError):
                result = None
            if result is None:
                rows[task] = {"status": "incomplete"}
            else:
                successes += result["successes"]
                episodes += total
                rows[task] = {"status": "complete", "successes": result["successes"], "episodes": total,
                              "success_rate": result["successes"] / total}
        missing = [task for task, row in rows.items() if row["status"] != "complete"]
        report["methods"][method] = {
            "status": "incomplete" if missing else "complete", "successes": successes, "episodes": episodes,
            "expected_episodes": len(tasks) * total, "tasks_complete": len(tasks) - len(missing),
            "tasks_expected": len(tasks), "missing_tasks": missing,
            "success_rate": successes / episodes if episodes else None, "tasks": rows,
            "joint_request_timing": joint_timings(root / method) if method == "joint_n4_e8" else {},
        }
    write_json(root / "comparison.json", report)
    lines = ["# RoboTwin Joint comparison", "", f"Diagnostic: {recorded['diagnostic']}; mode: demo_clean", "",
             "| Method | Status | Complete tasks | Success/Episodes | Rate (completed tasks only) |",
             "|---|---|---:|---:|---:|"]
    for method, row in report["methods"].items():
        rate = "-" if row["success_rate"] is None else f"{row['success_rate']:.2%}"
        text = f"{row['successes']}/{row['episodes']}"
        print(f"{method:20s} {row['status']:10s} {text} ({rate}); tasks {row['tasks_complete']}/{len(tasks)}", flush=True)
        lines.append(f"| {method} | {row['status']} | {row['tasks_complete']}/{len(tasks)} | {text} | {rate} |")
    lines += ["", "## Per-task successes", "", "| Task | Native N10 | Native N4 | Joint N4 |", "|---|---:|---:|---:|"]
    for task in tasks:
        values = [report["methods"][method]["tasks"][task].get("successes", "incomplete") for method in METHODS]
        lines.append(f"| {task} | " + " | ".join(map(str, values)) + " |")
    lines += ["", report["timing_note"]]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {root / 'comparison.md'}", flush=True)
    return all(row["status"] == "complete" for row in report["methods"].values())


def campaign(cfg):
    preflight(cfg)
    with output_lock(cfg.root):
        check_protocol(cfg.root, protocol(cfg))
        shutil.copyfile(HERE / "policy_config.yml", cfg.root / "policy_config.yml")
        shutil.copyfile(HERE / "step_limits.yml", cfg.root / "step_limits.yml")
        try:
            for task in cfg.tasks:
                path = cfg.root / "manifests" / f"{task}.json"
                if path.exists():
                    check_manifest(path, task, cfg.episodes)
                    continue
                directory = cfg.root / "manifest_attempts" / task / str(time.time_ns())
                print(f"Building manifest: {task}", flush=True)
                run_client(cfg, task, directory)
                manifest = check_manifest(directory / "result.json", task, cfg.episodes)
                write_json(path, manifest)
            for method in METHODS:
                pending = []
                for task in cfg.tasks:
                    path = cfg.root / "manifests" / f"{task}.json"
                    manifest = check_manifest(path, task, cfg.episodes)
                    if complete_result(cfg.root / method / "results" / f"{task}.json", manifest, cfg.episodes) is None:
                        pending.append(task)
                if not pending:
                    print(f"Skipping complete method: {method}", flush=True)
                    continue
                with policy_server(cfg, method) as server:
                    for task in pending:
                        print(f"Evaluating {method}: {task}", flush=True)
                        directory = cfg.root / method / "attempts" / task / str(time.time_ns())
                        path = cfg.root / "manifests" / f"{task}.json"
                        run_client(cfg, task, directory, path, server)
                        result = complete_result(directory / "result.json", check_manifest(path, task, cfg.episodes), cfg.episodes)
                        if result is None:
                            raise RuntimeError(f"Incomplete or invalid results: {directory}")
                        write_json(cfg.root / method / "results" / f"{task}.json", result)
        finally:
            summarize(cfg.root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("plan", "diagnostic", "full", "summary"), default="full")
    args = parser.parse_args(argv)
    cfg = configuration(args.mode)
    print(f"Comparison directory: {cfg.root}", flush=True)
    if args.mode == "plan":
        print("Full: 3 methods x 50 tasks x 100 episodes = 15000 episodes; demo_clean, shared seed-0 manifests.")
        print("Diagnostic: 3 methods x 3 tasks x 5 episodes = 45 episodes; separate default output directory.")
        print("Methods: native_n10_e8, native_n4_e8, joint_n4_e8; sync E8; compile/cache/decode off.")
        return 0
    if args.mode == "summary":
        if not (cfg.root / "protocol.json").is_file():
            raise FileNotFoundError(f"No protocol.json in {cfg.root}; set COMPARE_ROOT to an existing run")
        with output_lock(cfg.root):
            return 0 if summarize(cfg.root) else 1

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        campaign(cfg)
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; owned processes cleaned up. Reuse COMPARE_ROOT to resume.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
