"""Reuse sealed benchmark runners without changing their hashed sources."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from .experiment import ROOT, digest, write_json
from .runtime import method_spec


def parse_methods(value):
    names = value.split(",")
    if len(set(names)) != len(names) or "native_n10" not in names:
        raise ValueError("Use unique methods including native_n10")
    for name in names:
        method_spec(name)
    return names


def campaign_identity(args):
    from benchmarks.libero.replay_joint import resolve_checkpoint

    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [ROOT / "scripts/video_parallel_python.sh"]
    checkpoint = args.ckpt_dir.resolve()
    if args.benchmark == "robotwin":
        weight = max(checkpoint.glob("checkpoint_step_*.safetensors"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    else:
        weight = checkpoint / resolve_checkpoint(checkpoint, os.environ.get("CKPT_NAME"))
    return {"benchmark": args.benchmark, "methods": args.methods, "stage": args.stage,
            "overrides": args.overrides, "tasks": args.tasks, "gpus": args.gpus,
            "checkpoint": str(weight), "weight_size": weight.stat().st_size, "weight_mtime_ns": weight.stat().st_mtime_ns,
            "checkpoint_config_sha256": digest(checkpoint / "config.yaml"),
            "normalization_sha256": digest(checkpoint / "normalization_stats.npy"), "server_python": sys.executable,
            "sources": {str(path.relative_to(ROOT)): digest(path) for path in files}}


def libero(args):
    wrapper = ROOT / "scripts/video_parallel_python.sh"
    args.output.mkdir(parents=True, exist_ok=True)
    identity = campaign_identity(args)
    identity_file = args.output / "research_identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError("Research protocol changed; use a fresh output directory")
    write_json(identity_file, identity)
    try:
        for name in args.methods:
            steps, spec = method_spec(name)
            config = args.overrides if spec is not None and spec != "legacy" else {}
            environment = dict(os.environ, SERVER_PYTHON=str(wrapper), VIDEO_PARALLEL_PYTHON=sys.executable,
                               VIDEO_PARALLEL_METHOD=name, VIDEO_PARALLEL_OVERRIDES=json.dumps(config),
                               CKPT_DIR=str(args.ckpt_dir.resolve()), CKPT_NAME=Path(identity["checkpoint"]).name,
                               GPUS=args.gpus, OUTPUT_DIR=str(args.output / name))
            command = ["bash", str(ROOT / "scripts/eval_alpha_libero.sh"), "eval", "--sampler", "native",
                   "--denoise-steps", str(steps), "--inference-horizon", "8", "--inference-mode", "sync",
                   "--denoise-mode", "sync", "--extra-deploy-arg", "inference.cfg_scale=1.0",
                   "--extra-deploy-arg", "optimization.dit_cache.enabled=false",
                   "--extra-deploy-arg", "optimization.decode_video=false",
                   "--extra-deploy-arg", "inference.request_timing=true"]
            if args.stage == "screen":
                command += ["--diagnostic", "--num-trials", "5", "--task-ids", "0,1"]
            else:
                command += ["--num-trials", "50"]
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    finally:
        summarize(args.output)


def robotwin(args):
    from benchmarks.robotwin import compare_joint as base

    if "," in args.gpus:
        raise ValueError("RoboTwin launcher uses one GPU; choose --gpus with one index")
    config = base.configuration("diagnostic")
    config.root, config.checkpoint, config.gpu = args.output.resolve(), args.ckpt_dir.resolve(), args.gpus
    config.sim_gpu = os.environ.get("SIM_GPU", args.gpus)
    config.server_python = str(ROOT / "scripts/video_parallel_python.sh")
    config.episodes = 10 if args.stage == "screen" else 50
    tasks = args.tasks.split(",") if args.tasks else base.all_tasks()[:10]
    if len(set(tasks)) != len(tasks) or not set(tasks) <= set(base.all_tasks()):
        raise ValueError("Tasks must be unique RoboTwin task names")
    config.tasks = tasks
    config.methods = {name: ("native", method_spec(name)[0]) for name in args.methods}
    config.reference_sources = campaign_identity(args)
    config.deploy_overrides = ["inference.request_timing=true"]
    config.method_configs = {}
    for name in args.methods:
        selected = copy.copy(config)
        _, spec = method_spec(name)
        selected.server_environment = {"VIDEO_PARALLEL_PYTHON": sys.executable, "VIDEO_PARALLEL_METHOD": name,
                                       "VIDEO_PARALLEL_OVERRIDES": json.dumps(args.overrides if spec is not None and spec != "legacy" else {})}
        config.method_configs[name] = selected
    base.campaign(config)


def summarize(root):
    from .quality import clustered_interval

    root = Path(root)
    identity = json.loads((root / "research_identity.json").read_text())
    methods, tasks = {}, {}
    for name in identity["methods"]:
        path = root / name / "summary.json"
        if not path.exists():
            methods[name] = {"status": "pending"}
            continue
        measured = json.loads(path.read_text())
        methods[name] = measured["overall"]
        tasks[name] = {(row["suite"], row["task_id"]): row for row in measured["task_results"] if row["complete"]}
    reference = tasks.get("native_n10", {})
    paired = {}
    for name, rows in tasks.items():
        common = sorted(set(reference) & set(rows))
        differences = [{"scene": str(key), "difference": rows[key]["success_rate"] - reference[key]["success_rate"]} for key in common]
        interval = clustered_interval(differences, "difference")
        paired[name] = {"common_tasks": len(common), "difference": interval,
                        "noninferiority_margin": .02,
                        "interpretation": "Descriptive task-cluster bootstrap; incomplete and selected-task runs do not establish benchmark noninferiority."}
    result = {"methods": methods, "paired": paired, "closed_loop_timing": "not a formal speed comparison"}
    write_json(root / "quality_comparison.json", result)
    lines = ["# Joint video-parallel closed-loop comparison", "", "Missing trials are not failures. Timing requires separate exclusive replay.", "",
             "| Method | Successes | Trials | Complete tasks | Expected tasks |", "|---|---:|---:|---:|---:|"]
    for name, values in methods.items():
        lines.append(f"| {name} | {values.get('successes', 'pending')} | {values.get('trials', 'pending')} | {values.get('tasks_complete', 'pending')} | {values.get('tasks_expected', 'pending')} |")
    (root / "quality_comparison.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("libero", "robotwin"), default="libero")
    parser.add_argument("--stage", choices=("screen", "full"), default="screen")
    parser.add_argument("--methods", type=parse_methods, default=parse_methods("native_n10,native_n6,legacy_n10,adaptive"))
    parser.add_argument("--ckpt-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tasks", help="Predeclared RoboTwin task list")
    parser.add_argument("--overrides", type=json.loads, default={})
    parser.add_argument("--summary", action="store_true", help="Regenerate LIBERO quality comparison without running models")
    args = parser.parse_args(argv)
    if not isinstance(args.overrides, dict):
        parser.error("--overrides must be an object")
    if args.summary:
        summarize(args.output)
        return
    from .solver import SolverConfig
    from dataclasses import asdict

    for name in args.methods:
        _, spec = method_spec(name)
        if spec is not None and spec != "legacy":
            SolverConfig(**{**asdict(spec), **args.overrides})
    if args.ckpt_dir is None:
        args.ckpt_dir = Path("/data1/wam-ckpts/OpenWAM-Alpha-Sim-" + ("LIBERO" if args.benchmark == "libero" else "RoboTwin-Full"))
    if args.benchmark == "libero" and args.tasks:
        parser.error("--tasks is RoboTwin-only; LIBERO uses fixed suites and task IDs")
    (libero if args.benchmark == "libero" else robotwin)(args)


if __name__ == "__main__":
    main()
