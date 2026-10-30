"""Run the original ParaFlow benchmark and Joint as explicitly separate workloads."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MODES = ("eager", "cuda_graph", "model_fused", "model_deep", "model_deep_tuned")
SHARED = Path("/data2/jikangye/research/runs/paraflow/hardware-ablation-20261017/wam-port")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_command(python, options, output, *, warmup, samples, steps):
    return [python, "-m", "paraflow.infer.benchmark", "--factory", "paraflow.infer.openwam.replay:create_case",
            "--options", str(options), "--steps", *map(str, steps), "--warmup", str(warmup),
            "--samples", str(samples), "--require-validation", "--require-exclusive", "--output", str(output)]


def summarize(output):
    lines = ["# Joint / original ParaFlow optimization comparison", "",
             "These are separate workload panels, NOT an algorithm-equivalent head-to-head benchmark.",
             "Different checkpoint, attention semantics and input/return boundaries prohibit cross-panel speedup claims.",
             "Original ParaFlow execution modes below are current API modes, not the historical cumulative C1-C5 rows.", "",
             "## Original ParaFlow: first-frame / isolated", "",
             "| Mode | N | Samples | P50 ms | P95 ms | Eager-relative P50 ratio |", "|---|---:|---:|---:|---:|---:|"]
    records = {}
    for mode in MODES:
        path = output / f"paraflow_{mode}.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt.get("completed"):
                for row in receipt["results"]:
                    records[mode, row["steps"]] = row
    for (mode, steps), row in records.items():
        reference = records.get(("eager", steps))
        ratio = f"{reference['median_ms'] / row['median_ms']:.3f}" if reference else "not_run"
        lines.append(f"| {mode} | {steps} | {len(row['samples_ms'])} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} | {ratio} |")
    lines += ["", "## Current Joint: full future-video / mutual", "",
              "| Method | Samples | P50 ms | P95 ms |", "|---|---:|---:|---:|"]
    path = output / "joint" / "timing.json"
    joint = json.loads(path.read_text()) if path.exists() else None
    if joint:
        for name, row in joint["methods"].items():
            value = row["wall_ms"]
            lines.append(f"| {name} | {value['count']} | {value['p50']:.3f} | {value['p95']:.3f} |")
        lines += ["", f"Joint numerical equivalence validated: {joint.get('numerical_equivalence_validated', False)}."]
    else:
        lines += ["", "Joint timing: not_run or incomplete."]
    lines += ["", "Original: official raw-camera replay, first-frame path, N4/N10 H32/E8, per-request preprocessing included.",
              "Joint: fixed real composed-canvas trace, H32/E8, phase cold requests excluded from warm measurements.",
              "48 warmup / 120 samples by default, same explicitly selected GPU, exclusive-process checks.",
              "Separate processes and workload differences remain confounders. No closed-loop quality claim.",
              "For a causal optimization comparison, compare modes WITHIN each panel at the same N.",
              "A true same-workload head-to-head requires an original full-mutual backend or a validated first-frame port;",
              "neither is silently manufactured by this runner."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "preflight", "all", "summary"))
    args = parser.parse_args(argv)
    output = Path(os.environ.get("COMPARE_ROOT", f"outputs/joint_original_paraflow_{time.strftime('%Y%m%d_%H%M%S')}" )).resolve()
    if args.mode == "plan":
        print("Original ParaFlow official replay: eager/cuda_graph/model_fused/model_deep/model_deep_tuned, N4 and N10, H32/E8.")
        print("Then current full-mutual Joint replay; grouped report, no cross-semantics speedup ratio.")
        print("Required: TRACE. Original PF runtime/checkpoint/observation default to the local read-only shared wam-port installation.")
        print("Overrides: PF_PYTHON, PF_NATIVE_ROOT, PF_OPTIONS (official factory JSON).")
        print("Optional: PF_ROOT, GPU, COMPARE_ROOT, CKPT_DIR, CKPT_NAME, PF_MODES, PF_STEPS, ALLOW_PARITY_FAILURE.")
        print("No GPU work in plan/preflight. Default warmup=48, samples=120. all starts explicit GPU experiments.")
        return
    if args.mode == "summary":
        summarize(output)
        return
    required = ("TRACE",)
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError("Missing explicit original runtime/assets: " + ", ".join(missing))
    pf_root = Path(os.environ.get("PF_ROOT", "/home/zhijian/paraflow-main")).resolve()
    native = Path(os.environ.get("PF_NATIVE_ROOT", str(SHARED / "native/openwam"))).resolve()
    python = str(Path(os.environ.get("PF_PYTHON", str(SHARED / "runtime/wam/bin/python"))).absolute())
    options_path = Path(os.environ["PF_OPTIONS"]).resolve() if os.environ.get("PF_OPTIONS") else None
    trace = Path(os.environ["TRACE"]).resolve()
    for path in (Path(python), trace, pf_root / "src/paraflow/infer/openwam/replay.py",
                 native / "openwam/model/architectures/base.py"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if native == ROOT:
        raise ValueError("PF_NATIVE_ROOT must point to the original-compatible native runtime, not this Joint checkout")
    options = json.loads(options_path.read_text()) if options_path else {
        "checkpoint": str(SHARED / "assets/openwam"),
        "observation": str(SHARED / "original-options/openwam-observation.npz"),
        "operators": {"plan": "accepted", "refine_graph": True, "gpu_pixels": True, "early_vae": True},
    }
    for key in ("checkpoint", "observation"):
        path = Path(options[key]).expanduser()
        if not path.is_absolute() or not path.exists():
            raise ValueError(f"Original factory {key} must be an existing absolute path")
    weight = Path(options["checkpoint"]) / options.get("checkpoint_name", "checkpoint_step_9890.safetensors")
    if not weight.is_file():
        raise FileNotFoundError(weight)
    modes = os.environ.get("PF_MODES", ",".join(MODES)).split(",")
    if len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes) or "eager" not in modes:
        raise ValueError("PF_MODES must contain unique supported modes including eager")
    steps = [int(value) for value in os.environ.get("PF_STEPS", "4,10").split(",")]
    if len(set(steps)) != len(steps) or any(value < 2 for value in steps):
        raise ValueError("PF_STEPS must contain unique N>=2 values")
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
    env["PYTHONPATH"] = os.pathsep.join([str(pf_root / "src"), str(native), str(native.parent)])
    probe = ("import inspect,paraflow,openwam; from pathlib import Path; "
             f"assert Path(inspect.getfile(openwam)).resolve().is_relative_to(Path({str(native)!r})); "
             f"assert Path(inspect.getfile(paraflow)).resolve().is_relative_to(Path({str(pf_root / 'src')!r})); "
             "print('paraflow=',inspect.getfile(paraflow)); print('openwam=',inspect.getfile(openwam))")
    subprocess.run([python, "-c", probe], cwd=native, env=env, check=True)
    print("Asset/import preflight passed; GPU/model/kernel validation remains not_run.", flush=True)
    if args.mode == "preflight":
        return
    from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource

    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "pf_root": str(pf_root), "pf_native_root": str(native),
                "python": python, "options": options,
                "options_sha256": hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest(),
                "trace_sha256": digest(trace),
                "source_sha256": {name: digest(pf_root / name) for name in
                                  ("src/paraflow/infer/benchmark.py", "src/paraflow/infer/openwam/replay.py")},
                "same_workload": False, "cross_panel_speedup_valid": False, "closed_loop": "not_run"}
    save(output / "manifest.json", manifest)
    registry = ProcessRegistry()

    def run(command, cwd, environment, log_name):
        log = (output / log_name).open("x")
        print(f"Running {log_name}; output: {output}", flush=True)
        process = spawn_resource(command, cwd=cwd, env=environment, stdout=log)
        registry.add(process, log)
        if process.wait() != 0:
            raise RuntimeError(f"Experiment failed: see {output / log_name}")

    try:
        for mode in modes:
            local = dict(options, mode=mode)
            local_path = output / f"options_{mode}.json"
            save(local_path, local)
            run(original_command(python, local_path, output / f"paraflow_{mode}.json", warmup=48, samples=120, steps=steps),
                native, env, f"paraflow_{mode}.log")
        joint_env = dict(os.environ)
        joint_env.update(CUDA_VISIBLE_DEVICES=env["CUDA_VISIBLE_DEVICES"], DEVICE="cuda:0", TRACE=str(trace),
                         PROFILE_ROOT=str(output / "joint"), WARMUP="48", MEASURE="120")
        joint_env.pop("OPENWAM_REQUEST_TRACE_DIR", None)
        run(["bash", "scripts/profile_alpha_joint.sh", "all", "--require-exclusive"], ROOT, joint_env, "joint.log")
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        registry.terminate_all()
        save(output / "manifest.json", manifest)
        summarize(output)


if __name__ == "__main__":
    main()
