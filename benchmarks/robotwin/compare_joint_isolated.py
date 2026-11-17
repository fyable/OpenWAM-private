"""Four-arm RoboTwin comparison with checkpoint-matched attention semantics."""

import argparse
import copy
import os
import signal
from pathlib import Path

from benchmarks.robotwin import compare_joint as base


METHODS = {"joint_native_n10": ("native", 10), "joint_parallel_n10": ("joint-paraflow", 10),
           "isolated_native_n10": ("isolated-native", 10), "isolated_paraflow_n10": ("isolated-paraflow", 10)}
EXECUTION_METHODS = {"joint_native_n10": ("native", 10), "joint_parallel_n10": ("joint-paraflow", 10),
                     "joint_prepared_n10": ("joint-paraflow", 10), "joint_prefix_n10": ("joint-paraflow", 10),
                     "isolated_paraflow_n10": ("isolated-paraflow", 10)}
ISOLATED = "/data2/jikangye/research/runs/paraflow/hardware-ablation-20261017/wam-port/assets/openwam"


def configuration(mode):
    cfg = base.configuration(mode)
    execution = mode in ("execution-ablation", "execution-regression", "capture")
    cfg.root = Path(os.environ.get("COMPARE_ROOT", str(base.ROOT / "outputs/robotwin" /
                    ("joint_isolated_full" if mode == "full" else "joint_isolated_diagnostic")))).resolve()
    if execution and "COMPARE_ROOT" not in os.environ:
        cfg.root = base.ROOT / "outputs/robotwin" / ("joint_execution_" + mode.replace("execution-", ""))
    cfg.checkpoint = Path(os.environ.get("JOINT_CKPT_DIR", str(cfg.checkpoint))).resolve()
    cfg.methods = ({"joint_parallel_n10": ("joint-paraflow", 10)} if mode == "capture" else EXECUTION_METHODS if execution else METHODS)
    if mode != "full":
        default_tasks = ",".join(base.DIAGNOSTIC_TASKS) if execution else "adjust_bottle"
        cfg.tasks = os.environ.get("TASKS", default_tasks).split(",")
        cfg.episodes = int(os.environ.get("EPISODES", "20" if mode == "execution-regression" else "5"))
        if cfg.episodes <= 0 or not set(cfg.tasks).issubset(base.all_tasks()):
            raise ValueError("EPISODES must be positive and TASKS must contain RoboTwin task names")
    cfg.method_configs = {}
    if mode == "execution-regression":
        cfg.methods = {key: EXECUTION_METHODS[key] for key in ("joint_parallel_n10", "joint_prefix_n10")}
    requested_methods = os.environ.get("METHODS")
    if requested_methods:
        names = [name.strip() for name in requested_methods.split(",")]
        if not all(name in cfg.methods for name in names) or len(set(names)) != len(names):
            raise ValueError(f"METHODS must select unique names from: {', '.join(cfg.methods)}")
        cfg.methods = {name: cfg.methods[name] for name in names}
    cfg.diagnostic = mode != "full"
    for method in cfg.methods:
        selected = copy.copy(cfg)
        selected.expected_mask = "isolated" if method.startswith("isolated_") else "mutual"
        if selected.expected_mask == "isolated":
            selected.checkpoint = Path(os.environ.get("ISOLATED_CKPT_DIR", ISOLATED)).resolve()
        backend = {"joint_prepared_n10": "prepared", "joint_prefix_n10": "prefix"}.get(method, "legacy")
        selected.deploy_overrides = [f"inference.joint_execution_backend={backend}"] if method.startswith("joint_") else []
        if execution:
            selected.deploy_overrides.append("inference.request_timing=true")
            if os.environ.get("ALLOW_PARITY_FAILURE", "0") == "1":
                selected.deploy_overrides.append("inference.joint_execution_allow_parity_failure=true")
        if mode == "capture":
            selected.server_environment = {"OPENWAM_REQUEST_TRACE_DIR": str(cfg.root / "trace_sessions"),
                                           "OPENWAM_TRACE_WARMUP": os.environ.get("WARMUP", "48"),
                                           "OPENWAM_TRACE_MEASURE": os.environ.get("MEASURE", "120")}
        cfg.method_configs[method] = selected
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("plan", "preflight", "diagnostic", "full", "summary",
                                                   "execution-ablation", "execution-regression", "capture"), default="diagnostic")
    args = parser.parse_args(argv)
    cfg = configuration(args.mode)
    if args.mode == "plan":
        print("Four methods: " + ", ".join(METHODS))
        print("Default diagnostic: adjust_bottle, 5 episodes x 4 = 20 episodes; N10/H32/E8.")
        print("Full: 50 tasks x 100 episodes x 4 = 20000 episodes.")
        print("Joint checkpoint:", cfg.checkpoint)
        print("Isolated checkpoint:", cfg.method_configs["isolated_native_n10"].checkpoint)
        print("Same scene manifests; different trained checkpoints, not cross-mask algorithm speedups.")
        print("Output:", cfg.root)
        print("execution-ablation: 5 methods x 3 tasks x 5 = 75 episodes; execution-regression: legacy/prefix x 3 x 20 = 120.")
        return 0
    if args.mode == "summary":
        with base.output_lock(cfg.root):
            return 0 if base.summarize(cfg.root) else 1
    from openwam.deploy.isolated_reference import load_reference

    cfg.reference_sources = load_reference()[-1]
    if args.mode == "preflight":
        for selected in cfg.method_configs.values():
            base.preflight(selected)
        print("Checkpoint masks/assets/imports passed; GPU/model/operator validation not_run.")
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        base.campaign(cfg)
        if args.mode == "capture":
            from openwam.deploy.request_trace import assemble_trace

            trace = cfg.root / "observations.npz"
            if not trace.exists():
                sessions = sorted((cfg.root / "trace_sessions").glob("session_*/complete.json"))
                if not sessions:
                    raise RuntimeError("Capture incomplete: no session has enough warm requests; increase EPISODES or TASKS")
                assemble_trace(sessions[0].parent, trace)
            base.write_json(trace.with_suffix(".json"), {"benchmark": "robotwin", "source": "actual_E8_requests",
                            "checkpoint": str(cfg.weight), "checkpoint_config": base.digest(cfg.checkpoint / "config.yaml"),
                            "checkpoint_size": cfg.weight.stat().st_size, "checkpoint_mtime_ns": cfg.weight.stat().st_mtime_ns,
                            "trace_sha256": base.digest(trace), "tasks": cfg.tasks})
            print("Captured RoboTwin trace:", trace)
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
