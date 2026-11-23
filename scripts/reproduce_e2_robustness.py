"""Rebuild E2 prerequisites on one GPU, then run bounded robustness.

The source server's outputs are reference inputs only. Model identity checks in
the existing research commands remain mandatory throughout the fresh run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.libero.research import source_identity
from openwam.research.protocol import file_hash, read_sealed, seal, write_json
from scripts.research_e2_robustness import atomic_json, verify_result


LOCAL_CHECKPOINT = Path(
    "/mnt/storage2/users/hycui_data/yyf/OpenWAM/assets/openwam_ckpt/"
    "openwam_alpha/OpenWAM-Alpha-Sim-LIBERO"
)
BASE_NAME = "e2_exploratory_20261113_205159_787207"
BANK_NAME = "h2_states_20261113_175408_491989"
SPLIT_NAME = "h1h2_20261112_223445_299984"
SUBSETS = ("discovery", "calibration")


def select_gpu(rows, occupied, requested="auto"):
    """Choose by free memory, with a stable index tie-break; require an idle GPU."""
    candidates = []
    for row in rows.splitlines():
        index, uuid, name, free = [part.strip() for part in row.split(",")]
        gpu = {"index": int(index), "uuid": uuid, "name": name, "free_mib": int(free)}
        if uuid not in occupied and (requested == "auto" or index == str(requested)):
            candidates.append(gpu)
    if not candidates:
        raise RuntimeError("No idle GPU matches the request; refusing to share a busy GPU")
    return min(candidates, key=lambda gpu: (-gpu["free_mib"], gpu["index"]))


def check_assets(reference, ckpt_dir, ckpt_name):
    """Check every identity-bearing model asset, including unexpected extras."""
    from benchmarks.libero.research import assets_identity

    model = read_sealed(reference / BASE_NAME / "diagnostics/e2_discovery/manifest.json")["model"]
    if model["checkpoint_file"] != ckpt_name:
        raise ValueError("Checkpoint filename differs from the reference")
    actual = assets_identity(SimpleNamespace(ckpt_dir=ckpt_dir))
    expected = model["assets"]
    different = sorted(name for name in actual.keys() | expected.keys()
                       if actual.get(name) != expected.get(name))
    if different:
        raise ValueError(f"Local model asset hashes differ from reference: {different}")
    return {"directory": str(ckpt_dir), "assets": actual,
            "checkpoint_sha256": actual[ckpt_name], "reference_model_sha256": model["sha256"]}


def model_smoke(args):
    import numpy as np
    import torch

    from benchmarks.libero.research import build_server, load_states
    from openwam.research.runner import paired_generate

    server_args = SimpleNamespace(ckpt_dir=str(args.ckpt_dir), ckpt_name=args.ckpt_name,
                                  device="cuda:0", steps=10, horizon=None)
    server = build_server(server_args)
    split = read_sealed(args.reference_root / SPLIT_NAME / "manifests/split.json")
    _, states = load_states(args.reference_root / BANK_NAME / "state_bank/discovery", split)
    # Reference observation only: neither its baseline actions nor identity are reused.
    result, elapsed = paired_generate(server.engine, states[0][1], seed=43, steps=10)
    actions = np.asarray(result["actions"])
    if not np.isfinite(actions).all() or actions.ndim != 2 or actions.shape[0] < 1:
        raise ValueError("Model preflight produced invalid actions")
    write_json(args.run / "checks/model_smoke.json", {
        "status": "passed", "reference_state_id": states[0][0]["state_id"],
        "purpose": "hardware preflight only; not a fresh experimental baseline",
        "actions_shape": list(actions.shape), "diagnostic_server_ms": elapsed,
        "gpu": torch.cuda.get_device_name(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    })


def prepare_representatives(run, split):
    bank = read_sealed(run / "bank/state_bank/discovery/manifest.json")
    methods = [{"method_id": f"representative_L{layer}", "intervention": {
        "kind": "gate", "gate": 0.0, "layer_ids": [layer], "step_ids": [1],
        "query_modality": "action", "source_modality": "future_video"}}
        for layer in (0, 29)]
    write_json(run / "representatives/plan.json", seal({
        "experiment": "e1", "selected_split": "discovery", "fitted_split": "discovery",
        "task_split_sha256": split["sha256"], "state_ids": [bank["states"][0]["state_id"]],
        "purpose": "Observe token layouts at protocol-fixed layers; no outcome ranking",
        "methods": methods,
    }))


def freeze_selection(run, split):
    from openwam.research.exploratory import PROTOCOL, select_states, validate_coverage

    selection = {"fitted_split": "discovery", "task_split_sha256": split["sha256"],
                 "sampling_protocol": PROTOCOL, "layer_ids": [0, 29], "step_ids": [1, 5, 9],
                 "state_ids_by_split": {}, "sampling_coverage": {}, "bank_sha256_by_split": {},
                 "layouts": {}, "representative_evidence": {}}
    source = source_identity()
    for subset in SUBSETS:
        bank = read_sealed(run / f"bank/state_bank/{subset}/manifest.json")
        old = read_sealed(run / f"bank/state_bank/{subset}/model.json")
        if bank["model_sha256"] != old["sha256"] or old["source_sha256"] != source["source_sha256"]:
            raise ValueError("Fresh bank identity/source changed")
        if bank["task_split_sha256"] != split["sha256"]:
            raise ValueError("Fresh bank references a different split")
        ids, coverage = select_states(bank["states"])
        selected = [state for state in bank["states"] if state["state_id"] in ids]
        validate_coverage(selected, PROTOCOL)
        selection["state_ids_by_split"][subset] = ids
        selection["sampling_coverage"][subset] = coverage
        selection["bank_sha256_by_split"][subset] = bank["sha256"]
        write_json(run / f"base/manifests/source_review_{subset}.json", seal({
            "old_model_sha256": old["sha256"], "bank_sha256": bank["sha256"],
            "new_source_sha256": source["source_sha256"], "changes": {},
            "reason": "Fresh bank on the same GPU/runtime/source; no identity transition",
        }))
    for layer in (0, 29):
        directory = run / f"representatives/L{layer}"
        rows = [(path, json.loads(path.read_text())) for path in directory.glob("*.json")]
        rows = [(path, row) for path, row in rows if row.get("method_id") == f"representative_L{layer}"]
        if len(rows) != 1 or rows[0][1].get("status") != "completed" or not rows[0][1].get("valid_run"):
            raise ValueError("Missing or invalid fresh discovery representative")
        path, row = rows[0]
        selection["layouts"][str(layer)] = row["layouts"][str(layer)]
        selection["representative_evidence"][str(layer)] = {"file": str(path), "sha256": file_hash(path)}
    write_json(run / "base/manifests/selection.json", seal(selection))


def validate_baseline(run):
    selection = read_sealed(run / "base/manifests/selection.json")
    for subset in SUBSETS:
        directory = run / f"base/diagnostics/e2_{subset}"
        manifest = read_sealed(directory / "manifest.json")
        plan = read_sealed(directory / "plan.json")
        expected = {(state, method["method_id"]) for state in selection["state_ids_by_split"][subset]
                    for method in plan["methods"]}
        found = set()
        for path in directory.glob("*.json"):
            row = json.loads(path.read_text())
            if "method_id" not in row:
                continue
            key = row["state_id"], row["method_id"]
            if key not in expected or key in found or row["plan_sha256"] != plan["sha256"]:
                raise ValueError("Unexpected, duplicate or incompatible fresh baseline result")
            verify_result(row, *key, manifest["model"]["sha256"])
            found.add(key)
        if len(found) != 2304 or found != expected:
            raise ValueError(f"Incomplete fresh E2 baseline: {subset}, {len(found)}/2304")


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.run = args.run
        self.source = source_identity()["source_sha256"]
        self.completed = []
        self.current = "initializing"

    def status(self, status, **extra):
        atomic_json(self.run / "pipeline_status.json", {
            "status": status, "stage": self.current, "completed_stages": self.completed,
            "pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(), **extra,
        })

    def stage(self, name, command=None, operation=None, env=None):
        if source_identity()["source_sha256"] != self.source:
            raise ValueError("Research source changed during the experiment")
        self.current = name
        self.status("running")
        print(f"[{datetime.now().isoformat()}] Stage: {name}", flush=True)
        if command:
            print("Command:", json.dumps([str(part) for part in command]), flush=True)
            subprocess.run([str(part) for part in command], check=True, cwd=ROOT, env=env)
        if operation:
            operation()
        self.completed.append(name)
        self.status("running")

    def research(self, name, verb, *arguments):
        self.stage(name, [sys.executable, "-m", "benchmarks.libero.research", verb, *arguments])

    def execute(self):
        args, run = self.args, self.run
        rows = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.free", "--format=csv,noheader,nounits"
        ], text=True)
        busy = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"
        ], text=True)
        gpu = select_gpu(rows, set(busy.splitlines()), args.gpu)
        with (ROOT / f".cache/e2-reproduction-{gpu['uuid']}.lock").open("a") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.environ.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], MUJOCO_EGL_DEVICE_ID=str(gpu["index"]))
            os.environ["GPU"] = str(gpu["index"])
            print(f"GPU selected: {gpu}", flush=True)
            packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
            write_json(run / "environment.json", {"python": sys.version, "packages": packages,
                                                  "gpu": gpu, "source_sha256": self.source,
                                                  "checkpoint_dir": str(args.ckpt_dir)})
            self.stage("verify_local_assets", operation=lambda: write_json(
                run / "checks/local_assets.json", check_assets(args.reference_root, args.ckpt_dir, args.ckpt_name)))
            split_file = args.reference_root / SPLIT_NAME / "manifests/split.json"
            split = read_sealed(split_file)
            write_json(run / "manifests/split.json", split)
            common = ["--ckpt-dir", args.ckpt_dir, "--ckpt-name", args.ckpt_name, "--split", split_file]
            self.stage("egl_smoke", ["bash", "benchmarks/libero/run_smoke.sh", "env"],
                       env={**os.environ, "LIBERO_PYTHON": sys.executable})
            self.stage("model_smoke", [sys.executable, __file__, "--model-smoke", "--run", run,
                       "--ckpt-dir", args.ckpt_dir, "--ckpt-name", args.ckpt_name,
                       "--reference-root", args.reference_root])
            for subset in SUBSETS:
                self.research(f"collect_{subset}", "collect", *common, "--selected-split", subset,
                              "--seeds", 43, "--episodes", 10, "--output", run / f"bank/state_bank/{subset}")
                self.research(f"t0_{subset}", "t0", *common, "--bank", run / f"bank/state_bank/{subset}",
                              "--seeds", 43, 44, 45, "--output", run / f"base/diagnostics/t0_{subset}")
                certificate = read_sealed(run / f"base/diagnostics/t0_{subset}/t0_certificate.json")
                if certificate["status"] != "passed":
                    raise ValueError(f"Fresh T0 failed for {subset}")
            self.stage("representative_plan", operation=lambda: prepare_representatives(run, split))
            for layer in (0, 29):
                self.research(f"representative_L{layer}", "smoke", *common,
                              "--bank", run / "bank/state_bank/discovery", "--seeds", 43,
                              "--plan", run / "representatives/plan.json", "--method-id", f"representative_L{layer}",
                              "--output", run / f"representatives/L{layer}")
            self.stage("freeze_selection", operation=lambda: freeze_selection(run, split))
            for subset in ("calibration", "discovery"):
                self.research(f"features_{subset}", "features", *common,
                              "--bank", run / f"bank/state_bank/{subset}", "--selected-split", subset,
                              "--selection", run / "base/manifests/selection.json",
                              "--source-review", run / f"base/manifests/source_review_{subset}.json",
                              "--t0", run / f"base/diagnostics/t0_{subset}/t0_certificate.json", "--seeds", 43,
                              "--output", run / f"base/diagnostics/features_{subset}")
            for subset in SUBSETS:
                self.research(f"plan_{subset}", "plan", "--experiment", "e2", "--layers", 30,
                              "--selected-split", subset, "--split", split_file,
                              "--selection", run / "base/manifests/selection.json",
                              "--calibration", run / "base/diagnostics/features_calibration/calibration.json",
                              "--output", run / f"base/manifests/e2_{subset}.json")
                replay = [*common, "--bank", run / f"bank/state_bank/{subset}",
                          "--source-review", run / f"base/manifests/source_review_{subset}.json",
                          "--plan", run / f"base/manifests/e2_{subset}.json", "--seeds", 43]
                self.research(f"smoke_{subset}", "smoke", *replay, "--method-id", "E2_L0_S1_G0_sign-1",
                              "--output", run / f"base/diagnostics/smoke_{subset}")
                self.research(f"baseline_{subset}", "replay", *replay,
                              "--t0", run / f"base/diagnostics/t0_{subset}/t0_certificate.json",
                              "--output", run / f"base/diagnostics/e2_{subset}")
                self.research(f"summarize_{subset}", "summarize", "--runs", run / f"base/diagnostics/e2_{subset}",
                              "--features", run / f"base/diagnostics/features_{subset}/features.json",
                              "--calibration", run / "base/diagnostics/features_calibration/calibration.json",
                              "--output", run / f"base/tables/e2_{subset}")
            self.stage("validate_baseline", operation=lambda: validate_baseline(run))
            robust = ["bash", "scripts/research_alpha_libero_e2_robustness.sh", "--base", run / "base",
                      "--bank-run", run / "bank", "--split", split_file, "--ckpt-dir", args.ckpt_dir,
                      "--ckpt-name", args.ckpt_name]
            environment = {**os.environ, "RUN": str(run / "robustness"), "HOURS": str(args.hours)}
            self.stage("robustness_prepare", robust, env={**environment, "PREPARE_ONLY": "1"})
            self.stage("robustness", robust, env={**environment, "PREPARE_ONLY": "0"})
            final = json.loads((run / "robustness/status.json").read_text())
            self.status(final["status"], robustness=final)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu", default=os.environ.get("GPU", "auto"))
    overlay = ROOT / ".cache/models/OpenWAM-Alpha-Sim-LIBERO"
    default_checkpoint = overlay if overlay.is_dir() else LOCAL_CHECKPOINT
    parser.add_argument("--ckpt-dir", type=Path, default=Path(os.environ.get("CKPT_DIR", str(default_checkpoint))))
    parser.add_argument("--ckpt-name", default="checkpoint_step_10690.safetensors")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "experiment/reference/experiment")
    parser.add_argument("--hours", type=float, default=float(os.environ.get("HOURS", "10")))
    parser.add_argument("--model-smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error("--hours must be positive and finite")
    if args.gpu != "auto" and not args.gpu.isdigit():
        parser.error("--gpu must be auto or a physical numeric index")
    args.run, args.ckpt_dir, args.reference_root = args.run.resolve(), args.ckpt_dir.resolve(), args.reference_root.resolve()
    if args.model_smoke:
        model_smoke(args)
        return
    args.run.mkdir(parents=True, exist_ok=True)
    (ROOT / ".cache").mkdir(exist_ok=True)
    with (args.run / ".pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.run / "pipeline_status.json").exists():
            raise ValueError("Run already started; use a new RUN directory to preserve previous evidence")
        pipeline = Pipeline(args)
        def interrupted(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, interrupted)
        try:
            pipeline.execute()
        except BaseException as error:
            pipeline.status("failed", error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()
