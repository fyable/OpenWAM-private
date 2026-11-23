"""H1/H2 research CLI. Run with python -m benchmarks.libero.research --help."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
from pathlib import Path

import numpy as np

from openwam.research.protocol import (
    REFERENCE_COMMIT,
    cumulative_configs,
    digest,
    e1_configs,
    file_hash,
    read_sealed,
    require_development,
    retention_configs,
    seal,
    token_groups,
    write_json,
)


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(path for directory in ("openwam", "benchmarks/libero", "configs", "tests")
                   for path in (root / directory).rglob("*") if path.suffix in {".py", ".yaml", ".yml"})
    files = {str(path.relative_to(root)): file_hash(path) for path in paths}
    return {"reference_commit": REFERENCE_COMMIT, "upstream_equivalence_verified": False,
            "source_files": files, "source_sha256": digest(files)}


def assets_identity(args):
    root = Path(args.ckpt_dir)
    if not root.exists():
        return {}
    paths = []

    def raise_walk_error(error):
        raise error

    for directory, subdirectories, filenames in os.walk(root, onerror=raise_walk_error):
        subdirectories[:] = [name for name in subdirectories if not name.startswith(".")]
        for name in filenames:
            path = Path(directory) / name
            if not name.startswith(".") and path.suffix in {
                ".json", ".yaml", ".yml", ".safetensors", ".pt", ".pth", ".npy",
            }:
                paths.append(path)
    paths.sort()
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def preflight(args):
    import torch

    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("torch", "omegaconf", "safetensors", "libero", "scipy", "matplotlib")}
    result = {**source_identity(), "python": platform.python_version(), "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(), "dependencies": dependencies,
              "assets": assets_identity(args), "status": "not_run", "model_loaded": False,
              "reason": "Static preflight only; model identity is checked by t0/replay after loading"}
    result["missing_items"] = [name for name, available in dependencies.items() if not available]
    if not result["cuda_available"]:
        result["missing_items"].append("functional_cuda")
    if not any(Path(name).suffix in {".safetensors", ".pt", ".pth"} for name in result["assets"]):
        result["missing_items"].append("checkpoint_weights")
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name()
    write_json(args.output, result)


def build_server(args):
    import torch
    from omegaconf import OmegaConf

    from benchmarks.libero.replay_joint import resolve_checkpoint
    from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

    if not torch.cuda.is_available():
        raise RuntimeError("not_run: CUDA unavailable; refusing simulated model results")
    cfg = _load_deploy_yaml()
    for key, value in {"inference.sampler": "native", "inference.denoise_steps": args.steps,
                       "inference.inference_mode": "sync", "inference.denoise_mode": "sync",
                       "optimization.compile.enabled": False, "optimization.dit_cache.enabled": False,
                       "optimization.decode_video": False}.items():
        OmegaConf.update(cfg, key, value)
    if args.horizon is not None:
        OmegaConf.update(cfg, "inference.inference_horizon", args.horizon)
    checkpoint = resolve_checkpoint(args.ckpt_dir, args.ckpt_name)
    args.resolved_checkpoint = checkpoint
    server = build_server_from_config(cfg, args.ckpt_dir, device=args.device, ckpt_name=checkpoint)
    from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
    if not isinstance(server.engine.architecture, DualSystemSelfAttnArchitecture):
        raise ValueError("Research requires dual_system joint_self_attn")
    if server._pong_payload().get("representation") != "eef" or server.engine.architecture.normalizer is None:
        raise ValueError("Research requires a normalized LIBERO task checkpoint with the native EEF contract")
    if server.engine._cfg_scale != 1:
        raise ValueError("Research requires CFG=1")
    server._init_policy()
    return server


def config_identity(cfg):
    from omegaconf import OmegaConf

    unresolved = OmegaConf.to_container(cfg, resolve=False)
    resolved = {}
    for key in unresolved:
        if key in {"project", "training"}:
            continue
        value = cfg[key]
        resolved[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return {"config_unresolved": unresolved, "resolved_config": resolved,
            "resolved_config_excluded_sections": ["project", "training"],
            "resolved_config_sha256": digest(resolved)}


def model_identity(server, args):
    import torch

    from openwam.research.protocol import active_indices

    architecture = server.engine.architecture
    driver = architecture._mot_driver or architecture.build_mot_driver()
    assets = assets_identity(args)
    return seal({**source_identity(), "assets": assets, "torch_device": str(architecture.device),
                 "memory_mode": getattr(architecture, "research_memory_mode", "resident"),
                 "dtype": str(architecture.dtype), "architecture": type(architecture).__name__,
                 "num_layers": driver.num_layers, "num_heads": driver.num_heads, "head_dim": driver.head_dim,
                 "attention_mask_mode": driver.attention_mask_mode,
                 "video_attention_mask_mode": getattr(architecture.video_backbone, "video_attention_mask_mode", None),
                 "active_action_indices": active_indices(architecture),
                 "binary_command_dims": list(getattr(architecture, "binary_command_dims", ()) or ()),
                 **config_identity(server.cfg),
                 "checkpoint_file": args.resolved_checkpoint,
                 "checkpoint_sha256": assets[args.resolved_checkpoint],
                 "hardware_and_software": {"python": platform.python_version(), "platform": platform.platform(),
                                           "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                                           "gpu": torch.cuda.get_device_name(architecture.device)
                                           if torch.cuda.is_available() else None,
                                           "attention_environment": {key: os.environ.get(key) for key in
                                               ("WAM_ATTENTION_IMPL", "DIFFSYNTH_ATTENTION_IMPLEMENTATION")}},
                 "steps": args.steps, "execution_horizon": server._policy._executor.inference_horizon})


def load_states(bank, split):
    from PIL import Image

    manifest = read_sealed(Path(bank) / "manifest.json")
    if manifest["task_split_sha256"] != split["sha256"]:
        raise ValueError("Bank belongs to a different split")
    result = []
    seen = set()
    for entry in manifest["states"]:
        if entry["state_id"] in seen or not entry.get("valid_run", True):
            raise ValueError("Duplicate state ID or infrastructure-invalid state in bank")
        seen.add(entry["state_id"])
        require_development(split, entry)
        path = Path(bank) / entry["file"]
        if path.resolve().parent != Path(bank).resolve() or file_hash(path) != entry["sha256"]:
            raise ValueError("Invalid state file or hash")
        with np.load(path, allow_pickle=False) as data:
            image, proprio = data["image"], data["proprio"]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("State image must be uint8 RGB")
            if proprio.ndim != 1 or not np.isfinite(proprio).all():
                raise ValueError("State proprio must be a finite vector")
            conditions = {"first_frame_image": [Image.fromarray(image)],
                          "prompt": str(data["prompt"].item()), "proprio": proprio.copy()}
        result.append((entry, conditions))
    return manifest, result


def execution_horizon(server, conditions, args):
    horizon = server._policy._executor.inference_horizon
    if horizon is not None:
        return int(horizon)
    frames = getattr(server.cfg.inference, "num_frames", None)
    if frames is None:
        raise ValueError("Cannot resolve full execution horizon; provide --horizon matching the baseline")
    return int(frames) - 1


def validate_t0(certificate, identity, bank, steps):
    if (certificate["status"] != "passed" or certificate["states"] < 20 or len(set(certificate["seeds"])) != 3
        or certificate["model_sha256"] != identity["sha256"]
        or certificate["bank_sha256"] != bank["sha256"] or certificate["steps"] != steps):
        raise ValueError("Screening requires passing T0 for this model/source/bank/protocol")


def replay(args, t0=False, smoke=False):
    from openwam.research.runner import execute_method, infrastructure_retries, run_t0

    split = read_sealed(args.split)
    bank, states = load_states(args.bank, split)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    server = build_server(args)
    identity = model_identity(server, args)
    from openwam.research.exploratory import validate_bank_identity
    validate_bank_identity(args, bank, identity)
    write_json(output / "manifest.json", seal({"model": identity, "bank_sha256": bank["sha256"],
                                               "task_split_sha256": split["sha256"], "seeds": args.seeds,
                                               "source_review": read_sealed(args.source_review) if args.source_review else None}))
    if not states:
        raise ValueError("Empty state bank")
    horizon = execution_horizon(server, states[0][1], args)
    if t0:
        if len(states) < 20 or len(set(args.seeds)) != 3:
            raise ValueError("T0 requires at least 20 observations and exactly 3 distinct seeds")
        selected = [(entry["state_id"], condition) for entry, condition in states[:20]]
        recorded_outputs = {}
        for entry, _ in states[:20]:
            if entry["action_seed"] not in args.seeds or entry["action_seed"] != entry["video_seed"]:
                raise ValueError("T0 seeds must include each state's original coupled policy seed")
            with np.load(Path(args.bank) / entry["file"], allow_pickle=False) as data:
                recorded_outputs[entry["state_id"]] = (entry["action_seed"], data["actions"].copy())
        report = run_t0(server.engine, selected, seeds=args.seeds, steps=args.steps,
                        horizon=horizon, output=output / "t0.json", recorded_outputs=recorded_outputs)
        report.update(model_sha256=identity["sha256"], source_sha256=identity["source_sha256"],
                      bank_sha256=bank["sha256"], task_split_sha256=split["sha256"])
        write_json(output / "t0_certificate.json", seal(report))
        return
    if not smoke:
        validate_t0(read_sealed(args.t0), identity, bank, args.steps)
    plan = read_sealed(args.plan)
    if plan.get("task_split_sha256") != split["sha256"]:
        raise ValueError("Plan belongs to a different task split")
    if plan.get("selected_split") not in {"discovery", "calibration"}:
        raise ValueError("Plan must select a development split")
    selected = [(entry, condition) for entry, condition in states if entry["split"] == plan["selected_split"]]
    if "episode_ids" in plan:
        selected = [(entry, condition) for entry, condition in selected if entry["episode_id"] in plan["episode_ids"]]
    if plan.get("state_ids") is not None:
        ids = set(plan["state_ids"])
        selected = [(entry, condition) for entry, condition in selected if entry["state_id"] in ids]
        if {entry["state_id"] for entry, _ in selected} != ids:
            raise ValueError("Plan contains missing or out-of-split state IDs")
    if smoke:
        if len(args.seeds) != 1:
            raise ValueError("Smoke requires exactly one seed")
        selected = selected[:1]
        methods = [method for method in plan["methods"] if method["method_id"] == args.method_id]
        if len(methods) != 1:
            raise ValueError("Smoke method ID must uniquely identify a plan method")
    else:
        methods = plan["methods"]
    if plan.get("experiment") == "e2" and not smoke:
        from openwam.research.exploratory import validate_coverage
        validate_coverage([entry for entry, _ in selected], plan.get("sampling_protocol"))
    if not selected:
        raise ValueError("No states selected")
    write_json(output / "plan.json", plan)
    for entry, conditions in selected:
        for seed in args.seeds:
            for method in methods:
                row = {**entry, "method_id": method["method_id"], "action_seed": seed, "video_seed": seed,
                       "steps": args.steps, "prefix_horizon": horizon, "success": None,
                       "plan_sha256": plan["sha256"], "model_sha256": identity["sha256"]}
                path = output / f"{digest([entry['state_id'], seed, method['method_id']])}.json"
                try:
                    result, attempts = infrastructure_retries(lambda: execute_method(
                        server.engine, conditions, method, seed=seed, steps=args.steps, horizon=horizon))
                    row.update(valid_run=True, status="completed", retry_attempts=attempts, **result)
                except (ValueError, FloatingPointError) as error:
                    row.update(valid_run=isinstance(error, FloatingPointError), status="policy_failure" if isinstance(error, FloatingPointError)
                               else "configuration_error", error_type=type(error).__name__, error=str(error))
                    write_json(path, row)
                    raise
                except Exception as error:
                    row.update(valid_run=False, status="infrastructure_error" if isinstance(error, OSError)
                               else "unclassified_runtime_error", error_type=type(error).__name__,
                               error=str(error), retry_attempts=getattr(error, "research_attempts", []))
                    write_json(path, row)
                    raise
                write_json(path, row)


def make_plan(args):
    split = read_sealed(args.split)
    if args.experiment == "e1":
        methods = e1_configs(args.layers, args.steps)
        methods += [{"method_id": "B0", "steps": 10}, {"method_id": "B1", "steps": 10,
                    "cache": {"cosine_threshold": 0.99, "max_consecutive_skips": 3}}]
        methods += [{"method_id": f"B2_{steps}", "steps": steps} for steps in (4, 6, 8)]
    elif args.experiment in {"retention", "cumulative"}:
        ranking = read_sealed(args.selection)
        if ranking["task_split_sha256"] != split["sha256"] or ranking["fitted_split"] != "discovery":
            raise ValueError("Position ranking must be fitted on Discovery with this split")
        methods = retention_configs(list(range(args.layers)), list(range(args.steps)), ranking["positions"])
        if args.experiment == "cumulative":
            methods = cumulative_configs(ranking["positions"])
    else:
        selection = read_sealed(args.selection)
        calibration = read_sealed(args.calibration)
        if (selection["task_split_sha256"] != split["sha256"] or selection["fitted_split"] != "discovery"
            or calibration["task_split_sha256"] != split["sha256"] or calibration["fitted_split"] != "calibration"):
            raise ValueError("E2 requires Discovery selection and Calibration feature scale")
        state_ids = selection.get("state_ids_by_split", {}).get(args.selected_split, selection.get("state_ids", []))
        if len(state_ids) != 24 or len(set(state_ids)) != 24:
            raise ValueError("Initial E2 screening requires 24 explicitly frozen, distinct state IDs")
        if len(selection["layouts"]) != 2 or len(selection["step_ids"]) != 3:
            raise ValueError("Initial E2 screening requires 2 representative layers and 3 step positions")
        if (len(set(selection["step_ids"])) != 3
            or set(map(int, selection["layouts"])) != set(selection["layer_ids"])):
            raise ValueError("E2 selection must have distinct steps and consistent representative layer IDs")
        methods = []
        for layer, layout in selection["layouts"].items():
            for step in selection["step_ids"]:
                for group_id, tokens in enumerate(token_groups(layout)):
                    for sign in (-1, 1):
                        methods.append({"method_id": f"E2_L{layer}_S{step}_G{group_id}_sign{sign}",
                                        "intervention": {"layer_ids": [int(layer)], "step_ids": [step],
                                                         "kind": "value_noise", "token_ids": tokens,
                                                         "epsilon": 0.03, "floor": calibration["floor"], "sign": sign}})
    result = {"experiment": args.experiment, "task_split_sha256": split["sha256"],
              "selected_split": args.selected_split, "methods": methods}
    if args.experiment == "e1":
        result["episode_ids"] = list(range(5 if args.selected_split == "discovery" else 10))
    if args.selection:
        selection = read_sealed(args.selection)
        if "state_ids_by_split" in selection:
            result["state_ids"] = selection["state_ids_by_split"][args.selected_split]
        elif "state_ids" in selection:
            result["state_ids"] = selection["state_ids"]
        result["selection_sha256"] = selection["sha256"]
        if "sampling_protocol" in selection:
            result["sampling_protocol"] = selection["sampling_protocol"]
            result["sampling_coverage"] = selection["sampling_coverage"][args.selected_split]
    if args.calibration:
        result["calibration_sha256"] = read_sealed(args.calibration)["sha256"]
    write_json(args.output, seal(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--ckpt-dir", required=True)
    pre.add_argument("--output", required=True)
    split = commands.add_parser("split")
    split.add_argument("--task-map", required=True, help="JSON suite -> task names in original ID order")
    split.add_argument("--output", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--input", required=True, help="Unsealed selection/ranking JSON")
    freeze.add_argument("--split", required=True)
    freeze.add_argument("--output", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--experiment", choices=("e1", "e2", "retention", "cumulative"), required=True)
    plan.add_argument("--layers", type=int, required=True)
    plan.add_argument("--steps", type=int, default=10)
    plan.add_argument("--split", required=True)
    plan.add_argument("--selected-split", choices=("discovery", "calibration"), default="discovery")
    plan.add_argument("--selection")
    plan.add_argument("--calibration")
    plan.add_argument("--output", required=True)
    for name in ("t0", "replay", "smoke", "collect", "features"):
        sub = commands.add_parser(name)
        sub.add_argument("--ckpt-dir", required=True)
        sub.add_argument("--ckpt-name")
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--steps", type=int, default=10)
        sub.add_argument("--horizon", type=int)
        sub.add_argument("--split", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
        sub.add_argument("--source-review", help="Sealed, explicitly reviewed source-only bank transition")
        if name != "collect":
            sub.add_argument("--bank", required=True)
        if name in {"replay", "smoke"}:
            sub.add_argument("--plan", required=True)
        if name in {"replay", "features"}:
            sub.add_argument("--t0", required=True)
        if name == "smoke":
            sub.add_argument("--method-id", required=True)
        if name == "features":
            sub.add_argument("--selection", required=True)
            sub.add_argument("--selected-split", choices=("discovery", "calibration"), required=True)
        if name == "collect":
            sub.add_argument("--task-shard-index", type=int, default=0)
            sub.add_argument("--task-shard-count", type=int, default=1)
            sub.add_argument("--policy-config", default="benchmarks/libero/policy_config.yml")
            sub.add_argument("--selected-split", choices=("discovery", "calibration"), required=True)
            sub.add_argument("--episodes", type=int, default=10)
    summary = commands.add_parser("summarize")
    summary.add_argument("--runs", required=True)
    summary.add_argument("--output", required=True)
    summary.add_argument("--features")
    summary.add_argument("--calibration")
    args = parser.parse_args()
    if args.command == "collect" and not 0 <= args.task_shard_index < args.task_shard_count:
        parser.error("Invalid collection task shard")
    if hasattr(args, "steps") and args.steps < 1:
        parser.error("--steps must be positive")
    if hasattr(args, "horizon") and args.horizon is not None and args.horizon < 1:
        parser.error("--horizon must be positive")
    if hasattr(args, "seeds") and (len(set(args.seeds)) != len(args.seeds)
                                  or any(seed < 0 or seed >= 2**32 for seed in args.seeds)):
        parser.error("--seeds must be distinct uint32 values")
    if args.command == "preflight":
        preflight(args)
    elif args.command == "split":
        from openwam.research.protocol import split_tasks
        write_json(args.output, split_tasks(json.loads(Path(args.task_map).read_text())))
    elif args.command == "plan":
        make_plan(args)
    elif args.command == "freeze":
        value = json.loads(Path(args.input).read_text())
        split = read_sealed(args.split)
        if value.get("fitted_split") not in {"discovery", "calibration"} or "sha256" in value:
            raise ValueError("Freeze requires explicit development provenance and unsealed input")
        if value.get("task_split_sha256", split["sha256"]) != split["sha256"]:
            raise ValueError("Selection belongs to a different task split")
        value["task_split_sha256"] = split["sha256"]
        write_json(args.output, seal(value))
    elif args.command in {"t0", "replay", "smoke"}:
        replay(args, t0=args.command == "t0", smoke=args.command == "smoke")
    elif args.command == "collect":
        from openwam.research.collection import collect
        collect(args)
    elif args.command == "features":
        from openwam.research.features import collect_features
        collect_features(args)
    else:
        from openwam.research.analysis import summarize
        summarize(args.runs, args.output, features=args.features, calibration=args.calibration)


if __name__ == "__main__":
    main()
