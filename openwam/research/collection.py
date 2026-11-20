"""LIBERO development-state collection through the existing policy adapters."""

from __future__ import annotations

import copy
import importlib.metadata
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from openwam.research.protocol import (
    digest,
    file_hash,
    progress_indices,
    read_sealed,
    require_development,
    seal,
    write_json,
)


class EpisodeRecorder:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = metadata
        self.requests = []

    def record(self, conditions, actions, snapshot):
        images = conditions.get("first_frame_image")
        if not isinstance(images, (list, tuple)) or len(images) != 1:
            raise ValueError("Collection requires exactly one composed first-frame image")
        image = np.asarray(images[0])
        proprio = np.asarray(conditions.get("proprio"), dtype=np.float32)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Collection requires uint8 RGB")
        if proprio.ndim != 1 or not np.isfinite(proprio).all():
            raise ValueError("Collection requires finite proprio")
        index = len(self.requests)
        path = self.directory / f"request_{index:05d}.npz"
        arrays = {"image": image, "proprio": proprio, "prompt": np.array(conditions.get("prompt", "")),
                  "actions": np.asarray(actions)}
        if snapshot.get("sim_state") is not None:
            arrays["sim_state"] = np.asarray(snapshot.pop("sim_state"))
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        snapshot_path = path.with_suffix(".json")
        write_json(snapshot_path, snapshot)
        self.requests.append((path, snapshot_path))

    def finish(self, destination, outcome):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        entries = []
        selected = set(progress_indices(len(self.requests)))
        for index, (path, snapshot_path) in enumerate(self.requests):
            if index not in selected:
                path.unlink()
                snapshot_path.unlink()
                continue
            state_id = f"{self.metadata['suite']}_{self.metadata['task_id']}_ep{self.metadata['episode_id']}_r{index}"
            target = destination / f"{state_id}.npz"
            if target.exists():
                raise FileExistsError(target)
            path.rename(target)
            snapshot_target = target.with_suffix(".snapshot.json")
            snapshot_path.rename(snapshot_target)
            entries.append({**self.metadata, **outcome, "state_id": state_id, "request_index": index,
                            "phase": "progress_proxy", "progress": index / max(1, len(self.requests) - 1),
                            "file": target.name, "sha256": file_hash(target),
                            "snapshot_file": snapshot_target.name, "snapshot_sha256": file_hash(snapshot_target),
                            "open_loop_replayable": True, "closed_loop_restore_verified": False})
        write_json(self.directory / "episode.json", {**self.metadata, **outcome, "requests": len(self.requests),
                                                     "selected_state_ids": [entry["state_id"] for entry in entries]})
        return entries


class LocalPolicyClient:
    def __init__(self, server):
        self.server = server

    def ping(self):
        return {"type": "pong", **self.server._pong_payload()}

    def predict(self, payload):
        return self.server.predict(payload)

    def reset(self):
        self.server.reset()
        return {"type": "reset_ack"}

    def close(self):
        return None


def rng_snapshot():
    import torch

    numpy = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
            "torch_cpu": torch.get_rng_state().tolist(),
            "torch_cuda": [value.tolist() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def simulator_identity():
    root = Path(os.environ["LIBERO_PATH"]).resolve()
    files = {str(path.relative_to(root)): file_hash(path) for path in sorted((root / "libero").rglob("*.py"))}
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    versions = {}
    for name in ("libero", "mujoco", "robosuite", "bddl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_packaged_or_missing"
    return {"libero_root": str(root), "libero_commit": result.stdout.strip() if result.returncode == 0 else None,
            "libero_python_sources_sha256": digest(files), "libero_python_sources": files, "versions": versions}


def collect(args):
    import yaml

    from benchmarks.libero.research import build_server, model_identity

    if not 1 <= args.episodes <= 10 or len(args.seeds) != 1:
        raise ValueError("Collection requires 1–10 episodes/task and exactly one policy seed")
    split = read_sealed(args.split)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    server = build_server(args)
    identity = model_identity(server, args)
    write_json(root / "model.json", identity)
    cfg = yaml.safe_load(Path(args.policy_config).read_text())
    if cfg.get("action_mode") != "eef" or cfg.get("action_clip") is not None or cfg.get("action_indices") is not None:
        raise ValueError("Research collection requires the canonical unclipped EEF adapter")
    benchmark_dir = str(Path(__file__).resolve().parents[2] / "benchmarks/libero")
    if benchmark_dir not in sys.path:
        sys.path.insert(0, benchmark_dir)
    from benchmarks.libero.openwam2libero_interface import OpenWAMLiberoPolicy
    from benchmarks.libero.single_eval import (
        _make_env_with_randomization_retries,
        _parse_settle_action,
        _write_libero_config,
    )

    _write_libero_config()
    from libero.libero import benchmark

    from benchmarks.libero.eval_manifest import task_assets

    environment = seal({"simulator": simulator_identity(), "policy_config": cfg,
                        "policy_config_sha256": file_hash(args.policy_config)})
    write_json(root / "environment.json", environment)

    engine = server.engine
    original_generate = engine.generate
    current = {}
    entries = []

    def recorded_generate(conditions):
        request = copy.deepcopy(conditions)
        request["seed"] = args.seeds[0]
        snapshot = {"rng": rng_snapshot(), "policy_seed": args.seeds[0], "action_buffer": [],
                    "controller_state": "not_serialized", "environment_rng": "global_numpy_recorded; internal_rng_unverified",
                    "task_progress": current["step"], "closed_loop_restore_verified": False}
        try:
            snapshot["sim_state"] = current["env"].sim.get_state().flatten()
        except AttributeError:
            snapshot["sim_state"] = None
        result = original_generate(request)
        if not np.isfinite(result["actions"]).all():
            raise FloatingPointError("Non-finite generated action chunk")
        current["recorder"].record(request, result["actions"], snapshot)
        return result

    engine.generate = recorded_generate
    client = OpenWAMLiberoPolicy(_client=LocalPolicyClient(server),
                                head_camera_key=cfg.get("head_camera_key", "agentview_image"),
                                left_wrist_camera_key=cfg.get("left_wrist_camera_key", "robot0_eye_in_hand_image"),
                                right_wrist_camera_key=cfg.get("right_wrist_camera_key"),
                                image_transform=cfg.get("image_transform", "rotate_180"),
                                state_dim=int(cfg.get("state_dim", 10)))
    try:
        tasks = [task for task in split["tasks"] if task["split"] == args.selected_split]
        shard_count = getattr(args, "task_shard_count", 1)
        shard_index = getattr(args, "task_shard_index", 0)
        if not 0 <= shard_index < shard_count:
            raise ValueError("Invalid collection task shard")
        for task_record in tasks[shard_index::shard_count]:
            task_suite = benchmark.get_benchmark_dict()[task_record["suite"]]()
            task = task_suite.get_task(task_record["task_id"])
            if task.name != task_record["task_name"]:
                raise ValueError("Installed LIBERO task ordering differs from frozen split")
            init_states = task_suite.get_task_init_states(task_record["task_id"])
            init_path, bddl_path = task_assets(task)
            task_hashes = {"init_state_asset_sha256": file_hash(init_path), "bddl_sha256": file_hash(bddl_path)}
            if len(init_states) < args.episodes:
                raise ValueError("Insufficient distinct initial states")
            env = _make_env_with_randomization_retries(task, cfg)
            env_seed = int(cfg.get("seed", 42))
            env.seed(env_seed)
            try:
                for episode in range(args.episodes):
                    metadata = {**task_record, "task_split_sha256": split["sha256"], "episode_id": episode,
                                "initial_state_id": episode, "env_seed": env_seed,
                                "action_seed": args.seeds[0], "video_seed": args.seeds[0],
                                "model_sha256": identity["sha256"], "environment_sha256": environment["sha256"], **task_hashes}
                    require_development(split, metadata)
                    recorder = EpisodeRecorder(root / "episodes" / f"{task_record['suite']}_{task_record['task_id']}_{episode}", metadata)
                    outcome = {"success": False, "timeout": False, "policy_failure": False, "valid_run": True}
                    current.update(env=env, recorder=recorder, step=0)
                    try:
                        env.reset()
                        obs = env.set_init_state(init_states[episode])
                        for _ in range(int(cfg.get("settle_steps", 30))):
                            obs, _, _, _ = env.step(_parse_settle_action(cfg.get("settle_action")))
                        client.reset()
                        max_steps = int(cfg.get("max_steps_by_suite", {}).get(task_record["suite"], 600))
                        for step in range(max_steps):
                            current["step"] = step
                            action = client.act(obs, task.language)
                            if not np.isfinite(action).all():
                                raise FloatingPointError("Non-finite runtime action")
                            obs, _, done, _ = env.step(action)
                            if done:
                                outcome["success"] = True
                                break
                        outcome["timeout"] = not outcome["success"]
                    except FloatingPointError as error:
                        outcome.update(policy_failure=True, error_type=type(error).__name__, error=str(error))
                    except Exception as error:
                        outcome.update(valid_run=False, error_type=type(error).__name__, error=str(error))
                        raise
                    finally:
                        entries.extend(recorder.finish(root, outcome))
                        write_json(root / f"partial_{task_record['suite']}_{task_record['task_id']}_{episode}.json", entries)
            finally:
                env.close()
    finally:
        engine.generate = original_generate
        client.close()
        server._policy.shutdown()
    write_json(root / "manifest.json", seal({"schema_version": 1, "task_split_sha256": split["sha256"],
                                             "model_sha256": identity["sha256"], "states": entries,
                                             "environment_sha256": environment["sha256"],
                                             "protocol": "development_progress_proxy; not official full evaluation"}))
