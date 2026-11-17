import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openwam.research.collection import EpisodeRecorder, LocalPolicyClient
from openwam.research.protocol import (
    SUITES,
    action_metrics,
    e1_configs,
    progress_indices,
    read_sealed,
    require_development,
    retention_configs,
    split_tasks,
    token_groups,
    write_json,
)
from openwam.research.runner import paired_generate, run_t0


def test_config_identity_preserves_training_metadata_without_resolving_now():
    from omegaconf import OmegaConf

    from benchmarks.libero.research import config_identity

    cfg = OmegaConf.create({"project": {"output_dir": "outputs/${now:%Y-%m-%d_%H-%M-%S}"},
                            "training": {"unused": "${unregistered:unused}"},
                            "model": {"width": 32}, "inference": {"width": "${model.width}"}})
    first = config_identity(cfg)
    assert first == config_identity(cfg)
    assert first["config_unresolved"]["project"]["output_dir"] == "outputs/${now:%Y-%m-%d_%H-%M-%S}"
    assert first["resolved_config"]["inference"]["width"] == 32
    assert "project" not in first["resolved_config"]
    assert "training" not in first["resolved_config"]
    assert OmegaConf.to_container(cfg, resolve=False) == first["config_unresolved"]
    cfg.model.width = 64
    assert config_identity(cfg)["resolved_config_sha256"] != first["resolved_config_sha256"]


def test_config_identity_keeps_runtime_resolution_strict():
    from omegaconf import OmegaConf
    from omegaconf.errors import UnsupportedInterpolationType

    from benchmarks.libero.research import config_identity

    cfg = OmegaConf.create({"inference": {"steps": "${unregistered:runtime}"}})
    with pytest.raises(UnsupportedInterpolationType):
        config_identity(cfg)


def test_assets_identity_prunes_download_cache_and_hashes_normalization(tmp_path, monkeypatch):
    from benchmarks.libero import research

    (tmp_path / ".cache/huggingface/trees").mkdir(parents=True)
    (tmp_path / ".cache/huggingface/trees/private.json").write_text("cache")
    (tmp_path / "tokenizer").mkdir()
    for name in ("config.yaml", "checkpoint.safetensors", "normalization_stats.npy", "tokenizer/config.json"):
        (tmp_path / name).write_bytes(b"asset")
    hashed = []

    def checked_hash(path):
        assert ".cache" not in path.parts
        hashed.append(path.relative_to(tmp_path).as_posix())
        return "hash"

    monkeypatch.setattr(research, "file_hash", checked_hash)
    result = research.assets_identity(SimpleNamespace(ckpt_dir=tmp_path))
    assert set(result) == {"config.yaml", "checkpoint.safetensors", "normalization_stats.npy", "tokenizer/config.json"}
    assert len(hashed) == 4


def test_assets_identity_does_not_hide_required_asset_permission_error(tmp_path, monkeypatch):
    from benchmarks.libero import research

    (tmp_path / "config.yaml").write_text("model: {}")

    def unreadable(path):
        raise PermissionError(str(path))

    monkeypatch.setattr(research, "file_hash", unreadable)
    with pytest.raises(PermissionError):
        research.assets_identity(SimpleNamespace(ckpt_dir=tmp_path))


def test_split_frozen_and_heldout_protected(tmp_path):
    task_map = {suite: [f"task_{suite}_{index}" for index in range(10)] for suite in SUITES}
    split = split_tasks(task_map)
    assert split == split_tasks(task_map)
    assert [sum(task["split"] == group for task in split["tasks"]) for group in
            ("discovery", "calibration", "heldout")] == [8, 8, 24]
    identity = {(task["suite"], task["task_id"]) for task in split["tasks"]}
    assert len(identity) == 40
    for task in split["tasks"]:
        metadata = {**task, "task_split_sha256": split["sha256"]}
        if task["split"] == "heldout":
            with pytest.raises(ValueError, match="refuse"):
                require_development(split, metadata)
        else:
            require_development(split, metadata)
    path = tmp_path / "split.json"
    write_json(path, split)
    assert read_sealed(path) == split
    with pytest.raises(FileExistsError):
        write_json(path, split)
    changed = json.loads(path.read_text())
    changed["seed"] = 0
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hash"):
        read_sealed(path)


def test_screen_and_retention_positions():
    configs = e1_configs(30)
    assert len(configs) == 24
    for direction in ("action", "future_video"):
        positions = [(layer, step) for config in configs if config["intervention"]["query_modality"] == direction
                     for layer in config["intervention"]["layer_ids"] for step in config["intervention"]["step_ids"]]
        assert len(set(positions)) == len(positions) == 300
    positions = [(layer, step) for layer in range(4) for step in range(3)]
    methods = retention_configs(list(range(4)), list(range(3)), positions)
    assert len(methods) == 21
    for method in methods:
        assert len(method["kept_positions"]) + len(method["removed_positions"]) == 12
    assert methods == retention_configs(list(range(4)), list(range(3)), positions)


def test_latent_grid_groups_protect_prefix():
    layout = {"tokens_per_frame": 12, "prefix_tokens": 12, "video_tokens": 60,
              "grid_height": 3, "grid_width": 4}
    groups = token_groups(layout)
    assert len(groups) <= 8
    tokens = [token for group in groups for token in group]
    assert sorted(tokens) == list(range(12, 60))


@pytest.mark.parametrize("count", range(12))
def test_progress_sampling_short_trajectories(count):
    values = progress_indices(count)
    assert len(values) <= min(count, 3)
    assert len(values) == len(set(values))
    assert all(0 <= index < count for index in values)


def test_episode_failure_kept_and_snapshot_honest(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "episode", {"suite": "libero_goal", "task_id": 2, "episode_id": 0})
    for index in range(5):
        recorder.record({"first_frame_image": [np.zeros((4, 4, 3), dtype=np.uint8)],
                         "proprio": np.zeros(10), "prompt": "task"}, np.zeros((32, 10)),
                        {"sim_state": np.array([index]), "closed_loop_restore_verified": False})
    entries = recorder.finish(tmp_path / "states", {"success": False, "timeout": True})
    assert len(entries) == 3
    assert all(not entry["success"] and not entry["closed_loop_restore_verified"] for entry in entries)
    assert not list((tmp_path / "episode").glob("*.npz"))


def test_local_transport_preserves_policy_handshake_and_server_contract():
    from benchmarks.libero.openwam2libero_interface import OpenWAMLiberoPolicy
    resets = []
    server = SimpleNamespace(_pong_payload=lambda: {"representation": "eef"},
                             reset=lambda: resets.append(True), predict=lambda payload: {"action": [0] * 10})
    client = LocalPolicyClient(server)
    policy = OpenWAMLiberoPolicy(_client=client)
    policy.reset()
    assert resets == [True]
    assert len(client.predict({})["action"]) == 10


def test_action_metrics_ignore_inactive_and_runtime_rotation_conversion():
    normalized = np.zeros((4, 80))
    actual = normalized.copy()
    actual[:, 79] = 100
    native = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 1], (4, 1)).astype(np.float32)
    metrics = action_metrics(normalized, actual, native, native, active=list(range(10)), horizon=2)
    assert all(value == 0 for value in metrics.values())
    changed = native.copy()
    changed[-1, 9] = -1
    metrics = action_metrics(normalized, normalized, native, changed, active=list(range(10)), horizon=2)
    assert metrics["prefix_gripper_disagreement"] == 0
    assert metrics["full_gripper_disagreement"] == 0.25


class FakeEngine:
    def __init__(self, fail=False):
        self.architecture = SimpleNamespace(device="cpu")
        self._dit_cache = object()
        self._vace_cache = {"leak": True}
        self._prompt_embed_cache = {"leak": True}
        self.fail = fail
        self.requests = []

    def reset(self):
        return None

    def generate(self, request):
        assert not self._vace_cache and not self._prompt_embed_cache
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("branch failed")
        request["proprio"][0] = 100
        return {"actions": torch.randn(3, 10).numpy()}


def test_paired_rng_reset_input_and_exception_isolation():
    engine = FakeEngine()
    conditions = {"proprio": np.zeros(10)}
    original_cache = engine._dit_cache
    rng = torch.get_rng_state().clone()
    first, _ = paired_generate(engine, conditions, seed=42)
    second, _ = paired_generate(engine, conditions, seed=42)
    np.testing.assert_array_equal(first["actions"], second["actions"])
    assert conditions["proprio"][0] == 0
    assert engine._dit_cache is original_cache
    assert torch.equal(rng, torch.get_rng_state())


def test_branch_exception_restores_state():
    engine = FakeEngine(fail=True)
    original_cache = engine._dit_cache
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="branch failed"):
        paired_generate(engine, {"proprio": np.zeros(10)}, seed=42)
    assert engine._dit_cache is original_cache
    assert torch.equal(rng, torch.get_rng_state())


def test_t0_checks_stored_baseline_and_noop(tmp_path):
    class ConsumingEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.architecture.video_backbone = SimpleNamespace(num_layers=2)
            self.architecture.binary_command_dims = ()

        def generate(self, request):
            result = super().generate(request)
            context = request.get("research_context")
            if context is not None:
                context.begin(request["denoise_steps"], 2)
                context.records.append({"consumed": True})
                context.normalized_actions = torch.from_numpy(result["actions"])
            return result

    engine = ConsumingEngine()
    conditions = {"proprio": np.zeros(10)}
    result, _ = paired_generate(engine, conditions, seed=42)
    baseline = {"state": (42, result["actions"])}
    report = run_t0(engine, [("state", conditions)], seeds=[42, 43, 44], steps=10, horizon=2,
                    output=tmp_path / "pass.json", recorded_outputs=baseline)
    assert report["status"] == "passed"
    baseline["state"] = (42, result["actions"] + 1)
    report = run_t0(engine, [("state", conditions)], seeds=[42, 43, 44], steps=10, horizon=2,
                    output=tmp_path / "fail.json", recorded_outputs=baseline)
    assert report["status"] == "failed"
