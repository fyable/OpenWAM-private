import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional

from openwam.deploy.isolated_reference import IsolatedReferenceSampler, configure_isolated, load_reference


class Video:
    num_layers = 2
    video_attention_mask_mode = "first_frame_causal"
    dit = SimpleNamespace(fuse_vae_embedding_in_latents=True)

    def prepare(self, **values):
        latent = values["latents"]
        return SimpleNamespace(hidden_states=latent.permute(0, 2, 3, 4, 1).flatten(1, 3),
                               grid_frames=latent.shape[2], context=values["context"].mean((1, 2))[:, None, None])

    def pre_attn_at_layer(self, depth, state):
        hidden = state.hidden_states
        return torch.tanh(hidden), torch.sin(hidden), hidden * .7, hidden

    def post_attn_at_layer(self, depth, state, output, previous):
        state.hidden_states = torch.tanh(previous + .3 * output + state.context)
        return state

    def finalize(self, state):
        return state.hidden_states.unflatten(1, (state.grid_frames, 1, 2)).permute(0, 4, 1, 2, 3)


class Action:
    def prepare_state(self, actions, timestep, context, context_mask):
        return SimpleNamespace(payload=SimpleNamespace(x_action=actions, time=timestep[:, None, None] / 1000,
                                                      context=context.mean((1, 2))[:, None, None]))

    def pre_attn_at_layer(self, depth, state):
        hidden = state.payload.x_action + .2 * state.payload.time
        return torch.tanh(hidden), torch.sin(hidden), hidden * .6, state.payload.x_action

    def post_attn_at_layer(self, depth, state, output, previous):
        state.payload.x_action = torch.tanh(previous + .4 * output + state.payload.context)
        return state

    def extract_prediction(self, state):
        return state.payload.x_action * .5


class Driver:
    attention_mask_mode = "isolated"

    def __init__(self, video, action):
        self.video, self.action = video, action

    def _video_tokens_per_frame(self, state):
        return 2

    def _build_attention_mask(self, video_count, action_count, prefix, *, device):
        mask = torch.ones(video_count + action_count, video_count + action_count, device=device, dtype=torch.bool)
        mask[:video_count, video_count:] = False
        mask[:prefix, prefix:video_count] = False
        mask[video_count:, prefix:video_count] = False
        return mask

    def _mixed_attention(self, query, key, value, attn_mask=None):
        return functional.scaled_dot_product_attention(query[:, None], key[:, None], value[:, None], attn_mask=attn_mask)[:, 0]

    def step(self, depth, video, action, attn_mask):
        video_query, video_key, video_value, video_post = self.video.pre_attn_at_layer(depth, video)
        action_query, action_key, action_value, action_post = self.action.pre_attn_at_layer(depth, action)
        output = self._mixed_attention(torch.cat((video_query, action_query), 1), torch.cat((video_key, action_key), 1),
                                       torch.cat((video_value, action_value), 1), attn_mask)
        count = video_query.shape[1]
        return (self.video.post_attn_at_layer(depth, video, output[:, :count], video_post),
                self.action.post_attn_at_layer(depth, action, output[:, count:], action_post))


@pytest.fixture
def model():
    load_reference()
    architecture = type("DualSystemSelfAttnArchitecture", (SimpleNamespace,), {})()
    architecture.dtype = torch.float32
    architecture.video_backbone = Video()
    architecture.action_backbone = Action()
    architecture.mot_driver = Driver(architecture.video_backbone, architecture.action_backbone)
    architecture.video_scheduler = architecture.action_scheduler = SimpleNamespace(num_train_timesteps=1000)
    architecture._append_proprio_context_token = lambda values, proprio: dict(values, context=torch.cat((values["context"], proprio[:, None]), 1))
    return architecture


def request(prefix=1.):
    latent = torch.ones(1, 8, 3, 1, 2)
    latent[:, :, :1] = prefix
    return {"latents": latent, "first_frame_latents": latent[:, :, :1].clone(), "context": torch.ones(1, 2, 8),
            "proprio": torch.ones(1, 8), "seq_lens": torch.tensor([2])}


def invoke(sampler, model, prefix=1.):
    schedule = [(float(value), float(value)) for value in torch.linspace(1000, 0, 11)]
    return sampler.sample(model, request(prefix), torch.ones(1, 32, 8), schedule,
                          torch.tensor([False] * 6 + [True] * 2), prompt="task", seed=42)


def test_reference_import_is_read_only_and_does_not_shadow_openwam():
    import openwam
    before = openwam.__file__
    backend, _, _, identity = load_reference()
    assert "/paraflow-main/" in backend.__file__
    assert openwam.__file__ == before
    assert "openwam.deploy.denoising_common" not in sys.modules
    assert "openwam.deploy.history_denoising" not in sys.modules
    assert identity and all(Path(path).is_file() for path in identity)
    assert not any("wam-port/native" in value for value in sys.path)


@pytest.mark.parametrize("method", ["native", "paraflow"])
def test_reference_lifecycle_budget_constraints_and_current_condition(model, method):
    sampler = IsolatedReferenceSampler(method=method)
    video, action = invoke(sampler, model)
    assert sampler.validation["status"] == "passed"
    assert sampler.last_stats["cold"]
    assert sampler.last_stats["future_video_block_calls"] == 0
    assert sampler.last_stats["visual_prefills"] == 1
    assert sampler.last_stats["action_block_calls"] == 20
    assert torch.all(action[..., 6:] == 0)
    expected = action.clone()
    video, action = invoke(sampler, model, prefix=2.)
    assert not sampler.last_stats["cold"]
    assert sampler.last_stats["action_block_calls"] == (20 if method == "native" else 2)
    assert sampler.last_stats["video_prefill_block_calls"] == 2
    assert not torch.equal(expected, action)
    if method == "paraflow":
        assert sampler.batch_validation["status"] == "passed"
    sampler.reset()
    actual = invoke(sampler, model)[1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_failure_restores_reference_rng_history_and_methods(model):
    sampler = IsolatedReferenceSampler(method="paraflow")
    reference = IsolatedReferenceSampler(method="paraflow")
    for _ in range(2):
        invoke(sampler, model)
        invoke(reference, model)
    history = sampler.owner.history
    original = model.action_backbone.extract_prediction
    original_full = sampler.first_frame.full
    model.action_backbone.extract_prediction = lambda state: (_ for _ in ()).throw(RuntimeError("injected"))
    with pytest.raises(RuntimeError, match="injected"):
        invoke(sampler, model)
    assert sampler.owner.history is history
    assert sampler.first_frame.full is original_full
    model.action_backbone.extract_prediction = original
    actual = invoke(sampler, model)[1]
    expected = invoke(reference, model)[1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_isolated_plan_and_summary(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, COMPARE_ROOT=str(tmp_path / "output"))
    result = subprocess.run(["bash", "scripts/compare_alpha_libero_isolated.sh", "plan"], cwd=root,
                            env=env, capture_output=True, text=True, check=True)
    assert "15 episodes" in result.stdout and "NOT original isolated weights" in result.stdout
    assert not (tmp_path / "output").exists()
    from benchmarks.libero.isolated_experiment import METHODS, summary
    output = tmp_path / "output"
    output.mkdir()
    for method in METHODS:
        directory = output / method
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"diagnostic": True, "resume_signature": {"jobs_count": 1}, "trial_run": {"num_trials": 5}}))
        (directory / "summary.json").write_text(json.dumps({"overall": {"trials": 5, "successes": 3, "success_rate": .6,
                                                                        "tasks_complete": 1, "tasks_expected": 1},
                                                             "suite_results": [], "missing_runs": [], "task_results": []}))
    assert summary(output)
    assert "No cross-semantics" in (output / "comparison.md").read_text()


def test_isolated_resume_rejects_different_operator_source(tmp_path):
    from benchmarks.libero.scheduler import _validate_resume_signature
    signature = {"sampler": "isolated-paraflow", "diagnostic": True, "isolated_reference_identity": {"source.py": "old"}}
    (tmp_path / "manifest.json").write_text(json.dumps({"resume_signature": signature}))
    with pytest.raises(RuntimeError, match="isolated_reference_identity"):
        _validate_resume_signature(tmp_path, {**signature, "isolated_reference_identity": {"source.py": "new"}})


def test_operator_failure_is_not_silently_accepted(model, monkeypatch):
    sampler = IsolatedReferenceSampler(method="native")
    original = sampler.full_stages.full

    def wrong(*args, **kwargs):
        video, action, snapshots = original(*args, **kwargs)
        return video, action + 1, snapshots

    monkeypatch.setattr(sampler.full_stages, "full", wrong)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="action parity failed"):
            invoke(sampler, model)
        assert sampler.owner is None
    diagnostic = IsolatedReferenceSampler(method="native", allow_partition_failure=True)
    invoke(diagnostic, model)
    assert diagnostic.last_stats["operator_validation"] == "failed"


def test_actual_mask_configuration_is_explicit(model):
    model.mot_driver.attention_mask_mode = "mutual"
    model._mot_driver_kwargs = {"attention_mask_mode": "mutual", "video_attention_mask_mode": "first_frame_causal"}

    def build():
        model.mot_driver.attention_mask_mode = model._mot_driver_kwargs["attention_mask_mode"]
        return model.mot_driver

    model.build_mot_driver = build
    assert configure_isolated(model) == "mutual"
    assert model.mot_driver.attention_mask_mode == "isolated"
    assert model._mot_driver_kwargs["attention_mask_mode"] == "isolated"


def test_isolated_cli_and_launcher_parameter_wiring(tmp_path):
    from openwam.deploy.server import _build_argparser, _apply_inference_overrides
    from omegaconf import OmegaConf

    args = _build_argparser().parse_args(["--sampler", "isolated-paraflow"])
    assert _apply_inference_overrides(OmegaConf.create({}), args).inference.sampler == "isolated-paraflow"
    root = Path(__file__).resolve().parents[2]
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "bash"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", SERVER_PYTHON="/bin/echo",
               COMPARE_ROOT=str(tmp_path / "out"), GPU="6", ALLOW_PARITY_FAILURE="1")
    result = subprocess.run(["/bin/bash", "scripts/compare_alpha_libero_isolated.sh", "diagnostic"], cwd=root,
                            env=env, capture_output=True, text=True, check=True)
    commands = [line for line in result.stdout.splitlines() if line.startswith("scripts/eval_alpha_libero.sh")]
    assert len(commands) == 3
    for command, sampler in zip(commands, ["native", "isolated-native", "isolated-paraflow"]):
        assert f"--sampler {sampler} --denoise-steps 10" in command
        if sampler.startswith("isolated-"):
            assert "inference.isolated_allow_parity_failure=true" in command
    assert not (tmp_path / "out").exists()


def test_engine_isolated_path_does_not_modify_default_mutual(monkeypatch, model):
    from omegaconf import OmegaConf
    from openwam.deploy.engine import BaseInferenceEngine, JointInferenceEngine

    def initialize(self, cfg, **kwargs):
        self.cfg, self.architecture = cfg, kwargs["architecture"]

    def optimizations(self):
        self._decode_video, self._dit_cache = False, None

    monkeypatch.setattr(BaseInferenceEngine, "__init__", initialize)
    monkeypatch.setattr(JointInferenceEngine, "_init_optimizations", optimizations)
    monkeypatch.setattr(JointInferenceEngine, "_init_cfg", lambda self: setattr(self, "_cfg_scale", 1.0))
    model._mot_driver_kwargs = {"attention_mask_mode": "mutual"}
    model.mot_driver.attention_mask_mode = "mutual"
    model.build_mot_driver = lambda: setattr(model.mot_driver, "attention_mask_mode", model._mot_driver_kwargs["attention_mask_mode"])
    cfg = OmegaConf.create({"inference": {"sampler": "native", "inference_horizon": 8},
                           "optimization": {"compile": {"enabled": False}}})
    JointInferenceEngine(cfg, architecture=model)
    assert model.mot_driver.attention_mask_mode == "mutual"
    cfg.inference.sampler = "isolated-paraflow"
    engine = JointInferenceEngine(cfg, architecture=model)
    assert model.mot_driver.attention_mask_mode == "isolated"
    assert engine._joint_sampler.method == "paraflow"
