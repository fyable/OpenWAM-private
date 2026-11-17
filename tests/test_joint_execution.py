from types import SimpleNamespace

import pytest
import torch

from openwam.deploy.joint_execution import JointRequestExecutor, prediction_errors
from openwam.deploy.parallel_joint import JointParaFlowSampler, joint_velocity


@pytest.fixture
def model():
    from openwam.model.action_backbone.separate_action_dit import ActionDiT
    from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
    from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    torch.manual_seed(4)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)

    class Architecture(torch.nn.Module):
        forward = DualSystemSelfAttnArchitecture.forward
        dtype = torch.float32
        video_scheduler = action_scheduler = SimpleNamespace(num_train_timesteps=1000)

        def _append_proprio_context_token(self, inputs, proprio):
            result = dict(inputs)
            result["context"] = torch.cat((inputs["context"], proprio[:, None]), dim=1)
            result["context_mask"] = torch.ones(result["context"].shape[:2], dtype=torch.bool)
            return result

        @property
        def mot_driver(self):
            return self._mot_driver

    architecture = Architecture()
    video = WanModel(dim=32, in_dim=4, out_dim=4, ffn_dim=64, text_dim=32, freq_dim=16,
                     eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2, has_image_input=False,
                     seperated_timestep=True, fuse_vae_embedding_in_latents=True,
                     require_vae_embedding=False, require_clip_embedding=False)
    architecture.video_backbone = Wan22Ti2v(SimpleNamespace(dit=video))
    architecture.action_backbone = ActionDiT(action_dim=8, dim=32, ffn_dim=64, num_heads=4, num_layers=2,
                                            video_dim=32, bridge_layers=(0, 1), variant="joint_self_attn", text_dim=32)
    architecture._mot_driver = DualSystemMoTDriver(architecture.video_backbone, architecture.action_backbone,
                                                  attention_mask_mode="mutual", video_attention_mask_mode="first_frame_causal")
    architecture.eval()
    yield architecture
    torch.set_num_threads(previous)


def observation(offset=0.):
    generator = torch.Generator().manual_seed(12)
    video = torch.randn(1, 4, 3, 4, 4, generator=generator) + offset
    return {"latents": video, "first_frame_latents": video[:, :, :1].clone(),
            "context": torch.randn(1, 3, 32, generator=generator), "proprio": torch.ones(1, 32) * (1 + offset)}


@pytest.mark.parametrize("backend", ["prepared", "prefix"])
@torch.no_grad()
def test_real_wan_action_velocity_and_budget(model, backend):
    inputs = observation()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    executor = JointRequestExecutor(model, inputs, backend)
    videos = inputs["latents"].expand(4, -1, -1, -1, -1).clone()
    actions = torch.randn(4, 32, 8)
    schedule = [(1000., 1000.), (700., 700.), (300., 300.), (0., 0.)]
    actual = executor.velocity(videos, actions, schedule)
    expected = joint_velocity(model, inputs, videos, actions, schedule)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, rtol=1e-4, atol=1e-5)
    stats = executor.report()
    assert stats["text_projection_calls"] == 2
    assert stats["context_kv_projection_calls"] == 8
    assert stats["prefix_block_calls"] == (2 if backend == "prefix" else 0)
    assert stats["dynamic_token_instances"] == 4 * ((8 if backend == "prefix" else 12) + 32) * 2
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["prepared", "prefix"])
@torch.no_grad()
def test_sampler_trajectory_reset_and_changed_observation(model, backend):
    reference = JointParaFlowSampler()
    optimized = JointParaFlowSampler(execution_backend=backend)
    schedule = [(1000., 1000.), (700., 700.), (300., 300.), (0., 0.)]
    for index in range(5):
        inputs = observation(index * .1)
        arguments = (model, inputs, torch.ones(1, 32, 8), schedule, torch.arange(8) >= 6)
        expected = reference.sample(*arguments, prompt="task", seed=42)
        actual = optimized.sample(*arguments, prompt="task", seed=42)
        assert prediction_errors(actual, expected)["passed"]
        assert optimized.last_stats["joint_forward_calls"] == (3 if index < 2 else 1)
        assert optimized.last_stats["execution"]["prefix_block_calls"] == (2 if backend == "prefix" else 0)
    optimized.reset()
    assert optimized.history is None and optimized.execution_validation == {}


@torch.no_grad()
def test_strict_failure_retries_and_does_not_advance_history(model, monkeypatch):
    sampler = JointParaFlowSampler(execution_backend="prefix")
    original = JointRequestExecutor.velocity

    def wrong(self, *args):
        video, action = original(self, *args)
        return video, action + 1

    monkeypatch.setattr(JointRequestExecutor, "velocity", wrong)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="parity failed"):
            sampler.sample(model, observation(), torch.ones(1, 32, 8), [(1000., 1000.), (500., 500.), (0., 0.)],
                           None, prompt="task", seed=42)
        assert sampler.request == 0 and sampler.history is None and not sampler.execution_validation


@torch.no_grad()
def test_unsupported_mask_and_condition_rejected(model):
    inputs = observation()
    model.mot_driver.attention_mask_mode = "isolated"
    with pytest.raises(ValueError, match="mutual"):
        JointRequestExecutor(model, inputs, "prefix")
    model.mot_driver.attention_mask_mode = "mutual"
    with pytest.raises(ValueError, match="clip_feature"):
        JointRequestExecutor(model, dict(inputs, clip_feature=torch.ones(1)), "prefix")


@torch.no_grad()
def test_failed_warm_request_restores_generators_and_history(model, monkeypatch):
    sampler = JointParaFlowSampler(execution_backend="prefix")
    reference = JointParaFlowSampler(execution_backend="prefix")
    schedule = [(1000., 1000.), (500., 500.), (0., 0.)]

    def invoke(owner):
        return owner.sample(model, observation(), torch.ones(1, 32, 8), schedule, None, prompt="task", seed=42)

    for _ in range(3):
        invoke(sampler)
        invoke(reference)
    state = sampler.history
    original = JointRequestExecutor.velocity

    def fail(*args):
        raise RuntimeError("injected")

    monkeypatch.setattr(JointRequestExecutor, "velocity", fail)
    with pytest.raises(RuntimeError, match="injected"):
        invoke(sampler)
    assert sampler.request == 3 and sampler.history is state
    monkeypatch.setattr(JointRequestExecutor, "velocity", original)
    for actual, expected in zip(invoke(sampler), invoke(reference)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nonfinite_errors_are_json_safe():
    import json

    diagnostic = prediction_errors((torch.tensor([float("nan")]), torch.ones(1)), (torch.zeros(1), torch.ones(1)))
    assert not diagnostic["passed"] and not diagnostic["finite"]
    json.dumps(diagnostic, allow_nan=False)


@torch.no_grad()
def test_prepared_validation_changes_with_signature(model):
    sampler = JointParaFlowSampler(execution_backend="prefix")
    schedule = [(1000., 1000.), (500., 500.), (0., 0.)]
    first = observation()
    sampler.sample(model, first, torch.ones(1, 32, 8), schedule, None, prompt="first", seed=42)
    initial = sampler.history.clone()
    sampler.sample(model, observation(.4), torch.ones(1, 32, 8), schedule,
                   torch.arange(8) >= 6, prompt="second", seed=43)
    assert sampler.request == 1
    assert sampler.last_stats["reset_reason"] == "condition_signature_changed"
    assert sampler.execution_validation
    assert not torch.equal(initial, sampler.history)
    assert bool((sampler.history[-1, ..., 6:] == 0).all())


@torch.no_grad()
def test_bfloat16_velocities_are_finite_and_checked(model):
    model.to(dtype=torch.bfloat16)
    model.dtype = torch.bfloat16
    inputs = {key: value.to(torch.bfloat16) for key, value in observation().items()}
    executor = JointRequestExecutor(model, inputs, "prefix")
    actions = torch.ones(4, 32, 8, dtype=torch.bfloat16)
    videos = inputs["latents"].expand(4, -1, -1, -1, -1)
    schedule = [(1000., 1000.), (700., 700.), (300., 300.), (0., 0.)]
    actual = executor.velocity(videos, actions, schedule)
    expected = joint_velocity(model, inputs, videos, actions, schedule)
    assert prediction_errors(actual, expected)["passed"]


@torch.no_grad()
def test_sixty_four_request_history_regression(model):
    legacy = JointParaFlowSampler()
    prefix = JointParaFlowSampler(execution_backend="prefix")
    prefix.validate_execution = False
    schedule = [(1000., 1000.), (500., 500.), (0., 0.)]
    for index in range(64):
        if index == 32:
            legacy.reset()
            prefix.reset()
        inputs = observation(index / 100)
        arguments = (model, inputs, torch.ones(1, 32, 8), schedule, None)
        expected = legacy.sample(*arguments, prompt="task", seed=42)
        actual = prefix.sample(*arguments, prompt="task", seed=42)
        assert prediction_errors(actual, expected)["passed"]
