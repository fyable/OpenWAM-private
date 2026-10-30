"""CPU contracts for OpenWAM's local stage-batched joint sampler."""

from types import SimpleNamespace

import pytest
import torch

from openwam.deploy.parallel_joint import JointParaFlowSampler, joint_velocity, load_core


@pytest.fixture
def core():
    return load_core()


class FakeArchitecture:
    dtype = torch.float32
    video_scheduler = SimpleNamespace(num_train_timesteps=1000)
    action_scheduler = SimpleNamespace(num_train_timesteps=1000)

    def __init__(self):
        self.batches = []

    def forward(self, actions, action_time, *, latents, timestep, context, proprio, first_frame_latents):
        self.batches.append(len(actions))
        batch = len(actions)
        assert context.shape[0] == proprio.shape[0] == first_frame_latents.shape[0] == batch
        assert action_time.shape == timestep.shape == (batch,)
        video = latents * 0.05 + timestep.reshape(-1, 1, 1, 1, 1) / 1000
        action = actions * 0.1 + action_time.reshape(-1, 1, 1) / 1000
        return video, action


def inputs(prefix=2.0):
    return {
        "latents": torch.zeros(1, 2, 3, 2, 2),
        "first_frame_latents": torch.full((1, 2, 1, 2, 2), prefix),
        "context": torch.ones(1, 3, 4),
        "proprio": torch.ones(1, 80),
    }


SCHEDULE = [(1000, 1000), (800, 800), (500, 500), (200, 200), (0, 0)]


def sample(sampler, architecture, *, prefix=2.0, prompt="task", schedule=SCHEDULE):
    inactive = torch.arange(80) >= 10
    return sampler.sample(architecture, inputs(prefix), torch.ones(1, 32, 80), schedule,
                          inactive, prompt=prompt, seed=42)


def test_batch_velocity_matches_independent_calls():
    architecture = FakeArchitecture()
    generator = torch.Generator().manual_seed(7)
    videos = torch.randn(4, 2, 3, 2, 2, generator=generator)
    actions = torch.randn(4, 32, 80, generator=generator)
    batched = joint_velocity(architecture, inputs(), videos, actions, SCHEDULE[:-1])
    sequential = [joint_velocity(architecture, inputs(), videos[index:index + 1],
                                 actions[index:index + 1], [SCHEDULE[index]]) for index in range(4)]
    for modality in (0, 1):
        torch.testing.assert_close(batched[modality], torch.cat([value[modality] for value in sequential]))
    assert architecture.batches == [4, 1, 1, 1, 1]


def test_failure_rolls_back_rng_and_history():
    sampler = JointParaFlowSampler(cache_schedule=True)
    reference = JointParaFlowSampler(cache_schedule=True)
    architecture = FakeArchitecture()
    for _ in range(3):
        sample(sampler, architecture)
        sample(reference, architecture)
    history = sampler.history
    original = architecture.forward

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    architecture.forward = fail
    with pytest.raises(RuntimeError, match="injected"):
        sample(sampler, architecture)
    assert sampler.request == 3
    assert sampler.history is history
    architecture.forward = original
    actual = sample(sampler, architecture)
    expected = sample(reference, architecture)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, rtol=0, atol=0)


def test_cached_schedule_matches_default_over_resets():
    cached = JointParaFlowSampler(cache_schedule=True)
    reference = JointParaFlowSampler()
    architecture = FakeArchitecture()
    for index in range(64):
        if index in (8, 17):
            cached.reset()
            reference.reset()
        actual = sample(cached, architecture, prefix=float(index), prompt=str(index // 20))
        expected = sample(reference, architecture, prefix=float(index), prompt=str(index // 20))
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target, rtol=0, atol=0)
    assert len(cached._schedule_cache) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cached_schedule_cuda_matches_reference_exactly():
    architecture = FakeArchitecture()
    reference, cached = JointParaFlowSampler(), JointParaFlowSampler(cache_schedule=True)
    schedule = [(1000.0, 1000.0), (937.5, 917.0), (832.0, 713.0), (557.0, 327.0), (0.0, 0.0)]
    for index in range(32):
        values = {key: value.cuda() for key, value in inputs(float(index)).items()}
        results = [sampler.sample(architecture, values, torch.ones(1, 32, 80, device="cuda"),
                                  schedule, None, prompt="task", seed=42)
                   for sampler in (reference, cached)]
        for actual, expected in zip(*results):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_two_phase_bootstrap_then_single_call(core):
    sampler = JointParaFlowSampler()
    architecture = FakeArchitecture()
    for index in range(5):
        video, action = sample(sampler, architecture, prefix=float(index))
        assert sampler.last_stats["cold"] == (index < 2)
        assert sampler.last_stats["phase"] == index % 2
        assert sampler.last_stats["joint_forward_calls"] == (4 if index < 2 else 1)
        assert sampler.last_stats["stage_batch"] == (1 if index < 2 else 4)
        assert torch.all(video[:, :, :1] == index)
        assert torch.all(action[..., 10:] == 0)
        assert sampler.history.dtype == torch.float32
        for history in sampler.video_phases.values():
            assert history.dtype == torch.float32
    assert architecture.batches == [1] * 8 + [4] * 3


def test_reset_and_prompt_change(core):
    sampler = JointParaFlowSampler()
    architecture = FakeArchitecture()
    first = sample(sampler, architecture)
    sample(sampler, architecture)
    sample(sampler, architecture)
    sample(sampler, architecture, prompt="different")
    assert sampler.last_stats["cold"]
    assert sampler.last_stats["reset_reason"] == "condition_signature_changed"
    sampler.reset()
    repeated = sample(sampler, architecture)
    for expected, actual in zip(first, repeated):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert len(sampler.video_phases) == 1


def test_alignment_and_scan_use_local_core(core):
    alignment, scan, video_core = core
    plan = video_core.VideoPhasePlan(32, 2, 8)
    assert (plan.phases, plan.shift_latents) == (2, 1)
    with pytest.raises(ValueError):
        video_core.VideoPhasePlan(32, 2, 10)
    sigmas = torch.tensor([1.0, 0.7, 0.2, 0.0])
    noise = torch.arange(32).reshape(1, 32, 1).float()
    velocity = torch.full((3, 1, 32, 1), 2.0)
    trajectory = noise[None] + (sigmas - 1).reshape(-1, 1, 1, 1) * 2
    torch.testing.assert_close(scan.AffineScan(sigmas)(trajectory, velocity, velocity_sign=-1), trajectory)
    torch.testing.assert_close(scan.euler_prefix(noise, velocity, sigmas=sigmas), trajectory)
    aligned = alignment.align_trajectory(trajectory, torch.zeros_like(noise), sigmas, 8, 32)
    torch.testing.assert_close(aligned[..., :24, :], trajectory[..., 8:, :])
    residual = trajectory[..., -1:, :] - sigmas.reshape(-1, 1, 1, 1) * trajectory[0, ..., -1:, :]
    torch.testing.assert_close(aligned[..., 24:, :], residual.expand(-1, -1, 8, -1))


def test_no_nonfinite_or_invalid_grid_fallback(core):
    sampler = JointParaFlowSampler()
    with pytest.raises(ValueError, match="inference_horizon=8"):
        JointParaFlowSampler(advance=10)
    with pytest.raises(ValueError, match="descending"):
        sample(sampler, FakeArchitecture(), schedule=[(1000, 1000), (1000, 500), (0, 0)])
    architecture = FakeArchitecture()
    architecture.forward = lambda *args, **kwargs: (torch.full_like(kwargs["latents"], float("nan")), args[0])
    with pytest.raises(FloatingPointError):
        sample(sampler, architecture)


def test_condition_batch_mismatch_is_rejected():
    conditions = inputs()
    conditions["proprio"] = torch.ones(2, 80)
    with pytest.raises(ValueError, match="batch-one"):
        joint_velocity(FakeArchitecture(), conditions, torch.zeros(4, 2, 3, 2, 2),
                       torch.zeros(4, 32, 80), SCHEDULE[:-1])


def test_unbatched_normalized_proprio():
    conditions = inputs()
    conditions["proprio"] = torch.ones(80)
    architecture = FakeArchitecture()
    joint_velocity(architecture, conditions, torch.zeros(4, 2, 3, 2, 2),
                   torch.zeros(4, 32, 80), SCHEDULE[:-1])
    assert architecture.batches == [4]


def test_server_sampler_cli_override():
    from omegaconf import OmegaConf
    from openwam.deploy.server import _apply_inference_overrides, _build_argparser

    args = _build_argparser().parse_args(["--sampler", "joint-paraflow"])
    cfg = _apply_inference_overrides(OmegaConf.create({}), args)
    assert cfg.inference.sampler == "joint-paraflow"


def test_policy_reset_propagates_to_engine():
    from unittest.mock import Mock
    from openwam.deploy.policy import WAMPolicy

    policy = object.__new__(WAMPolicy)
    policy.engine = Mock()
    policy._executor = Mock()
    policy.reset()
    policy._executor.reset.assert_called_once_with()
    policy.engine.reset.assert_called_once_with()


def test_local_affine_matches_triangular_definition(core):
    _, scan, _ = core
    generator = torch.Generator().manual_seed(91)
    grid = torch.tensor([1., .85, .4, .1, 0.])
    candidates = torch.randn(5, 1, 32, 8, generator=generator)
    velocities = torch.randn(4, 1, 32, 8, generator=generator)
    endpoints = candidates[:-1] - grid[:-1, None, None, None] * velocities
    expected = [candidates[0]]
    for target in range(4):
        weights = []
        for source in range(target + 1):
            weight = 1 - grid[source + 1] / grid[source]
            for intermediate in range(source + 1, target + 1):
                weight = weight * grid[intermediate + 1] / grid[intermediate]
            weights.append(weight)
        value = grid[target + 1] / grid[0] * candidates[0]
        for source, weight in enumerate(weights):
            value = value + weight * endpoints[source]
        expected.append(value)
    torch.testing.assert_close(scan.AffineScan(grid)(candidates, velocities, velocity_sign=-1),
                               torch.stack(expected))
