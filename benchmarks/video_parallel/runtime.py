"""Adapters installed only by the independent research entrypoints."""

from contextlib import contextmanager
from dataclasses import asdict

import torch

from .solver import SolverConfig, VideoGuidedSampler


def method_spec(name):
    if name.startswith("native_n") and name.removeprefix("native_n") in ("4", "6", "8", "10"):
        return int(name.removeprefix("native_n")), None
    if name == "legacy_n10":
        return 10, "legacy"
    if name == "adaptive":
        return 10, SolverConfig()
    if name.startswith("residual_k") and name[-1:] in ("2", "4") and name == f"residual_k{name[-1]}":
        return 10, SolverConfig(policy="residual", window=int(name[-1]))
    for size in (1, 2, 4):
        for rounds in range(1, size + 1):
            if name == f"fixed_k{size}_r{rounds}":
                return 10, SolverConfig(policy="fixed", window=size, rounds=rounds)
    raise ValueError(f"Unknown method: {name}")


def configure_engine(engine, method, overrides=None):
    from openwam.deploy.parallel_joint import JointParaFlowSampler
    from omegaconf import OmegaConf

    architecture = engine.architecture
    if architecture.mot_driver.attention_mask_mode != "mutual":
        raise ValueError("This experiment requires an unchanged mutual Joint checkpoint")
    if engine._dit_cache is not None or engine._cfg_scale != 1.:
        raise ValueError("Disable DiT cache and CFG for controlled sampling experiments")
    if (OmegaConf.select(engine.cfg, "optimization.compile.enabled", default=False)
            or OmegaConf.select(engine.cfg, "inference.denoise_mode", default="sync") != "sync"
            or OmegaConf.select(engine.cfg, "inference.inference_mode", default="sync") != "sync"):
        raise ValueError("Controlled experiments require eager, synchronous inference and denoising")
    steps, config = method_spec(method)
    engine.cfg.inference.denoise_steps = steps
    if config == "legacy":
        if overrides:
            raise ValueError("Solver overrides cannot be applied to legacy")
        sampler = JointParaFlowSampler()
    elif config is None:
        if overrides:
            raise ValueError("Solver overrides cannot be applied to Native")
        sampler = None
    else:
        sampler = VideoGuidedSampler(SolverConfig(**{**asdict(config), **(overrides or {})}))
    engine._joint_sampler = sampler
    return sampler


@contextmanager
def capture_video(engine, decode=False):
    backbone = engine.architecture.video_backbone
    previous = backbone.__dict__.get("decode_video")
    present = "decode_video" in backbone.__dict__
    original = backbone.decode_video
    previous_decode = engine._decode_video
    captured = {}

    def decode_or_capture(latents, *args, **kwargs):
        captured["latents"] = latents.detach()
        return original(latents, *args, **kwargs) if decode else None

    backbone.decode_video = decode_or_capture
    engine._decode_video = True
    try:
        yield captured
    finally:
        engine._decode_video = previous_decode
        if present:
            backbone.decode_video = previous
        else:
            del backbone.decode_video


def tensor_error(actual, expected):
    actual, expected = torch.as_tensor(actual).float(), torch.as_tensor(expected).float()
    if actual.shape != expected.shape:
        raise ValueError("Reference and candidate tensor shapes differ")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {"finite": False, "passed": False, "rmse": None, "max_abs": None, "relative_rms": None}
    difference = actual - expected
    return {"finite": True, "passed": bool(torch.allclose(actual, expected, rtol=.05, atol=.05)),
            "rtol": .05, "atol": .05, "rmse": float(difference.square().mean().sqrt()),
            "max_abs": float(difference.abs().max()),
            "relative_rms": float(difference.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-6))}


class PartitionValidator:
    def __init__(self):
        self.checks = []
        self.shapes = set()

    def __call__(self, architecture, inputs, videos, actions, timesteps):
        from openwam.deploy.parallel_joint import joint_velocity

        predictions = joint_velocity(architecture, inputs, videos, actions, timesteps)
        key = (tuple(videos.shape), tuple(actions.shape), str(videos.dtype))
        if len(actions) > 1 and key not in self.shapes:
            self.shapes.add(key)
            separate = [joint_velocity(architecture, inputs, videos[index:index + 1], actions[index:index + 1],
                                       timesteps[index:index + 1]) for index in range(len(actions))]
            self.checks.append({"batch": len(actions), "video": tensor_error(predictions[0], torch.cat([pair[0] for pair in separate])),
                                "action": tensor_error(predictions[1], torch.cat([pair[1] for pair in separate]))})
        return predictions
