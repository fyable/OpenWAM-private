"""OpenWAM-local stage alignment and updates, inspired by ParaFlow's method.

Action stages continue their own residual tails. The descending-noise affine
recurrence combines stage endpoint estimates; video uses frozen-velocity Euler
increments. This implementation is owned and executed by OpenWAM, with no
imports from the reference repository.
"""

from dataclasses import dataclass
from math import gcd

import torch


def couple_noise(previous, fresh, advance, horizon):
    if previous.shape != fresh.shape or not 0 <= advance <= horizon <= fresh.shape[-2]:
        raise ValueError("Invalid noise shape or execution horizon")
    result = fresh.float().clone()
    result[..., :horizon - advance, :] = previous[..., advance:horizon, :]
    result[..., horizon:, :] = previous[..., horizon:, :]
    return result


def align_trajectory(trajectory, fresh, remaining, advance, horizon):
    if len(trajectory) != len(remaining):
        raise ValueError("Trajectory and sigma counts disagree")
    noise = couple_noise(trajectory[0], fresh, advance, horizon)
    result = trajectory.float().clone()
    result[..., :horizon - advance, :] = trajectory[..., advance:horizon, :]
    if advance:
        scales = (remaining / remaining[0]).to(noise).reshape(-1, *([1] * noise.ndim))
        residual = trajectory[..., horizon - 1:horizon, :].float()
        residual = residual - scales * trajectory[0, ..., horizon - 1:horizon, :].float()
        result[..., horizon - advance:horizon, :] = residual + scales * noise[..., horizon - advance:horizon, :]
    result[0] = noise
    return result


class AffineScan:
    """FP32 causal affine update on a strictly descending noise grid."""

    def __init__(self, remaining):
        grid = torch.as_tensor(remaining, dtype=torch.float64).detach().cpu()
        if (grid.ndim != 1 or len(grid) < 2 or not torch.isfinite(grid).all()
                or grid[-1] != 0 or not (grid[:-1] > grid[1:]).all()):
            raise ValueError("Expected strictly descending sigmas ending at zero")
        self.remaining = grid

    def __call__(self, candidates, velocities, velocity_sign=1):
        if len(candidates) != len(self.remaining) or velocities.shape != candidates[:-1].shape:
            raise ValueError("Invalid affine candidate or velocity shape")
        grid = self.remaining.to(device=candidates.device, dtype=torch.float32)
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            current = candidates[0].float()
            nodes = [current]
            for stage in range(len(velocities)):
                ratio = grid[stage + 1] / grid[stage]
                endpoint = candidates[stage].float() + velocity_sign * grid[stage] * velocities[stage].float()
                current = ratio * current + (1 - ratio) * endpoint
                nodes.append(current)
            return torch.stack(nodes)


def euler_prefix(noise, velocities, *, sigmas):
    if len(sigmas) != len(velocities) + 1:
        raise ValueError("Sigma and velocity counts disagree")
    intervals = torch.diff(sigmas).to(device=velocities.device, dtype=torch.float32)
    intervals = intervals.reshape(-1, *([1] * (velocities.ndim - 1)))
    first = noise.float().unsqueeze(0)
    return torch.cat((first, first + (velocities.float() * intervals).cumsum(0)))


@dataclass(frozen=True)
class VideoPhasePlan:
    action_horizon: int
    future_latents: int
    advance: int

    def __post_init__(self):
        if (self.future_latents < 1 or self.action_horizon < 1
                or self.action_horizon % self.future_latents
                or not 0 < self.advance <= self.action_horizon):
            raise ValueError("Invalid integral video/action phase grid")
        if self.shift_latents > self.future_latents:
            raise ValueError("Phase history shift exceeds video horizon")

    @property
    def actions_per_latent(self):
        return self.action_horizon // self.future_latents

    @property
    def phases(self):
        return self.actions_per_latent // gcd(self.actions_per_latent, self.advance)

    @property
    def shift_latents(self):
        return self.advance * self.phases // self.actions_per_latent


def align_video_trajectory(trajectory, fresh, remaining, shift, clean_prefix):
    if trajectory.ndim != 6 or fresh.ndim != 5 or trajectory.shape[1:] != fresh.shape:
        raise ValueError("Video history shape mismatch")
    if clean_prefix is None or clean_prefix.shape != fresh[:, :, :1].shape:
        raise ValueError("Expected one clean observed prefix")
    future = trajectory[:, :, :, 1:].movedim(3, 2)
    noise = fresh[:, :, 1:].movedim(2, 1).flatten(2)
    aligned = align_trajectory(future.flatten(3), noise, remaining, shift, noise.shape[-2])
    aligned = aligned.reshape(future.shape).movedim(2, 3)
    prefix = clean_prefix.float().unsqueeze(0).expand(len(trajectory), -1, -1, -1, -1, -1)
    return torch.cat((prefix, aligned), dim=3)
