"""Experimental H32/E8 joint sampler using OpenWAM's local solver."""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_core():
    """Return the repository-local implementation of all solver operations."""
    from openwam.deploy import joint_solver

    return joint_solver, joint_solver, joint_solver


def core_identity():
    return {
        str(Path(module.__file__).resolve()): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (*load_core(), sys.modules[__name__])
    }


def _expand_conditions(inputs, batch):
    expanded = {}
    for key, value in inputs.items():
        if key == "latents":
            continue
        if key == "proprio" and torch.is_tensor(value) and value.ndim == 1:
            value = value.unsqueeze(0)
        if torch.is_tensor(value) and value.ndim:
            if value.shape[0] != 1:
                raise ValueError(f"joint-paraflow requires batch-one condition {key}, got {value.shape}")
            expanded[key] = value.expand(batch, *value.shape[1:])
        else:
            expanded[key] = value
    return expanded


def joint_velocity(architecture, inputs, videos, actions, timesteps):
    """One full joint traversal; each batch item is an independent stage."""
    batch = len(actions)
    if len(videos) != batch or len(timesteps) != batch:
        raise ValueError("video, action, and timestep batch sizes disagree")
    packed = _expand_conditions(inputs, batch)
    packed["latents"] = videos.to(dtype=architecture.dtype)
    times = torch.as_tensor(timesteps, device=actions.device, dtype=architecture.dtype)
    predictions = architecture.forward(
        actions.to(dtype=architecture.dtype), times[:, 1], **packed, timestep=times[:, 0]
    )
    if predictions[1] is None or predictions[0].shape != videos.shape or predictions[1].shape != actions.shape:
        raise RuntimeError("joint forward returned invalid velocity shapes")
    if not all(torch.isfinite(value).all() for value in predictions):
        raise FloatingPointError("non-finite joint velocities")
    return predictions


class JointParaFlowSampler:
    """Two video-phase bootstraps, then one batched joint traversal per call."""

    def __init__(self, advance=8, *, cache_schedule=False):
        if advance != 8:
            raise ValueError("joint-paraflow v1 requires inference_horizon=8")
        self.alignment, self.scan_core, self.video_core = load_core()
        self.advance = advance
        self.cache_schedule = cache_schedule
        self.identity = core_identity()
        self.reset()

    def reset(self):
        self.history = None
        self.video_phases = {}
        self.request = 0
        self.signature = None
        self.action_generator = None
        self.video_generator = None
        self.last_stats = None
        self._schedule_cache = {}

    def _constrain(self, videos, actions, ref, sigmas, noise, inactive):
        videos = videos.float().clone()
        actions = actions.float().clone()
        videos[:, :, :, :1] = ref.float().unsqueeze(0)
        if inactive is not None:
            scale = (sigmas / sigmas[0]).reshape(-1, 1, 1, 1)
            actions[..., inactive] = scale * noise[..., inactive].float().unsqueeze(0)
        if not torch.isfinite(videos).all() or not torch.isfinite(actions).all():
            raise FloatingPointError("non-finite joint trajectory")
        return videos, actions

    @torch.no_grad()
    def sample(self, architecture, inputs, action_noise, schedule, inactive, *, prompt, seed):
        snapshot = dict(self.__dict__)
        snapshot["video_phases"] = dict(self.video_phases)
        snapshot["_schedule_cache"] = dict(self._schedule_cache)
        generators = [(generator, generator.get_state()) for generator in
                      (self.action_generator, self.video_generator) if generator is not None]
        try:
            return self._sample(architecture, inputs, action_noise, schedule, inactive,
                                prompt=prompt, seed=seed)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            for generator, state in generators:
                generator.set_state(state)
            raise

    def _sample(self, architecture, inputs, action_noise, schedule, inactive, *, prompt, seed):
        videos = inputs["latents"]
        ref = inputs.get("first_frame_latents")
        if videos.ndim != 5 or videos.shape[0] != 1 or videos.shape[2] != 3:
            raise ValueError("joint-paraflow requires one episode and one clean plus two future video latents")
        if action_noise.ndim != 3 or action_noise.shape[:2] != (1, 32):
            raise ValueError("joint-paraflow requires an H32 action chunk")
        if ref is None or ref.shape != videos[:, :, :1].shape:
            raise ValueError("joint-paraflow requires exactly one current clean video prefix")
        schedule = tuple(tuple(float(value) for value in pair) for pair in schedule)
        if len(schedule) < 3 or schedule[-1] != (0.0, 0.0):
            raise ValueError("joint-paraflow requires at least two stages and a zero endpoint")
        signature = (prompt, seed, schedule, tuple(videos.shape), tuple(action_noise.shape),
                     str(videos.device), str(architecture.dtype),
                     None if inactive is None else tuple(inactive.tolist()))
        reset_reason = None
        if self.signature != signature:
            reset_reason = "initial" if self.signature is None else "condition_signature_changed"
            self.reset()
            self.signature = signature
            self.action_generator = torch.Generator(device=action_noise.device).manual_seed(seed)
            self.video_generator = torch.Generator(device=videos.device).manual_seed((seed + 104729) % (2**63))
        if videos.is_cuda:
            torch.cuda.synchronize(videos.device)
            torch.cuda.reset_peak_memory_stats(videos.device)
        start = time.perf_counter()
        schedule_key = (schedule, str(videos.device),
                        float(architecture.video_scheduler.num_train_timesteps),
                        float(architecture.action_scheduler.num_train_timesteps))
        cached = self._schedule_cache.get(schedule_key) if self.cache_schedule else None
        if cached is None:
            grid = torch.tensor(schedule, dtype=torch.float32, device=videos.device)
            video_grid = grid[:, 0] / schedule_key[2]
            action_grid = grid[:, 1] / schedule_key[3]
            if not all(torch.isfinite(value).all() and (value[:-1] > value[1:]).all()
                       for value in (video_grid, action_grid)):
                raise ValueError("joint-paraflow requires strictly descending synchronous grids")
            cached = (video_grid.to(videos.device), action_grid.to(videos.device),
                      self.scan_core.AffineScan(action_grid))
            if self.cache_schedule:
                self._schedule_cache[schedule_key] = cached
        video_sigmas, action_sigmas, scan = cached
        plan = self.video_core.VideoPhasePlan(32, 2, self.advance)
        phase = self.request % plan.phases
        cold = phase not in self.video_phases
        fresh_actions = torch.randn(action_noise.shape, device=action_noise.device,
                                    dtype=action_noise.dtype, generator=self.action_generator)
        if self.request:
            action_noise = fresh_actions
            videos = torch.randn(videos.shape, device=videos.device, dtype=videos.dtype,
                                 generator=self.video_generator)
        videos = videos.clone()
        videos[:, :, :1] = ref
        calls = 0
        if cold:
            if self.history is not None:
                action_noise = self.alignment.couple_noise(self.history[0], action_noise, self.advance, 32)
            video_nodes, action_nodes = [videos.float()], [action_noise.float()]
            for index, pair in enumerate(schedule[:-1]):
                velocity_video, velocity_action = joint_velocity(
                    architecture, inputs, video_nodes[-1], action_nodes[-1], [pair]
                )
                calls += 1
                next_video = video_nodes[-1] + velocity_video.float() * (video_sigmas[index + 1] - video_sigmas[index])
                next_action = action_nodes[-1] + velocity_action.float() * (action_sigmas[index + 1] - action_sigmas[index])
                constrained_video, constrained_action = self._constrain(
                    next_video[None], next_action[None], ref, action_sigmas[index + 1:index + 2],
                    action_noise, None,
                )
                if inactive is not None:
                    constrained_action[..., inactive] = (
                        action_sigmas[index + 1] / action_sigmas[0] * action_noise[..., inactive].float()
                    )
                video_nodes.append(constrained_video[0])
                action_nodes.append(constrained_action[0])
            video_nodes, action_nodes = torch.stack(video_nodes), torch.stack(action_nodes)
        else:
            video_nodes = self.video_core.align_video_trajectory(
                self.video_phases[phase], videos, video_sigmas, plan.shift_latents, ref
            )
            action_nodes = self.alignment.align_trajectory(
                self.history, action_noise, action_sigmas, self.advance, 32
            )
            video_nodes, action_nodes = self._constrain(
                video_nodes, action_nodes, ref, action_sigmas, action_nodes[0], inactive
            )
            velocity_video, velocity_action = joint_velocity(
                architecture, inputs, video_nodes[:-1, 0], action_nodes[:-1, 0], schedule[:-1]
            )
            calls += 1
            video_nodes = self.scan_core.euler_prefix(video_nodes[0], velocity_video[:, None], sigmas=video_sigmas)
            action_nodes = scan(action_nodes, velocity_action[:, None], velocity_sign=-1)
        video_nodes, action_nodes = self._constrain(
            video_nodes, action_nodes, ref, action_sigmas, action_nodes[0], inactive
        )
        self.history = action_nodes.detach()
        self.video_phases[phase] = video_nodes.detach()
        self.request += 1
        if videos.is_cuda:
            torch.cuda.synchronize(videos.device)
        self.last_stats = {
            "sampler": "joint-paraflow", "algorithm": "phase-euler-v1-native-adapter",
            "cold": cold, "phase": phase, "phases": plan.phases, "reset_reason": reset_reason,
            "joint_forward_calls": calls, "stage_batch": 1 if cold else len(schedule) - 1,
            "action_advance": self.advance, "video_shift_latents": plan.shift_latents,
            "video_history_age_actions": 0 if cold else plan.phases * self.advance,
            "denoise_ms": 1000 * (time.perf_counter() - start),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(videos.device) if videos.is_cuda else None,
        }
        logger.info("joint-paraflow stats=%s", self.last_stats)
        return video_nodes[-1].to(architecture.dtype), action_nodes[-1].to(architecture.dtype)
