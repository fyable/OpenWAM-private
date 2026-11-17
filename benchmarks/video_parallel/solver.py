"""Video-guided Picard sampling without changing the frozen Joint architecture."""

from dataclasses import asdict, dataclass
import math
import time

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class SolverConfig:
    policy: str = "adaptive"
    window: int = 4
    rounds: int = 1
    video_tolerance: float = 0.003
    local_tolerance: float = 0.01
    action_tolerance: float = 0.003
    action_guard: bool = True
    local_guard: bool = True
    block_size: int = 4

    def __post_init__(self):
        if self.policy not in ("serial", "fixed", "residual", "adaptive"):
            raise ValueError("Unknown window policy")
        if self.window not in (1, 2, 4) or not 1 <= self.rounds <= self.window or self.block_size < 1:
            raise ValueError("Expected window 1/2/4, rounds <= window, and positive block size")
        tolerances = (self.video_tolerance, self.local_tolerance, self.action_tolerance)
        if any(not math.isfinite(value) or value < 0 for value in tolerances):
            raise ValueError("Residual tolerances must be finite and nonnegative")


def residuals(previous_video, video, previous_action, action, active, block_size):
    future = video[:, :, 1:].float()
    delta = future - previous_video[:, :, 1:].float()
    scale = future.square().mean().sqrt().clamp_min(1e-6)
    global_error = delta.square().mean().sqrt() / scale
    spatial = delta.square().mean(dim=1).reshape(-1, 1, delta.shape[-2], delta.shape[-1])
    local = functional.avg_pool2d(spatial, block_size, ceil_mode=True).sqrt() / scale
    local_error = torch.quantile(local.flatten(), 0.95)
    if active is None:
        active = torch.ones(action.shape[-1], dtype=torch.bool, device=action.device)
    if bool(active.any()):
        current = action[..., active].float()
        action_error = ((current - previous_action[..., active].float()).square().mean().sqrt()
                        / current.square().mean().sqrt().clamp_min(1e-6))
    else:
        action_error = action.new_zeros((), dtype=torch.float32)
    return {"video_rms": float(global_error), "video_local_p95": float(local_error),
            "action_rms": float(action_error)}


class VideoGuidedSampler:
    """Same-request window solver; never consumes prior-request trajectories."""

    def __init__(self, config=None, velocity=None):
        self.config = config or SolverConfig()
        self.velocity = velocity
        self.reset()

    def reset(self):
        self.last_stats = {}
        self.last_video = None
        self.last_action = None

    @torch.no_grad()
    def sample(self, architecture, inputs, action_noise, schedule, inactive, *, prompt, seed):
        from openwam.deploy.parallel_joint import joint_velocity

        velocity = self.velocity or joint_velocity
        config = self.config
        video = inputs["latents"].clone()
        action = action_noise.clone()
        reference = inputs.get("first_frame_latents")
        grid = tuple(tuple(float(value) for value in pair) for pair in schedule)
        if (video.ndim != 5 or video.shape[0] != 1 or video.shape[2] < 2
                or action.ndim != 3 or action.shape[0] != 1 or reference is None
                or reference.shape != video[:, :, :1].shape):
            raise ValueError("Expected batch-one Joint inputs with one clean prefix and future video")
        if (len(grid) < 2 or any(len(pair) != 2 for pair in grid)
                or any(not math.isfinite(value) for pair in grid for value in pair)
                or grid[-1] != (0., 0.)):
            raise ValueError("Expected finite paired schedule ending at zero")
        scales = (float(architecture.video_scheduler.num_train_timesteps),
                  float(architecture.action_scheduler.num_train_timesteps))
        sigmas = tuple((pair[0] / scales[0], pair[1] / scales[1]) for pair in grid)
        if any(left[modality] <= right[modality] or right[modality] < 0
               for left, right in zip(sigmas, sigmas[1:]) for modality in (0, 1)):
            raise ValueError("Only strictly descending synchronous schedules are supported")
        if inactive is not None:
            inactive = torch.as_tensor(inactive, dtype=torch.bool, device=action.device)
            if inactive.shape != (action.shape[-1],):
                raise ValueError("Inactive mask dimension mismatch")
        active = None if inactive is None else ~inactive
        video[:, :, :1] = reference
        calls, work, fallbacks = 0, 0, 0
        windows = []
        initial_action_noise = action.clone()
        started = time.perf_counter()

        def evaluate(videos, actions, start):
            nonlocal calls, work
            calls += 1
            work += len(actions)
            result = velocity(architecture, inputs, videos, actions, grid[start:start + len(actions)])
            if (result[0].shape != videos.shape or result[1].shape != actions.shape
                    or not all(bool(torch.isfinite(value).all()) for value in result)):
                raise FloatingPointError("Invalid or nonfinite Joint velocity")
            return result

        def update(current_video, current_action, video_velocity, action_velocity, index):
            next_video = current_video + video_velocity * (sigmas[index + 1][0] - sigmas[index][0])
            next_action = architecture.action_scheduler.flow_step(
                action_velocity, sigmas[index][1], sigmas[index + 1][1], current_action)
            next_video = next_video.clone()
            next_video[:, :, :1] = reference
            if inactive is not None:
                next_action = next_action.clone()
                next_action[..., inactive] = initial_action_noise[..., inactive] * (sigmas[index + 1][1] / sigmas[0][1])
            if not all(bool(torch.isfinite(value).all()) for value in (next_video, next_action)):
                raise FloatingPointError("Nonfinite Joint state")
            return next_video, next_action

        def serial(current_video, current_action, start, count):
            for index in range(start, start + count):
                predicted = evaluate(current_video, current_action, index)
                current_video, current_action = update(current_video, current_action, *predicted, index)
            return current_video, current_action

        position, current_window, easy = 0, min(2, config.window), 0
        while position < len(grid) - 1:
            size = min(current_window if config.policy == "adaptive" else config.window, len(grid) - 1 - position)
            if config.policy == "serial" or size == 1:
                video, action = serial(video, action, position, 1)
                windows.append({"start": position, "size": 1, "rounds": 1, "accepted": 1, "reason": "serial"})
                position += 1
                if config.policy == "adaptive":
                    current_window = min(2, config.window)
                continue
            previous_videos = [video.clone() for _ in range(size + 1)]
            previous_actions = [action.clone() for _ in range(size + 1)]
            for offset in range(1, size + 1):
                if inactive is not None:
                    previous_actions[offset][..., inactive] = (initial_action_noise[..., inactive]
                                                               * sigmas[position + offset][1] / sigmas[0][1])
            accepted, reason, errors = 0, "residual_rejected", []
            limit = min(config.rounds, size) if config.policy == "fixed" else size
            for iteration in range(limit):
                try:
                    predicted_videos, predicted_actions = evaluate(
                        torch.cat(previous_videos[:-1]), torch.cat(previous_actions[:-1]), position)
                    new_videos, new_actions = [video], [action]
                    errors = []
                    for offset in range(size):
                        next_video, next_action = update(new_videos[-1], new_actions[-1],
                            predicted_videos[offset:offset + 1], predicted_actions[offset:offset + 1], position + offset)
                        new_videos.append(next_video)
                        new_actions.append(next_action)
                        errors.append(residuals(previous_videos[offset + 1], next_video,
                                                previous_actions[offset + 1], next_action, active, config.block_size))
                except FloatingPointError:
                    reason = "nonfinite_parallel"
                    break
                previous_videos, previous_actions = new_videos, new_actions
                if config.policy == "fixed":
                    if iteration == limit - 1:
                        accepted, reason = size, "fixed_budget"
                    continue
                for error in errors:
                    passed = (error["video_rms"] <= config.video_tolerance
                              and (not config.local_guard or error["video_local_p95"] <= config.local_tolerance)
                              and (not config.action_guard or error["action_rms"] <= config.action_tolerance))
                    if not passed:
                        break
                    accepted += 1
                if accepted:
                    reason = "residual_prefix"
                    break
            rounds = iteration + 1
            if accepted:
                video, action = previous_videos[accepted], previous_actions[accepted]
            else:
                fallbacks += 1
                video, action = serial(video, action, position, size)
                accepted = size
            windows.append({"start": position, "size": size, "rounds": rounds, "accepted": accepted,
                            "reason": reason, "residuals": errors})
            if config.policy == "adaptive":
                easy = easy + 1 if reason == "residual_prefix" and rounds == 1 and accepted == size else 0
                if reason != "residual_prefix":
                    current_window = 1
                elif easy >= 2:
                    current_window = min(4, config.window)
            position += accepted
        self.last_video, self.last_action = video.detach(), action.detach()
        self.last_stats = {"config": asdict(config), "cold": True, "history_reuse": False,
                           "joint_forward_calls": calls, "stage_evaluations": work, "fallbacks": fallbacks,
                           "windows": windows, "solver_host_ms": (time.perf_counter() - started) * 1000,
                           "residual_is_equivalence_proof": False}
        return video, action
