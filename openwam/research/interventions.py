"""Bounded-memory, post-softmax interventions; not an acceleration backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class Intervention:
    layer_ids: tuple[int, ...]
    step_ids: tuple[int, ...]
    query_modality: str = "action"
    source_modality: str = "future_video"
    kind: str = "gate"
    gate: float = 1.0
    token_ids: tuple[int, ...] | None = None
    epsilon: float = 0.03
    floor: float | None = None
    sign: int = 1
    noise_seed: int = 20261112
    chunk_size: int = 32

    def __post_init__(self):
        for name in ("layer_ids", "step_ids"):
            values = tuple(getattr(self, name))
            if not values or any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"{name} must contain nonnegative integer IDs")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates")
            object.__setattr__(self, name, values)
        if (self.query_modality, self.source_modality) not in {
            ("action", "future_video"), ("action", "video"), ("future_video", "action")
        }:
            raise ValueError("Unsupported query/source direction")
        if self.kind not in {"gate", "value_noise", "value_stale"}:
            raise ValueError("Unknown intervention kind")
        if not math.isfinite(self.gate) or not 0 <= self.gate <= 1:
            raise ValueError("gate must be finite and in [0,1]")
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.token_ids is not None:
            values = tuple(self.token_ids)
            if not values or any(type(value) is not int or value < 0 for value in values):
                raise ValueError("token_ids must be nonnegative integer IDs")
            if len(set(values)) != len(values):
                raise ValueError("token_ids contains duplicates")
            object.__setattr__(self, "token_ids", values)
        if self.kind != "gate":
            if self.query_modality != "action":
                raise ValueError("V perturbations support action queries only")
            if self.floor is None or not math.isfinite(self.floor) or self.floor <= 0:
                raise ValueError("V perturbations require a positive frozen calibration floor")
            if not math.isfinite(self.epsilon) or self.epsilon <= 0 or self.sign not in {-1, 1}:
                raise ValueError("epsilon must be positive and sign must be +/-1")
            if self.kind == "value_stale" and self.sign != 1:
                raise ValueError("Stale replacement requires sign=1")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise TypeError("intervention must be an object")
        return cls(**values)


@dataclass
class ResearchContext:
    intervention: Intervention
    capture_values: bool = False
    stale_values: dict = field(default_factory=dict)
    records: list = field(default_factory=list)
    values: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    layouts: dict = field(default_factory=dict)
    normalized_actions: torch.Tensor | None = None
    _started: bool = False

    def begin(self, steps, layers):
        if self._started:
            raise ValueError("ResearchContext is single-use; create a new context for every branch")
        if max(self.intervention.step_ids) >= steps or max(self.intervention.layer_ids) >= layers:
            raise ValueError("Intervention layer/step IDs exceed the actual model/schedule")
        self._started = True

    def step(self, index, sigma_video, sigma_action, prefix_frames, video_stepping, action_stepping, latent_frames=None):
        return ResearchStep(self, index, float(sigma_video), float(sigma_action), prefix_frames,
                            bool(video_stepping), bool(action_stepping), latent_frames)


@dataclass(frozen=True)
class ResearchStep:
    context: ResearchContext
    index: int
    sigma_video: float
    sigma_action: float
    prefix_frames: int
    video_stepping: bool
    action_stepping: bool
    latent_frames: int | None = None

    def selected(self, layer):
        config = self.context.intervention
        return layer in config.layer_ids and self.index in config.step_ids


def intervene_attention(q_cat, k_cat, v_cat, mask, *, heads, video_tokens, tokens_per_frame,
                        grid_frames, layer, step):
    config = step.context.intervention
    batch, total, channels = q_cat.shape
    if batch != 1 or k_cat.shape != q_cat.shape or v_cat.shape != q_cat.shape:
        raise ValueError("Research attention requires batch=1 and matching Q/K/V; prefix KV is unsupported")
    prefix = step.prefix_frames * tokens_per_frame
    if step.latent_frames is not None and step.latent_frames != grid_frames:
        raise ValueError("Temporal patching is unsupported: latent/grid prefix mapping is not verified")
    if grid_frames * tokens_per_frame != video_tokens or not 0 < prefix < video_tokens:
        raise ValueError("Cannot resolve a protected prefix and future video token layout")
    if channels % heads or not 0 < video_tokens < total:
        raise ValueError("Invalid attention shape")
    if not all(torch.isfinite(value).all() for value in (q_cat, k_cat, v_cat)):
        raise FloatingPointError("Non-finite research Q/K/V")
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError("Research attention requires a boolean visibility mask")
    layout = {"grid_frames": grid_frames, "tokens_per_frame": tokens_per_frame,
              "prefix_tokens": prefix, "video_tokens": video_tokens,
              "action_tokens": total - video_tokens, "channels": channels}
    step.context.layouts[layer] = layout
    query_start = video_tokens if config.query_modality == "action" else prefix
    query_stop = total if config.query_modality == "action" else video_tokens
    source_start = video_tokens if config.source_modality == "action" else (
        prefix if config.source_modality == "future_video" else 0)
    source_stop = total if config.source_modality == "action" else video_tokens
    source_ids = list(range(source_start, source_stop)) if config.token_ids is None else list(config.token_ids)
    if any(value < source_start or value >= source_stop for value in source_ids):
        raise ValueError("Selected source tokens cross source modality/protected prefix boundaries")
    source = torch.tensor(source_ids, device=q_cat.device, dtype=torch.long)
    query = q_cat.reshape(batch, total, heads, -1).transpose(1, 2)
    keys = k_cat.reshape(batch, total, heads, -1).transpose(1, 2)
    values = v_cat.reshape(batch, total, heads, -1).transpose(1, 2)
    selected_values = values.index_select(2, source)
    key = (layer, step.index)
    if step.context.capture_values:
        step.context.values[key] = selected_values.detach().float().cpu().clone()
    delta = None
    if config.kind != "gate":
        scale = selected_values.float().norm() + config.floor
        if config.kind == "value_noise":
            generator = torch.Generator(device=values.device).manual_seed(config.noise_seed)
            direction = torch.randn(selected_values.shape, generator=generator, device=values.device,
                                    dtype=torch.float32)
        else:
            previous = step.context.stale_values.get((layer, step.index - 1))
            if previous is None or previous.shape != selected_values.shape:
                raise ValueError("Missing matching baseline V at the preceding denoise step")
            direction = previous.to(values.device) - selected_values.float()
        norm = direction.norm()
        if not torch.isfinite(norm) or (norm <= 0 and config.kind == "value_noise"):
            raise ValueError("Perturbation direction has zero or non-finite norm")
        delta = direction if config.kind == "value_stale" else direction * (config.sign * config.epsilon * scale / norm)
    original_output = functional.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
    outputs = []
    mass_sum = 0.0
    contribution_sq = 0.0
    for start, stop in ((0, query_start), (query_start, query_stop), (query_stop, total)):
        if start == stop:
            continue
        if start != query_start:
            outputs.append(original_output[:, :, start:stop])
            continue
        for offset in range(start, stop, config.chunk_size):
            end = min(stop, offset + config.chunk_size)
            query_rows = query[:, :, offset:end]
            row_mask = None if mask is None else mask[..., offset:end, :]
            original = original_output[:, :, offset:end]
            logits = query_rows.float() @ keys.float().transpose(-1, -2) / math.sqrt(query.shape[-1])
            if row_mask is not None:
                logits = logits.masked_fill(~row_mask, -torch.inf)
            probabilities = torch.softmax(logits, dim=-1).nan_to_num(0.0).index_select(-1, source)
            contribution = probabilities @ selected_values.float()
            mass_sum += probabilities.sum().item()
            contribution_sq += contribution.square().sum().item()
            change = contribution * (config.gate - 1) if delta is None else probabilities @ delta
            outputs.append(original + change.to(original.dtype))
    step.context.records.append({"layer_id": layer, "step_id": step.index,
                                 "sigma_video": step.sigma_video, "sigma_action": step.sigma_action,
                                 "video_stepping": step.video_stepping, "action_stepping": step.action_stepping,
                                 "branch": "conditional", "source_tokens": len(source_ids),
                                 "relative_perturbation_energy": None if delta is None else float(delta.norm() / scale),
                                 "reference_step_id": step.index - 1 if config.kind == "value_stale" else None,
                                 "attention_mass": mass_sum / (heads * (query_stop - query_start)),
                                 "contribution_norm": math.sqrt(contribution_sq), **layout})
    return torch.cat(outputs, dim=2).transpose(1, 2).reshape(batch, total, channels)
