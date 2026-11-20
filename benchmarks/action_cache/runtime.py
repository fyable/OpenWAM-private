"""Request-local video K/V and residual reuse with fresh prefix/action computation.

This is an approximate Wan mutual-Joint backend, not an equivalence claim.
No state from a reference trajectory is available to a candidate execution.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import math
import threading

import torch


@dataclass(frozen=True)
class CacheConfig:
    policy: str = "fixed"
    interval: int = 2
    max_age: int = 2
    threshold: float = 0.1
    probes: int = 16
    layers: tuple[int, ...] = ()
    reuse_steps: tuple[int, ...] = ()
    random_seed: int = 20261119

    def __post_init__(self):
        if self.policy not in {"refresh", "fixed", "change", "static", "adaptive", "random", "probe"}:
            raise ValueError("Unknown cache policy")
        for name in ("interval", "max_age", "probes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("Threshold must be finite and nonnegative")
        for name in ("layers", "reuse_steps"):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, values)
        if self.policy == "probe" and (not self.layers or not self.reuse_steps):
            raise ValueError("Direct reuse probes require explicit layers and steps")


def prefix_state(state, prefix):
    result = copy.copy(state)
    result.hidden_states = state.hidden_states[:, :prefix]
    # Native Wan RoPE: (tokens, 1, head_dim/2), no token renumbering.
    if state.rope_freqs.shape[0] != state.hidden_states.shape[1]:
        raise ValueError("Unsupported RoPE layout")
    result.rope_freqs = state.rope_freqs[:prefix]
    if state.time_mod.ndim == 4:
        if state.time_mod.shape[1] != state.hidden_states.shape[1]:
            raise ValueError("Unsupported per-token time modulation")
        result.time_mod = state.time_mod[:, :prefix]
    return result


class VideoReuse:
    def __init__(self, driver, config, steps, sensitivities=None):
        self.driver, self.config, self.steps = driver, config, steps
        self.sensitivities = sensitivities or {}
        if config.policy in {"static", "adaptive"}:
            required = config.layers or tuple(range(driver.num_layers))
            for layer in required:
                for stage in range(3):
                    value = self.sensitivities.get(f"{layer}:{stage}")
                    if value is None or not math.isfinite(value) or value < 0:
                        raise ValueError("Static/adaptive reuse requires measured Discovery sensitivity for every eligible layer/stage")
        self.cache = {}
        self.step = -1
        self.stats = dict(full_layers=0, reused_layers=0, video_query_rows=0,
                          native_video_query_rows=0, action_layers=0, controller_evaluations=0)

    def probe(self, state, prefix):
        count = state.hidden_states.shape[1] - prefix
        ids = torch.linspace(prefix, prefix + count - 1, min(count, self.config.probes),
                             device=state.hidden_states.device).long()
        return state.hidden_states.index_select(1, ids).detach().float().clone()

    def should_store(self):
        return True

    def capture_probe(self, state, prefix):
        return self.probe(state, prefix)

    def store(self, layer, k, v, hidden, initial, prefix, probe):
        self.cache[layer] = dict(
            k=k[:, prefix:].detach().clone(), v=v[:, prefix:].detach().clone(),
            delta=(hidden[:, prefix:] - initial[:, prefix:]).detach().clone(),
            probe=probe, step=self.step,
            signature=(tuple(initial.shape), initial.dtype, initial.device),
        )

    def assemble_reused_kv(self, kp, vp, ka, va, entry):
        return (torch.cat([kp, entry["k"], ka], dim=1),
                torch.cat([vp, entry["v"], va], dim=1))

    def slice_reuse_mask(self, mask, prefix, tokens):
        if mask is None:
            return None
        return torch.cat([mask[..., :prefix, :], mask[..., tokens:, :]], dim=-2)

    def should_refresh(self, layer, state, prefix):
        cfg = self.config
        entry = self.cache.get(layer)
        if (entry is None or self.step == self.steps - 1 or cfg.policy == "refresh"
            or (cfg.layers and layer not in cfg.layers)):
            return True
        signature = (tuple(state.hidden_states.shape), state.hidden_states.dtype, state.hidden_states.device)
        if entry["signature"] != signature:
            return True
        age = self.step - entry["step"]
        if age >= cfg.max_age:
            return True
        if cfg.policy == "probe":
            return self.step not in cfg.reuse_steps
        if cfg.policy == "fixed":
            return self.step % cfg.interval == 0
        if cfg.policy == "random":
            # Deterministic local RNG; never advance the policy noise generator.
            import random
            return random.Random(cfg.random_seed + self.step * self.driver.num_layers + layer).random() < 1 / cfg.interval
        stage = min(2, self.step * 3 // self.steps)
        sensitivity = self.sensitivities.get(f"{layer}:{stage}", 1.)
        self.stats["controller_evaluations"] += 1
        if cfg.policy == "static":
            risk = sensitivity * age / cfg.max_age
        else:
            probe = self.probe(state, prefix)
            relative = (probe - entry["probe"]).square().mean().sqrt() / entry["probe"].square().mean().sqrt().clamp_min(1e-6)
            risk = float(relative) * (1 + self.step / max(1, self.steps - 1))
            if cfg.policy == "adaptive":
                risk *= sensitivity
        if not math.isfinite(risk):
            raise FloatingPointError("Nonfinite cache risk")
        return risk > cfg.threshold

    def layer(self, layer, vstate, astate, *, attn_mask, **kwargs):
        if kwargs.get("research_step") is not None or any(kwargs.get(k, False) for k in
                ("use_gradient_checkpointing", "use_gradient_checkpointing_offload")):
            raise ValueError("Cache backend cannot combine with research intervention or training")
        driver, vb, ab = self.driver, self.driver.vb, self.driver.ab
        if (vstate.hidden_states.ndim != 3 or vstate.hidden_states.shape[0] != 1
            or vstate.grid_frames < 2 or vstate.prefix_kv_len or vstate.vace_hints is not None):
            raise ValueError("Only batch-one Wan states with one clean prefix and no VACE/prefix-KV are supported")
        prefix = driver._video_tokens_per_frame(vstate)
        tokens = vstate.hidden_states.shape[1]
        if prefix * vstate.grid_frames != tokens or attn_mask is not None and attn_mask.dtype != torch.bool:
            raise ValueError("Unsupported token grid or attention mask")
        self.stats["native_video_query_rows"] += tokens
        self.stats["action_layers"] += 1
        if self.should_refresh(layer, vstate, prefix):
            # Use exactly the native SDPA shapes/ordering on refreshed layers.
            initial = vstate.hidden_states
            store = self.should_store()
            probe = self.capture_probe(vstate, prefix) if store else None
            q_v, k_v, v_v, vpost = vb.pre_attn_at_layer(layer, vstate)
            q_a, k_a, v_a, apost = ab.pre_attn_at_layer(layer, astate)
            mixed = driver._mixed_attention(torch.cat([q_v, q_a], 1), torch.cat([k_v, k_a], 1),
                                            torch.cat([v_v, v_a], 1), attn_mask)
            av, aa = mixed.split([tokens, q_a.shape[1]], dim=1)
            vstate = vb.post_attn_at_layer(layer, vstate, av.contiguous(), vpost)
            astate = ab.post_attn_at_layer(layer, astate, aa.contiguous(), apost)
            if store:
                self.store(layer, k_v, v_v, vstate.hidden_states, initial, prefix, probe)
            self.stats["full_layers"] += 1
            self.stats["video_query_rows"] += tokens
        else:
            entry = self.cache[layer]
            # Recompute clean-prefix QKV, cross-attention and MLP; skip future
            # video work. Prefix queries retain the original K/V column layout.
            clean = prefix_state(vstate, prefix)
            q_p, k_p, v_p, ppost = vb.pre_attn_at_layer(layer, clean)
            q_a, k_a, v_a, apost = ab.pre_attn_at_layer(layer, astate)
            keys, values = self.assemble_reused_kv(k_p, v_p, k_a, v_a, entry)
            reduced_mask = self.slice_reuse_mask(attn_mask, prefix, tokens)
            mixed = driver._mixed_attention(torch.cat([q_p, q_a], 1), keys, values, reduced_mask)
            ap, aa = mixed.split([prefix, q_a.shape[1]], dim=1)
            clean = vb.post_attn_at_layer(layer, clean, ap.contiguous(), ppost)
            vstate.hidden_states = torch.cat([clean.hidden_states,
                vstate.hidden_states[:, prefix:] + entry["delta"]], dim=1)
            astate = ab.post_attn_at_layer(layer, astate, aa.contiguous(), apost)
            self.stats["reused_layers"] += 1
            self.stats["video_query_rows"] += prefix
        return vstate, astate

    def report(self):
        tensors = [entry[key] for entry in self.cache.values() for key in ("k", "v", "delta", "probe")
                   if entry[key] is not None]
        total = self.stats["full_layers"] + self.stats["reused_layers"]
        return {**self.stats, "steps": self.step + 1,
                "refresh_fraction": self.stats["full_layers"] / max(1, total),
                "video_query_fraction": self.stats["video_query_rows"] / max(1, self.stats["native_video_query_rows"]),
                "cache_bytes": sum(t.numel() * t.element_size() for t in tensors),
                "interpretation": "Actual future-video QKV/attention/MLP skipped; query fraction is not a FLOPs or speedup measurement"}


@contextmanager
def cached_request(engine, config, *, steps=10, sensitivities=None, reuse_factory=VideoReuse,
                   record_schedule=False):
    """Install only for one generate call; always restore, release all tensors."""
    architecture = engine.architecture
    driver = architecture.mot_driver
    if (architecture.training or driver.ab.training or driver.vb.training
        or getattr(engine, "_dit_cache", None) is not None or getattr(engine, "_joint_sampler", None) is not None
        or engine._cfg_scale != 1 or getattr(architecture, "_compiled_mot_run_joint_loop", None) is not None):
        raise ValueError("Cache research requires eager native evaluation, CFG=1, no other cache/sampler")
    if getattr(driver.vb, "dit_patch_size", (1, 2, 2))[0] != 1 or driver.attention_mask_mode != "mutual":
        raise ValueError("Only unchanged mutual Wan checkpoints with temporal patch size one are supported")
    if steps != 10:
        raise ValueError("Video reuse experiments keep the native ten-step schedule")
    if getattr(driver, "_action_cache_active", False):
        raise RuntimeError("Concurrent/nested requests on one cache engine are unsupported")
    if config.layers and max(config.layers) >= driver.num_layers or config.reuse_steps and max(config.reuse_steps) >= steps:
        raise ValueError("Cache layer/step exceeds model schedule")
    reuse = reuse_factory(driver, config, steps, sensitivities)
    if record_schedule:
        reuse.refresh_counts = []
    previous = {name: (name in driver.__dict__, driver.__dict__.get(name)) for name in ("step", "run_joint_loop")}
    original_loop = driver.run_joint_loop
    def loop(vstate, astate, **kwargs):
        reuse.step += 1
        if reuse.step >= steps:
            raise ValueError("Unexpected extra forward; asynchronous schedules and CFG are unsupported")
        before = reuse.stats["full_layers"]
        result = original_loop(vstate, astate, **kwargs)
        if record_schedule:
            reuse.refresh_counts.append(reuse.stats["full_layers"] - before)
        return result
    driver._action_cache_active = True
    driver.step, driver.run_joint_loop = reuse.layer, loop
    try:
        yield reuse
        if reuse.step + 1 != steps:
            raise ValueError("Cache backend did not observe exactly ten forwards")
    finally:
        for name, (present, value) in previous.items():
            if present:
                setattr(driver, name, value)
            else:
                delattr(driver, name)
        del driver._action_cache_active
        reuse.cache.clear()


def install(engine, config, *, sensitivities=None, policy_seed=None):
    """Opt-in server adapter; cache lifetime remains one generate request."""
    original = engine.generate
    from omegaconf import OmegaConf
    if (OmegaConf.select(engine.cfg, "optimization.compile.enabled", default=False)
        or OmegaConf.select(engine.cfg, "inference.inference_mode", default="sync") != "sync"
        or OmegaConf.select(engine.cfg, "inference.denoise_mode", default="sync") != "sync"):
        raise ValueError("Cache deployment requires synchronous eager execution")
    lock = threading.Lock()
    def generate(conditions):
        if not lock.acquire(blocking=False):
            raise RuntimeError("Concurrent cache requests require independent engines")
        try:
            request = dict(conditions)
            if policy_seed is not None:
                request["seed"] = policy_seed
            if request.get("research_context") is not None or request.get("denoise_mode", "sync") != "sync":
                raise ValueError("Cached requests cannot use intervention/asynchronous mode")
            steps = request.get("denoise_steps", engine.cfg.inference.denoise_steps)
            with cached_request(engine, config, steps=steps, sensitivities=sensitivities) as reuse:
                result = original(request)
                import numpy as np
                if not np.isfinite(result["actions"]).all():
                    raise FloatingPointError("Nonfinite cached policy actions")
                result["action_cache_stats"] = reuse.report()
            return result
        finally:
            lock.release()
    engine.generate = generate
    return original
