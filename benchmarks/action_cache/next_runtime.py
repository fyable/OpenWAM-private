"""Fixed schedule optimization and requests for the last-step group experiment."""

from contextlib import contextmanager
import copy

import torch

from .runtime import VideoReuse


class FixedReuse(VideoReuse):
    def __init__(self, driver, config, steps, sensitivities):
        super().__init__(driver, config, steps, sensitivities)
        if config.policy != "fixed" or config.max_age != config.interval:
            raise ValueError("Write elision requires a fixed schedule with max age equal to the interval")
        self.mask_step = -1
        self.saved_mask = None

    def should_store(self):
        next_step = self.step + 1
        return (next_step < self.steps - 1
                and next_step % self.config.interval != 0
                and self.config.max_age > 1)

    def capture_probe(self, state, prefix):
        return None

    def slice_reuse_mask(self, mask, prefix, tokens):
        # The native driver passes the same mask to every layer in one forward.
        if self.mask_step != self.step:
            self.saved_mask = super().slice_reuse_mask(mask, prefix, tokens)
            self.mask_step = self.step
        return self.saved_mask


@contextmanager
def replace_method(owner, name, make_wrapper):
    """Temporarily replace a bound method, preserving descriptor lookup."""
    present = name in vars(owner)
    saved = vars(owner).get(name)
    setattr(owner, name, make_wrapper(getattr(owner, name)))
    try:
        yield
    finally:
        if present:
            setattr(owner, name, saved)
        else:
            delattr(owner, name)


def attention_group_scores(q, k, mask, *, heads, video_tokens, horizon, groups):
    """Mean attention mass from executed action queries to each video group."""
    head_width = q.shape[-1] // heads
    queries = q[:, video_tokens:video_tokens + horizon]
    queries = queries.reshape(q.shape[0], -1, heads, head_width).transpose(1, 2)
    keys = k.reshape(k.shape[0], -1, heads, head_width).transpose(1, 2)
    logits = (queries.float() @ keys.float().transpose(-2, -1)) * head_width**-0.5
    if mask is not None:
        logits.masked_fill_(~mask[..., video_tokens:video_tokens + queries.shape[-2], :], -torch.inf)
    mass = logits.softmax(dim=-1).mean(dim=(0, 1, 2))
    return torch.stack([mass[ids].sum() for ids in groups])


class GroupReuse(VideoReuse):
    """Reuse an equal-sized subset at a single layer on the last forward."""

    def __init__(self, driver, config, steps, sensitivities, *, groups, chosen, horizon,
                 target_layer=29):
        super().__init__(driver, config, steps, sensitivities)
        if len(groups) != 8 or len({len(group) for group in groups}) != 1:
            raise ValueError("Group reuse requires eight equal-sized groups")
        if chosen != "attention" and (len(set(chosen)) != len(chosen)
                                      or any(not 0 <= group < 8 for group in chosen)
                                      or len(chosen) > 2):
            raise ValueError("Expected up to two distinct group indices")
        if not 0 <= target_layer < driver.num_layers or horizon < 1:
            raise ValueError("Invalid group target or action horizon")
        self.target_layer = target_layer
        self.groups = groups
        self.chosen = chosen
        self.horizon = horizon
        self.native_layer = driver.step
        self.used_groups = ()
        self.skipped_rows = 0
        self.anchor_seen = False
        self.target_seen = False

    def layer(self, layer, vstate, astate, *, attn_mask, **kwargs):
        native_args = dict(attn_mask=attn_mask, **kwargs)
        if layer != self.target_layer or self.step < self.steps - 2:
            return self.native_layer(layer, vstate, astate, **native_args)
        prefix = self.driver._video_tokens_per_frame(vstate)
        initial = vstate.hidden_states
        tokens = initial.shape[1]
        if (initial.ndim != 3 or initial.shape[0] != 1 or vstate.prefix_kv_len
                or vstate.vace_hints is not None or vstate.rope_freqs.shape[0] != tokens
                or (attn_mask is not None and attn_mask.dtype != torch.bool)
                or sorted(i for group in self.groups for i in group) != list(range(prefix, tokens))):
            raise ValueError("Live video layout differs from the sealed group partition")
        if self.step == self.steps - 2:
            self.anchor_seen = True
            return self.anchor(layer, vstate, astate, initial, prefix, native_args)
        self.target_seen = True
        if not self.anchor_seen:
            raise RuntimeError("Group reuse requires the preceding forward in the same request")
        if not self.chosen:
            return self.native_layer(layer, vstate, astate, **native_args)
        return self.reuse_group(layer, vstate, astate, initial, prefix, attn_mask)

    def anchor(self, layer, vstate, astate, initial, prefix, native_args):
        def wrap(original):
            def observe(q, k, v, mask):
                self.cache["k"] = k[:, prefix:initial.shape[1]].clone()
                self.cache["v"] = v[:, prefix:initial.shape[1]].clone()
                if self.chosen == "attention":
                    group_ids = [torch.as_tensor(g, device=q.device) for g in self.groups]
                    scores = attention_group_scores(q, k, mask, heads=self.driver.num_heads,
                                                    video_tokens=initial.shape[1],
                                                    horizon=self.horizon, groups=group_ids)
                    self.cache["chosen"] = tuple(scores.argsort(stable=True)[:2].tolist())
                return original(q, k, v, mask)
            return observe

        with replace_method(self.driver, "_mixed_attention", wrap):
            vstate, astate = self.native_layer(layer, vstate, astate, **native_args)
        self.cache["delta"] = vstate.hidden_states[:, prefix:] - initial[:, prefix:]
        return vstate, astate

    def reuse_group(self, layer, vstate, astate, initial, prefix, mask):
        chosen = self.cache["chosen"] if self.chosen == "attention" else self.chosen
        skipped = sorted(i for group in chosen for i in self.groups[group])
        skipped_set = set(skipped)
        tokens = initial.shape[1]
        active_ids = torch.tensor([i for i in range(tokens) if i not in skipped_set],
                                  device=initial.device)
        skipped_ids = torch.tensor(skipped, device=initial.device)
        stale_ids = skipped_ids - prefix
        active = copy.copy(vstate)
        active.hidden_states = initial.index_select(1, active_ids)
        active.rope_freqs = vstate.rope_freqs.index_select(0, active_ids)
        if vstate.time_mod.ndim == 4:
            active.time_mod = vstate.time_mod.index_select(1, active_ids)

        vb, ab = self.driver.vb, self.driver.ab
        qv, kv, vv, video_post = vb.pre_attn_at_layer(layer, active)
        qa, ka, va, action_post = ab.pre_attn_at_layer(layer, astate)

        def restore_columns(fresh, old):
            dense = fresh.new_empty((fresh.shape[0], tokens, fresh.shape[-1]))
            dense.index_copy_(1, active_ids, fresh)
            return dense.index_copy_(1, skipped_ids, old.index_select(1, stale_ids))

        keys = torch.cat((restore_columns(kv, self.cache["k"]), ka), dim=1)
        values = torch.cat((restore_columns(vv, self.cache["v"]), va), dim=1)
        queries = torch.cat((qv, qa), dim=1)
        if mask is not None:
            action_ids = torch.arange(tokens, tokens + qa.shape[1], device=initial.device)
            mask = mask.index_select(-2, torch.cat((active_ids, action_ids)))
        av, aa = self.driver._mixed_attention(queries, keys, values, mask).split(
            (len(active_ids), qa.shape[1]), dim=1)
        active = vb.post_attn_at_layer(layer, active, av.contiguous(), video_post)
        stale = initial.index_select(1, skipped_ids) + self.cache["delta"].index_select(1, stale_ids)
        vstate.hidden_states = initial.clone().index_copy_(1, active_ids, active.hidden_states)
        vstate.hidden_states.index_copy_(1, skipped_ids, stale)
        self.used_groups = tuple(chosen)
        self.skipped_rows = len(skipped)
        return vstate, ab.post_attn_at_layer(layer, astate, aa.contiguous(), action_post)
