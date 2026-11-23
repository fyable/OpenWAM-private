"""Reuse the clean video prefix in the native serial Joint loop."""

from contextlib import contextmanager
import copy

import torch

from benchmarks.action_cache.next_runtime import replace_method


@contextmanager
def clean_prefix_kv(architecture, *, steps, verify_prefix=False, keep_attention_shape=False, diagnostic=None):
    driver = architecture.mot_driver
    video, action = driver.vb, driver.ab
    if (driver.attention_mask_mode != "mutual" or video.video_attention_mask_mode != "first_frame_causal"
            or video.dit_patch_size[0] != 1 or not video.dit.seperated_timestep):
        raise ValueError("Expected a mutual Wan TI2V model with an independent, clean first frame")

    prefix_kv = []
    first_layer_inputs = []
    prefix_input = None
    prefix_context = None
    last_hidden = None
    mask = None
    reduced = None
    prefix = None
    current_step = 0
    stats = {"initial_video_rows": 0, "reused_video_rows": 0, "dynamic_video_rows": 0}

    def capture_kv(original):
        def attention(q, k, v, attn_mask):
            nonlocal mask, reduced
            if mask is None:
                if attn_mask is None or attn_mask.dtype != torch.bool:
                    raise ValueError("Expected the native boolean attention mask")
                if bool(attn_mask[..., :prefix, prefix:].any()):
                    raise ValueError("First-frame queries can see dynamic tokens")
                mask = attn_mask
                reduced = attn_mask[..., prefix:, :].contiguous()
            prefix_kv.append((k[:, :prefix].clone(), v[:, :prefix].clone()))
            return original(q, k, v, attn_mask)
        return attention

    def wrap_loop(original):
        def loop(vstate, astate, *, use_gradient_checkpointing=False,
                 use_gradient_checkpointing_offload=False, research_step=None):
            nonlocal current_step, prefix_input, prefix_context, last_hidden, prefix
            if (use_gradient_checkpointing or use_gradient_checkpointing_offload or research_step is not None
                    or vstate.hidden_states.shape[0] != 1 or vstate.prefix_kv_len
                    or vstate.vace_hints is not None or vstate.time_mod.ndim != 4):
                raise ValueError("Prefix reuse requires native batch-one inference")
            prefix = driver._video_tokens_per_frame(vstate)
            video_rows = vstate.hidden_states.shape[1]
            action_rows = astate.payload.x_action.shape[1]
            if video_rows <= prefix or vstate.rope_freqs.shape[0] != video_rows:
                raise ValueError("Invalid video prefix layout")
            current_step += 1
            if current_step == 1:
                prefix_context = vstate.context
                if verify_prefix:
                    prefix_input = (vstate.hidden_states[:, :prefix].clone(), vstate.time_mod[:, :prefix].clone())
                def capture_layer(original_pre):
                    def pre(layer, state):
                        first_layer_inputs.append(state.hidden_states[:, :prefix].clone())
                        return original_pre(layer, state)
                    return pre
                with replace_method(driver, "_mixed_attention", capture_kv):
                    if diagnostic is None:
                        vstate, astate = original(vstate, astate)
                    else:
                        with replace_method(video, "pre_attn_at_layer", capture_layer):
                            vstate, astate = original(vstate, astate)
                if len(prefix_kv) != driver.num_layers or mask.shape[-2:] != (video_rows + action_rows,) * 2:
                    raise ValueError("Native prefix capture did not cover the expected Joint layout")
                last_hidden = vstate.hidden_states[:, :prefix].clone()
                stats["initial_video_rows"] = video_rows * driver.num_layers
                return vstate, astate

            if (video_rows + action_rows != mask.shape[-1] or prefix != last_hidden.shape[1]
                    or vstate.context is not prefix_context):
                raise ValueError("Joint layout or conditioning changed within the request")
            # Checking values here would add a device synchronization per solver step.
            # Verify every step in the untimed trajectory check instead.
            if verify_prefix and (not torch.equal(vstate.hidden_states[:, :prefix], prefix_input[0])
                                  or not torch.equal(vstate.time_mod[:, :prefix], prefix_input[1])):
                raise ValueError("First-frame input or timestep modulation changed")
            active = copy.copy(vstate)
            active.hidden_states = vstate.hidden_states[:, prefix:]
            active.time_mod = vstate.time_mod[:, prefix:]
            active.rope_freqs = vstate.rope_freqs[prefix:]
            for layer, (kp, vp) in enumerate(prefix_kv):
                qv, kv, vv, post_v = video.pre_attn_at_layer(layer, active)
                qa, ka, va, post_a = action.pre_attn_at_layer(layer, astate)
                queries = torch.cat((qv, qa), dim=1)
                keys = torch.cat((kp, kv, ka), dim=1)
                values = torch.cat((vp, vv, va), dim=1)
                if keep_attention_shape:
                    padding = queries.new_zeros(queries.shape[0], prefix, queries.shape[-1])
                    output = driver._mixed_attention(torch.cat((padding, queries), dim=1), keys, values, mask)
                    output = output[:, prefix:]
                else:
                    output = driver._mixed_attention(queries, keys, values, reduced)
                av, aa = output.split((video_rows - prefix, action_rows), dim=1)
                if diagnostic is not None and current_step == 2:
                    full = copy.copy(vstate)
                    full.hidden_states = torch.cat((first_layer_inputs[layer], active.hidden_states), dim=1)
                    qf, kf, vf, post_f = video.pre_attn_at_layer(layer, full)
                    same_kv = driver._mixed_attention(torch.cat((qf[:, :prefix], queries), dim=1), keys, values, mask)
                    native = driver._mixed_attention(torch.cat((qf, qa), dim=1), torch.cat((kf, ka), dim=1),
                                                     torch.cat((vf, va), dim=1), mask)
                    full = video.post_attn_at_layer(layer, full, native[:, :video_rows].contiguous(), post_f)
                active = video.post_attn_at_layer(layer, active, av.contiguous(), post_v)
                if diagnostic is not None and current_step == 2:
                    def error(a, b):
                        return float((a.float() - b.float()).abs().max())
                    diagnostic.append({"layer": layer, "q": error(qv, qf[:, prefix:]),
                                       "k": error(kv, kf[:, prefix:]), "v": error(vv, vf[:, prefix:]),
                                       "cached_prefix_k": error(kp, kf[:, :prefix]),
                                       "cached_prefix_v": error(vp, vf[:, :prefix]),
                                       "attention_shape": error(output, same_kv[:, prefix:]),
                                       "joint_attention": error(output, native[:, prefix:]),
                                       "block_output": error(active.hidden_states, full.hidden_states[:, prefix:])})
                astate = action.post_attn_at_layer(layer, astate, aa.contiguous(), post_a)
            vstate.hidden_states = torch.cat((last_hidden, active.hidden_states), dim=1)
            stats["reused_video_rows"] += prefix * driver.num_layers
            stats["dynamic_video_rows"] += (video_rows - prefix) * driver.num_layers
            return vstate, astate
        return loop

    with replace_method(driver, "run_joint_loop", wrap_loop):
        yield stats
    if current_step != steps or not stats["reused_video_rows"]:
        raise ValueError("Prefix reuse did not preserve the native denoising schedule")
