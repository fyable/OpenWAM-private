"""Refresh the video expert on selected native N10 steps."""

from contextlib import ExitStack, contextmanager

import torch

from benchmarks.action_cache.next_runtime import replace_method


@contextmanager
def video_refresh(architecture, steps, refresh_steps, *, inplace_kv=False, video_key_stride=1, selection=None, action_graph=None, projections=None):
    """Use each full forward's video features until the next refresh in the request."""
    refresh_steps = frozenset(refresh_steps)
    if 0 not in refresh_steps or not refresh_steps <= set(range(steps)):
        raise ValueError("Video refresh must start at step zero and use the native grid")
    if video_key_stride < 1 or (video_key_stride > 1 and inplace_kv):
        raise ValueError("Video key thinning requires a positive stride and a separate cache")
    driver = architecture.mot_driver
    if driver.attention_mask_mode != "mutual" or driver.vb.dit_patch_size[0] != 1:
        raise ValueError("Expected native mutual Wan Joint inference")
    if selection is not None and (inplace_kv or video_key_stride != 1):
        raise ValueError("Explicit row selection requires the full separate cache")
    if action_graph is not None and (refresh_steps != {0} or inplace_kv or video_key_stride != 1
                                     or selection is not None or projections is None):
        raise ValueError("Graph replay requires complete video K/V with refresh at step zero")
    action = driver.ab
    video_kv = []
    video_velocity = None
    action_mask = None
    step = 0
    stats = {"video_refresh_steps": [], "video_reuse_steps": [], "action_forwards": 0}

    def capture_attention(original):
        def attention(q, k, v, mask):
            nonlocal action_mask
            if mask is None or mask.dtype != torch.bool or mask.ndim != 2:
                raise ValueError("Expected the native Joint attention mask")
            if action_mask is None:
                if not bool(mask[-action_rows:, :].all()):
                    raise ValueError("Expected action queries to see all video and action keys")
                action_mask = mask[-action_rows:, :].contiguous()
            if inplace_kv:
                # Retain the full concatenation. Its video rows stay fixed,
                # and its action rows are overwritten before each attention.
                video_kv.append((k, v))
            else:
                video_kv.append((k[:, :-action_rows:video_key_stride].clone(memory_format=torch.contiguous_format),
                                 v[:, :-action_rows:video_key_stride].clone(memory_format=torch.contiguous_format)))
            return original(q, k, v, mask)

        return attention

    def check_layout(original):
        def run(vstate, astate, **kwargs):
            layout = selection[3]
            if tuple(int(getattr(vstate, key)) for key in ("grid_frames", "grid_height", "grid_width")) != tuple(
                    layout[key] for key in ("grid_frames", "grid_height", "grid_width")):
                raise ValueError("Frozen E2 grid differs from the live video grid")
            return original(vstate, astate, **kwargs)
        return run

    action_rows = 0

    def capture_action_state(original):
        def prepare(noisy_actions, timestep, **kwargs):
            nonlocal action_rows
            state = original(noisy_actions, timestep, **kwargs)
            action_rows = state.payload.x_action.shape[1]
            return state

        return prepare

    def run_action(noisy_actions, action_timestep, context, context_mask, keys, mixed_mask):
        state = action.prepare_state(noisy_actions, action_timestep, context=context,
                                     context_mask=context_mask)
        if state.payload.x_action.shape[1] != action_rows:
            raise ValueError("Joint layout changed after video refresh")
        for layer, (video_key, video_value) in enumerate(keys):
            query, key, value, post = action.pre_attn_at_layer(layer, state)
            mask = mixed_mask
            if selection is not None and (layer, step) == selection[:2]:
                rows, layout = selection[2:]
                if (video_key.shape[1], action_rows, video_key.shape[2]) != (
                        layout["video_tokens"], layout["action_tokens"], layout["channels"]):
                    raise ValueError("Frozen E2 layout does not match cached K/V")
                indices = torch.as_tensor(rows, device=video_key.device, dtype=torch.long)
                video_key = video_key.index_select(1, indices)
                video_value = video_value.index_select(1, indices)
                mask = None if len(rows) != layout["video_tokens"] else action_mask
                stats["selected_rows"] = [layer, step, len(rows)]
            if inplace_kv:
                video_key[:, -action_rows:].copy_(key)
                video_value[:, -action_rows:].copy_(value)
                output = driver._mixed_attention(query, video_key, video_value, None)
            else:
                output = driver._mixed_attention(query, torch.cat((video_key, key), dim=1),
                                                 torch.cat((video_value, value), dim=1),
                                                 mask if video_key_stride == 1 else None)
            state = action.post_attn_at_layer(layer, state, output.contiguous(), post)
        return action.extract_prediction(state)

    def wrap(original):
        def forward(noisy_actions, action_timestep, *, proprio=None, research_step=None,
                    use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False, **inputs):
            nonlocal step, video_velocity
            if (step >= steps or noisy_actions is None or noisy_actions.shape[0] != 1
                    or research_step is not None or use_gradient_checkpointing or use_gradient_checkpointing_offload
                    or inputs.get("first_frame_latents") is None or inputs.get("vace_context") is not None):
                raise ValueError("Video refresh requires ordinary batch-one native Joint inference")
            if step in refresh_steps:
                # Capture only when the next step will read this cache.
                if step + 1 < steps and step + 1 not in refresh_steps:
                    video_kv.clear()
                    with replace_method(action, "prepare_state", capture_action_state):
                        with replace_method(driver, "_mixed_attention", capture_attention):
                            video_velocity, prediction = original(noisy_actions, action_timestep, proprio=proprio,
                                                                  **inputs)
                    if len(video_kv) != driver.num_layers:
                        raise ValueError("Video cache did not cover the native layer count")
                else:
                    video_velocity, prediction = original(noisy_actions, action_timestep, proprio=proprio, **inputs)
                stats["video_refresh_steps"].append(step)
            else:
                condition = architecture._append_proprio_context_token(inputs, proprio)
                context_mask = condition.get("context_mask")
                if context_mask is None and condition.get("seq_lens") is not None:
                    context = condition["context"]
                    positions = torch.arange(context.shape[1], device=context.device)
                    context_mask = positions.unsqueeze(0) < condition["seq_lens"].to(context.device).unsqueeze(1)
                if video_velocity.shape != inputs["latents"].shape:
                    raise ValueError("Joint video layout changed after video refresh")
                action_inputs = (noisy_actions, action_timestep, condition["context"], context_mask)
                if action_graph is None:
                    prediction = run_action(*action_inputs, video_kv, action_mask)
                else:
                    prediction = action_graph.run(run_action, action_inputs, video_kv, action_mask,
                                                  projections, new_request=step == 1)
                    stats["graph_replays"] = stats.get("graph_replays", 0) + 1
                stats["video_reuse_steps"].append(step)
            step += 1
            stats["action_forwards"] += 1
            # The native solver applies both updates at each original time point.
            return video_velocity, prediction

        return forward

    with ExitStack() as stack:
        stack.enter_context(replace_method(architecture, "forward", wrap))
        if selection is not None:
            stack.enter_context(replace_method(driver, "run_joint_loop", check_layout))
        yield stats
    if step != steps:
        raise ValueError(f"Expected {steps} serial action forwards, got {step}")
