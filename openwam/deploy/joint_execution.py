"""Observation-local execution of the unchanged mutual Joint velocity field."""

import copy
from contextlib import contextmanager

import torch


def prediction_errors(actual, expected):
    rows = []
    for value, target in zip(actual, expected):
        difference = (value.float() - target.float()).abs()
        finite = bool(torch.isfinite(value).all() and torch.isfinite(target).all())
        mismatch = difference > (.05 + .05 * target.float().abs())
        rows.append({"finite": finite, "passed": finite and not bool(mismatch.any()),
                     "max_absolute_error": float(difference.max()) if finite else None,
                     "rmse": float(difference.square().mean().sqrt()) if finite else None,
                     "mismatches": int(mismatch.sum()), "elements": value.numel()})
    return {"passed": all(row["passed"] for row in rows), "finite": all(row["finite"] for row in rows),
            "rtol": .05, "atol": .05, "video": rows[0], "action": rows[1]}


def module_view(module):
    result = copy.copy(module)
    result._modules = module._modules.copy()
    return result


class BroadcastProjection(torch.nn.Module):
    def __init__(self, value, shape):
        super().__init__()
        self.register_buffer("value", value.detach(), persistent=False)
        self.input_shape = tuple(shape)

    def forward(self, inputs):
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != self.input_shape:
            raise ValueError("Prepared Joint projection received incompatible conditioning")
        return self.value.expand(inputs.shape[0], -1, -1)


def prepare_backend(backend, context, stats):
    result = module_view(backend)
    projected = backend.text_embedding(context)
    stats["text_projection_calls"] += 1
    result.text_embedding = BroadcastProjection(projected, context.shape[1:])
    blocks = []
    for block in backend.blocks:
        attention = block.cross_attn
        if getattr(attention, "has_image_input", False):
            raise ValueError("Prepared Joint does not support image cross-attention")
        cached = module_view(attention)
        key = attention.norm_k(attention.k(projected))
        value = attention.v(projected)
        stats["context_kv_projection_calls"] += 2
        cached.k = BroadcastProjection(key, projected.shape[1:])
        cached.norm_k = torch.nn.Identity()
        cached.v = BroadcastProjection(value, projected.shape[1:])
        current = module_view(block)
        current.cross_attn = cached
        blocks.append(current)
    result.blocks = torch.nn.ModuleList(blocks)
    return result


class JointRequestExecutor:
    def __init__(self, architecture, inputs, backend, *, profile=False):
        if backend not in ("prepared", "prefix"):
            raise ValueError(backend)
        self.architecture = architecture
        self.backend = backend
        self.profile = profile
        self.events = []
        self.driver = architecture.mot_driver
        video = architecture.video_backbone
        action = architecture.action_backbone
        if (architecture.training or video.training or action.training or torch.is_grad_enabled()
                or getattr(architecture, "_compiled_mot_run_joint_loop", None) is not None
                or getattr(video, "_wan_blocks_compile_enabled", False)):
            raise ValueError("Prepared Joint requires eager eval inference without gradients")
        if self.driver.attention_mask_mode != "mutual" or video.video_attention_mask_mode != "first_frame_causal":
            raise ValueError("Prepared Joint requires mutual / first_frame_causal masks")
        for key in ("vace_context", "control_camera_latents_input", "motion_bucket_id", "y", "clip_feature"):
            if inputs.get(key) is not None:
                raise ValueError(f"Prepared Joint does not support {key}")
        if inputs.get("force_per_token_t_mod", True) is not True or inputs.get("zero_clean_prefix_t_mod", True) is not True:
            raise ValueError("Prepared Joint requires clean-prefix per-token modulation")
        self.ref = inputs.get("first_frame_latents")
        if (self.ref is None or inputs["latents"].ndim != 5 or inputs["latents"].shape[0] != 1
                or self.ref.shape != inputs["latents"][:, :, :1].shape
                or not torch.equal(inputs["latents"][:, :, :1], self.ref)):
            raise ValueError("Prepared Joint requires one identical clean first-frame boundary")
        self.stats = {"backend": backend, "text_projection_calls": 0, "context_kv_projection_calls": 0,
                      "prefix_block_calls": 0, "dynamic_block_calls": 0, "dynamic_token_instances": 0,
                      "prefix_token_instances": 0, "velocity_calls": 0, "timing_mode": "cuda_events_profile_only" if profile else "not_recorded"}
        with self.region("conditioning"):
            self.condition = architecture._append_proprio_context_token(
                {key: value for key, value in inputs.items() if key != "proprio"}, inputs.get("proprio"))
            self.condition.setdefault("force_per_token_t_mod", True)
            self.condition.setdefault("zero_clean_prefix_t_mod", True)
            context = self.condition.get("context")
            if context is None or context.shape[0] != 1:
                raise ValueError("Prepared Joint requires one context batch")
            if self.condition.get("context_mask") is None and self.condition.get("seq_lens") is not None:
                self.condition["context_mask"] = (torch.arange(context.shape[1], device=context.device)[None]
                                                   < self.condition["seq_lens"].to(context.device)[:, None])
            self.video = module_view(video)
            self.video.dit = prepare_backend(video.dit, context.to(architecture.dtype), self.stats)
            self.action = prepare_backend(action, context.to(architecture.dtype), self.stats)
        self.prefill_state = None
        self.prefix_kv = []

    @contextmanager
    def region(self, name):
        begin = end = None
        if self.profile and self.ref.is_cuda:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(torch.cuda.current_stream(self.ref.device))
        with torch.profiler.record_function(f"joint_execution.{name}"):
            yield
        if begin is not None:
            end.record(torch.cuda.current_stream(self.ref.device))
            self.events.append((name, begin, end))

    def prefill(self, timestep):
        with self.region("prefix_prefill"):
            condition = dict(self.condition, latents=self.ref, timestep=timestep[:1], num_clean_prefix_frames=1)
            state = self.video.prepare(**condition)
            if state.grid_frames != 1 or state.prefix_kv_len or state.vace_hints is not None:
                raise ValueError("Unsupported Joint prefix state")
            self.prefix_tokens = state.hidden_states.shape[1]
            for depth in range(self.video.num_layers):
                query, key, value, post = self.video.pre_attn_at_layer(depth, state)
                self.prefix_kv.append((key, value))
                output = self.driver._mixed_attention(query, key, value, attn_mask=None)
                state = self.video.post_attn_at_layer(depth, state, output, post)
                self.stats["prefix_block_calls"] += 1
                self.stats["prefix_token_instances"] += self.prefix_tokens
            self.prefill_state = state

    def velocity(self, videos, actions, timesteps):
        count = len(actions)
        if len(videos) != count or len(timesteps) != count:
            raise ValueError("Prepared Joint stage batch dimensions disagree")
        times = torch.as_tensor(timesteps, device=actions.device, dtype=self.architecture.dtype)
        condition = {key: value.expand(count, *value.shape[1:]) if torch.is_tensor(value) and value.ndim else value
                     for key, value in self.condition.items() if key not in ("latents", "timestep")}
        if self.backend == "prefix" and self.prefill_state is None:
            self.prefill(times[:, 0])
        with self.region("prepare_states"):
            state = self.video.prepare(**dict(condition, latents=videos.to(self.architecture.dtype), timestep=times[:, 0]))
            action = self.action.prepare_state(actions.to(self.architecture.dtype), times[:, 1],
                                               context=condition.get("context"), context_mask=condition.get("context_mask"))
        video_tokens = state.hidden_states.shape[1]
        action_tokens = action.payload.x_action.shape[1]
        if state.prefix_kv_len or state.vace_hints is not None:
            raise ValueError("Unsupported prepared Joint video state")
        prefix = self.prefix_tokens if self.backend == "prefix" else 0
        mask = self.driver._build_attention_mask(s_video=video_tokens, s_action=action_tokens,
                                                video_tokens_per_frame=state.grid_height * state.grid_width,
                                                device=actions.device)
        if prefix:
            if state.time_mod.ndim != 4 or state.time_mod.shape[1] != video_tokens:
                raise ValueError("Prefix Joint requires token-local time modulation")
            if video_tokens != prefix * state.grid_frames or state.grid_frames != 3:
                raise ValueError("Prefix Joint requires one clean plus two future frames")
            state.hidden_states = state.hidden_states[:, prefix:]
            state.time_mod = state.time_mod[:, prefix:]
            state.rope_freqs = state.rope_freqs[prefix:]
            mask = mask[..., prefix:, :]
        with self.region("joint_blocks"):
            for depth in range(self.video.num_layers):
                query_video, key_video, value_video, post_video = self.video.pre_attn_at_layer(depth, state)
                query_action, key_action, value_action, post_action = self.action.pre_attn_at_layer(depth, action)
                query = torch.cat((query_video, query_action), dim=1)
                if prefix:
                    prefix_key, prefix_value = self.prefix_kv[depth]
                    key_video = torch.cat((prefix_key.expand(count, -1, -1), key_video), dim=1)
                    value_video = torch.cat((prefix_value.expand(count, -1, -1), value_video), dim=1)
                output = self.driver._mixed_attention(query, torch.cat((key_video, key_action), dim=1),
                                                      torch.cat((value_video, value_action), dim=1), attn_mask=mask)
                video_output, action_output = output.split((video_tokens - prefix, action_tokens), dim=1)
                state = self.video.post_attn_at_layer(depth, state, video_output.contiguous(), post_video)
                action = self.action.post_attn_at_layer(depth, action, action_output.contiguous(), post_action)
                self.stats["dynamic_block_calls"] += 1
                self.stats["dynamic_token_instances"] += count * (video_tokens - prefix + action_tokens)
        with self.region("output_heads"):
            if prefix:
                state.hidden_states = torch.cat((self.prefill_state.hidden_states.expand(count, -1, -1), state.hidden_states), dim=1)
            prediction = self.video.finalize(state), self.action.extract_prediction(action)
        self.stats["velocity_calls"] += 1
        if prediction[0].shape != videos.shape or prediction[1].shape != actions.shape:
            raise RuntimeError("Prepared Joint prediction shape mismatch")
        if not all(torch.isfinite(value).all() for value in prediction):
            raise FloatingPointError("Prepared Joint produced non-finite velocities")
        return prediction

    def report(self):
        result = dict(self.stats)
        if self.events:
            self.events[-1][2].synchronize()
            timings = {}
            for name, begin, end in self.events:
                timings[name] = timings.get(name, 0.) + begin.elapsed_time(end)
            result["components_ms"] = timings
        return result
