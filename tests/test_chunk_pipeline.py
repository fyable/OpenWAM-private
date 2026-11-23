from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openwam.deploy.chunk_pipeline import Chunk, ChunkPipeline, ChunkWorker


def make_request():
    return {
        "prompt": "test", "height": 16, "width": 16, "seed": 42,
        "tiled": True, "shift": 5.0, "first_frame_image": [100],
        "schedule": tuple((float(stage), float(stage)) for stage in range(10, -1, -1)),
    }


class FakeWorker:
    def __init__(self):
        self.prepared = []

    def prepare(self, chunk, request, image):
        self.prepared.append((chunk.start, chunk.stage, image))
        return None

    def step(self, chunk, request, prepared):
        chunk.stage += 1

    def actions(self, chunk):
        return np.zeros((32, 7), dtype=np.float32)

    def decode(self, chunk, request):
        return {chunk.start + offset: chunk.start + offset for offset in (8, 16, 24)}

    def device_context(self):
        return nullcontext()

    def sync(self):
        pass

    def memory(self):
        return None


def finish_background(pipeline):
    for _, future in list(pipeline.pending.values()):
        future.result(timeout=5)


def test_rotation_stage_budget_and_reset():
    workers = [FakeWorker() for _ in range(4)]
    pipeline = ChunkPipeline(workers)
    request = make_request()
    try:
        for request_index in range(5):
            result = pipeline.generate(request)
            stats = result["sampler_stats"]
            assert stats["head_worker"] == request_index % 4
            assert stats["online_steps"] == 10 - min(request_index * 3, 9)
            assert stats["cold"] == (request_index < 3)
            assert result["video"] is None
            assert result["actions"].shape == (32, 7)
            finish_background(pipeline)
            assert all(chunk.stage <= 9 for chunk in pipeline.chunks.values())
        pipeline.reset()
        assert pipeline.start == 0
        assert not pipeline.chunks
        assert not pipeline.pending
        assert not pipeline.forecasts
        assert pipeline.generate(request)["sampler_stats"]["online_steps"] == 10
    finally:
        pipeline.close()
    assert all(executor._shutdown for executor in pipeline.executors)


def test_observation_timestamp_is_distinct_from_takeover_stage():
    pipeline = ChunkPipeline([FakeWorker() for _ in range(4)])
    try:
        pipeline.generate(make_request())
        finish_background(pipeline)
        head = pipeline.chunks[8]
        head.completed_at = [perf_counter() + 60] * head.stage
        stats = pipeline.generate(make_request())["sampler_stats"]
        assert stats["ready_stage_at_observation"] == 0
        assert stats["ready_stage_at_takeover"] == 3
        assert stats["online_steps"] == 7
    finally:
        pipeline.close()


@pytest.mark.parametrize("worker_count", [1, 4])
@pytest.mark.parametrize("speculative_steps", [2, 3])
def test_round_barrier_completes_budget_without_client_wait(worker_count, speculative_steps):
    pipeline = ChunkPipeline([FakeWorker() for _ in range(worker_count)], speculative_steps)
    try:
        for request_index in range(6):
            stats = pipeline.generate(make_request())["sampler_stats"]
            expected_stage = min(request_index * speculative_steps, 3 * speculative_steps)
            assert stats["ready_stage_at_takeover"] == expected_stage
            assert stats["online_steps"] == 10 - expected_stage
    finally:
        pipeline.close()


def test_round_barrier_waits_for_non_head_worker_before_real_conditioning():
    background_entered = Event()
    release_background = Event()
    barrier_entered = Event()
    real_conditioning = Event()
    round_completed = Event()

    class SlowWorker(FakeWorker):
        def step(self, chunk, request, prepared):
            background_entered.set()
            if not release_background.wait(timeout=10):
                raise TimeoutError("background was not released")
            super().step(chunk, request, prepared)
            if chunk.stage == 3:
                round_completed.set()

    class HeadWorker(FakeWorker):
        def prepare(self, chunk, request, image):
            if image == request["first_frame_image"]:
                assert round_completed.is_set()
                real_conditioning.set()
            return super().prepare(chunk, request, image)

    class ObservedFuture:
        def __init__(self, future):
            self.future = future

        def result(self):
            barrier_entered.set()
            return self.future.result()

    workers = [FakeWorker(), HeadWorker(), FakeWorker(), SlowWorker()]
    pipeline = ChunkPipeline(workers)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        pipeline.generate(make_request())
        assert background_entered.wait(timeout=5)
        stop, future = pipeline.pending[3]
        pipeline.pending[3] = (stop, ObservedFuture(future))
        next_request = executor.submit(pipeline.generate, make_request())
        assert barrier_entered.wait(timeout=5)
        assert not stop.is_set()
        assert not real_conditioning.is_set()
        assert not next_request.done()
        release_background.set()
        stats = next_request.result(timeout=5)["sampler_stats"]
        assert real_conditioning.is_set()
        assert stats["ready_stage_at_takeover"] == 3
        assert stats["online_steps"] == 7
        assert round_completed.is_set()
    finally:
        release_background.set()
        executor.shutdown(wait=True)
        pipeline.close()


def test_round_barrier_failure_preserves_tasks_for_cleanup():
    class FailingWorker(FakeWorker):
        def prepare(self, chunk, request, image):
            raise RuntimeError("background preparation failed")

    pipeline = ChunkPipeline([FakeWorker(), FakeWorker(), FakeWorker(), FailingWorker()])
    pipeline.generate(make_request())
    with pytest.raises(RuntimeError, match="background preparation failed"):
        pipeline.generate(make_request())
    assert pipeline.start == 8
    assert len(pipeline.pending) == 3
    with pytest.raises(RuntimeError, match="background preparation failed"):
        pipeline.close()
    assert not pipeline.pending
    assert all(executor._shutdown for executor in pipeline.executors)


def test_signature_change_starts_new_pipeline():
    pipeline = ChunkPipeline([FakeWorker() for _ in range(4)])
    try:
        pipeline.generate(make_request())
        finish_background(pipeline)
        request = {**make_request(), "prompt": "new episode"}
        stats = pipeline.generate(request)["sampler_stats"]
        assert stats["chunk_start"] == 0
        assert stats["online_steps"] == 10
    finally:
        pipeline.close()


def test_background_failure_propagates_and_cleanup_completes():
    class FailingWorker(FakeWorker):
        def decode(self, chunk, request):
            raise RuntimeError("decode failed")

    pipeline = ChunkPipeline([FailingWorker() for _ in range(4)])
    pipeline.generate(make_request())
    with pytest.raises(RuntimeError, match="decode failed"):
        pipeline.close()
    assert not pipeline.chunks
    assert not pipeline.pending
    assert all(executor._shutdown for executor in pipeline.executors)


def test_prepare_preserves_latents_actions_and_stage():
    reference = torch.full((1, 2, 1, 2, 2), 7.0)

    def preprocess(**kwargs):
        return {"first_frame_latents": reference, "latents": torch.zeros((1, 2, 3, 2, 2))}

    model = SimpleNamespace(
        video_backbone=SimpleNamespace(preprocess_input_for_inference=preprocess),
        uses_proprioception=False, device="cpu", dtype=torch.float32,
        action_dim=7, _resolve_inactive_action_dims=lambda *args: None,
    )
    chunk = Chunk(8, 1, stage=6, video=torch.ones((1, 2, 3, 2, 2)),
                  action=torch.ones((1, 32, 7)))
    video, actions = chunk.video, chunk.action
    inputs, _ = ChunkWorker(model).prepare(chunk, make_request(), [100])
    assert chunk.video is video
    assert chunk.action is actions
    assert chunk.stage == 6
    torch.testing.assert_close(chunk.video[:, :, :1], reference)
    torch.testing.assert_close(chunk.video[:, :, 1:], torch.ones((1, 2, 2, 2, 2)))
    assert "latents" not in inputs


def test_decode_selects_prediction_offsets():
    model = SimpleNamespace(video_backbone=SimpleNamespace(decode_video=lambda *args, **kwargs: list(range(9))))
    assert ChunkWorker(model).decode(Chunk(8, 1), make_request()) == {16: 2, 24: 4, 32: 6}


def test_current_wan_preprocessor_is_compatible(monkeypatch):
    from openwam.model.video_backbone import wan_backbone

    reference = torch.full((1, 2, 1, 2, 2), 7.0)
    monkeypatch.setattr(wan_backbone, "check_resize_height_width", lambda height, width, frames, **kwargs: (height, width, frames))
    monkeypatch.setattr(wan_backbone.wan_encode, "encode_text_for_inference", lambda *args, **kwargs: (torch.zeros(1), torch.ones(1)))
    monkeypatch.setattr(wan_backbone.wan_encode, "preprocess_video", lambda *args, **kwargs: reference)
    monkeypatch.setattr(wan_backbone.wan_encode, "encode_video", lambda *args, **kwargs: reference)
    monkeypatch.setattr(wan_backbone.wan_conditioning, "build_deploy_noise", lambda **kwargs: torch.zeros((1, 2, 3, 2, 2)))
    monkeypatch.setattr(wan_backbone.wan_conditioning, "build_vace_context_for_deploy", lambda *args, **kwargs: None)
    backbone = SimpleNamespace(
        scheduler=SimpleNamespace(set_timesteps=lambda **kwargs: None),
        _height_division_factor=16, _width_division_factor=16,
        _time_division_factor=4, _time_division_remainder=1,
        _tokenizer=None, text_encoder=None, device="cpu", dtype=torch.float32,
        _latent_spec=None, vae=None, _dit=SimpleNamespace(has_image_input=False),
        _is_ti2v=True, _has_vace=False, video_encoder=None,
    )
    inputs = wan_backbone.WanBase.preprocess_input_for_inference(
        backbone, prompt="test", first_frame_image=[100], height=16, width=16,
        num_frames=9, pin_first_frame=True, initial_latents=torch.ones((1, 2, 3, 2, 2)),
    )
    assert inputs["fuse_vae_embedding_in_latents"] is True
    torch.testing.assert_close(inputs["first_frame_latents"], reference)


def test_step_pins_first_frame_and_advances_state():
    reference = torch.full((1, 2, 1, 2, 2), 7.0)
    model = SimpleNamespace(
        device="cpu", dtype=torch.float32,
        video_scheduler=SimpleNamespace(num_train_timesteps=10),
        action_scheduler=SimpleNamespace(
            num_train_timesteps=10,
            flow_step=lambda flow, current, following, action: action + flow * (following - current),
        ),
        forward=lambda action, action_timestep, **kwargs: (torch.ones_like(kwargs["latents"]), torch.ones_like(action)),
    )
    chunk = Chunk(0, 0, video=torch.zeros((1, 2, 3, 2, 2)), action=torch.zeros((1, 32, 7)))
    ChunkWorker(model).step(chunk, make_request(), ({"first_frame_latents": reference}, None))
    assert chunk.stage == 1
    torch.testing.assert_close(chunk.video[:, :, :1], reference)
    torch.testing.assert_close(chunk.video[:, :, 1:], torch.full((1, 2, 2, 2, 2), -0.1))
    torch.testing.assert_close(chunk.action, torch.full((1, 32, 7), -0.1))


@pytest.mark.parametrize("worker_count,steps", [(2, 3), (4, 1), (4, 4)])
def test_invalid_configuration(worker_count, steps):
    with pytest.raises(ValueError):
        ChunkPipeline([FakeWorker() for _ in range(worker_count)], steps)


def test_scheduler_passes_four_visible_devices(tmp_path):
    from benchmarks.libero import scheduler

    args = SimpleNamespace(
        server_python="python", ckpt_dir=tmp_path, ckpt_name="checkpoint.safetensors",
        host="127.0.0.1", denoise_steps=10, denoise_mode="sync",
        inference_mode="sync", inference_horizon=8, compile_enabled="false",
        sampler="chunk-pipeline", gpus=[1, 2, 3, 0], chunk_speculative_steps=3,
        extra_deploy_arg=["optimization.decode_video=false"],
    )
    command = scheduler._server_command(args, 8993)
    assert command[command.index("--chunk-devices") + 1] == "cuda:0,cuda:1,cuda:2,cuda:3"
    assert command[command.index("--chunk-speculative-steps") + 1] == "3"
    assert "optimization.dit_cache.enabled=false" in command
