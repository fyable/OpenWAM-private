from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event

import pytest

from openwam.deploy.chunk_pipeline import ChunkPipeline


class Worker:
    def __init__(self):
        self.steps = defaultdict(list)
        self.forecast_started = Event()
        self.release_forecast = Event()
        self.release_forecast.set()

    def device_context(self):
        return nullcontext()

    def prepare(self, chunk, request, image):
        return image[0], request['proprio']

    def step(self, chunk, request, prepared):
        self.steps[chunk.start].append((chunk.stage, *prepared))
        chunk.stage += 1

    def actions(self, chunk):
        return chunk.start, chunk.stage

    def decode(self, chunk, request):
        self.forecast_started.set()
        assert self.release_forecast.wait(10)
        return {chunk.start + i: f'forecast:{chunk.start}:{i}' for i in (8, 16, 24)}

    def memory(self):
        return None

    def sync(self):
        pass


def observation(start):
    return dict(prompt='task', height=1, width=1, seed=42, tiled=True, shift=5,
                proprio=start, schedule=tuple(range(11)), first_frame_image=[f'real:{start}'])


@pytest.mark.parametrize('worker_count,budget,online', [(4, 3, [10, 7, 4, 1, 1]),
                                                       (1, 3, [10, 7, 4, 1, 1]),
                                                       (4, 2, [10, 8, 6, 4, 4])])
def test_all_stages_complete_on_same_worker_and_new_observation_replaces_forecast(worker_count, budget, online):
    workers = [Worker() for _ in range(worker_count)]
    pipeline = ChunkPipeline(workers, speculative_steps=budget)
    try:
        for index, expected in enumerate(online):
            start = index * 8
            result = pipeline.generate(observation(start))
            assert result['actions'] == (start, 10)
            stats = result['sampler_stats']
            assert stats['online_steps'] == expected
            assert stats['ready_stage_at_observation'] == 10 - expected
            assert stats['head_worker'] == index % worker_count
            steps = workers[index % worker_count].steps[start]
            assert [stage for stage, _, _ in steps] == list(range(10))
            assert steps[-expected:] == [(stage, f'real:{start}', start) for stage in range(10 - expected, 10)]
            for _, frame, proprio in steps[:-expected]:
                assert frame.startswith(f'forecast:{proprio}:')
                assert proprio + int(frame.rsplit(':', 1)[1]) == start
            for _, pending in pipeline.pending.values():
                pending.result(timeout=5)
        pipeline.reset()
        assert pipeline.generate(observation(0))['sampler_stats']['online_steps'] == 10
    finally:
        pipeline.close()


def test_next_head_does_not_wait_for_another_gpus_forecast():
    workers = [Worker() for _ in range(4)]
    pipeline = ChunkPipeline(workers)
    workers[0].release_forecast.clear()
    try:
        pipeline.generate(observation(0))
        assert workers[0].forecast_started.wait(5)
        with ThreadPoolExecutor(max_workers=1) as caller:
            try:
                result = caller.submit(pipeline.generate, observation(8)).result(timeout=5)
                assert result['sampler_stats']['online_steps'] == 10
                assert all(frame == 'real:8' for _, frame, _ in workers[1].steps[8])
            finally:
                workers[0].release_forecast.set()
    finally:
        workers[0].release_forecast.set()
        pipeline.close()


def test_single_node_joint_matches_native_with_fixed_conditions():
    from types import SimpleNamespace

    import numpy as np
    import torch

    from openwam.deploy.chunk_pipeline import Chunk, ChunkWorker
    from openwam.model.architectures.base import BaseWAMArchitecture

    class Model:
        dtype = torch.bfloat16
        device = torch.device('cpu')
        action_dim = 3
        uses_proprioception = True
        video_scheduler = SimpleNamespace(num_train_timesteps=1000)
        action_scheduler = SimpleNamespace(
            num_train_timesteps=1000, flow_step=lambda flow, sigma, next_sigma, x: x + flow * (next_sigma - sigma)
        )

        def __init__(self):
            self.video_backbone = SimpleNamespace(preprocess_input_for_inference=self.preprocess)

        def eval(self):
            return self

        def _resolve_inactive_action_dims(self, _active, _device):
            return torch.tensor([False, True, False])

        def preprocess(self, **kwargs):
            rng = torch.Generator().manual_seed(kwargs['seed'])
            return {'latents': torch.randn((1, 2, 3, 2, 2), dtype=self.dtype, generator=rng),
                    'first_frame_latents': torch.full((1, 2, 1, 2, 2), 0.5, dtype=self.dtype)}

        def forward(self, actions, action_time, *, latents, timestep, proprio, first_frame_latents):
            assert torch.equal(latents[:, :, :1], first_frame_latents)
            return latents * 0.25 + actions.mean(), actions * 0.1 + latents.mean() + proprio.mean()

    model = Model()
    schedule = tuple((1000 - stage * 100, 920 - stage * 92) for stage in range(11))
    request = dict(observation(0), schedule=schedule, proprio=torch.tensor([0.1]))
    expected = BaseWAMArchitecture.generate(
        model, **request, num_frames=9, action_num_frames=33, num_inference_steps=10, decode_video=False
    )['actions']
    worker = ChunkWorker(model)
    chunk = Chunk(0, 0)
    prepared = worker.prepare(chunk, request, request['first_frame_image'])
    while chunk.stage < 10:
        worker.step(chunk, request, prepared)
    np.testing.assert_array_equal(worker.actions(chunk), expected)
