"""Overlapped H32/E8 inference with one resident Joint state per chunk."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import nullcontext
from dataclasses import dataclass, field
from threading import Event, Lock
from time import perf_counter
import os

import torch

from openwam import chunk_diagnostics as diagnostics


@dataclass
class Chunk:
    start: int
    worker: int
    stage: int = 0
    video: torch.Tensor | None = None
    action: torch.Tensor | None = None
    inactive_noise: torch.Tensor | None = None
    speculative_ms: float = 0.0
    completed_at: list[float] = field(default_factory=list)
    online_step_host_ms: list[float] = field(default_factory=list)


class ChunkWorker:
    def __init__(self, model):
        self.model = model
        self.prompt_cache = OrderedDict()
        self.timestep_cache = {}
        self.forward_graph = None
        from openwam.deploy.chunk_affinity import configure
        configure(getattr(model, 'device', 'cpu'))

    @diagnostics.profiled('prepare')
    def prepare(self, chunk, request, image):
        model = self.model
        inputs = model.video_backbone.preprocess_input_for_inference(
            prompt=request['prompt'], first_frame_image=image, num_frames=9,
            height=request['height'], width=request['width'], seed=request['seed'],
            num_inference_steps=10, shift=request['shift'], tiled=request['tiled'],
            prompt_embed_cache=self.prompt_cache, pin_first_frame=True,
            initial_latents=chunk.video,
        )
        if len(self.prompt_cache) > 32:
            self.prompt_cache.popitem(last=False)
        ref = inputs['first_frame_latents']
        if chunk.video is None:
            chunk.video = inputs['latents']
            chunk.action = torch.randn(
                (1, 32, model.action_dim), dtype=model.dtype, device=model.device,
                generator=torch.Generator(device=model.device).manual_seed(request['seed']),
            )
        chunk.video[:, :, :ref.shape[2]] = ref
        if model.uses_proprioception:
            inputs['proprio'] = request['proprio'].to(device=model.device, dtype=model.dtype)
        inputs.pop('latents')
        inputs.pop('noise', None)
        return inputs, model._resolve_inactive_action_dims(None, model.device)

    @diagnostics.profiled('denoise_step')
    def step(self, chunk, request, prepared):
        model = self.model
        inputs, inactive = prepared
        index = chunk.stage
        (tv, ta), (nv, na) = request['schedule'][index:index + 2]
        train_v = float(model.video_scheduler.num_train_timesteps)
        train_a = float(model.action_scheduler.num_train_timesteps)
        dv = nv / train_v - tv / train_v
        # The native sampler keeps masked dimensions on the analytic noise path.
        if inactive is not None and chunk.inactive_noise is None:
            chunk.inactive_noise = chunk.action[..., inactive].detach().clone() / (ta / train_a)
        if os.environ.get('OPENWAM_CHUNK_TIMESTEP_CACHE') == '1':
            key = (ta, tv, str(model.device), model.dtype)
            if key not in self.timestep_cache:
                self.timestep_cache[key] = (
                    torch.tensor([ta], device=model.device, dtype=model.dtype),
                    torch.tensor([tv], device=model.device, dtype=model.dtype))
            action_time, video_time = self.timestep_cache[key]
        else:
            action_time = torch.tensor([ta], device=model.device, dtype=model.dtype)
            video_time = torch.tensor([tv], device=model.device, dtype=model.dtype)
        with diagnostics.region('model_forward'):
            if os.environ.get('OPENWAM_CHUNK_CUDA_GRAPH') == '1':
                from openwam.deploy.chunk_graph import ForwardGraph, signature
                forward_inputs = dict(inputs, latents=chunk.video, timestep=video_time)
                key = signature((chunk.action, action_time, forward_inputs))
                if self.forward_graph is None or self.forward_graph.key != key:
                    self.forward_graph = None
                    self.forward_graph = ForwardGraph(model, chunk.action, action_time, forward_inputs)
                video_flow, action_flow = self.forward_graph.replay(chunk.action, action_time, forward_inputs)
            else:
                video_flow, action_flow = model.forward(
                    chunk.action, action_time,
                    **inputs, latents=chunk.video,
                    timestep=video_time,
                )
        with diagnostics.region('latent_action_update'):
            chunk.video = chunk.video + video_flow * dv
            ref = inputs['first_frame_latents']
            chunk.video[:, :, :ref.shape[2]] = ref
            chunk.action = model.action_scheduler.flow_step(action_flow, ta / train_a, na / train_a, chunk.action)
            if chunk.inactive_noise is not None:
                chunk.action[..., inactive] = chunk.inactive_noise * (na / train_a)
        chunk.stage += 1

    @diagnostics.profiled('actions_d2h_convert')
    def actions(self, chunk):
        actions = chunk.action.squeeze(0).float().cpu().numpy()
        normalizer = getattr(self.model, 'normalizer', None)
        return normalizer.unnormalize(actions) if normalizer is not None else actions

    @diagnostics.profiled('forecast_decode')
    def decode(self, chunk, request):
        if os.environ.get('OPENWAM_CHUNK_SELECTED_FRAMES') == '1':
            frames = self.model.video_backbone.decode_selected_video_frames(
                chunk.video, indices=(2, 4, 6), tiled=request['tiled'])
            return dict(zip((chunk.start + offset for offset in (8, 16, 24)), frames))
        frames = self.model.video_backbone.decode_video(chunk.video, tiled=request['tiled'])
        if len(frames) != 9:
            raise RuntimeError(f'Expected 9 decoded frames, got {len(frames)}')
        diagnostics.mark('decoded_frame_selection', decoded=9, used_indices=[2, 4, 6])
        return {chunk.start + offset: frames[offset // 4] for offset in (8, 16, 24)}

    def device_context(self):
        device = torch.device(self.model.device)
        return torch.cuda.device(device) if device.type == 'cuda' else nullcontext()

    def sync(self):
        device = torch.device(self.model.device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def memory(self):
        device = torch.device(self.model.device)
        if device.type == 'cuda':
            return {'allocated': torch.cuda.max_memory_allocated(device),
                    'reserved': torch.cuda.max_memory_reserved(device)}
        return None


class ChunkPipeline:
    def __init__(self, workers, speculative_steps=3):
        if len(workers) not in (1, 4) or speculative_steps not in (2, 3):
            raise ValueError('Chunk pipeline requires 1 or 4 workers and 2 or 3 speculative steps per interval')
        self.workers = workers
        self.speculative_steps = speculative_steps
        self.executors = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'joint-{i}')
                          for i in range(len(workers))]
        self.lock = Lock()
        self.chunks = {}
        self.pending = {}
        self.forecasts = []
        self.start = 0
        self.signature = None
        self.episode = -1
        self.trace_profiler = None
        self.trace_done = False
        self.capture_path = os.environ.get('OPENWAM_CHUNK_CAPTURE_PATH')

    def _reset(self):
        for stop, _ in self.pending.values():
            stop.set()
        error = None
        try:
            for future in [job for _, job in self.pending.values()] + self.forecasts:
                try:
                    future.result()
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
        finally:
            if diagnostics.ENABLED:
                for executor in self.executors:
                    executor.submit(diagnostics.flush, True).result()
            self._stop_trace()
            self.pending.clear()
            self.forecasts.clear()
            self.chunks.clear()
            self.start = 0
            self.signature = None
            self.episode += 1
            for worker in self.workers:
                if hasattr(worker, 'command'):
                    worker.reset()

    def _stop_trace(self):
        if self.trace_profiler is not None:
            import os
            from pathlib import Path
            profiler, self.trace_profiler = self.trace_profiler, None
            profiler.stop()
            profiler.export_chrome_trace(str(Path(os.environ['OPENWAM_CHUNK_DIAGNOSTICS_DIR']) /
                                            f'trace-{os.getpid()}.json'))
            self.trace_done = True

    def _submit(self, worker, function, *args):
        if not diagnostics.ENABLED:
            return self.executors[worker].submit(self._run, worker, function, *args)
        submitted = perf_counter()
        metadata = {**diagnostics.metadata(), 'episode': self.episode, 'round_start': self.start,
                    'worker': worker, 'phase': function.__name__.lstrip('_')}
        def run():
            with diagnostics.context(**metadata):
                diagnostics.mark('task_start', submitted_s=submitted,
                                 queue_ms=(perf_counter() - submitted) * 1000)
                try:
                    return self._run(worker, function, *args)
                finally:
                    diagnostics.mark('task_end')
        return self.executors[worker].submit(run)

    def _run(self, worker, function, *args):
        with self.workers[worker].device_context():
            if not diagnostics.ENABLED:
                return function(*args)
            model = getattr(self.workers[worker], 'model', None)
            import os
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
            physical_gpu = visible[worker] if worker < len(visible) else None
            with diagnostics.context(device=str(getattr(model, 'device', 'cpu')), physical_gpu=physical_gpu):
                diagnostics.flush()
                try:
                    return function(*args)
                finally:
                    diagnostics.flush()

    def reset(self):
        with self.lock:
            self._reset()

    def close(self):
        with self.lock:
            try:
                self._reset()
            finally:
                for executor in self.executors:
                    executor.shutdown(wait=True)
                for worker in self.workers:
                    if hasattr(worker, 'command'):
                        worker.close(force=True)

    @torch.inference_mode()
    def _finish(self, chunk, request):
        worker = self.workers[chunk.worker]
        if hasattr(worker, 'command'):
            return worker.command('finish', [chunk], request)
        chunk.online_step_host_ms.clear()
        with diagnostics.context(chunk_start=chunk.start, condition='real'):
            prepared = worker.prepare(chunk, request, request['first_frame_image'])
        while chunk.stage < 10:
            step_started = perf_counter()
            with diagnostics.context(chunk_start=chunk.start, stage=chunk.stage, condition='real'):
                worker.step(chunk, request, prepared)
            chunk.online_step_host_ms.append((perf_counter() - step_started) * 1000)
        actions = worker.actions(chunk)
        worker.sync()
        return actions

    @torch.inference_mode()
    def _forecast(self, chunk, request):
        started = perf_counter()
        worker = self.workers[chunk.worker]
        if hasattr(worker, 'command'):
            return worker.command('forecast', [chunk], request)
        with diagnostics.context(chunk_start=chunk.start):
            frames = worker.decode(chunk, request)
        return frames, (perf_counter() - started) * 1000

    @torch.inference_mode()
    def _advance(self, worker_index, chunks, request, forecast, stop):
        worker = self.workers[worker_index]
        forecast_wait_started = perf_counter()
        while not stop.is_set():
            try:
                frames, decode_ms = forecast.result(timeout=0.001)
                break
            except TimeoutError:
                if forecast.done():
                    raise
        else:
            return None, None
        diagnostics.mark('forecast_ready', wait_ms=(perf_counter() - forecast_wait_started) * 1000)
        if hasattr(worker, 'command'):
            return worker.command('advance', chunks, request, stop=stop,
                                  forecast=(frames, decode_ms), budget=self.speculative_steps)
        started = perf_counter()
        prepared = {}
        # Round robin gives the one-card comparison the same stage budget and
        # prioritizes the nearest chunk on each pass.
        for _ in range(self.speculative_steps):
            for chunk in chunks:
                if stop.is_set():
                    return decode_ms, (perf_counter() - started) * 1000
                if chunk.stage >= 3 * self.speculative_steps:
                    continue
                elapsed = perf_counter()
                if chunk.start not in prepared:
                    with diagnostics.context(chunk_start=chunk.start, condition='predicted'):
                        prepared[chunk.start] = worker.prepare(chunk, request, [frames[chunk.start]])
                    if stop.is_set():
                        return decode_ms, (perf_counter() - started) * 1000
                with diagnostics.context(chunk_start=chunk.start, stage=chunk.stage, condition='predicted'):
                    worker.step(chunk, request, prepared[chunk.start])
                # Bound the outstanding work to one GPU step and timestamp the
                # point at which this node really became available.
                with diagnostics.region('step_sync', gpu=False, chunk_start=chunk.start):
                    worker.sync()
                chunk.completed_at.append(perf_counter())
                chunk.speculative_ms += (perf_counter() - elapsed) * 1000
                diagnostics.flush()
        return decode_ms, (perf_counter() - started) * 1000

    def generate(self, request, *, arrived=None):
        arrived = perf_counter() if arrived is None else arrived
        with self.lock:
            if self.capture_path:
                from pathlib import Path
                path = Path(self.capture_path)
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    snapshot = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                                for key, value in request.items()}
                    torch.save(snapshot, path)
                self.capture_path = None
            with diagnostics.context(episode=self.episode, round_start=self.start, phase='request'):
                try:
                    with diagnostics.region('generate', gpu=False):
                        return self._generate(request, arrived)
                finally:
                    diagnostics.flush()

    def _generate(self, request, arrived):
        signature = tuple(request[key] for key in ('prompt', 'height', 'width', 'seed', 'tiled', 'shift', 'schedule'))
        if signature != self.signature:
            self._reset()
            self.signature = signature

        start = self.start
        if diagnostics.TRACE and not self.trace_done and start == 16:
            self.trace_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False,
                experimental_config=torch._C._profiler._ExperimentalConfig(profile_all_threads=True))
            self.trace_profiler.start()
        slot = (start // 8) % len(self.workers)
        head = self.chunks.get(start, Chunk(start, slot))
        wait_started = perf_counter()
        decode_ms = None
        background_ms = None
        for worker, (_, future) in self.pending.items():
            result = future.result()
            if worker == slot:
                decode_ms, background_ms = result
        self.pending.clear()
        if start == 56:
            self._stop_trace()
        ready_stage = sum(completed <= arrived for completed in head.completed_at)
        takeover_stage = head.stage
        online_steps = 10 - head.stage
        wait_ms = (perf_counter() - wait_started) * 1000
        diagnostics.mark('barrier_done', request_arrived_s=arrived, wait_begin_s=wait_started,
                         wait_ms=wait_ms, head_worker=slot, takeover_stage=takeover_stage,
                         actual_episode=self.episode, actual_round_start=start)
        online_started = perf_counter()
        actions = self._submit(slot, self._finish, head, request).result()
        actions_ready = perf_counter()
        diagnostics.mark('actions_ready', actual_episode=self.episode, actual_round_start=start,
                         request_arrived_s=arrived, action_ready_s=actions_ready)

        self.chunks.pop(start, None)
        self.start += 8
        groups = {}
        for offset in (8, 16, 24):
            future_start = start + offset
            worker = (future_start // 8) % len(self.workers)
            chunk = self.chunks.setdefault(future_start, Chunk(future_start, worker))
            groups.setdefault(worker, []).append(chunk)
        # Retire completed decode tasks and surface failures even if every
        # consumer was interrupted before the frames arrived.
        remaining_forecasts = []
        for previous in self.forecasts:
            if previous.done():
                previous.result()
            else:
                remaining_forecasts.append(previous)
        forecast = self._submit(slot, self._forecast, head, request)
        self.forecasts = [*remaining_forecasts, forecast]
        for worker, chunks in groups.items():
            stop = Event()
            future = self._submit(worker, self._advance, worker, chunks, request, forecast, stop)
            self.pending[worker] = (stop, future)

        return {'video': None, 'actions': actions, 'sampler_stats': {
            'sampler': 'chunk-pipeline', 'cold': start < 24, 'chunk_start': start,
            'head_worker': slot, 'ready_stage_at_observation': ready_stage,
            'ready_stage_at_takeover': takeover_stage,
            'online_steps': online_steps, 'background_wait_ms': wait_ms,
            'online_finish_ms': (actions_ready - online_started) * 1000,
            'online_step_host_ms': list(head.online_step_host_ms),
            'head_speculative_ms': head.speculative_ms, 'prior_decode_ms': decode_ms,
            'prior_worker_background_ms': background_ms,
            'action_ready_ms': (actions_ready - arrived) * 1000,
            'worker_peak_memory': [worker.memory() for worker in self.workers],
        }}
