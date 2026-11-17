"""Controlled extension of the frozen baseline to future-video latent generation."""

import ast
import copy
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import time


@contextmanager
def count_video_stages(stage_type, model, counts):
    original = stage_type.full

    def counted(stage, videos, *args, **kwargs):
        result = original(stage, videos, *args, **kwargs)
        if stage.arch is model and videos.ndim == 5 and videos.shape[2] > 1:
            counts['full_video_calls'] += 1
            counts['future_latents'] = videos.shape[2] - 1
        return result

    stage_type.full = counted
    try:
        yield
    finally:
        stage_type.full = original


def instrument(source):
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    stop = next(index for index, node in enumerate(main.body)
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'cases'
                                                       for target in node.targets))
    main.body = main.body[:stop] + ast.parse('run_video_experiment(locals())').body
    return ast.fix_missing_locations(tree)


def launch(script):
    script = Path(script)
    def run(context):
        context['__file__'] = str(script)
        context['save'] = namespace['save']
        return run_video_experiment(context)

    namespace = {'__name__': '__main__', '__file__': str(script), 'run_video_experiment': run}
    exec(compile(instrument(script.read_text()), str(script), 'exec'), namespace)


def run_video_experiment(context):
    import numpy as np
    import torch
    from paraflow.infer.benchmark import exclusive_gpu_guard
    from openwam.deploy.denoising_common import JointStages

    args = context['args']
    output = args.output
    guard = exclusive_gpu_guard()
    case = context['Case'](0)
    original_config = case.sampler.config
    model = case.model
    original_sample = case.sampler.sample
    counts = {'full_video_calls': 0, 'future_latents': 0}

    def sample(*positional, **keywords):
        result = original_sample(*positional, **keywords)
        if case.sampler.config.visual_conditioning == 'joint':
            video = result[0]
            if video.ndim != 5 or video.shape[2] <= 1 or not torch.isfinite(video).all():
                raise RuntimeError('Missing or nonfinite future-video output')
        return result

    records = {}
    receipt = {'completed': False, 'timings': records, 'closed_loop': 'not_run',
               'boundary': 'CPU cameras/state to CPU actions plus generated video latents; no pixel decode',
               'attention_mask_mode': model.mot_driver.attention_mask_mode,
               'same_checkpoint': True, 'mutual_checkpoint': False,
               'historical_script_sha256': hashlib.sha256(Path(context['__file__']).read_bytes()).hexdigest()
               if '__file__' in context else 'see frozen source manifest',
               'execution_note': 'first-frame uses historical controlled engine; full-video uses original native operators without isolated engine',
               'measurement_overhead': 'includes per-request finite checks on actions and generated video latents',
               'correctness': 'repeatability/finite checks only, not cross-algorithm equivalence'}
    save = context['save']
    case.sampler.sample = sample

    def request(index):
        value = np.asarray(case.request(index))
        if not np.isfinite(value).all():
            raise RuntimeError('Nonfinite actions')
        return value

    try:
        stage_counter = count_video_stages(JointStages, model, counts)
        stage_counter.__enter__()
        for name in ('first_frame_controlled', 'full_video_action_history', 'full_video_phase_history'):
            guard()
            if name != 'first_frame_controlled':
                if case.engine is not None:
                    native_vae = case.engine.native_vae
                    case.engine.close()
                    model.video_backbone.vae = native_vae
                    case.engine = None
                case.sampler.engine = None
                case.sampler.config = replace(original_config, visual_conditioning='joint',
                    history_scope='joint_video_action' if name.endswith('phase_history') else 'action_only',
                    algorithm_revision='phase_euler_v1' if name.endswith('phase_history') else original_config.algorithm_revision)
            reference = []
            case.reset()
            torch.manual_seed(52)
            for index in range(4):
                reference.append(request(index).copy())
            case.reset()
            torch.manual_seed(52)
            for index, expected in enumerate(reference):
                np.testing.assert_allclose(request(index), expected, atol=0.04, rtol=0.02)
            case.reset()
            torch.manual_seed(52)
            for index in range(args.warmup):
                request(index)
            counts.update(full_video_calls=0, future_latents=0)
            elapsed = []
            for index in range(args.samples):
                if index % 10 == 0:
                    guard()
                torch.cuda.synchronize()
                started = time.perf_counter()
                request(args.warmup + index)
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter() - started) * 1000)
            guard()
            if name != 'first_frame_controlled' and not counts['full_video_calls']:
                raise RuntimeError('No full-video forward executed; refusing to label this video generation')
            records[name] = {'median_ms': statistics.median(elapsed), 'p95_ms': float(np.percentile(elapsed, 95)),
                             'samples_ms': elapsed, 'video_work': dict(counts),
                             'config': copy.copy(vars(case.sampler.config)),
                             'repeatability': 'passed', 'warmup': args.warmup}
            save(output / 'video_summary.json', receipt)
            print(name, records[name]['median_ms'], 'ms', flush=True)
        receipt['completed'] = True
        receipt['timings']['baseline'] = records['first_frame_controlled']
        save(output / 'summary.json', receipt)
        save(output / 'video_summary.json', receipt)
    finally:
        stage_counter.__exit__(None, None, None)
        case.sampler.sample = original_sample
        if case.engine is not None:
            case.engine.close()
        for lock in context['locks']:
            lock.close()
