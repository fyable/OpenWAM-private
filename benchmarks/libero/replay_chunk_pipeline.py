"""Replay a fixed E8 observation trace with a controlled action execution interval."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from benchmarks.libero.replay_joint import load_observations, resolve_checkpoint
from openwam.deploy.server import _load_deploy_yaml, build_server_from_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--ckpt-dir', required=True)
    parser.add_argument('--ckpt-name')
    parser.add_argument('--sampler', choices=('native', 'chunk-pipeline'), required=True)
    parser.add_argument('--devices', default='cuda:0')
    parser.add_argument('--speculative-steps', type=int, choices=(2, 3), default=3)
    parser.add_argument('--execution-ms', type=float, required=True, help='Measured robot time for eight actions')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Native replay NPZ (same observation trace)')
    args = parser.parse_args()
    if not np.isfinite(args.execution_ms) or args.execution_ms < 0:
        parser.error('--execution-ms must be finite and nonnegative')
    actions_path = args.output.with_suffix('.npz')
    if args.output.suffix != '.json' or any(path.exists() for path in (args.output, actions_path)):
        parser.error('Use a new .json output path and a new matching .npz path')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU replay requires CUDA')

    ckpt_name = resolve_checkpoint(args.ckpt_dir, args.ckpt_name)
    observations = load_observations(args.observations)
    trace_hash = hashlib.sha256()
    with args.observations.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            trace_hash.update(block)
    observation_sha256 = trace_hash.hexdigest()
    devices = args.devices.split(',')
    if args.sampler == 'native' and len(devices) != 1:
        parser.error('Native replay uses one device')
    cfg = _load_deploy_yaml()
    for key, value in {
        'inference.sampler': args.sampler, 'inference.denoise_steps': 10,
        'inference.inference_horizon': 8, 'inference.inference_mode': 'sync',
        'inference.denoise_mode': 'sync', 'inference.chunk_devices': devices,
        'inference.chunk_speculative_steps': args.speculative_steps,
        'optimization.compile.enabled': False, 'optimization.dit_cache.enabled': False,
        'optimization.decode_video': False,
    }.items():
        OmegaConf.update(cfg, key, value)
    server = build_server_from_config(cfg, args.ckpt_dir, device=devices[0], ckpt_name=ckpt_name)
    engine = server.engine
    rows, actions = [], []
    episode_index = 0
    try:
        for index, (image, prompt, state, episode_start) in enumerate(zip(*observations)):
            if episode_start:
                engine.reset()
                episode_index = 0
            elif args.execution_ms:
                time.sleep(args.execution_ms / 1000)
            conditions = {'first_frame_image': [Image.fromarray(image)], 'prompt': str(prompt),
                          'proprio': state, 'seed': 42}
            started = time.perf_counter()
            result = engine.generate(conditions)
            elapsed = (time.perf_counter() - started) * 1000
            stats = result.get('sampler_stats', {})
            rows.append({'index': index, 'episode_index': episode_index, 'full_request_ms': elapsed,
                         'action_ready_ms': elapsed, **stats})
            actions.append(result['actions'][:8])
            episode_index += 1
    finally:
        server.shutdown()

    warm = [row for row in rows if row['episode_index'] >= 3]
    latency = [row['action_ready_ms'] for row in warm]
    target_stage = 3 * args.speculative_steps
    report = {
        'sampler': args.sampler, 'devices': devices, 'speculative_steps': args.speculative_steps,
        'execution_ms': args.execution_ms, 'observation_sha256': observation_sha256,
        'observations': str(args.observations.resolve()), 'rows': rows,
        'steady_requests': len(warm),
        'p50_ms': float(np.percentile(latency, 50)) if latency else None,
        'p95_ms': float(np.percentile(latency, 95)) if latency else None,
        'ready_at_target_rate': (sum(row['ready_stage_at_observation'] == target_stage for row in warm) / len(warm)
                                 if args.sampler == 'chunk-pipeline' and warm else None),
    }
    actions = np.stack(actions)
    if args.baseline:
        with np.load(args.baseline, allow_pickle=False) as baseline:
            if (str(baseline['observation_sha256']) != observation_sha256
                    or baseline['actions'].shape != actions.shape):
                raise ValueError('Baseline must use the same observation trace and action shape')
            native_actions = baseline['actions']
        baseline_report = json.loads(args.baseline.with_suffix('.json').read_text())
        if baseline_report['sampler'] != 'native' or baseline_report['execution_ms'] != args.execution_ms:
            raise ValueError('Baseline must be native N10 with the same action execution interval')
        if report['p50_ms'] and baseline_report['p50_ms']:
            report['p50_speedup_vs_native'] = baseline_report['p50_ms'] / report['p50_ms']
        absolute_error = np.abs(actions - native_actions)
        cold = observations[3]
        report['first_chunk_max_abs_error'] = float(absolute_error[cold].max())
        report['executed_actions_mean_abs_error'] = float(absolute_error.mean())
        report['gripper_sign_agreement'] = float(np.mean((actions[..., -1] > 0) == (native_actions[..., -1] > 0)))
        for row, error in zip(rows, absolute_error):
            row['executed_actions_max_abs_error'] = float(error.max())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(actions_path, actions=actions, observation_sha256=observation_sha256)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
