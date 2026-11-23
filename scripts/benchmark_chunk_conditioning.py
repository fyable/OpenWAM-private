"""Real-model frame, condition and step equivalence gate for opt-in candidates."""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwam.deploy.chunk_pipeline import Chunk, ChunkWorker
from openwam.deploy.model_loader import load_from_checkpoint_dir
from scripts.benchmark_chunk_submission import difference, distribution, move_tree


FLAGS = ('OPENWAM_CHUNK_SELECTED_FRAMES', 'OPENWAM_CHUNK_REUSE_NOISE', 'OPENWAM_CHUNK_TIMESTEP_CACHE',
         'OPENWAM_CHUNK_CUDA_GRAPH', 'OPENWAM_CHUNK_VAE_GPU_ACCUM', 'OPENWAM_CHUNK_VAE_GRAPH')


def compare(actual, expected, path='root'):
    if isinstance(actual, torch.Tensor):
        return {path: difference(actual.float().cpu().numpy(), expected.float().cpu().numpy())}
    if isinstance(actual, np.ndarray):
        return {path: difference(actual, expected)}
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f'Condition keys differ: {path}')
        return {name: check for key in actual for name, check in compare(actual[key], expected[key], f'{path}.{key}').items()}
    if isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f'Condition length differs: {path}')
        return {name: check for index, value in enumerate(actual)
                for name, check in compare(value, expected[index], f'{path}.{index}').items()}
    if actual != expected:
        raise AssertionError(f'Condition scalar differs: {path}')
    return {}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    torch.set_num_interop_threads(1)
    desired = {key: os.environ.get(key, '0') for key in FLAGS}
    _, model = load_from_checkpoint_dir(args.checkpoint, 'cuda:0', 'checkpoint_step_10690.safetensors')
    worker = ChunkWorker(model)
    snapshots = {stage: move_tree(torch.load(args.reference / f'stage-{stage}.pt', weights_only=False), 'cuda:0')
                 for stage in (0, 3, 6, 9)}
    outcomes = {}
    modes = [('reference', {}), *[(key, {key: '1'}) for key, value in desired.items() if value == '1'],
             ('candidate', desired)]
    for mode, enabled in modes:
        for key in FLAGS:
            os.environ[key] = enabled.get(key, '0')
        records, preparation_times, decode_times = {}, [], []
        for stage, snapshot in snapshots.items():
            request = snapshot['request']
            chunk = Chunk(0, 0, stage=stage, video=snapshot['video'].clone(), action=snapshot['action'].clone(),
                          inactive_noise=move_tree(snapshot['inactive_noise'], 'cuda:0'))
            prepared = worker.prepare(chunk, request, request['first_frame_image'])
            records[f'conditions-stage{stage}'] = move_tree(prepared, 'cpu')
            worker.step(chunk, request, prepared)
            worker.sync()
            records[f'step-stage{stage}'] = {'video': chunk.video.cpu(), 'action': chunk.action.cpu()}
        for sample in range(40):
            snapshot = snapshots[9]
            request = snapshot['request']
            chunk = Chunk(0, 0, stage=9, video=snapshot['video'].clone(), action=snapshot['action'].clone())
            worker.sync()
            begin = perf_counter()
            frames = worker.decode(chunk, request)
            worker.sync()
            decoded_ms = (perf_counter() - begin) * 1000
            begin = perf_counter()
            prepared = worker.prepare(chunk, request, [frames[8]])
            worker.sync()
            prepared_ms = (perf_counter() - begin) * 1000
            if sample >= 10:
                decode_times.append(decoded_ms)
                preparation_times.append(prepared_ms)
            if sample == 39:
                records['selected_frames'] = {key: np.asarray(frame) for key, frame in frames.items()}
                records['forecast_condition'] = move_tree(prepared, 'cpu')
        cold = Chunk(0, 0)
        prepared = worker.prepare(cold, request, request['first_frame_image'])
        worker.sync()
        records['cold_start'] = {'video': cold.video.cpu(), 'action': cold.action.cpu(), 'prepared': move_tree(prepared, 'cpu')}
        outcomes[mode] = {'records': records, 'decode': distribution(decode_times), 'prepare': distribution(preparation_times)}
    checks = {f'{mode}.{name}': check for mode, row in outcomes.items() if mode != 'reference'
              for name, check in compare(row['records'], outcomes['reference']['records']).items()}
    passed = bool(checks) and all(row['passed'] for row in checks.values())
    result = {'passed': passed, 'checks': checks, 'flags': desired,
              'timings': {mode: {key: row[key] for key in ('decode', 'prepare')} for mode, row in outcomes.items()},
              'note': 'Full VAE decoded before frame selection; cold start and every schedule stage checked.'}
    (args.output / 'parity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'checks'}), flush=True)
    if not passed:
        raise AssertionError('Candidate differs from unchanged real-model reference')


if __name__ == '__main__':
    main()
