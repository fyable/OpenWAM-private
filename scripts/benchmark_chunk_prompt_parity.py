"""Check complete ten-step trajectories across the actual coverage prompts."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiment/chunk_autonomous_20261121/repair_source'
sys.path.insert(0, str(SOURCE))

import numpy as np
import torch
from openwam.deploy.chunk_pipeline import Chunk, ChunkWorker
from openwam.deploy.model_loader import load_from_checkpoint_dir
from scripts.benchmark_chunk_submission import difference, move_tree


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, model = load_from_checkpoint_dir(args.checkpoint, 'cuda:0', 'checkpoint_step_10690.safetensors')
    worker = ChunkWorker(model)
    request = torch.load(ROOT / 'experiment/chunk_acceleration_20261120_203804/request.pt', weights_only=False)
    request = move_tree(request, 'cuda:0')
    instructions = set()
    coverage = ROOT / 'experiment/chunk_coverage_20261121'
    for summary_path in coverage.glob('tasks-*-A-attempt*/evaluation/summary.json'):
        summary = json.loads(summary_path.read_text())
        for task in summary['task_results']:
            result = json.loads(Path(task['result_paths'][0]).read_text())
            instructions.add(result['instruction'])
    flags = ('OPENWAM_CHUNK_SELECTED_FRAMES', 'OPENWAM_CHUNK_REUSE_NOISE', 'OPENWAM_CHUNK_TIMESTEP_CACHE',
             'OPENWAM_CHUNK_CUDA_GRAPH', 'OPENWAM_CHUNK_VAE_GPU_ACCUM', 'OPENWAM_CHUNK_VAE_GRAPH')
    results = []
    for prompt in sorted(instructions):
        request['prompt'] = prompt
        trajectories = {}
        for mode in ('reference', 'graph'):
            for key in flags:
                os.environ[key] = '1' if mode == 'graph' else '0'
            worker.forward_graph = None
            chunk = Chunk(0, 0)
            prepared = worker.prepare(chunk, request, request['first_frame_image'])
            rows = []
            for _ in range(10):
                worker.step(chunk, request, prepared)
                worker.sync()
                rows.append({'video': chunk.video.float().cpu().numpy(), 'action': chunk.action.float().cpu().numpy()})
            trajectories[mode] = rows
        checks = [{name: difference(actual[name], reference[name]) for name in reference}
                  for actual, reference in zip(trajectories['graph'], trajectories['reference'])]
        row = {'prompt': prompt, 'checks': checks, 'passed': all(check['passed'] for step in checks for check in step.values())}
        results.append(row)
        print(json.dumps({'prompt': prompt, 'passed': row['passed'],
                          'max_abs': max(check['max_abs'] for step in checks for check in step.values())}), flush=True)
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    if not results or not all(row['passed'] for row in results):
        raise AssertionError('Full trajectory prompt parity failed')


if __name__ == '__main__':
    main()
