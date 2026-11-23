"""Replay identical resident chunk states under single/thread/process submission."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import sys
from threading import Barrier
from time import perf_counter
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openwam.deploy.chunk_pipeline import Chunk, ChunkWorker
from openwam.deploy.model_loader import load_from_checkpoint_dir


STAGES = (0, 3, 6, 9)
MODES = ('thread4', 'thread1', 'process1')


def move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).clone()
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    return copy.deepcopy(value)


def distribution(values):
    return {'count': len(values), 'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95))}


def difference(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    return {'exact': bool(np.array_equal(actual, expected)),
            'passed': finite and bool(np.allclose(actual, expected, rtol=1e-3, atol=1e-4)),
            'max_abs': float(np.max(np.abs(delta))), 'rmse': float(np.sqrt(np.mean(delta ** 2)))}


def make_snapshots(args):
    request = torch.load(args.request, map_location='cpu', weights_only=False)
    with torch.cuda.device(0), torch.inference_mode():
        _, model = load_from_checkpoint_dir(args.checkpoint, 'cuda:0', args.checkpoint_name)
        worker = ChunkWorker(model)
        chunk = Chunk(0, 0)
        prepared = worker.prepare(chunk, request, request['first_frame_image'])
        for stage in STAGES:
            while chunk.stage < stage:
                worker.step(chunk, request, prepared)
            worker.sync()
            snapshot = {'stage': stage, 'request': move_tree(request, 'cpu'),
                        'prepared': move_tree(prepared, 'cpu'),
                        'video': move_tree(chunk.video, 'cpu'),
                        'action': move_tree(chunk.action, 'cpu'),
                        'inactive_noise': move_tree(chunk.inactive_noise, 'cpu')}
            torch.save(snapshot, args.output / f'stage-{stage}.pt')
            replay = Chunk(0, 0, stage=stage, video=chunk.video.clone(), action=chunk.action.clone(),
                           inactive_noise=move_tree(chunk.inactive_noise, model.device))
            for _ in range(1 if stage == 9 else 3):
                worker.step(replay, request, prepared)
            worker.sync()
            np.savez(args.output / f'reference-{stage}.npz',
                     video=replay.video.float().cpu().numpy(), action=replay.action.float().cpu().numpy())


class ReplayWorker:
    def __init__(self, args, rank):
        self.device = f'cuda:{rank}'
        self.rank = rank
        with torch.cuda.device(self.device):
            _, model = load_from_checkpoint_dir(args.checkpoint, self.device, args.checkpoint_name)
        self.worker = ChunkWorker(model)
        self.snapshots = {stage: move_tree(torch.load(
            args.output / f'stage-{stage}.pt', map_location='cpu', weights_only=False), self.device)
            for stage in STAGES}
        self.warmup = args.warmup
        self.measurements = args.measurements

    @torch.inference_mode()
    def run(self, stage, active, barrier):
        rows = []
        snapshot = self.snapshots[stage]
        selected = self.rank in active
        started_cpu = resource.getrusage(resource.RUSAGE_SELF)
        with torch.cuda.device(self.device):
            for sample in range(self.warmup + self.measurements):
                if selected:
                    chunk = Chunk(0, self.rank, stage=stage,
                                  video=snapshot['video'].clone(), action=snapshot['action'].clone(),
                                  inactive_noise=move_tree(snapshot['inactive_noise'], self.device))
                    self.worker.sync()
                barrier.wait(timeout=300)
                if selected:
                    begin = perf_counter()
                    steps = []
                    for _ in range(1 if stage == 9 else 3):
                        step_started = perf_counter()
                        self.worker.step(chunk, snapshot['request'], snapshot['prepared'])
                        self.worker.sync()
                        steps.append((perf_counter() - step_started) * 1000)
                    end = perf_counter()
                    if sample >= self.warmup:
                        rows.append({'begin_s': begin, 'end_s': end, 'steps_ms': steps,
                                     'round_ms': (end - begin) * 1000})
                barrier.wait(timeout=300)
            output = None
            if selected:
                output = {'video': chunk.video.float().cpu().numpy(),
                          'action': chunk.action.float().cpu().numpy()}
        ended_cpu = resource.getrusage(resource.RUSAGE_SELF)
        return {'rank': self.rank, 'pid': os.getpid(), 'samples': rows, 'output': output,
                'cpu_user_s': ended_cpu.ru_utime - started_cpu.ru_utime,
                'cpu_system_s': ended_cpu.ru_stime - started_cpu.ru_stime,
                'affinity': sorted(os.sched_getaffinity(0)),
                'os_threads': len(list(Path('/proc/self/task').iterdir())),
                'torch_threads': torch.get_num_threads()}


def serve_replay(connection, args, rank, barrier):
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        worker = ReplayWorker(args, rank)
        connection.send({'ready': True, 'pid': os.getpid()})
        while True:
            command = connection.recv()
            if command is None:
                break
            connection.send(worker.run(*command, barrier))
    except BaseException:
        barrier.abort()
        try:
            connection.send({'error': traceback.format_exc()})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


def receive(connection, process):
    deadline = perf_counter() + 1200
    while not connection.poll(0.1):
        if not process.is_alive() or perf_counter() > deadline:
            raise RuntimeError(f'Replay worker exited/timed out: {process.exitcode}')
    result = connection.recv()
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result


def evaluate_scenario(args, mode, repeat, stage, active, results):
    selected = [row for row in results if row['rank'] in active]
    slowest = [max(row['samples'][sample]['end_s'] for row in selected) -
               min(row['samples'][sample]['begin_s'] for row in selected)
               for sample in range(args.measurements)]
    reference = np.load(args.output / f'reference-{stage}.npz')
    for row in results:
        output = row.pop('output')
        if output is not None:
            row['parity'] = {name: difference(output[name], reference[name]) for name in output}
            row['step'] = distribution([value for sample in row['samples'] for value in sample['steps_ms']])
            row['round'] = distribution([sample['round_ms'] for sample in row['samples']])
    return {'mode': mode, 'repeat': repeat, 'stage': stage, 'active': list(active),
            'slowest': distribution([value * 1000 for value in slowest]),
            'batch_throughput_per_s': len(active) / (float(np.mean(slowest))),
            'workers': results}


def session(args):
    session_started = perf_counter()
    mode = args.mode
    torch.set_num_threads(4 if mode == 'thread4' else 1)
    torch.set_num_interop_threads(1)
    processes, connections, workers = [], [], []
    executors = [ThreadPoolExecutor(max_workers=1) for _ in range(4)]
    context = mp.get_context('spawn')
    barrier = context.Barrier(4) if mode == 'process1' else Barrier(4)
    try:
        if mode == 'process1':
            for rank in range(4):
                parent, child = context.Pipe()
                process = context.Process(target=serve_replay, args=(child, args, rank, barrier))
                process.start()
                child.close()
                connections.append(parent)
                processes.append(process)
                receive(parent, process)
        else:
            for rank in range(4):
                workers.append(executors[rank].submit(ReplayWorker, args, rank).result())
        (args.output / f'{args.repeat}-{mode}-startup.json').write_text(json.dumps({
            'startup_s': perf_counter() - session_started, 'pid': os.getpid(),
            'mode': mode, 'repeat': args.repeat, 'affinity': sorted(os.sched_getaffinity(0)),
            'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        }, indent=2) + '\n')
        scenarios = list(itertools.product(STAGES, [(rank,) for rank in range(4)] +
                                            list(itertools.combinations(range(4), 3))))
        offset = args.repeat * 7
        scenarios = scenarios[offset:] + scenarios[:offset]
        path = args.output / f'{args.repeat}-{mode}.jsonl'
        with path.open('x') as output:
            for stage, active in scenarios:
                begin = perf_counter()
                if mode == 'process1':
                    for connection in connections:
                        connection.send((stage, active))
                    results = [receive(connection, process) for connection, process in zip(connections, processes)]
                else:
                    jobs = [executor.submit(worker.run, stage, active, barrier)
                            for executor, worker in zip(executors, workers)]
                    results = [job.result() for job in jobs]
                row = evaluate_scenario(args, mode, args.repeat, stage, active, results)
                output.write(json.dumps(row) + '\n')
                output.flush()
                print(json.dumps({'mode': mode, 'repeat': args.repeat, 'stage': stage,
                                  'active': active, 'slowest': row['slowest'],
                                  'scenario_s': perf_counter() - begin}), flush=True)
    finally:
        barrier.abort()
        for connection, process in zip(connections, processes):
            try:
                if process.is_alive():
                    connection.send(None)
            except (OSError, EOFError):
                pass
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            connection.close()
        for executor in executors:
            executor.shutdown(wait=True)


def summarize(output):
    rows = [json.loads(line) for path in sorted(output.glob('[0-2]-*.jsonl'))
            for line in path.read_text().splitlines()]
    expected = {(repeat, mode, stage, active) for repeat in range(3) for mode in MODES
                for stage in STAGES for active in
                [*(tuple([rank]) for rank in range(4)), *itertools.combinations(range(4), 3)]}
    keys = [(row['repeat'], row['mode'], row['stage'], tuple(row['active'])) for row in rows]
    complete = len(keys) == len(expected) and set(keys) == expected
    parity = bool(rows) and all(
        sorted(worker['rank'] for worker in row['workers']) == list(range(4)) and all(
            set(worker.get('parity', {})) == {'video', 'action'} and
            all(check['passed'] for check in worker['parity'].values()) and bool(worker['samples'])
            for worker in row['workers'] if worker['rank'] in row['active']) for row in rows)
    comparisons = []
    for stage in (0, 3, 6):
        for active in itertools.combinations(range(4), 3):
            distributions = {}
            for mode in MODES:
                matching = [row for row in rows if row['stage'] == stage and
                            tuple(row['active']) == active and row['mode'] == mode]
                samples = []
                for row in matching:
                    selected = [worker for worker in row['workers'] if worker['rank'] in active]
                    for index in range(len(selected[0]['samples'])):
                        samples.append((max(worker['samples'][index]['end_s'] for worker in selected) -
                                        min(worker['samples'][index]['begin_s'] for worker in selected)) * 1000)
                if samples:
                    distributions[mode] = distribution(samples)
            if set(distributions) != set(MODES):
                continue
            baseline, candidate = distributions['thread1'], distributions['process1']
            gain = 1 - candidate['p50_ms'] / baseline['p50_ms']
            tail_ratio = candidate['p95_ms'] / baseline['p95_ms']
            comparisons.append({'stage': stage, 'active': active, 'modes': distributions,
                                'p50_reduction': gain, 'p95_ratio': tail_ratio,
                                'passed': gain >= .15 and tail_ratio <= 1.05})
    result = {'complete': complete, 'parity_passed': parity, 'comparisons': comparisons,
              'process_gate_passed': complete and parity and len(comparisons) == 12 and
              all(row['passed'] for row in comparisons)}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--checkpoint', default='/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO')
    parser.add_argument('--checkpoint-name', default='checkpoint_step_10690.safetensors')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--measurements', type=int, default=30)
    parser.add_argument('--phase', choices=('snapshots', 'session', 'all', 'summary'), default='all')
    parser.add_argument('--mode', choices=MODES)
    parser.add_argument('--repeat', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.warmup < 1 or args.measurements < 1:
        parser.error('warmup and measurements must be positive')
    if args.phase == 'snapshots':
        make_snapshots(args)
    elif args.phase == 'session':
        session(args)
    elif args.phase == 'summary':
        print(json.dumps(summarize(args.output), indent=2))
    else:
        if args.request is None:
            parser.error('--request is required')
        manifest = {'request_sha256': hashlib.sha256(args.request.read_bytes()).hexdigest(),
                    'warmup': args.warmup, 'measurements': args.measurements,
                    'affinity': sorted(os.sched_getaffinity(0)), 'torch': torch.__version__,
                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        base = [sys.executable, str(Path(__file__).resolve()), str(args.output),
                '--checkpoint', args.checkpoint, '--checkpoint-name', args.checkpoint_name,
                '--warmup', str(args.warmup), '--measurements', str(args.measurements)]
        subprocess.run([*base, '--phase', 'snapshots', '--request', str(args.request)], check=True)
        for repeat in range(3):
            order = MODES[repeat:] + MODES[:repeat]
            for mode in order:
                env = dict(os.environ)
                for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                    env[name] = '4' if mode == 'thread4' else '1'
                subprocess.run([*base, '--phase', 'session', '--mode', mode,
                                '--repeat', str(repeat)], env=env, check=True)
        print(json.dumps(summarize(args.output), indent=2))


if __name__ == '__main__':
    main()
