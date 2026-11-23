"""Gated submission experiment and paired LIBERO acceptance, on physical 0/2/4/6."""

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from threading import Event, Thread
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_chunk_pipeline_smoke import summarize_method


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = '34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def own_pids():
    output = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
    parents = {int(parts[0]): int(parts[1]) for line in output.splitlines() if (parts := line.split())}
    owned = {os.getpid()}
    while True:
        expanded = owned | {pid for pid, parent in parents.items() if parent in owned}
        if expanded == owned:
            return owned
        owned = expanded


def gpu_snapshot(uuids):
    utilization = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,clocks.sm,power.draw',
        '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    parsed = [row for row in csv.reader(io.StringIO(processes), skipinitialspace=True) if row]
    owned = own_pids()
    external = [row for row in parsed if row[0] in uuids and int(row[1]) not in owned]
    return {'time_s': time.time(), 'perf_counter_s': time.perf_counter(),
            'utilization_csv': utilization, 'processes': parsed, 'external': external}


def monitored_run(command, environment, directory, uuids):
    directory.mkdir(parents=True)
    while True:
        snapshot = gpu_snapshot(uuids)
        if not snapshot['external']:
            break
        print('Waiting for target GPUs:', snapshot['external'], flush=True)
        with (directory / 'waiting.jsonl').open('a') as output:
            output.write(json.dumps(snapshot) + '\n')
        time.sleep(10)
    stopped = Event()
    interference = []
    errors = []

    def monitor():
        with (directory / 'gpu-monitor.jsonl').open('w') as output:
            while not stopped.is_set():
                try:
                    row = gpu_snapshot(uuids)
                    output.write(json.dumps(row) + '\n')
                    output.flush()
                    interference.extend(row['external'])
                except Exception as error:
                    errors.append(repr(error))
                stopped.wait(1)

    write_json(directory / 'command.json', {'argv': command,
               'environment': {key: environment.get(key) for key in (
                   'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                   'CUDA_VISIBLE_DEVICES', 'GPUS', 'SERVER_PYTHON', 'LIBERO_PYTHON',
                   'OPENWAM_CHUNK_DIAGNOSTICS', 'OPENWAM_CHUNK_TRACE')},
               'begin_time_s': time.time()})
    watcher = Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        with (directory / 'run.log').open('w') as output:
            result = subprocess.run(command, env=environment, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    finally:
        stopped.set()
        watcher.join()
    validation = {'valid': not interference and not errors and result.returncode == 0,
                  'exit_code': result.returncode, 'external': interference, 'monitor_errors': errors,
                  'end_time_s': time.time()}
    write_json(directory / 'validation.json', validation)
    if result.returncode != 0:
        raise RuntimeError(f'Experiment failed; inspect {directory / "run.log"}')
    return validation['valid']


def read_phase(directory):
    trials, metrics = summarize_method(directory)
    states, lengths, episodes = {}, {}, {}
    summary = json.loads((directory / 'summary.json').read_text())
    for task in summary['task_results']:
        result = json.loads(Path(task['result_paths'][0]).read_text())
        for trial in result['trials']:
            key = (result['suite'], result['task_id'], trial['trial'])
            states[key] = trial['init_state_sha256']
            lengths[key] = trial['policy_steps']
    for path in sorted((directory / 'logs' / 'clients').glob('*.log')):
        for line in path.read_text().splitlines():
            if line.startswith('[CLIENT_TIMING] '):
                row = json.loads(line.split(' ', 1)[1])
                if row['step'] >= 24 and row['step'] % 8 == 0:
                    episodes.setdefault((path.name, row['episode']), []).append(row['observation_to_action_ms'])
    if len(trials) != 20 or len(episodes) != 20:
        raise ValueError('Expected exactly 20 complete paired episodes with steady requests')
    return {'metrics': metrics, 'successes': sum(trials.values()), 'states': states,
            'steps': lengths, 'episodes': episodes}


def eligible(candidate, baseline):
    return (candidate['successes'] == 20 and candidate['states'] == baseline['states'] and
            candidate['metrics']['client_chunk_requests_excluding_first_three']['p95_ms'] <=
            baseline['metrics']['client_chunk_requests_excluding_first_three']['p95_ms'] * 1.05)


def paired_bootstrap(candidate, baseline):
    def indexed(phase):
        values = {}
        for (filename, episode), samples in phase['episodes'].items():
            stem = filename.split('_attempt')[0]
            values[(stem, episode)] = samples
        return values
    candidate_rows, baseline_rows = indexed(candidate), indexed(baseline)
    if candidate_rows.keys() != baseline_rows.keys():
        raise ValueError('Episode log identities are not paired')
    keys = sorted(candidate_rows)
    rng = np.random.default_rng(42)
    gains = {50: [], 95: []}
    for _ in range(2000):
        selection = rng.integers(0, len(keys), size=len(keys))
        actual = np.concatenate([candidate_rows[keys[index]] for index in selection])
        reference = np.concatenate([baseline_rows[keys[index]] for index in selection])
        for quantile in gains:
            gains[quantile].append(1 - np.percentile(actual, quantile) / np.percentile(reference, quantile))
    return {f'p{quantile}_reduction_95ci': np.percentile(values, [2.5, 97.5]).tolist()
            for quantile, values in gains.items()}


def report(run, phases, candidate, gate):
    reference = phases['A1']
    for name, phase in phases.items():
        if phase['states'] != reference['states']:
            raise ValueError(f'Initial state mismatch: {name}')
    comparisons = []
    if candidate is not None:
        for actual, baseline in ((candidate + '1', 'A1'), (candidate + '2', 'A2')):
            current, control = phases[actual], phases[baseline]
            current_ms = current['metrics']['client_chunk_requests_excluding_first_three']
            control_ms = control['metrics']['client_chunk_requests_excluding_first_three']
            gain = 1 - current_ms['p50_ms'] / control_ms['p50_ms']
            tail = current_ms['p95_ms'] / control_ms['p95_ms']
            comparisons.append({'candidate': actual, 'baseline': baseline, 'p50_reduction': gain,
                                'p95_ratio': tail, 'passed': eligible(current, control) and gain >= .10,
                                'bootstrap': paired_bootstrap(current, control)})
    payload = {'candidate': candidate, 'process_gate': gate,
               'accepted': len(comparisons) == 2 and all(row['passed'] for row in comparisons),
               'comparisons': comparisons,
               'phases': {name: {'metrics': phase['metrics'], 'successes': phase['successes'],
                                'steps': {str(key): value for key, value in phase['steps'].items()}}
                          for name, phase in phases.items()}}
    write_json(run / 'acceptance.json', payload)
    lines = ['# 四卡提交优化实验', '', '| 配置 | 成功 | 客户端P50 ms | P95 ms | 屏障P50 ms |',
             '|---|---:|---:|---:|---:|']
    for name, phase in phases.items():
        metrics = phase['metrics']
        latency = metrics['client_chunk_requests_excluding_first_three']
        wait = metrics['server_pipeline_background_wait_ms']
        lines.append(f'| {name} | {phase["successes"]}/20 | {latency["p50_ms"]:.3f} | '
                     f'{latency["p95_ms"]:.3f} | {wait["p50_ms"]:.3f} |')
    lines += ['', f'多进程微基准门槛：{gate["process_gate_passed"]}。',
              f'最终候选：{candidate}；加速验收：{payload["accepted"]}。',
              'A=线程/CPU算子线程4，B=线程/CPU算子线程1，C=常驻进程/CPU算子线程1。',
              '两次比较和回合级配对bootstrap区间见 acceptance.json；不将请求视作独立样本。',
              'N10/H32/E8、后台三步、完整屏障不变；无CPU卸载、compile、DiT cache或环境睡眠。',
              '约1秒GPU监控未采样到外部计算进程不等于绝对独占；本实验仅覆盖20个固定回合。']
    (run / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'experiment' /
                        ('chunk_acceleration_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--checkpoint', default='/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO')
    parser.add_argument('--base-port', default='8995')
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 12:
        raise RuntimeError('Experiment requires at least 12 allowed CPUs')
    server_cpus, client_cpus = cpus[:8], cpus[8:12]
    wrappers = {}
    for name, affinity, interpreter in (
            ('server', server_cpus, ROOT / '.venv-openwam/bin/python'),
            ('client', client_cpus, ROOT / '.venv-libero/bin/python')):
        wrapper = run / f'{name}-python'
        prefix = 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n' if name == 'client' else ''
        wrapper.write_text('#!/bin/sh\n' + prefix + 'exec taskset -c ' +
                           shlex.quote(','.join(map(str, affinity))) + ' ' + shlex.quote(str(interpreter)) + ' "$@"\n')
        wrapper.chmod(0o755)
        wrappers[name] = str(wrapper)
    checkpoint_name = 'checkpoint_step_10690.safetensors'
    digest = hashlib.sha256()
    with (Path(args.checkpoint) / checkpoint_name).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint fingerprint mismatch')
    sources = [path for folder in ('openwam', 'benchmarks/libero', 'scripts', 'configs')
               for path in (ROOT / folder).rglob('*') if path.suffix in ('.py', '.sh', '.yaml', '.yml')]
    (run / 'source-sha256.txt').write_text(''.join(
        f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}\n' for path in sorted(sources)))
    gpu_rows = list(csv.reader(io.StringIO(subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True)),
        skipinitialspace=True))
    uuids = {row[1] for row in gpu_rows if int(row[0]) in (0, 2, 4, 6)}
    if len(uuids) != 4:
        raise ValueError('Physical GPUs 0,2,4,6 are required')
    write_json(run / 'manifest.json', {'gpus': gpu_rows, 'selected': [0, 2, 4, 6],
               'server_cpus': server_cpus, 'client_cpus': client_cpus,
               'checkpoint_sha256': digest.hexdigest(), 'python': sys.version,
               'created_at': datetime.now().isoformat(),
               'lscpu': subprocess.check_output(['lscpu', '-e=CPU,CORE,SOCKET,NODE'], text=True)})
    environment = dict(os.environ)
    environment.update({'SERVER_PYTHON': wrappers['server'], 'LIBERO_PYTHON': wrappers['client'],
                        'LIBERO_PATH': str(ROOT / 'third_party/LIBERO'),
                        'PYTHONPATH': f'{ROOT}:{ROOT / "third_party/LIBERO"}',
                        'GPUS': '0,2,4,6', 'REPLICAS_PER_GPU': '1', 'BASE_PORT': args.base_port,
                        'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
                        'OPENWAM_LIBERO_REQUEST_TIMING': '1', 'OPENWAM_REQUIRE_GPU_RESIDENT': '1',
                        'OPENWAM_LIBERO_STATE_HASH': '1', 'OPENWAM_CHUNK_INTEROP_THREADS': '1',
                        'OPENWAM_CHUNK_DIAGNOSTICS': '0', 'OPENWAM_CHUNK_TRACE': '0',
                        'OPENWAM_EXPERIMENT_SERVER_CPUS': ','.join(map(str, server_cpus))})
    for name in ('E2_MEMORY_MODE', 'OPENWAM_CHUNK_CAPTURE_PATH'):
        environment.pop(name, None)
    common = ['bash', 'benchmarks/libero/run_eval.sh', args.checkpoint, checkpoint_name,
              '--sampler', 'chunk-pipeline', '--chunk-speculative-steps', '3',
              '--denoise-steps', '10', '--inference-horizon', '8', '--inference-mode', 'sync',
              '--denoise-mode', 'sync', '--compile-enabled', 'false', '--seed', '42',
              '--render-gpus', '0', '--extra-deploy-arg', 'optimization.dit_cache.enabled=false',
              '--extra-deploy-arg', 'optimization.decode_video=false', '--extra-deploy-arg',
              'inference.request_timing=true', '--diagnostic', '--task-ids', '0', '--client-max-attempts', '1']

    def evaluate(name, smoke=False, capture=False):
        mode = name[0]
        env = dict(environment)
        for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[variable] = '4' if mode == 'A' else '1'
        if capture:
            env['OPENWAM_CHUNK_CAPTURE_PATH'] = str(run / 'request.pt')
        extra = ['--suites', 'spatial', '--num-trials', '1', '--smoke'] if smoke else [
            '--suites', 'spatial,goal,object,long', '--num-trials', '5']
        command = [*common, '--extra-deploy-arg',
                   'inference.chunk_worker_backend=' + ('process' if mode == 'C' else 'thread'), *extra]
        for attempt in range(1, 4):
            directory = run / f'{name}-attempt{attempt}'
            env['OUTPUT_DIR'] = str(directory / 'evaluation')
            print(f'Running {name}, attempt {attempt}: {directory}', flush=True)
            if monitored_run(command, env, directory, uuids):
                if smoke:
                    return directory / 'evaluation'
                phase = read_phase(directory / 'evaluation')
                phase['directory'] = str(directory / 'evaluation')
                return phase
        raise RuntimeError(f'{name}: three runs invalidated by interference; see monitoring logs')

    evaluate('A-fixture', smoke=True, capture=True)
    micro_environment = dict(environment)
    micro_environment['CUDA_VISIBLE_DEVICES'] = '0,2,4,6'
    micro_environment['OMP_NUM_THREADS'] = micro_environment['MKL_NUM_THREADS'] = '4'
    micro_environment['OPENBLAS_NUM_THREADS'] = '4'
    gate = None
    for attempt in range(1, 4):
        directory = run / f'micro-attempt{attempt}'
        command = [wrappers['server'], 'scripts/benchmark_chunk_submission.py', str(directory / 'results'),
                   '--request', str(run / 'request.pt'), '--checkpoint', args.checkpoint]
        print(f'Running controlled submission benchmark: {directory}', flush=True)
        if monitored_run(command, micro_environment, directory, uuids):
            gate = json.loads((directory / 'results/summary.json').read_text())
            break
    if gate is None or not gate['complete']:
        raise RuntimeError('No valid complete microbenchmark')
    write_json(run / 'micro-gate.json', gate)
    if not gate['parity_passed']:
        raise RuntimeError('Numerical parity failed; do not adopt a new backend')
    phases = {'A1': evaluate('A1'), 'B1': evaluate('B1')}
    if gate['process_gate_passed']:
        evaluate('C-smoke', smoke=True)
        phases['C1'] = evaluate('C1')
    candidates = [name[0] for name in ('B1', 'C1') if name in phases and eligible(phases[name], phases['A1'])]
    candidate = min(candidates, key=lambda name: (
        phases[name + '1']['metrics']['client_chunk_requests_excluding_first_three']['p50_ms'], name)) if candidates else None
    if candidate is not None:
        phases[candidate + '2'] = evaluate(candidate + '2')
        phases['A2'] = evaluate('A2')
    result = report(run, phases, candidate, gate)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
