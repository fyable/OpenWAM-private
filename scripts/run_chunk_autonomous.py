"""Deadline-bound, resumable four-GPU optimization campaign."""

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_chunk_acceleration import CHECKPOINT_SHA256, gpu_snapshot, read_phase, paired_bootstrap
from scripts.summarize_chunk_pipeline_smoke import summarize_method


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants(pid):
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
    parents = {int(parts[0]): int(parts[1]) for row in rows.splitlines() if (parts := row.split())}
    owned = {pid}
    while True:
        expanded = owned | {child for child, parent in parents.items() if parent in owned}
        if expanded == owned:
            return owned
        owned = expanded


def stop_owned(pid, identity):
    if process_identity(pid) != identity:
        return
    owned = {child: process_identity(child) for child in descendants(pid)}
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for child, started in owned.items():
            if started is not None and process_identity(child) == started:
                try:
                    os.kill(child, signum)
                except ProcessLookupError:
                    pass
        if signum == signal.SIGTERM:
            time.sleep(5)


def micro_summary(directory):
    files = sorted(directory.glob('[0-2]-process1.jsonl'))
    rows = [json.loads(line) for path in files for line in path.read_text().splitlines()]
    keys = {(row['repeat'], row['stage'], tuple(row['active'])) for row in rows}
    import itertools
    expected = {(repeat, stage, active) for repeat in range(len(files)) for stage in (0, 3, 6, 9)
                for active in [*((rank,) for rank in range(4)), *itertools.combinations(range(4), 3)]}
    passed = bool(files) and len(rows) == 32 * len(files) and keys == expected and all(
        len(worker['samples']) == 30 and set(worker['parity']) == {'video', 'action'} and
        all(check['passed'] for check in worker['parity'].values())
        for row in rows for worker in row['workers'] if worker['rank'] in row['active'])
    values = [value * 1000 for row in rows if row['stage'] < 9 and len(row['active']) == 3
              for value in [max(worker['samples'][sample]['end_s'] for worker in row['workers']
                               if worker['rank'] in row['active']) -
                            min(worker['samples'][sample]['begin_s'] for worker in row['workers']
                                if worker['rank'] in row['active']) for sample in range(30)]]
    import numpy as np
    return {'parity': passed, 'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95)), 'scenarios': len(rows)}


def read_extended_phase(directory, expected):
    trials, metrics = summarize_method(directory)
    states, lengths, episodes = {}, {}, {}
    summary = json.loads((directory / 'summary.json').read_text())
    for task in summary['task_results']:
        result = json.loads(Path(task['result_paths'][0]).read_text())
        for trial in result['trials']:
            key = (result['suite'], result['task_id'], trial['trial'])
            states[key] = trial['init_state_sha256']
            lengths[key] = trial['policy_steps']
    for path in sorted((directory / 'logs/clients').glob('*.log')):
        for line in path.read_text().splitlines():
            if line.startswith('[CLIENT_TIMING] '):
                row = json.loads(line.split(' ', 1)[1])
                if row['step'] >= 24 and row['step'] % 8 == 0:
                    episodes.setdefault((path.name, row['episode']), []).append(row['observation_to_action_ms'])
    if len(trials) != expected or len(episodes) != expected or len(states) != expected:
        raise ValueError(f'Expected {expected} complete episodes with initial hashes and steady requests')
    return {'metrics': metrics, 'successes': sum(trials.values()), 'states': states,
            'steps': lengths, 'episodes': episodes}


class Campaign:
    def __init__(self, args):
        self.args = args
        self.run = args.run.resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run / 'state.json'
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {'created_s': args.started_at, 'deadline_s': args.started_at + args.budget_hours * 3600,
                          'stages': {}, 'status': 'running', 'adopt_pid': args.adopt_pid,
                          'adopt_identity': process_identity(args.adopt_pid)}
        self.source = self.run / 'source'
        rows = csv.reader(io.StringIO(subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)),
            skipinitialspace=True)
        self.uuids = {row[1] for row in rows if int(row[0]) in (0, 2, 4, 6)}
        if len(self.uuids) != 4:
            raise RuntimeError('Four physical GPUs 0,2,4,6 are required')
        self.heartbeat('starting')

    def heartbeat(self, activity):
        self.state.update({'heartbeat_s': time.time(), 'activity': activity, 'manager_pid': os.getpid()})
        save(self.state_path, self.state)
        remaining = max(0, self.state['deadline_s'] - time.time()) / 3600
        (self.run / 'STATUS.md').write_text(
            f'# 自主优化状态\n\n更新时间：{datetime.now().isoformat()}\n\n'
            f'当前：{activity}\n\n剩余预算：{remaining:.2f} 小时\n\n'
            f'阶段数：{len(self.state["stages"])}；详见 state.json 和各阶段原始日志。\n')

    def remaining(self):
        return self.state['deadline_s'] - time.time()

    def execute(self, name, command, env, timeout=3600):
        original_command = list(command)
        previous = self.state['stages'].get(name, {})
        if previous.get('valid'):
            return Path(previous['directory'])
        if previous.get('pid'):
            stop_owned(previous['pid'], previous.get('identity'))
        for attempt in range(previous.get('attempt', 0) + 1, 3):
            directory = self.run / f'{name}-attempt{attempt}'
            directory.mkdir()
            row = {'attempt': attempt, 'directory': str(directory), 'command': command,
                   'environment': {key: value for key, value in env.items()
                                   if key.startswith('OPENWAM_') or key in ('OMP_NUM_THREADS', 'GPUS', 'CUDA_VISIBLE_DEVICES')},
                   'start_s': time.time(), 'valid': False}
            self.state['stages'][name] = row
            self.heartbeat(f'{name}: waiting for GPUs')
            while self.remaining() > 60:
                sample = gpu_snapshot(self.uuids)
                if not sample['external']:
                    break
                self.heartbeat(f'{name}: external GPU activity, waiting')
                time.sleep(10)
            if self.remaining() <= 60:
                return None
            command = [part.replace('{output}', str(directory / 'evaluation')) for part in original_command]
            stage_env = dict(env, OUTPUT_DIR=str(directory / 'evaluation'))
            started = time.time()
            last_progress = started
            previous_size = -1
            if name.startswith('micro-') and attempt > 1:
                destination = Path(command[2])
                repeat = command[command.index('--repeat') + 1] if '--repeat' in command else '0'
                for filename in (f'{repeat}-process1.jsonl', f'{repeat}-process1-startup.json'):
                    incomplete = destination / filename
                    if incomplete.exists():
                        incomplete.rename(directory / ('previous-' + filename))
            with (directory / 'run.log').open('w') as output, (directory / 'gpu-monitor.jsonl').open('w') as monitor:
                process = subprocess.Popen(command, env=stage_env, cwd=self.source, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                row.update(pid=process.pid, identity=process_identity(process.pid))
                invalid = []
                children = {}
                while process.poll() is None:
                    now = time.time()
                    try:
                        children.update({child: process_identity(child) for child in descendants(process.pid)})
                        sample = gpu_snapshot(self.uuids)
                        remaining_external = []
                        owned_teardown = []
                        for entry in sample['external']:
                            child = int(entry[1])
                            identity = process_identity(child)
                            if child in children and (identity is None or identity == children[child]):
                                owned_teardown.append(entry)
                            else:
                                remaining_external.append(entry)
                        sample['owned_teardown'] = owned_teardown
                        sample['external'] = remaining_external
                        monitor.write(json.dumps(sample) + '\n')
                        monitor.flush()
                        if sample['external']:
                            invalid.append('external GPU activity')
                    except Exception as error:
                        invalid.append(repr(error))
                    size = sum(path.stat().st_size for path in directory.rglob('*.log'))
                    if size != previous_size:
                        last_progress, previous_size = now, size
                    reason = None
                    if now - started > timeout or self.remaining() < 60:
                        reason = 'deadline/phase timeout'
                    elif now - last_progress > 1800:
                        reason = 'no log progress for 30 minutes'
                    elif shutil.disk_usage(self.run).free < 5 * 1024 ** 3:
                        reason = 'free disk below 5 GiB'
                    if reason:
                        invalid.append(reason)
                        stop_owned(process.pid, row['identity'])
                        break
                    self.heartbeat(name)
                    time.sleep(1)
                process.wait()
            stop_owned(process.pid, row['identity'])
            for child, identity in children.items():
                if child != process.pid and identity is not None and process_identity(child) == identity:
                    stop_owned(child, identity)
            row.update(exit_code=process.returncode, end_s=time.time(), errors=invalid,
                       valid=process.returncode == 0 and not invalid)
            save(directory / 'validation.json', row)
            self.heartbeat(name + ': finished')
            if row['valid']:
                return directory
            if self.remaining() < 5400:
                break
        return None

    def environment(self, config):
        env = dict(os.environ)
        env.update({'CUDA_VISIBLE_DEVICES': '0,2,4,6', 'GPUS': '0,2,4,6', 'REPLICAS_PER_GPU': '1',
                    'BASE_PORT': '9015', 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
                    'PYTHONPATH': f'{self.source}:{self.source / "third_party/LIBERO"}',
                    'LIBERO_PATH': str(self.source / 'third_party/LIBERO'),
                    'SERVER_PYTHON': str(self.run / 'server-python'),
                    'LIBERO_PYTHON': str(self.run / 'client-python'),
                    'OPENWAM_LIBERO_REQUEST_TIMING': '1', 'OPENWAM_LIBERO_STATE_HASH': '1',
                    'OPENWAM_REQUIRE_GPU_RESIDENT': '1', 'OPENWAM_CHUNK_INTEROP_THREADS': '1',
                    'OPENWAM_CHUNK_DIAGNOSTICS': '0', 'OPENWAM_CHUNK_TRACE': '0',
                    'OPENWAM_CHUNK_CPU_GROUPS': config.get('groups', ''),
                    'OPENWAM_EXPERIMENT_SERVER_CPUS': '0-31',
                    'OPENWAM_CHUNK_SELECTED_FRAMES': str(int(config.get('frames', False))),
                    'OPENWAM_CHUNK_REUSE_NOISE': str(int(config.get('noise', False))),
                    'OPENWAM_CHUNK_TIMESTEP_CACHE': str(int(config.get('timestep', False)))})
        env['OPENWAM_CHUNK_CUDA_GRAPH'] = str(int(config.get('graph', False)))
        env['OPENWAM_CHUNK_VAE_GPU_ACCUM'] = str(int(config.get('gpu_accum', False)))
        env['OPENWAM_CHUNK_VAE_GRAPH'] = str(int(config.get('vae_graph', False)))
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[key] = str(config.get('threads', 1))
        env.pop('OPENWAM_CHUNK_CAPTURE_PATH', None)
        env.pop('E2_MEMORY_MODE', None)
        return env

    def evaluate(self, name, config, task_ids='0', num_trials=5):
        command = ['bash', 'benchmarks/libero/run_eval.sh', self.args.checkpoint,
                   'checkpoint_step_10690.safetensors', '--sampler', 'chunk-pipeline',
                   '--chunk-speculative-steps', '3', '--denoise-steps', '10', '--inference-horizon', '8',
                   '--inference-mode', 'sync', '--denoise-mode', 'sync', '--compile-enabled', 'false',
                   '--seed', '42', '--render-gpus', '0', '--client-max-attempts', '1',
                   '--server-start-timeout', '1800', '--diagnostic', '--task-ids', task_ids,
                   '--suites', 'spatial,goal,object,long', '--num-trials', str(num_trials)]
        for value in ('optimization.dit_cache.enabled=false', 'optimization.decode_video=false',
                      'inference.request_timing=true', 'inference.chunk_worker_backend=' + config.get('backend', 'process')):
            command.extend(['--extra-deploy-arg', value])
        directory = self.execute(name, command, self.environment(config), timeout=5400)
        if directory:
            try:
                expected = 4 * len(task_ids.split(',')) * num_trials
                phase = read_phase(directory / 'evaluation') if expected == 20 else read_extended_phase(directory / 'evaluation', expected)
                self.state['stages'][name]['metrics'] = phase['metrics']
                self.state['stages'][name]['successes'] = phase['successes']
                self.heartbeat(name + ': analyzed')
                return phase
            except Exception as error:
                self.state['stages'][name]['analysis_error'] = repr(error)
                self.heartbeat(name + ': analysis failed')
        return None

    def run_all(self):
        pid = self.state['adopt_pid']
        while self.state['adopt_identity'] and process_identity(pid) == self.state['adopt_identity']:
            self.heartbeat('waiting for existing paired experiment (not modifying its sources)')
            if self.remaining() < 5400:
                self.state['status'] = 'budget_exhausted_waiting_for_adopted_run'
                return
            time.sleep(15)
        for name, cpus, interpreter in (('server', '0-31', '.venv-openwam'), ('client', '40-43', '.venv-libero')):
            path = self.run / f'{name}-python'
            prefix = 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n' if name == 'client' else ''
            path.write_text('#!/bin/sh\n' + prefix + f'exec taskset -c {cpus} {ROOT / interpreter / "bin/python"} "$@"\n')
            path.chmod(0o755)
        sources = [path for folder in ('openwam', 'scripts', 'configs', 'benchmarks')
                   for path in (self.source / folder).rglob('*') if path.suffix in ('.py', '.yaml', '.sh')]
        (self.run / 'source-sha256.txt').write_text(''.join(
            f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(self.source)}\n' for path in sorted(sources)))
        self.heartbeat('verifying checkpoint fingerprint')
        checkpoint = Path(self.args.checkpoint) / 'checkpoint_step_10690.safetensors'
        digest = hashlib.sha256()
        with checkpoint.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise RuntimeError('Checkpoint fingerprint mismatch')
        save(self.run / 'manifest.json', {
            'checkpoint_sha256': digest.hexdigest(), 'gpus': [0, 2, 4, 6], 'gpu_uuids': sorted(self.uuids),
            'deadline_s': self.state['deadline_s'], 'python': sys.version, 'source': str(self.source),
            'lscpu': subprocess.check_output(['lscpu', '-e=CPU,CORE,SOCKET,NODE'], text=True),
            'constraints': {'denoise': 10, 'horizon': 32, 'execute': 8, 'speculative_steps': 3,
                            'all_worker_barrier': True, 'cpu_model_offload': False}})
        configs = [
            ('shared8', {'groups': '0-7;0-7;0-7;0-7'}),
            ('shared32', {}),
            ('split2', {'groups': '0-1;2-3;4-5;6-7'}),
            ('split4', {'groups': '0-3;4-7;8-11;12-15'}),
            ('split8', {'groups': '0-7;8-15;16-23;24-31'}),
            ('split4t2', {'groups': '0-3;4-7;8-11;12-15', 'threads': 2}),
            ('numa4', {'groups': '0-3;4-7;24-27;28-31'}),
            ('numa8', {'groups': '0-7;8-15;24-31;32-39'}),
        ]
        reference = self.args.previous / 'micro-attempt1/results'
        ranked = []
        baseline_micro = None
        for name, config in configs:
            if self.remaining() < 10800:
                break
            destination = self.run / ('microdata-' + name)
            destination.mkdir(exist_ok=True)
            for path in reference.iterdir():
                if path.name.startswith(('stage-', 'reference-')) and not (destination / path.name).exists():
                    (destination / path.name).symlink_to(path.resolve())
            command = [str(ROOT / '.venv-openwam/bin/python'), 'scripts/benchmark_chunk_submission.py',
                       str(destination), '--phase', 'session', '--mode', 'process1',
                       '--checkpoint', self.args.checkpoint, '--warmup', '10', '--measurements', '30']
            directory = self.execute('micro-' + name, command, self.environment(config), timeout=5400)
            if directory:
                summary = micro_summary(destination)
                save(destination / 'summary.json', summary)
                if name == 'shared8':
                    baseline_micro = summary
                if summary['parity']:
                    ranked.append((summary['p50_ms'], name, config, summary))
            save(self.run / 'rankings.json', sorted(ranked))
        if not ranked:
            self.state['status'] = 'no_valid_candidate'
            return
        ranked.sort()
        finalists = ranked[:3]
        for repeat in (1, 2):
            for _, name, config, _ in finalists[repeat - 1:] + finalists[:repeat - 1]:
                if self.remaining() < 14400:
                    break
                destination = self.run / ('microdata-' + name)
                command = [str(ROOT / '.venv-openwam/bin/python'), 'scripts/benchmark_chunk_submission.py',
                           str(destination), '--phase', 'session', '--mode', 'process1', '--repeat', str(repeat),
                           '--checkpoint', self.args.checkpoint, '--warmup', '10', '--measurements', '30']
                self.execute(f'micro-{name}-repeat{repeat}', command, self.environment(config), timeout=5400)
        stable = []
        for _, name, config, _ in finalists:
            summary = micro_summary(self.run / ('microdata-' + name))
            if summary['parity']:
                stable.append((summary['p50_ms'], name, config, summary))
        if stable:
            ranked = sorted(stable)
        save(self.run / 'stable-rankings.json', ranked)
        winner = ranked[0]
        screen_config = winner[2]
        baseline = self.evaluate('control-screen', {'groups': '0-7;0-7;0-7;0-7'})
        screened = []
        for name, config in [(winner[1], screen_config),
                             (winner[1] + '-frames', dict(screen_config, frames=True)),
                             (winner[1] + '-reuse', dict(screen_config, frames=True, noise=True, timestep=True)),
                             (winner[1] + '-graph', dict(screen_config, frames=True, noise=True, timestep=True, graph=True)),
                             (winner[1] + '-gpuaccum', dict(screen_config, frames=True, noise=True, timestep=True, gpu_accum=True)),
                             (winner[1] + '-vaegraph', dict(screen_config, frames=True, noise=True, timestep=True, vae_graph=True)),
                             (winner[1] + '-allgraphs', dict(screen_config, frames=True, noise=True, timestep=True,
                                                           graph=True, gpu_accum=True, vae_graph=True))]:
            if self.remaining() < 9000:
                break
            if name == winner[1] and baseline_micro and 1 - winner[0] / baseline_micro['p50_ms'] < .03:
                continue
            parity_command = [str(ROOT / '.venv-openwam/bin/python'), 'scripts/benchmark_chunk_conditioning.py',
                              '--reference', str(reference), '--output', '{output}', '--checkpoint', self.args.checkpoint]
            parity_directory = self.execute('parity-' + name, parity_command, self.environment(config), timeout=2700)
            if not parity_directory:
                continue
            parity_result = json.loads((parity_directory / 'evaluation/parity.json').read_text())
            if not parity_result['passed']:
                continue
            if config.get('gpu_accum') or config.get('vae_graph'):
                timing = parity_result['timings']
                reference_dependency = sum(timing['reference'][key]['p50_ms'] for key in ('decode', 'prepare'))
                candidate_dependency = sum(timing['candidate'][key]['p50_ms'] for key in ('decode', 'prepare'))
                if 1 - candidate_dependency / reference_dependency < .03 and not config.get('graph'):
                    continue
            if config.get('graph'):
                destination = self.run / ('microdata-' + name)
                destination.mkdir(exist_ok=True)
                for path in reference.iterdir():
                    if path.name.startswith(('stage-', 'reference-')) and not (destination / path.name).exists():
                        (destination / path.name).symlink_to(path.resolve())
                command = [str(ROOT / '.venv-openwam/bin/python'), 'scripts/benchmark_chunk_submission.py',
                           str(destination), '--phase', 'session', '--mode', 'process1',
                           '--checkpoint', self.args.checkpoint, '--warmup', '10', '--measurements', '30']
                if not self.execute('micro-' + name, command, self.environment(config), timeout=5400):
                    continue
                graph_summary = micro_summary(destination)
                save(destination / 'summary.json', graph_summary)
                if not graph_summary['parity'] or 1 - graph_summary['p50_ms'] / winner[0] < .03:
                    continue
            phase = self.evaluate('screen-' + name, config)
            if phase and baseline and phase['successes'] == 20 and phase['states'] == baseline['states']:
                metric = phase['metrics']['client_chunk_requests_excluding_first_three']
                control = baseline['metrics']['client_chunk_requests_excluding_first_three']
                if metric['p95_ms'] <= control['p95_ms'] * 1.05:
                    screened.append((metric['p50_ms'], name, config))
        save(self.run / 'screened.json', sorted(screened))
        if screened and self.remaining() > 5400:
            _, name, config = min(screened)
            phases = []
            for label, current in [('A1', {'backend': 'thread', 'threads': 4, 'groups': '0-7;0-7;0-7;0-7'}),
                                   ('C1', config), ('C2', config),
                                   ('A2', {'backend': 'thread', 'threads': 4, 'groups': '0-7;0-7;0-7;0-7'})]:
                if self.remaining() < 900:
                    break
                phases.append(self.evaluate('confirm-' + label, current))
            comparisons = []
            if len(phases) == 4 and all(phase is not None for phase in phases):
                for actual, control in ((phases[1], phases[0]), (phases[2], phases[3])):
                    metric = actual['metrics']['client_chunk_requests_excluding_first_three']
                    reference_metric = control['metrics']['client_chunk_requests_excluding_first_three']
                    gain = 1 - metric['p50_ms'] / reference_metric['p50_ms']
                    tail = metric['p95_ms'] / reference_metric['p95_ms']
                    comparisons.append({'p50_reduction': gain, 'p95_ratio': tail,
                                        'passed': gain >= .10 and tail <= 1.05 and actual['successes'] == 20 and
                                        actual['states'] == control['states'],
                                        'bootstrap': paired_bootstrap(actual, control)})
            save(self.run / 'acceptance.json', {'candidate': name, 'config': config, 'comparisons': comparisons,
                                              'accepted': len(comparisons) == 2 and all(row['passed'] for row in comparisons)})
            if len(comparisons) == 2 and all(row['passed'] for row in comparisons) and self.remaining() > 14400:
                extended = []
                for label, current in [('A1', {'backend': 'thread', 'threads': 4, 'groups': '0-7;0-7;0-7;0-7'}),
                                       ('C1', config), ('C2', config),
                                       ('A2', {'backend': 'thread', 'threads': 4, 'groups': '0-7;0-7;0-7;0-7'})]:
                    if self.remaining() < 5400:
                        break
                    extended.append(self.evaluate('extended-' + label, current, task_ids='1,2', num_trials=3))
                extra = []
                if len(extended) == 4 and all(phase is not None for phase in extended):
                    for actual, control in ((extended[1], extended[0]), (extended[2], extended[3])):
                        metric = actual['metrics']['client_chunk_requests_excluding_first_three']
                        reference_metric = control['metrics']['client_chunk_requests_excluding_first_three']
                        gain = 1 - metric['p50_ms'] / reference_metric['p50_ms']
                        tail = metric['p95_ms'] / reference_metric['p95_ms']
                        extra.append({'p50_reduction': gain, 'p95_ratio': tail,
                                      'candidate_successes': actual['successes'], 'baseline_successes': control['successes'],
                                      'passed': gain >= .10 and tail <= 1.05 and actual['successes'] >= control['successes'] and
                                      actual['states'] == control['states'], 'bootstrap': paired_bootstrap(actual, control)})
                save(self.run / 'extended-acceptance.json', {'candidate': name, 'config': config, 'comparisons': extra,
                                                           'accepted': len(extra) == 2 and all(row['passed'] for row in extra)})
            if len(comparisons) == 2 and all(row['passed'] for row in comparisons):
                for tasks in ('3,4', '5,6', '7,8'):
                    if self.remaining() < 10800:
                        break
                    coverage = []
                    for label, current in [('A', {'backend': 'thread', 'threads': 4, 'groups': '0-7;0-7;0-7;0-7'}),
                                           ('C', config)]:
                        coverage.append(self.evaluate('coverage-' + tasks.replace(',', '-') + '-' + label,
                                                      current, task_ids=tasks, num_trials=3))
                    if all(phase is not None for phase in coverage):
                        actual, control = coverage[1], coverage[0]
                        save(self.run / ('coverage-' + tasks.replace(',', '-') + '.json'), {
                            'tasks': tasks, 'candidate_successes': actual['successes'],
                            'baseline_successes': control['successes'], 'states_match': actual['states'] == control['states'],
                            'candidate_metrics': actual['metrics'], 'baseline_metrics': control['metrics'],
                            'bootstrap': paired_bootstrap(actual, control)})
        self.state['status'] = 'completed_available_candidates'

    def report(self):
        lines = ['# 四卡自主优化实验', '', '固定物理 GPU 0/2/4/6；N10/H32/E8；完整后台屏障；无CPU模型卸载。', '',
                 '| 阶段 | 有效 | 成功 | 客户端P50 ms | P95 ms |', '|---|---|---|---|---|']
        for name, row in self.state['stages'].items():
            metric = row.get('metrics', {}).get('client_chunk_requests_excluding_first_three', {})
            lines.append(f'| {name} | {row.get("valid")} | {row.get("successes", "-")} | '
                         f'{metric.get("p50_ms", "-")} | {metric.get("p95_ms", "-")} |')
        lines.extend(['', '筛选不等同最终确认。确认采用 A→C→C→A、回合级配对bootstrap；见 acceptance.json。',
                      '数值门槛 rtol=1e-3/atol=1e-4；未完成或失败候选不视作通过。',
                      '完整命令、监控、失败和源代码指纹分别保存在各阶段目录、state.json、source-sha256.txt。',
                      '仅采样监控外部GPU进程，不能保证绝对无干扰。', '', f'终态：{self.state["status"]}'])
        (self.run / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'experiment/chunk_autonomous_20261121')
    parser.add_argument('--previous', type=Path, default=ROOT / 'experiment/chunk_acceleration_20261120_203804')
    parser.add_argument('--budget-hours', type=float, default=20)
    parser.add_argument('--started-at', type=float, default=1795192418)
    parser.add_argument('--adopt-pid', type=int, default=662346)
    parser.add_argument('--checkpoint', default='/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO')
    parser.add_argument('--replace-waiting-manager', action='store_true')
    args = parser.parse_args()
    import fcntl
    args.run.mkdir(parents=True, exist_ok=True)
    if args.replace_waiting_manager and (args.run / 'state.json').exists():
        previous = json.loads((args.run / 'state.json').read_text())
        pid = previous.get('manager_pid')
        if pid and process_identity(pid):
            command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if previous['stages'] or not previous.get('activity', '').startswith('waiting for existing'):
                raise RuntimeError('Only an idle waiting manager can be replaced')
            if b'scripts/run_chunk_autonomous.py' not in command:
                raise RuntimeError('Manager command identity mismatch')
            stop_owned(pid, process_identity(pid))
    with (args.run / 'manager.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(args)
        try:
            campaign.run_all()
        except BaseException as error:
            campaign.state.update(status='failed', error=repr(error))
            for row in campaign.state['stages'].values():
                if row.get('pid'):
                    stop_owned(row['pid'], row.get('identity'))
            raise
        finally:
            campaign.report()
            campaign.heartbeat(campaign.state['status'])


if __name__ == '__main__':
    main()
