"""Continue the same deadline after the first campaign, using isolated repairs."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_chunk_autonomous import Campaign, micro_summary, process_identity, save
from scripts.run_chunk_acceleration import paired_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'experiment/chunk_graph_repair_20261121')
    parser.add_argument('--parent', type=Path, default=ROOT / 'experiment/chunk_autonomous_20261121')
    parser.add_argument('--checkpoint', default='/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO')
    parser.add_argument('--budget-hours', type=float, default=20)
    parser.add_argument('--started-at', type=float, default=1795192418)
    parser.add_argument('--adopt-pid', type=int, default=22233)
    args = parser.parse_args()
    import fcntl
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / 'manager.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Campaign(args)
        try:
            while manager.state['adopt_identity'] and process_identity(args.adopt_pid) == manager.state['adopt_identity']:
                manager.heartbeat('waiting for original campaign; no simultaneous GPU experiments')
                if manager.remaining() < 5400:
                    manager.state['status'] = 'insufficient_budget_after_parent'
                    return
                time.sleep(15)
            source = args.parent / 'repair_source'
            manager.source = source.resolve()
            for filename in ('server-python', 'client-python'):
                shutil.copy2(args.parent / filename, args.run / filename)
            paths = [path for folder in ('openwam', 'scripts', 'configs', 'benchmarks')
                     for path in (manager.source / folder).rglob('*') if path.suffix in ('.py', '.sh', '.yaml')]
            import hashlib
            (args.run / 'source-sha256.txt').write_text(''.join(
                f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(manager.source)}\n'
                for path in sorted(paths)))
            config = {'groups': '0-7;8-15;16-23;24-31', 'frames': True, 'noise': True,
                      'timestep': True, 'graph': True, 'gpu_accum': True, 'vae_graph': True}
            reference = ROOT / 'experiment/chunk_acceleration_20261120_203804/micro-attempt1/results'
            command = [str(ROOT / '.venv-openwam/bin/python'), 'scripts/benchmark_chunk_conditioning.py',
                       '--reference', str(reference), '--output', '{output}', '--checkpoint', args.checkpoint]
            parity = manager.execute('repaired-forward-parity', command, manager.environment(config), timeout=2700)
            passed = False
            if parity:
                result = json.loads((parity / 'evaluation/parity.json').read_text())
                passed = result['passed']
            if passed and manager.remaining() > 10800:
                destination = args.run / 'microdata-repaired'
                destination.mkdir(exist_ok=True)
                for path in reference.iterdir():
                    if path.name.startswith(('stage-', 'reference-')) and not (destination / path.name).exists():
                        (destination / path.name).symlink_to(path.resolve())
                command = [str(ROOT / '.venv-openwam/bin/python'), 'scripts/benchmark_chunk_submission.py',
                           str(destination), '--phase', 'session', '--mode', 'process1',
                           '--checkpoint', args.checkpoint, '--warmup', '10', '--measurements', '30']
                micro = manager.execute('repaired-forward-micro', command, manager.environment(config), timeout=5400)
                if micro:
                    summary = micro_summary(destination)
                    save(args.run / 'micro-summary.json', summary)
                    passed = summary['parity'] and summary['p50_ms'] < 245.102 * .97
                else:
                    passed = False
            if passed and manager.remaining() > 9000:
                control_config = dict(config, graph=False, gpu_accum=False)
                phases = []
                for label, current in [('A1', control_config), ('C1', config), ('C2', config), ('A2', control_config)]:
                    if manager.remaining() < 900:
                        break
                    phases.append(manager.evaluate('repair-confirm-' + label, current))
                comparisons = []
                if len(phases) == 4 and all(phase is not None for phase in phases):
                    for actual, control in ((phases[1], phases[0]), (phases[2], phases[3])):
                        metric = actual['metrics']['client_chunk_requests_excluding_first_three']
                        baseline = control['metrics']['client_chunk_requests_excluding_first_three']
                        gain = 1 - metric['p50_ms'] / baseline['p50_ms']
                        tail = metric['p95_ms'] / baseline['p95_ms']
                        comparisons.append({'p50_reduction': gain, 'p95_ratio': tail,
                                            'passed': gain >= .10 and tail <= 1.05 and actual['successes'] == 20 and
                                            actual['states'] == control['states'], 'bootstrap': paired_bootstrap(actual, control)})
                save(args.run / 'acceptance.json', {'config': config, 'comparisons': comparisons,
                                                  'accepted': len(comparisons) == 2 and all(row['passed'] for row in comparisons)})
            elif manager.remaining() > 7200:
                config = dict(config, graph=False, vae_graph=False)
                manager.evaluate('gpuaccum-clean-screen', config)
                manager.evaluate('gpuaccum-clean-control', dict(config, gpu_accum=False))
            manager.state['status'] = 'completed_repair_candidates'
        except BaseException as error:
            manager.state.update(status='failed', error=repr(error))
            from scripts.run_chunk_autonomous import stop_owned
            for row in manager.state['stages'].values():
                if row.get('pid'):
                    stop_owned(row['pid'], row.get('identity'))
            raise
        finally:
            manager.report()
            manager.heartbeat(manager.state['status'])


if __name__ == '__main__':
    main()
