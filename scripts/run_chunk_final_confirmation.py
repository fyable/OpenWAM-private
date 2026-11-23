"""Direct original-thread versus integrated-graph A-C-C-A confirmation."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_chunk_autonomous import Campaign, save
from scripts.run_chunk_acceleration import paired_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'experiment/chunk_final_confirmation_20261121')
    parser.add_argument('--checkpoint', default='/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO')
    parser.add_argument('--budget-hours', type=float, default=20)
    parser.add_argument('--started-at', type=float, default=1795192418)
    parser.add_argument('--adopt-pid', type=int, default=0)
    args = parser.parse_args()
    import fcntl
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / 'manager.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Campaign(args)
        try:
            manager.source = ROOT
            parent = ROOT / 'experiment/chunk_autonomous_20261121'
            for filename in ('server-python', 'client-python'):
                shutil.copy2(parent / filename, args.run / filename)
            config = {'groups': '0-7;8-15;16-23;24-31', 'frames': True, 'noise': True,
                      'timestep': True, 'graph': True, 'gpu_accum': True, 'vae_graph': True}
            baseline = {'backend': 'thread', 'threads': 4, 'groups': '0-7;0-7;0-7;0-7'}
            phases = []
            for label, current in [('A1', baseline), ('C1', config), ('C2', config), ('A2', baseline)]:
                if manager.remaining() < 5400:
                    break
                phases.append(manager.evaluate(label, current))
            comparisons = []
            if len(phases) == 4 and all(phase is not None for phase in phases):
                for actual, control in ((phases[1], phases[0]), (phases[2], phases[3])):
                    metric = actual['metrics']['client_chunk_requests_excluding_first_three']
                    baseline_metric = control['metrics']['client_chunk_requests_excluding_first_three']
                    gain = 1 - metric['p50_ms'] / baseline_metric['p50_ms']
                    tail = metric['p95_ms'] / baseline_metric['p95_ms']
                    comparisons.append({'p50_reduction': gain, 'p95_ratio': tail,
                                        'passed': gain >= .10 and tail <= 1.05 and actual['successes'] == 20 and
                                        actual['states'] == control['states'], 'bootstrap': paired_bootstrap(actual, control)})
            save(args.run / 'acceptance.json', {'config': config, 'comparisons': comparisons,
                                              'accepted': len(comparisons) == 2 and all(row['passed'] for row in comparisons),
                                              'scope': 'task0 only; does not override 154/160 versus 155/160 coverage regression'})
            manager.state['status'] = 'completed_direct_confirmation'
        finally:
            manager.report()
            manager.heartbeat(manager.state['status'])


if __name__ == '__main__':
    main()
