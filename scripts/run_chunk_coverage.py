"""Pinned-protocol coverage after graph repair, within the original deadline."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_chunk_autonomous import Campaign, process_identity, save
from scripts.run_chunk_acceleration import paired_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'experiment/chunk_coverage_20261121')
    parser.add_argument('--parent', type=Path, default=ROOT / 'experiment/chunk_graph_repair_20261121')
    parser.add_argument('--checkpoint', default='/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO')
    parser.add_argument('--budget-hours', type=float, default=20)
    parser.add_argument('--started-at', type=float, default=1795192418)
    parser.add_argument('--adopt-pid', type=int, required=True)
    args = parser.parse_args()
    import fcntl
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / 'manager.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Campaign(args)
        try:
            while manager.state['adopt_identity'] and process_identity(args.adopt_pid) == manager.state['adopt_identity']:
                manager.heartbeat('waiting for graph repair before pinned-protocol coverage')
                if manager.remaining() < 5400:
                    manager.state['status'] = 'budget_reserved_for_report'
                    return
                time.sleep(15)
            original = ROOT / 'experiment/chunk_autonomous_20261121'
            manager.source = (original / 'source').resolve()
            for filename in ('server-python', 'client-python'):
                shutil.copy2(original / filename, args.run / filename)
            config = json.loads((original / 'acceptance.json').read_text())['config']
            repairs = args.parent / 'acceptance.json'
            if repairs.exists() and json.loads(repairs.read_text()).get('accepted'):
                config = json.loads(repairs.read_text())['config']
                manager.source = (original / 'repair_source').resolve()
            for tasks in ('1,2', '3,4', '5,6', '7,8'):
                if manager.remaining() < 9000:
                    break
                phases = []
                for label, current in [('A', {'backend': 'thread', 'threads': 4, 'groups': '0-7;0-7;0-7;0-7'}), ('C', config)]:
                    phases.append(manager.evaluate('tasks-' + tasks.replace(',', '-') + '-' + label,
                                                   current, task_ids=tasks, num_trials=5))
                if all(phase is not None for phase in phases):
                    actual, control = phases[1], phases[0]
                    save(args.run / ('tasks-' + tasks.replace(',', '-') + '.json'), {
                        'config': config, 'candidate_successes': actual['successes'],
                        'baseline_successes': control['successes'], 'states_match': actual['states'] == control['states'],
                        'candidate_metrics': actual['metrics'], 'baseline_metrics': control['metrics'],
                        'bootstrap': paired_bootstrap(actual, control)})
            manager.state['status'] = 'completed_pinned_coverage'
        finally:
            manager.report()
            manager.heartbeat(manager.state['status'])


if __name__ == '__main__':
    main()
