"""Sequential regression audit under the original fixed wall-clock deadline."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_chunk_autonomous import Campaign, process_identity, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'experiment/chunk_regression_20261121')
    parser.add_argument('--checkpoint', default='/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO')
    parser.add_argument('--budget-hours', type=float, default=20)
    parser.add_argument('--started-at', type=float, default=1795192418)
    parser.add_argument('--adopt-pid', type=int, default=651950)
    args = parser.parse_args()
    import fcntl
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / 'manager.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Campaign(args)
        try:
            while manager.state['adopt_identity'] and process_identity(args.adopt_pid) == manager.state['adopt_identity']:
                manager.heartbeat('waiting for coverage; audit does not overlap GPU use')
                if manager.remaining() < 5400:
                    manager.state['status'] = 'budget_reserved_for_final_report'
                    return
                time.sleep(15)
            parent = ROOT / 'experiment/chunk_autonomous_20261121'
            manager.source = (parent / 'repair_source').resolve()
            for filename in ('server-python', 'client-python'):
                shutil.copy2(parent / filename, args.run / filename)
            command = [str(ROOT / '.venv-openwam/bin/python'), str(ROOT / 'scripts/benchmark_chunk_prompt_parity.py'),
                       '--output', '{output}', '--checkpoint', args.checkpoint]
            manager.execute('full-trajectory-prompts', command, manager.environment({}), timeout=2700)
            config = {'groups': '0-7;8-15;16-23;24-31'}
            for name, current in [('process-eager', config),
                                  ('graph-repeat', dict(config, frames=True, noise=True, timestep=True,
                                                        graph=True, gpu_accum=True, vae_graph=True))]:
                if manager.remaining() < 7200:
                    break
                phase = manager.evaluate('tasks-1-2-' + name, current, task_ids='1,2', num_trials=5)
                if phase:
                    save(args.run / (name + '.json'), {'successes': phase['successes'], 'metrics': phase['metrics'],
                                                      'steps': {str(key): value for key, value in phase['steps'].items()}})
            manager.state['status'] = 'completed_regression_audit'
        finally:
            manager.report()
            manager.heartbeat(manager.state['status'])


if __name__ == '__main__':
    main()
