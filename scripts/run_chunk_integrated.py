"""Final opt-in verification from the integrated repository source."""

import argparse
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_chunk_autonomous import Campaign, process_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'experiment/chunk_integrated_20261121')
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
                manager.heartbeat('waiting for shadow audit before integrated source verification')
                if manager.remaining() < 5400:
                    manager.state['status'] = 'budget_reserved_for_report'
                    return
                time.sleep(15)
            manager.source = ROOT
            parent = ROOT / 'experiment/chunk_autonomous_20261121'
            for filename in ('server-python', 'client-python'):
                shutil.copy2(parent / filename, args.run / filename)
            config = {'groups': '0-7;8-15;16-23;24-31', 'frames': True, 'noise': True,
                      'timestep': True, 'graph': True, 'gpu_accum': True, 'vae_graph': True}
            if manager.remaining() > 7200:
                manager.evaluate('integrated-opt-in', config)
            manager.state['status'] = 'completed_integrated_verification'
        finally:
            manager.report()
            manager.heartbeat(manager.state['status'])


if __name__ == '__main__':
    main()
