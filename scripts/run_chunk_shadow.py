"""Audit graph versus eager on every real closed-loop forward; not a latency run."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_chunk_autonomous import Campaign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'experiment/chunk_shadow_20261121')
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
            parent = ROOT / 'experiment/chunk_autonomous_20261121'
            for filename in ('server-python', 'client-python'):
                shutil.copy2(parent / filename, args.run / filename)
            config = {'groups': '0-7;8-15;16-23;24-31', 'frames': True, 'noise': True,
                      'timestep': True, 'graph': True, 'gpu_accum': True, 'vae_graph': True}
            import os
            os.environ['OPENWAM_CHUNK_SHADOW_PARITY'] = '1'
            if manager.remaining() > 7200:
                manager.evaluate('real-closed-loop-shadow', config, task_ids='1,2', num_trials=5)
            manager.state['status'] = 'completed_shadow_audit_not_latency'
        finally:
            manager.report()
            manager.heartbeat(manager.state['status'])


if __name__ == '__main__':
    main()
