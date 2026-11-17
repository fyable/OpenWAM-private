"""Launch the frozen historical OpenWAM controlled baseline, not current API modes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
SHARED = Path('/data2/jikangye/research/runs/paraflow/hardware-ablation-20261017/wam-port')


def video_launcher(worker):
    return (
        f'import runpy,sys; worker=runpy.run_path({str(worker)!r}); '
        'worker["launch"](sys.argv.pop(1))'
    )


def prepare(output, pf_root):
    packet = pf_root / 'results/20261017-hardware-ablation/evidence'
    archive = packet / 'frozen-seven-row-source.tar.gz'
    identity = json.loads((packet / 'openwam-formal-host2-v6r/identity.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'frozen'
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            path = Path(member.name)
            if not member.isfile() or path.parts[0] != 'wam' or '..' in path.parts:
                continue
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            relative = Path(*path.parts[1:])
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = bundle.extractfile(member).read()
            expected = identity['source'].get(str(relative))
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'Frozen source identity mismatch: {relative}')
            target.write_bytes(data)
    for filename, key in [('wam-diagnostic-overlay.json', 'overlay'),
                          ('wam-additional-controls-overlay.json', 'additional_overlay')]:
        (source / filename).write_text(json.dumps(identity[key], indent=2))
    (output / 'assets-openwam-identity.json').write_text(json.dumps(identity['assets'], indent=2))
    (output / 'options.json').write_text(json.dumps(identity['options'], indent=2))
    manifest = {'status': 'prepared_not_run', 'historical_median_ms': 110.624094,
                'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'source': 'frozen historical wam source, unmodified',
                'selection': 'baseline only plus original Native N10 reference; C1-C5 excluded',
                'samples': 120, 'warmup': 48, 'stream_length': 64,
                'historical_overlay_records': 'copied provenance, not new validation',
                'exact_latency_reproduction': 'not_run'}
    (output / 'reproduction.json').write_text(json.dumps(manifest, indent=2))
    return source, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'all', 'video'])
    args = parser.parse_args()
    output = Path(os.environ.get('REPRO_ROOT', str(ROOT / 'outputs' / time.strftime('paraflow_110_%Y%m%d_%H%M%S')))).resolve()
    source, manifest = prepare(output, Path(os.environ.get('PF_ROOT', '/home/zhijian/paraflow-main')))
    print(f'Prepared historical source: {output}', flush=True)
    if args.mode == 'prepare':
        return
    native = Path(os.environ.get('PF_NATIVE_ROOT', str(SHARED / 'native/openwam'))).resolve()
    python = os.environ.get('PF_PYTHON', str(SHARED / 'runtime/wam/bin/python'))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join([str(source / 'scripts/diagnostics'), str(source / 'src'),
                                       str(native), str(native.parent)])
    env['PARAFLOW_ABLATION_LOCK_DIR'] = str(output / '.gpu-locks')
    launcher = (
        'import runpy,sys,cumulative_common; '
        'cumulative_common.LABELS[:]=["baseline"]; '
        'runpy.run_path(sys.argv.pop(1),run_name="__main__")'
    )
    if args.mode == 'video':
        launcher = video_launcher(ROOT / 'benchmarks/libero/paraflow_video_worker.py')
        manifest['selection'] = 'historical first-frame baseline plus original full-video sampler variants'
        manifest['video_boundary'] = 'future latent denoising included; pixel decoding excluded; isolated masks unchanged'
        manifest['orchestration'] = 'historical main setup retained; AST replaces campaign tail with local video measurement worker'
        manifest['worker_sha256'] = hashlib.sha256((ROOT / 'benchmarks/libero/paraflow_video_worker.py').read_bytes()).hexdigest()
    command = [python, '-c', launcher, str(source / 'scripts/diagnostics/benchmark_wam_cumulative.py'),
               '--model', 'openwam', '--options', str(output / 'options.json'),
               '--physical-gpu', os.environ.get('GPU', '0'), '--output', str(output / 'measurement'),
               '--samples', '120', '--warmup', '48', '--stream-length', '64']
    manifest.update(status='running', command=command)
    receipt = output / 'reproduction.json'
    receipt.write_text(json.dumps(manifest, indent=2))
    try:
        with (output / 'run.log').open('x') as log:
            label = 'historical baseline and full-video variants' if args.mode == 'video' else 'historical baseline and Native reference'
            print(f'Running {label}; log: {output / "run.log"}', flush=True)
            subprocess.run(command, cwd=native, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((output / 'measurement/summary.json').read_text())
        if not summary.get('completed'):
            raise RuntimeError('Historical measurement did not complete')
        median = summary['timings']['baseline']['median_ms']
        manifest.update(status='completed', measured_median_ms=median,
                        relative_to_historical=median / manifest['historical_median_ms'],
                        exact_latency_reproduction='inspect measured latency and environment; not guaranteed')
        print(f'Baseline P50: {median:.3f} ms; historical: 110.624 ms', flush=True)
        if args.mode == 'video':
            for name, row in summary['timings'].items():
                if name != 'baseline':
                    print(f"{name}: P50={row['median_ms']:.3f} ms; P95={row['p95_ms']:.3f} ms", flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        print(f'Failed; inspect {output / "run.log"}', flush=True)
        if (output / 'run.log').exists():
            print('\n'.join((output / 'run.log').read_text(errors='replace').splitlines()[-30:]), flush=True)
        raise
    finally:
        receipt.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
