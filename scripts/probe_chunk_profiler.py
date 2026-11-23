"""Validate all-thread, four-device CUDA activity capture before loading models."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


def coverage(path, events=None):
    if events is None:
        events = json.loads(Path(path).read_text())['traceEvents']
    kernels = [row for row in events if row.get('cat') == 'kernel']
    copies = [row for row in events if 'memcpy' in row.get('cat', '').lower()]
    return {
        'kernel_devices': sorted({row['args']['device'] for row in kernels}),
        'copy_devices': sorted({row['args']['device'] for row in copies}),
        'copy_names': sorted({row['name'] for row in copies}),
        'worker_threads': sorted({row['tid'] for row in events
                                  if row.get('name', '').startswith('probe.worker.')
                                  and row.get('cat') == 'user_annotation'}),
        'chunk_regions': sum(row.get('name', '').startswith('chunk.') for row in events),
        'kernel_count': len(kernels), 'copy_count': len(copies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'complete': False}
    executors = [ThreadPoolExecutor(max_workers=1) for _ in range(4)]
    buffers = {}

    def work(device):
        with torch.cuda.device(device), torch.profiler.record_function(f'probe.worker.{device}'):
            tensor = buffers[device].to(f'cuda:{device}')
            output = tensor @ tensor
            output.cpu()
            torch.cuda.synchronize(device)

    try:
        assert torch.cuda.device_count() == 4
        for device in range(4):
            buffers[device] = torch.ones((128, 128))
            executors[device].submit(work, device).result()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(profile_all_threads=True),
        ) as profiler:
            jobs = [executor.submit(work, device) for device, executor in enumerate(executors)]
            for job in jobs:
                job.result()
        path = args.output / 'probe-trace.json'
        profiler.export_chrome_trace(str(path))
        result.update(coverage(path))
        result['complete'] = (result['kernel_devices'] == list(range(4))
                              and result['copy_devices'] == list(range(4))
                              and len(result['worker_threads']) == 4
                              and any('HtoD' in name for name in result['copy_names'])
                              and any('DtoH' in name for name in result['copy_names']))
    except Exception as error:
        result['error'] = repr(error)
    finally:
        for executor in executors:
            executor.shutdown(wait=True)
    (args.output / 'probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
