"""Summarize paired diagnostics; inclusive intervals are never added as totals."""

import argparse
import html
import json
import sys
from collections import Counter, defaultdict
from bisect import bisect_left
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_chunk_profiler import coverage
from scripts.summarize_chunk_pipeline_smoke import percentiles, summarize_method


def tagged(directory, marker):
    rows = []
    for path in sorted((directory / 'logs' / 'clients').glob('*.log')):
        for line in path.read_text().splitlines():
            if line.startswith(marker + ' '):
                rows.append({**json.loads(line[len(marker) + 1:]), 'source': path.name})
    return rows


def segment_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get('round_start', -1) < 24 or row['name'] in ('task_start', 'task_end'):
            continue
        key = (row.get('phase'), row.get('condition'), row.get('worker'), row['name'])
        groups[key].append(row)
    return [dict(phase=key[0], condition=key[1], worker=key[2], name=key[3],
                 wall=percentiles([row['wall_ms'] for row in values]),
                 gpu_event_span=percentiles([row['gpu_elapsed_ms'] for row in values
                                            if row.get('gpu_elapsed_ms') is not None]))
            for key, values in sorted(groups.items(), key=lambda item: str(item[0]))]


def aggregate_segments(rows):
    groups = defaultdict(list)
    rounds = defaultdict(float)
    frame_groups = defaultdict(float)
    for row in rows:
        if row.get('round_start', -1) < 24:
            continue
        groups[(row.get('phase'), row['name'])].append(row)
        if row.get('phase') == 'advance':
            key = (row['pid'], row['episode'], row['round_start'], row['worker'], row['name'])
            rounds[key] += row['wall_ms']
        if row.get('phase') == 'forecast' and row['name'] == 'decoded_frame':
            key = (row['pid'], row['episode'], row['round_start'], row.get('selected_for_future', False))
            frame_groups[key] += row['wall_ms']
    round_groups = defaultdict(list)
    for key, value in rounds.items():
        round_groups[key[-1]].append(value)
    return {
        'regions': [{'phase': key[0], 'name': key[1],
                     'wall': percentiles([row['wall_ms'] for row in values]),
                     'gpu_event_span': percentiles([row['gpu_elapsed_ms'] for row in values
                                                    if row.get('gpu_elapsed_ms') is not None])}
                    for key, values in sorted(groups.items(), key=lambda item: str(item[0]))],
        'per_background_worker_round': {name: percentiles(values) for name, values in round_groups.items()},
        'frame_materialization_per_forecast': {
            str(selected): percentiles([value for key, value in frame_groups.items() if key[-1] == selected])
            for selected in (False, True)},
    }


def critical_paths(rows):
    jobs = defaultdict(list)
    for row in rows:
        if row.get('phase') in ('forecast', 'advance'):
            jobs[(row['pid'], row['episode'], row['round_start'])].append(row)
    results = []
    for barrier in rows:
        if barrier['name'] != 'barrier_done' or barrier['actual_round_start'] < 24:
            continue
        key = (barrier['pid'], barrier['actual_episode'], barrier['actual_round_start'])
        candidates = jobs[key]
        endings = [row for row in candidates if row['phase'] == 'advance' and row['name'] == 'task_end']
        if not endings:
            raise ValueError(f'Missing worker completion records for {key}')
        slowest = max(endings, key=lambda row: row['end_s'])
        begin, end = barrier.get('wait_begin_s', barrier['request_arrived_s']), barrier['begin_s']
        intervals = []
        for row in candidates:
            name = row['name']
            if row['phase'] == 'forecast' and name == 'forecast_decode':
                category = 'forecast_decode_and_materialize'
            elif row['phase'] == 'advance' and row['worker'] == slowest['worker']:
                category = {'prepare': 'predicted_condition_prepare',
                            'denoise_step': 'denoise_host_submission',
                            'step_sync': 'denoise_completion_wait'}.get(name)
            else:
                category = None
            if category:
                left, right = max(begin, row['begin_s']), min(end, row['end_s'])
                if right > left:
                    intervals.append((left, right, category))
        boundaries = sorted({begin, end, *(point for interval in intervals for point in interval[:2])})
        visible = defaultdict(float)
        for left, right in zip(boundaries, boundaries[1:]):
            active = [category for start, stop, category in intervals if start <= left and stop >= right]
            category = active[0] if len(active) == 1 else ('overlap' if active else 'other_or_scheduling')
            visible[category] += (right - left) * 1000
        results.append({'pid': key[0], 'episode': key[1], 'round_start': key[2],
                        'slowest_worker': slowest['worker'], 'wait_ms': barrier['wait_ms'],
                        'exposed_wall_ms': dict(visible),
                        'background_done_minus_arrival_ms': (slowest['end_s'] - barrier['request_arrived_s']) * 1000})
    return results


def trace_summary(path):
    events = json.loads(path.read_text())['traceEvents']
    result = coverage(path, events=events)
    totals = defaultdict(float)
    for row in events:
        if 'memcpy' in row.get('cat', '').lower():
            totals[(row['args']['device'], row['name'])] += row.get('dur', 0) / 1000
    result['copy_activity_ms'] = [{'device': key[0], 'name': key[1], 'sum_ms': value}
                                  for key, value in sorted(totals.items())]
    result['complete'] = (result['kernel_devices'] == list(range(4))
                          and result['copy_devices'] == list(range(4))
                          and result['chunk_regions'] > 0
                          and any('HtoD' in name for name in result['copy_names'])
                          and any('DtoH' in name for name in result['copy_names']))
    result['activity'] = trace_activity(events)
    result['complete'] = result['complete'] and len(result['activity']['forward_cpu_threads']) == 4
    result['note'] = 'Activity sums per device over the entire sample; not additive request latency.'
    return result


def union_intervals(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((begin, end))
    return merged


def trace_activity(events):
    kernels = defaultdict(list)
    forwards = defaultdict(list)
    for row in events:
        if row.get('cat') == 'kernel':
            kernels[row['args']['device']].append((row['ts'], row['ts'] + row['dur']))
        if row.get('cat') == 'gpu_user_annotation' and row['name'] == 'chunk.model_forward':
            forwards[row['pid']].append((row['ts'], row['ts'] + row['dur']))
    devices = []
    for device, intervals in sorted(kernels.items()):
        merged = union_intervals(intervals)
        ends = [end for _, end in merged]
        spans, active_times = [], []
        for begin, end in forwards[device]:
            active = 0.0
            index = bisect_left(ends, begin)
            while index < len(merged) and merged[index][0] < end:
                left, right = merged[index]
                active += max(0, min(end, right) - max(begin, left))
                index += 1
            spans.append((end - begin) / 1000)
            active_times.append(active / 1000)
        window = (merged[-1][1] - merged[0][0]) / 1000
        active = sum(end - begin for begin, end in merged) / 1000
        devices.append({'device': device, 'kernel_window_ms': window, 'kernel_active_union_ms': active,
                        'kernel_active_fraction': active / window,
                        'forward_gpu_span': percentiles(spans),
                        'forward_kernel_active': percentiles(active_times)})
    return {'forward_cpu_threads': sorted({row['tid'] for row in events
                                           if row.get('cat') == 'user_annotation'
                                           and row.get('name') == 'chunk.model_forward'}),
            'devices': devices,
            'caveat': 'Heavily instrumented sample; kernel gaps do not prove GIL contention or predict unprofiled utilization.'}


def timeline(rows, output):
    requests = [row for row in rows if row['name'] == 'barrier_done' and row['actual_round_start'] == 24]
    if not requests:
        return
    request = requests[0]
    selected = [row for row in rows if row['pid'] == request['pid']
                and row.get('episode') == request['actual_episode']
                and row.get('round_start') in (24, 32, 40, 48)
                and row['name'] in ('forecast_decode', 'prepare', 'denoise_step', 'step_sync')]
    if not selected:
        return
    begin = min(row['begin_s'] for row in selected)
    end = max(row['end_s'] for row in selected)
    colors = {'forecast_decode': '#9467bd', 'prepare': '#ff7f0e',
              'denoise_step': '#1f77b4', 'step_sync': '#2ca02c'}
    elements = ['<html><meta charset="utf-8"><h2>Wall-clock timeline (inclusive host regions)</h2>',
                '<p>Blue: denoise host submission; green: completion wait; orange: prepare; purple: forecast. '
                'Nested regions and GPUs must not be summed.</p><svg width="1240" height="440">']
    for worker in range(4):
        elements.append(f'<text x="0" y="{70 + worker * 90}">worker {worker}</text>')
    for row in selected:
        worker = row.get('worker', 0)
        horizontal = 100 + (row['begin_s'] - begin) / (end - begin) * 1100
        width = max(1, row['wall_ms'] / 1000 / (end - begin) * 1100)
        vertical = 40 + worker * 90 + (20 if row.get('phase') == 'finish' else 0)
        title = html.escape(f"{row['name']} round={row['round_start']} {row['wall_ms']:.3f}ms")
        elements.append(f'<rect x="{horizontal:.2f}" y="{vertical}" width="{width:.2f}" height="16" '
                        f'fill="{colors[row["name"]]}"><title>{title}</title></rect>')
    elements.append('</svg></html>')
    output.write_text('\n'.join(elements))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    methods, trials = {}, {}
    for phase in ('control', 'segments'):
        trials[phase], methods[phase] = summarize_method(args.run / phase)
    if len(trials['control']) != 20 or trials['control'].keys() != trials['segments'].keys():
        raise ValueError('Expected paired 20-trial experiment')
    if methods['control']['resume_signature'] != methods['segments']['resume_signature']:
        raise ValueError('Paired algorithm/evaluation signatures differ')
    rows = [json.loads(line) for path in sorted((args.run / 'segments' / 'diagnostics').glob('segments-*.jsonl'))
            for line in path.read_text().splitlines()]
    if not rows:
        raise ValueError('No diagnostic records')
    paths = critical_paths(rows)
    if not paths:
        raise ValueError('No steady critical paths')
    baseline = methods['control']['client_chunk_requests_excluding_first_three']
    measured = methods['segments']['client_chunk_requests_excluding_first_three']
    overhead = {name: (measured[name] / baseline[name] - 1) * 100 for name in ('p50_ms', 'p95_ms')}
    env = tagged(args.run / 'segments', '[CHUNK_ENV]')
    windows = [row for row in env if row['kind'] == 'execution_window' and row['step'] >= 24]
    steps = [row for row in env if row['kind'] == 'step' and row['step'] >= 24]
    categories = sorted({category for row in paths for category in row['exposed_wall_ms']})
    exposure = {category: percentiles([row['exposed_wall_ms'].get(category, 0) for row in paths])
                for category in categories}
    trial_details = {}
    for phase in ('control', 'segments'):
        summary = json.loads((args.run / phase / 'summary.json').read_text())
        trial_details[phase] = {}
        for task in summary['task_results']:
            result = json.loads(Path(task['result_paths'][0]).read_text())
            for trial in result['trials']:
                key = f"{result['suite']}/{result['task_id']}/{trial['trial']}"
                trial_details[phase][key] = {name: trial.get(name) for name in ('success', 'policy_steps', 'init_state_index')}
    differences = [key for key in trial_details['control']
                   if trial_details['control'][key] != trial_details['segments'][key]]
    traces = {path.name: trace_summary(path) for path in (args.run / 'trace' / 'diagnostics').glob('trace-*.json')}
    result = {'methods': methods, 'instrumentation_delta_percent': overhead,
              'instrumentation_perturbed_over_5_percent': any(abs(value) > 5 for value in overhead.values()),
              'paired_trial_details': trial_details, 'trial_differences': differences,
              'segments': segment_summary(rows), 'aggregated_segments': aggregate_segments(rows),
              'critical_paths': paths,
              'slowest_worker_counts': dict(Counter(row['slowest_worker'] for row in paths)),
              'exposed_wall_categories': exposure,
              'environment': {'step': percentiles([row['env_ms'] for row in steps]),
                              'eight_env_steps_sum': percentiles([row['env_sum_ms'] for row in windows]),
                              'execution_window': percentiles([row['window_ms'] for row in windows])},
              'probe': json.loads((args.run / 'probe' / 'probe.json').read_text()), 'traces': traces,
              'interpretation': ['GPU event spans include host submission gaps, not pure kernel time.',
                                 'Transfer-call wall time includes synchronization and casting.',
                                 'Missing or inapplicable regions are not zero-cost measurements.',
                                 'Critical-path categories partition each observed barrier window; no median subtraction.',
                                 'Paired runs are sequential, not a randomized repeated benchmark.']}
    (args.run / 'diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    timeline(rows, args.run / 'timeline.html')
    lines = ['# Chunk pipeline latency diagnostics', '',
             '| Run | Success | Steady client P50 ms | P95 ms |', '|---|---:|---:|---:|']
    for phase, method in methods.items():
        latency = method['client_chunk_requests_excluding_first_three']
        lines.append(f"| {phase} | {sum(trials[phase].values())}/20 | {latency['p50_ms']:.3f} | {latency['p95_ms']:.3f} |")
    lines += ['', f'Instrumentation delta: {overhead}; paired trial differences: {len(differences)}.',
              '', '## Exposed barrier wall time', '', '| Category | P50 ms | P95 ms |', '|---|---:|---:|']
    for category, values in exposure.items():
        lines.append(f"| {category} | {values['p50_ms']:.3f} | {values['p95_ms']:.3f} |")
    lines += ['', 'Each request is partitioned before percentiles; the table medians do not add to median latency.',
              '', '## Measured segments (pooled workers)', '',
              '| Phase | Region | Wall P50 ms | Wall P95 ms |', '|---|---|---:|---:|']
    selected_regions = {'forecast_decode', 'vae_decode', 'frames_materialize', 'frame_d2h_cast_call',
                        'frame_numpy_pil', 'prepare', 'vae_encode', 'image_h2d_call', 'noise_total',
                        'denoise_step', 'model_forward', 'step_sync', 'actions_d2h_convert'}
    for region in result['aggregated_segments']['regions']:
        if region['name'] in selected_regions:
            values = region['wall']
            lines.append(f"| {region['phase']} | {region['name']} | {values['p50_ms']:.3f} | {values['p95_ms']:.3f} |")
    lines += ['', 'These are inclusive call intervals, not additive latency components. '
              'Background statistics include terminal speculative work, not only consumed chunks.',
              '', '## Detailed CUDA activity', '', '```json', json.dumps(traces, indent=2), '```',
              '', '## Environment', '', '```json', json.dumps(result['environment'], indent=2), '```',
              '', '## Limitations', '', *['- ' + note for note in result['interpretation']],
              '- Review gpu-processes.csv and gpu-util.csv before attributing differences to instrumentation.',
              f'- Profiler probe complete: {result["probe"]["complete"]}; detailed traces: {len(traces)}.',
              '- Optimization decisions require inspecting segment GPU spans alongside exposed wall time; '
              'D2H-call time alone is not evidence of a bandwidth bottleneck.']
    (args.run / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
