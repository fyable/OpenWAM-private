import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from openwam import chunk_diagnostics as diagnostics
from scripts.summarize_chunk_diagnostics import aggregate_segments, critical_paths, segment_summary, trace_activity
from scripts.probe_chunk_profiler import coverage


def test_disabled_decorator_is_identity(monkeypatch):
    monkeypatch.setattr(diagnostics, 'ENABLED', False)
    def operation():
        return 42
    assert diagnostics.profiled('test')(operation) is operation


def test_context_is_thread_local_and_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, 'ENABLED', True)
    monkeypatch.setenv('OPENWAM_CHUNK_DIAGNOSTICS_DIR', str(tmp_path))
    def operation(worker):
        with diagnostics.context(worker=worker, device='cpu'):
            with diagnostics.region('outer'):
                with diagnostics.context(chunk_start=8):
                    with diagnostics.region('inner'):
                        pass
                with pytest.raises(ValueError):
                    with diagnostics.region('failure'):
                        raise ValueError('expected')
            diagnostics.flush()
        assert getattr(diagnostics._state, 'metadata', None) is None
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(operation, range(4)))
    rows = [json.loads(line) for path in tmp_path.glob('*.jsonl') for line in path.read_text().splitlines()]
    assert len(rows) == 12
    for row in rows:
        assert row['gpu_elapsed_ms'] is None
        assert row['wall_ms'] >= 0
        assert ('chunk_start' in row) == (row['name'] == 'inner')
        assert row.get('failed', False) == (row['name'] == 'failure')
    assert {row['worker'] for row in rows} == set(range(4))


def test_flush_defers_unfinished_events_without_synchronizing(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, 'ENABLED', True)
    monkeypatch.setenv('OPENWAM_CHUNK_DIAGNOSTICS_DIR', str(tmp_path))
    class Event:
        ready = False
        def query(self):
            return self.ready
        def elapsed_time(self, other):
            return 2.5
    begin, end = Event(), Event()
    monkeypatch.setattr(diagnostics._state, 'pending', [({'name': 'gpu'}, begin, end)], raising=False)
    diagnostics.flush()
    assert not list(tmp_path.glob('*.jsonl'))
    assert len(diagnostics._state.pending) == 1
    end.ready = True
    diagnostics.flush()
    assert not diagnostics._state.pending
    assert json.loads(next(tmp_path.glob('*.jsonl')).read_text())['gpu_elapsed_ms'] == 2.5


def test_critical_path_uses_slowest_worker_and_partitions_window():
    common = {'pid': 1, 'episode': 2, 'round_start': 24}
    rows = [dict(common, phase='advance', worker=worker, name='task_end', begin_s=finished, end_s=finished)
            for worker, finished in ((1, 1.1), (2, 1.4), (3, 1.2))]
    rows += [dict(common, phase='advance', worker=2, name='prepare', begin_s=.9, end_s=1.1),
             dict(common, phase='advance', worker=2, name='denoise_step', begin_s=1.1, end_s=1.2),
             dict(common, phase='advance', worker=2, name='step_sync', begin_s=1.2, end_s=1.4),
             dict(common, name='barrier_done', actual_episode=2, actual_round_start=24,
                  begin_s=1.5, request_arrived_s=1., wait_ms=500.)]
    result = critical_paths(rows)[0]
    assert result['slowest_worker'] == 2
    assert sum(result['exposed_wall_ms'].values()) == pytest.approx(500)
    assert result['exposed_wall_ms']['predicted_condition_prepare'] == pytest.approx(100)
    assert result['exposed_wall_ms']['denoise_completion_wait'] == pytest.approx(200)
    assert result['exposed_wall_ms']['other_or_scheduling'] == pytest.approx(100)
    rows[-1]['wait_begin_s'] = 1.05
    rows[-1]['wait_ms'] = 450
    result = critical_paths(rows)[0]
    assert sum(result['exposed_wall_ms'].values()) == pytest.approx(450)


def test_summary_separates_condition_and_worker():
    rows = [dict(name='prepare', phase='advance', condition='predicted', worker=worker,
                 round_start=24, wall_ms=worker + 1, gpu_elapsed_ms=None) for worker in range(4)]
    summary = segment_summary(rows)
    assert len(summary) == 4
    assert all(row['gpu_event_span'] is None for row in summary)


def test_trace_coverage_does_not_count_gpu_annotations_as_cpu_threads(tmp_path):
    events = []
    for device in range(4):
        events.extend([
            {'cat': 'user_annotation', 'name': f'probe.worker.{device}', 'tid': 100 + device},
            {'cat': 'gpu_user_annotation', 'name': f'probe.worker.{device}', 'tid': device},
            {'cat': 'kernel', 'name': 'kernel', 'args': {'device': device}},
            {'cat': 'gpu_memcpy', 'name': 'Memcpy HtoD', 'args': {'device': device}},
        ])
    path = tmp_path / 'trace.json'
    path.write_text(json.dumps({'traceEvents': events}))
    result = coverage(path)
    assert result['worker_threads'] == [100, 101, 102, 103]
    assert result['kernel_devices'] == result['copy_devices'] == list(range(4))


def test_trace_activity_unions_overlapping_kernels():
    events = [{'cat': 'kernel', 'args': {'device': 0}, 'ts': start, 'dur': duration}
              for start, duration in ((0, 1000), (500, 1000), (2000, 1000))]
    events.append({'cat': 'gpu_user_annotation', 'name': 'chunk.model_forward',
                   'pid': 0, 'ts': 1000, 'dur': 1500})
    result = trace_activity(events)['devices'][0]
    assert result['kernel_active_union_ms'] == 2.5
    assert result['forward_gpu_span']['p50_ms'] == 1.5
    assert result['forward_kernel_active']['p50_ms'] == 1.0


def test_aggregate_sums_each_round_before_percentiles():
    rows = [dict(pid=1, episode=0, round_start=24, worker=worker, phase='advance',
                 name='denoise_step', wall_ms=duration, gpu_elapsed_ms=duration)
            for worker, durations in ((0, (1, 2, 3)), (1, (4, 5, 6))) for duration in durations]
    result = aggregate_segments(rows)['per_background_worker_round']['denoise_step']
    assert result['count'] == 2
    assert result['p50_ms'] == 10.5
