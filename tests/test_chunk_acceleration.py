import itertools
import json

import numpy as np
import torch

from scripts import benchmark_chunk_submission as benchmark
from scripts import run_chunk_acceleration as experiment
from openwam.deploy.chunk_pipeline import ChunkPipeline
from tests.test_chunk_pipeline import FakeWorker, make_request


def test_snapshot_tree_preserves_dtype_and_independent_storage():
    original = {'nested': (torch.ones(2, dtype=torch.bfloat16), ['image'])}
    result = benchmark.move_tree(original, 'cpu')
    result['nested'][0].zero_()
    assert original['nested'][0].sum() == 2
    assert result['nested'][0].dtype == torch.bfloat16


def test_numeric_difference_rejects_nonfinite_or_divergent():
    reference = np.array([1., 2.])
    assert benchmark.difference(reference, reference)['exact']
    assert not benchmark.difference(np.array([np.nan, 2.]), reference)['passed']
    assert not benchmark.difference(reference + .1, reference)['passed']


def write_micro(directory, process_ms=70, corrupt=False):
    for repeat in range(3):
        for mode in benchmark.MODES:
            lines = []
            for stage in benchmark.STAGES:
                for active in [*(tuple([rank]) for rank in range(4)),
                               *itertools.combinations(range(4), 3)]:
                    duration = process_ms if mode == 'process1' else 100
                    workers = [{'rank': rank, 'samples': [{'begin_s': 0, 'end_s': duration / 1000}],
                                'parity': {'video': {'passed': not corrupt}, 'action': {'passed': True}}}
                               for rank in range(4)]
                    lines.append(json.dumps({'repeat': repeat, 'mode': mode, 'stage': stage,
                                             'active': active, 'workers': workers}))
            (directory / f'{repeat}-{mode}.jsonl').write_text('\n'.join(lines))


def test_gate_requires_complete_and_numerically_equal_samples(tmp_path):
    write_micro(tmp_path)
    summary = benchmark.summarize(tmp_path)
    assert summary['process_gate_passed']
    assert len(summary['comparisons']) == 12
    (tmp_path / '2-process1.jsonl').unlink()
    assert not benchmark.summarize(tmp_path)['process_gate_passed']


def test_gate_rejects_insufficient_speed_or_parity(tmp_path):
    write_micro(tmp_path, process_ms=90)
    assert not benchmark.summarize(tmp_path)['process_gate_passed']
    write_micro(tmp_path, corrupt=True)
    assert not benchmark.summarize(tmp_path)['process_gate_passed']


def phase(latency, success=20):
    return {'successes': success, 'states': {'trial': 'sha'},
            'metrics': {'client_chunk_requests_excluding_first_three': {
                'p50_ms': latency, 'p95_ms': latency}},
            'episodes': {(f'suite_task00_attempt01.log', index): [latency] * (index + 1)
                         for index in range(20)}}


def test_acceptance_rejects_success_regression_and_tail_latency():
    assert experiment.eligible(phase(80), phase(100))
    assert not experiment.eligible(phase(80, 19), phase(100))
    assert not experiment.eligible(phase(110), phase(100))


def test_bootstrap_samples_paired_episodes():
    result = experiment.paired_bootstrap(phase(80), phase(100))
    np.testing.assert_allclose(result['p50_reduction_95ci'], [.2, .2])
    np.testing.assert_allclose(result['p95_reduction_95ci'], [.2, .2])


def test_request_capture_is_opt_in_and_does_not_overwrite(tmp_path, monkeypatch):
    path = tmp_path / 'request.pt'
    monkeypatch.setenv('OPENWAM_CHUNK_CAPTURE_PATH', str(path))
    pipeline = ChunkPipeline([FakeWorker() for _ in range(4)])
    try:
        pipeline.generate(make_request())
        original = path.read_bytes()
        assert torch.load(path, weights_only=False)['prompt'] == 'test'
        pipeline.generate({**make_request(), 'prompt': 'other'})
        assert path.read_bytes() == original
    finally:
        pipeline.close()
