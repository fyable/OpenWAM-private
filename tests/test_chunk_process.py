from contextlib import nullcontext
from threading import Event
import time

import numpy as np
import pytest

from openwam.deploy import chunk_process
from openwam.deploy.chunk_pipeline import Chunk, ChunkPipeline
from tests.test_chunk_pipeline import FakeWorker, make_request


def fake_service(connection, stop, checkpoint, checkpoint_name, device, threads):
    connection.send({'ready': True, 'device': device})
    generation = 0
    while True:
        message = connection.recv()
        operation = message['operation']
        if operation == 'reset':
            generation = message['generation']
        if (message.get('request') or {}).get('fail'):
            connection.send({'error': 'injected worker failure'})
            break
        if (message.get('request') or {}).get('stale'):
            connection.send({'sequence': message['sequence'], 'generation': generation - 1})
            continue
        statuses = message['chunks']
        result = None
        if operation == 'finish':
            statuses[0]['stage'] = 10
            result = np.zeros((32, 7), dtype=np.float32)
        elif operation == 'forecast':
            result = ({statuses[0]['start'] + offset: offset for offset in (8, 16, 24)}, 0.0)
        elif operation == 'advance':
            for _ in range(message['budget']):
                if stop.is_set():
                    break
                time.sleep(.005)
                for status in statuses:
                    status['stage'] += 1
                    status['completed_at'].append(time.perf_counter())
            result = (0.0, 0.0)
        connection.send({'sequence': message['sequence'], 'generation': generation,
                         'result': result, 'chunks': statuses})
        if operation == 'close':
            break
    connection.close()


@pytest.fixture
def remote(monkeypatch):
    monkeypatch.setattr(chunk_process, '_serve', fake_service)
    worker = chunk_process.ProcessChunkWorker('unused', 'unused', 'cuda:1', timeout=10)
    try:
        yield worker
    finally:
        worker.close(force=True)


def test_remote_commands_never_return_latents(remote):
    chunk = Chunk(8, 1)
    stop = Event()
    remote.command('advance', [chunk], make_request(), stop=stop, forecast=({}, 0), budget=3)
    assert chunk.stage == 3
    assert len(chunk.completed_at) == 3
    assert chunk.video is None and chunk.action is None
    actions = remote.command('finish', [chunk], make_request())
    assert chunk.stage == 10 and actions.shape == (32, 7)
    remote.reset()
    assert remote.generation == 1


def test_remote_full_round_barrier_rotation_and_reset(remote):
    pipeline = ChunkPipeline([FakeWorker(), remote, FakeWorker(), FakeWorker()])
    try:
        for index in range(7):
            result = pipeline.generate(make_request())
            stats = result['sampler_stats']
            assert stats['head_worker'] == index % 4
            assert stats['online_steps'] == 10 - min(index * 3, 9)
        pipeline.reset()
        assert not pipeline.pending and not pipeline.chunks
        assert remote.generation >= 2
        assert pipeline.generate(make_request())['sampler_stats']['online_steps'] == 10
    finally:
        pipeline.close()
    assert not remote.process.is_alive()


def test_remote_cancellation_does_not_start_new_work(remote):
    stop = Event()
    stop.set()
    chunk = Chunk(8, 1)
    assert remote.command('advance', [chunk], make_request(), stop=stop, budget=3) == (None, None)
    assert chunk.stage == 0


def test_remote_cancellation_interrupts_at_step_boundary(remote):
    from concurrent.futures import ThreadPoolExecutor
    stop = Event()
    chunk = Chunk(8, 1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(remote.command, 'advance', [chunk], make_request(),
                                 stop=stop, forecast=({}, 0), budget=300)
        time.sleep(.05)
        stop.set()
        future.result(timeout=5)
    assert chunk.stage < 300
    remote.reset()


def test_stale_response_is_rejected(remote):
    with pytest.raises(RuntimeError, match='Stale chunk response'):
        remote.command('finish', [Chunk(8, 1)], {**make_request(), 'stale': True})


def test_worker_error_propagates_and_cleanup(remote):
    with pytest.raises(RuntimeError, match='injected worker failure'):
        remote.command('finish', [Chunk(8, 1)], {**make_request(), 'fail': True})
    remote.close(force=True)
    assert not remote.process.is_alive()


def test_exited_worker_is_not_silently_replaced(remote):
    remote.process.terminate()
    remote.process.join(timeout=5)
    with pytest.raises((RuntimeError, BrokenPipeError, EOFError)):
        remote.command('finish', [Chunk(8, 1)], make_request())


def test_status_rejects_wrong_identity():
    with pytest.raises(RuntimeError, match='identity'):
        chunk_process.apply_status(Chunk(0, 0), chunk_process.chunk_status(Chunk(8, 1)))


def test_real_service_protocol_and_state_retention(monkeypatch):
    class Connection:
        def __init__(self):
            self.results = []
            self.messages = iter([
                {'operation': 'reset', 'sequence': 1, 'generation': 1},
                {'operation': 'advance', 'sequence': 2, 'generation': 1,
                 'chunks': [chunk_process.chunk_status(Chunk(8, 1))], 'metadata': {},
                 'request': make_request(), 'forecast': ({8: 100}, 0), 'budget': 3},
                {'operation': 'finish', 'sequence': 3, 'generation': 1,
                 'chunks': [chunk_process.chunk_status(Chunk(8, 1, stage=3))],
                 'metadata': {}, 'request': make_request()},
                {'operation': 'forecast', 'sequence': 4, 'generation': 1,
                 'chunks': [chunk_process.chunk_status(Chunk(8, 1, stage=10))],
                 'metadata': {}, 'request': make_request()},
                {'operation': 'close', 'sequence': 5, 'generation': 1},
            ])

        def recv(self):
            return next(self.messages)

        def send(self, value):
            self.results.append(value)

        def close(self):
            pass

    monkeypatch.setattr(chunk_process.torch, 'set_num_threads', lambda value: None)
    monkeypatch.setattr(chunk_process.torch, 'set_num_interop_threads', lambda value: None)
    monkeypatch.setattr(chunk_process.torch.cuda, 'device', lambda value: nullcontext())
    monkeypatch.setattr('openwam.deploy.model_loader.load_from_checkpoint_dir', lambda *args: (None, object()))
    monkeypatch.setattr(chunk_process, 'ChunkWorker', lambda model: FakeWorker())
    connection = Connection()
    chunk_process._serve(connection, Event(), 'unused', 'unused', 'cuda:1', 1)
    assert not any('error' in response for response in connection.results)
    assert connection.results[2]['chunks'][0]['stage'] == 3
    assert connection.results[3]['chunks'][0]['stage'] == 10
    assert set(connection.results[4]['result'][0]) == {16, 24, 32}


def test_stale_generation_fails_service(monkeypatch):
    class Connection:
        def __init__(self):
            self.results = []

        def recv(self):
            return {'sequence': 1, 'operation': 'finish', 'generation': 99}

        def send(self, value):
            self.results.append(value)

        def close(self):
            pass

    monkeypatch.setattr(chunk_process.torch, 'set_num_threads', lambda value: None)
    monkeypatch.setattr(chunk_process.torch, 'set_num_interop_threads', lambda value: None)
    monkeypatch.setattr(chunk_process.torch.cuda, 'device', lambda value: nullcontext())
    monkeypatch.setattr('openwam.deploy.model_loader.load_from_checkpoint_dir', lambda *args: (None, object()))
    monkeypatch.setattr(chunk_process, 'ChunkWorker', lambda model: FakeWorker())
    connection = Connection()
    chunk_process._serve(connection, Event(), 'unused', 'unused', 'cuda:1', 1)
    assert 'Stale chunk generation' in connection.results[-1]['error']
