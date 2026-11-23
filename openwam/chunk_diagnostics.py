"""Opt-in, thread-local chunk timing without additional CUDA synchronization."""

import json
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from threading import Lock, local
from time import perf_counter

import torch


ENABLED = os.environ.get('OPENWAM_CHUNK_DIAGNOSTICS') == '1'
TRACE = os.environ.get('OPENWAM_CHUNK_TRACE') == '1'
_state = local()
_lock = Lock()


def metadata():
    return dict(getattr(_state, 'metadata', None) or {})


def emit(rows):
    if not rows:
        return
    directory = Path(os.environ['OPENWAM_CHUNK_DIAGNOSTICS_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    with _lock, (directory / f'segments-{os.getpid()}.jsonl').open('a') as output:
        output.write(''.join(json.dumps(row) + '\n' for row in rows))


@contextmanager
def context(**metadata):
    previous = getattr(_state, 'metadata', None)
    _state.metadata = {**(previous or {}), **metadata}
    try:
        yield
    finally:
        _state.metadata = previous


@contextmanager
def region(name, gpu=True, **metadata):
    parent = getattr(_state, 'metadata', None)
    if not ENABLED or parent is None:
        yield
        return
    row = {**parent, **metadata, 'name': name, 'pid': os.getpid()}
    use_events = gpu and torch.device(row.get('device', 'cpu')).type == 'cuda'
    begin = end = None
    if use_events:
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record(torch.cuda.current_stream(row['device']))
    row['begin_s'] = perf_counter()
    annotation = torch.profiler.record_function('chunk.' + name) if TRACE else None
    if annotation is not None:
        annotation.__enter__()
    try:
        yield
    except BaseException:
        row['failed'] = True
        raise
    finally:
        row['end_s'] = perf_counter()
        row['wall_ms'] = (row['end_s'] - row['begin_s']) * 1000
        if end is not None:
            end.record(torch.cuda.current_stream(row['device']))
        if annotation is not None:
            annotation.__exit__(None, None, None)
        if not hasattr(_state, 'pending'):
            _state.pending = []
        _state.pending.append((row, begin, end))


def flush(wait=False):
    if not ENABLED:
        return
    remaining, rows = [], []
    for row, begin, end in getattr(_state, 'pending', []):
        if wait and end is not None:
            end.synchronize()
        if end is not None and not end.query():
            remaining.append((row, begin, end))
            continue
        row['gpu_elapsed_ms'] = begin.elapsed_time(end) if end is not None else None
        rows.append(row)
    _state.pending = remaining
    emit(rows)


def profiled(name, gpu=True):
    def decorate(function):
        if not ENABLED:
            return function

        @wraps(function)
        def measured(*args, **kwargs):
            with region(name, gpu=gpu):
                return function(*args, **kwargs)
        return measured
    return decorate


def mark(name, **metadata):
    if ENABLED:
        now = perf_counter()
        emit([{**(getattr(_state, 'metadata', None) or {}), **metadata,
               'name': name, 'pid': os.getpid(), 'begin_s': now, 'end_s': now,
               'wall_ms': 0, 'gpu_elapsed_ms': None}])
