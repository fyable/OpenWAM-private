"""Spawned, GPU-resident chunk workers with coarse-grained commands."""

from concurrent.futures import Future
from contextlib import nullcontext
import multiprocessing as mp
import os
from time import monotonic
import traceback
from types import SimpleNamespace

import torch

from openwam import chunk_diagnostics as diagnostics
from openwam.deploy.chunk_pipeline import Chunk, ChunkPipeline, ChunkWorker


def chunk_status(chunk):
    return {name: getattr(chunk, name) for name in
            ('start', 'worker', 'stage', 'speculative_ms', 'completed_at', 'online_step_host_ms')}


def apply_status(chunk, status):
    if (chunk.start, chunk.worker) != (status['start'], status['worker']):
        raise RuntimeError('Remote chunk identity mismatch')
    for name in ('stage', 'speculative_ms', 'completed_at', 'online_step_host_ms'):
        setattr(chunk, name, status[name])


def _serve(connection, stop, checkpoint, checkpoint_name, device, threads):
    try:
        from openwam.deploy.chunk_affinity import configure
        configure(device)
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        from openwam.deploy.model_loader import load_from_checkpoint_dir

        with torch.cuda.device(device):
            _, model = load_from_checkpoint_dir(checkpoint, device, checkpoint_name)
            worker = ChunkWorker(model)
            service = SimpleNamespace(workers=[worker] * 4, speculative_steps=3)
            chunks = {}
            generation = 0
            connection.send({'ready': True, 'pid': os.getpid(), 'device': device})
            while True:
                message = connection.recv()
                sequence, operation = message['sequence'], message['operation']
                if operation == 'close':
                    connection.send({'sequence': sequence, 'generation': generation, 'result': None})
                    break
                if operation == 'reset':
                    generation = message['generation']
                    chunks.clear()
                    diagnostics.flush(True)
                    connection.send({'sequence': sequence, 'generation': generation, 'result': None})
                    continue
                if message['generation'] != generation:
                    raise RuntimeError('Stale chunk generation')
                selected = []
                for status in message['chunks']:
                    chunk = chunks.setdefault(status['start'], Chunk(status['start'], status['worker']))
                    if chunk.stage != status['stage']:
                        raise RuntimeError('Coordinator/worker stage mismatch')
                    selected.append(chunk)
                with torch.inference_mode(), diagnostics.context(**message['metadata']):
                    if operation == 'finish':
                        result = ChunkPipeline._finish(service, selected[0], message['request'])
                    elif operation == 'forecast':
                        result = ChunkPipeline._forecast(service, selected[0], message['request'])
                        del chunks[selected[0].start]
                    elif operation == 'advance':
                        service.speculative_steps = message['budget']
                        forecast = Future()
                        forecast.set_result(message['forecast'])
                        result = ChunkPipeline._advance(
                            service, selected[0].worker, selected, message['request'], forecast, stop)
                    else:
                        raise ValueError(f'Unknown chunk command: {operation}')
                    diagnostics.flush()
                connection.send({'sequence': sequence, 'generation': generation, 'result': result,
                                 'chunks': [chunk_status(chunk) for chunk in selected],
                                 'memory': worker.memory()})
    except BaseException:
        try:
            connection.send({'error': traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class ProcessChunkWorker:
    def __init__(self, checkpoint, checkpoint_name, device, threads=1, timeout=900):
        if torch.device(device).type != 'cuda':
            raise ValueError('Process chunk workers require CUDA')
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.stop = context.Event()
        self.process = context.Process(target=_serve, args=(
            child, self.stop, checkpoint, checkpoint_name, device, threads), daemon=True)
        self.model = SimpleNamespace(device=device)
        self.generation = 0
        self.sequence = 0
        self.timeout = timeout
        self.peak_memory = None
        self.closed = False
        self.process.start()
        child.close()
        try:
            self.readiness = self._receive()
            if not self.readiness.get('ready'):
                raise RuntimeError('Remote chunk worker did not become ready')
        except BaseException:
            self.close(force=True)
            raise

    def _receive(self, stop=None):
        deadline = monotonic() + self.timeout
        while not self.connection.poll(0.01):
            if stop is not None and stop.is_set():
                self.stop.set()
            if not self.process.is_alive():
                raise RuntimeError(f'Chunk worker exited: {self.process.exitcode}')
            if monotonic() >= deadline:
                raise TimeoutError('Chunk worker command timed out')
        try:
            response = self.connection.recv()
        except EOFError as error:
            raise RuntimeError('Chunk worker closed its connection') from error
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response

    def command(self, operation, chunks=(), request=None, stop=None, **extra):
        if self.closed:
            raise RuntimeError('Chunk worker is closed')
        if stop is not None and stop.is_set():
            return None, None
        self.stop.clear()
        self.sequence += 1
        self.connection.send({'sequence': self.sequence, 'generation': self.generation,
                              'operation': operation, 'chunks': [chunk_status(chunk) for chunk in chunks],
                              'request': request, 'metadata': diagnostics.metadata(), **extra})
        response = self._receive(stop)
        if response['sequence'] != self.sequence or response['generation'] != self.generation:
            raise RuntimeError('Stale chunk response')
        if len(response.get('chunks', [])) != len(chunks):
            raise RuntimeError('Remote chunk result count mismatch')
        for chunk, status in zip(chunks, response.get('chunks', [])):
            apply_status(chunk, status)
        self.peak_memory = response.get('memory', self.peak_memory)
        return response['result']

    def reset(self):
        self.generation += 1
        self.command('reset')

    def close(self, force=False):
        if self.closed:
            return
        try:
            if not force and self.process.is_alive():
                self.command('close')
        finally:
            self.closed = True
            self.connection.close()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()

    def device_context(self):
        return nullcontext()

    def memory(self):
        return self.peak_memory
