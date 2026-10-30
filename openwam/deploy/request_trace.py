"""Opt-in recording at the actual chunk-generation boundary."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np


class RequestTrace:
    def __init__(self, directory, *, warmup=48, measure=120, max_requests=10000, metadata=None):
        if warmup < 0 or measure < 1 or max_requests < 3:
            raise ValueError("Invalid trace limits")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        temporary = self.directory / "session.tmp"
        temporary.write_text(json.dumps({"created_ns": time.time_ns(), "source": metadata or {}}))
        temporary.replace(self.directory / "session.json")
        self.warmup = warmup
        self.measure = measure
        self.max_requests = max_requests
        self.count = 0
        self.warm_count = 0
        self.episode_request = 0
        self.previous_prompt = None
        self.pending_reset = True
        self.complete = False

    def reset(self):
        self.pending_reset = True

    def record(self, conditions):
        if self.complete:
            return
        if self.count >= self.max_requests:
            raise RuntimeError("Trace reached max_requests before collecting enough warm observations")
        images = conditions.get("first_frame_image")
        if not isinstance(images, (list, tuple)) or len(images) != 1:
            raise ValueError("Trace requires exactly one composed first-frame image")
        image = np.asarray(images[0])
        state = np.asarray(conditions.get("proprio"), dtype=np.float32)
        prompt = conditions.get("prompt")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Trace requires uint8 RGB canvas")
        if state.ndim != 1 or not np.isfinite(state).all() or not isinstance(prompt, str):
            raise ValueError("Trace requires a finite state vector and a string prompt")
        if conditions.get("seed", 42) != 42:
            raise ValueError("Experiment trace supports seed=42 only")
        start = self.pending_reset or prompt != self.previous_prompt
        if start:
            self.episode_request = 0
        path = self.directory / f"request_{self.count:06d}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("xb") as stream:
            np.savez(stream, image=image, state=state, prompt=np.array(prompt),
                     episode_start=np.array(start))
        temporary.replace(path)
        if self.count >= self.warmup and self.episode_request >= 2:
            self.warm_count += 1
        self.count += 1
        self.episode_request += 1
        self.previous_prompt = prompt
        self.pending_reset = False
        if self.warm_count >= self.measure:
            marker = self.directory / "complete.json"
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps({"requests": self.count, "warm_requests": self.warm_count,
                                             "warmup": self.warmup, "measure": self.measure}))
            temporary.replace(marker)
            self.complete = True


def attach_request_trace(engine):
    directory = os.environ.get("OPENWAM_REQUEST_TRACE_DIR")
    if not directory:
        return None
    if getattr(engine.cfg.inference, "inference_mode", "sync") != "sync":
        raise ValueError("Request tracing requires synchronous E8 inference")
    if getattr(engine.cfg.inference, "inference_horizon", None) != 8:
        raise ValueError("Request tracing requires inference_horizon=8")
    trace = RequestTrace(Path(directory) / f"session_{uuid.uuid4().hex}",
                         warmup=int(os.environ.get("OPENWAM_TRACE_WARMUP", "48")),
                         measure=int(os.environ.get("OPENWAM_TRACE_MEASURE", "120")),
                         metadata={"sampler": getattr(engine.cfg.inference, "sampler", "native"),
                                   "steps": getattr(engine.cfg.inference, "denoise_steps", None), "advance": 8})
    original_generate, original_reset = engine.generate, engine.reset

    def generate(conditions):
        result = original_generate(conditions)
        trace.record(conditions)
        return result

    def reset():
        result = original_reset()
        trace.reset()
        return result

    engine.generate, engine.reset = generate, reset
    return trace


def assemble_trace(session, output):
    session, output = Path(session), Path(output)
    metadata = json.loads((session / "complete.json").read_text())
    paths = [session / f"request_{index:06d}.npz" for index in range(metadata["requests"])]
    _write_records(paths, output)
    return metadata


def _write_records(paths, output):
    output = Path(output)
    images, prompts, states, starts = [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            images.append(data["image"])
            prompts.append(str(data["prompt"].item()))
            states.append(data["state"])
            starts.append(bool(data["episode_start"]))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, images=np.stack(images), prompts=np.array(prompts),
                            states=np.stack(states), episode_starts=np.array(starts, dtype=np.bool_))
    temporary.replace(output)


def assemble_sessions(directory, output, *, warmup, measure):
    sessions = sorted((path for path in Path(directory).glob("session_*") if (path / "session.json").is_file()),
                      key=lambda path: json.loads((path / "session.json").read_text())["created_ns"])
    records = []
    warm_count = 0
    previous_prompt = None
    episode_request = 0
    for session in sessions:
        for path in sorted(session.glob("request_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                prompt, start = str(data["prompt"].item()), bool(data["episode_start"])
            if start or prompt != previous_prompt:
                episode_request = 0
            if len(records) >= warmup and episode_request >= 2:
                warm_count += 1
            records.append(path)
            previous_prompt = prompt
            episode_request += 1
            if warm_count == measure:
                _write_records(records, output)
                return {"requests": len(records), "warm_requests": warm_count, "warmup": warmup,
                        "measure": measure, "sessions": list(dict.fromkeys(str(path.parent) for path in records))}
    return None
