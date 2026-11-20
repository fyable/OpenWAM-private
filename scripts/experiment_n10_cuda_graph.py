"""Measure CUDA Graph replay for the nine action-only steps of N10."""

import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks.action_cache.next_experiment import load_selected
from benchmarks.action_cache.next_runtime import replace_method
from benchmarks.libero import research
from scripts.experiment_native_n10_kv import (
    ROOT, Replay, action_metrics, alarm, atomic_json, choose_gpu, file_hash,
    generate, inputs, performance, read_sealed, sample_order, seal, write_json,
)

NATIVE = "native_n10"
EAGER = "video_refresh_0"
GRAPH = "video_refresh_0_cuda_graph"
METHODS = (NATIVE, EAGER, GRAPH)
SEEDS = (43, 44)
REPEATS = 3


class ActionGraph:
    """Own the buffers captured by one fixed-layout, batch-one forward."""

    def __init__(self):
        self.graph = None
        self.setup_ms = None
        self.pool_reserved_bytes = None

    @staticmethod
    def layout(values):
        return tuple(None if x is None else (x.shape, x.stride(), x.dtype, x.device) for x in values)

    def initialize(self, forward, args, keys, mask, projections, values):
        device = args[0].device
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        reserved_before = torch.cuda.memory_reserved(device)
        self.static = tuple(None if x is None else x.clone() for x in values)
        static_args = self.static[:len(args)]
        static_mask = self.static[len(args)]
        flat_keys = self.static[len(args) + 1:len(args) + 1 + 2 * len(keys)]
        static_keys = tuple(zip(flat_keys[::2], flat_keys[1::2]))
        static_projections = self.static[len(args) + 1 + 2 * len(keys):]
        self.signature = self.layout(values)
        self.modules = tuple(projections)
        stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        with ExitStack() as stack:
            for module, cached in zip(self.modules, static_projections):
                stack.enter_context(replace_method(
                    module, "forward", lambda _, cached=cached: lambda _inputs: cached))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward(*static_args, static_keys, static_mask)
            current.wait_stream(stream)
            with torch.cuda.graph(graph, stream=stream):
                self.output = forward(*static_args, static_keys, static_mask)
        torch.cuda.synchronize(device)
        self.graph = graph
        self.setup_ms = (time.perf_counter() - start) * 1000
        self.pool_reserved_bytes = torch.cuda.memory_reserved(device) - reserved_before

    def run(self, forward, args, keys, mask, projections, *, new_request):
        if new_request:
            values = (*args, mask, *(x for pair in keys for x in pair), *projections.values())
            if self.graph is None:
                self.initialize(forward, args, keys, mask, projections, values)
            else:
                if self.layout(values) != self.signature or tuple(projections) != self.modules:
                    raise ValueError("Action graph request layout changed")
                for target, source in zip(self.static, values):
                    if target is not None:
                        target.copy_(source)
        else:
            dynamic = args[:2]
            if self.layout(dynamic) != self.signature[:2]:
                raise ValueError("Action graph step layout changed")
            for target, source in zip(self.static[:2], dynamic):
                target.copy_(source)
        self.graph.replay()
        return self.output


def validate(replay, states, graph, out):
    first = states[0]
    second = next(item for item in states if (item[0]['suite'], item[0]['task_id']) !=
                  (first[0]['suite'], first[0]['task_id']))
    checked = []
    for index, (entry, conditions) in enumerate((first, second, first)):
        seed = SEEDS[1 if index == 1 else 0]
        expected, reference, _ = generate(replay, conditions, seed, EAGER, capture=True)
        actual, trajectory, details = generate(replay, conditions, seed, EAGER, capture=True,
                                               action_graph=graph)
        validate_counts(details)
        for step, (before, after) in enumerate(zip(reference, trajectory, strict=True)):
            for modality, left, right in zip(("video", "action"), before, after, strict=True):
                if not torch.equal(left, right):
                    delta = (left.float() - right.float()).abs().max().item()
                    raise ValueError(f"Smoke {index + 1}, {entry['state_id']}, step {step}, {modality}: max abs {delta}")
        np.testing.assert_array_equal(actual[:8], expected[:8])
        checked.append({"state": entry["state_id"], "seed": seed, "steps": details["video_refresh"]})
        atomic_json(out / "validation.json", checked)


def validate_counts(details):
    counts = details["video_refresh"]
    if (counts["action_forwards"] != 10 or counts["video_refresh_steps"] != [0]
            or counts["video_reuse_steps"] != list(range(1, 10)) or counts.get("graph_replays") != 9):
        raise ValueError(f"Unexpected graph schedule: {counts}")


def timing_pairs(replay, entry, conditions, graph):
    for seed in SEEDS:
        for rep in range(REPEATS):
            cases = {}
            for name in sample_order(METHODS, entry, seed, rep):
                actions, _, details = generate(
                    replay, conditions, seed, EAGER if name == GRAPH else name,
                    action_graph=graph if name == GRAPH else None)
                if name == GRAPH:
                    validate_counts(details)
                cases[name] = (actions, details)
            np.testing.assert_array_equal(cases[GRAPH][0][:8], cases[EAGER][0][:8])
            yield {"state": entry["state_id"], "task": [entry["suite"], entry["task_id"]],
                   "seed": seed, "repeat": rep, "cases": {
                       name: {**details, **action_metrics(cases[NATIVE][0], actions, 8)}
                       for name, (actions, details) in cases.items()}}


def summarize(rows):
    methods = performance(rows, NATIVE)
    if methods:
        native_ms = methods[NATIVE]["request_ms"]["p50"]
        for name, method in methods.items():
            method["request_ms"]["p50_ratio"] = native_ms / method["request_ms"]["p50"]
            remaining = [sum(row["cases"][name]["step_ms"][1:]) for row in rows]
            method["steps_1_to_9_ms"] = {"p50": float(np.median(remaining)),
                                          "p95": float(np.percentile(remaining, 95))}
    return methods


def run(args):
    plan, protocol, source, split = inputs(args.reference_run)
    if research.source_identity()["source_sha256"] != protocol["research_source_sha256"]:
        raise ValueError("Model source differs from the sealed E2 reference")
    states = load_selected(source, split, plan, "calibration")
    if len(states) != 16:
        raise ValueError("Expected 16 frozen Calibration observations")
    gpu, _ = choose_gpu(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, E2_MEMORY_MODE="text-offload")
    lock_path = ROOT / ".cache" / f"e2-reproduction-{gpu}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args, gpu, plan, protocol, source, states)


def run_locked(args, gpu, plan, protocol, source, states):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    expected = len(states) * len(SEEDS) * REPEATS
    checkpoint = SimpleNamespace(ckpt_dir=protocol["checkpoint_dir"], ckpt_name=protocol["checkpoint_name"],
                                 device="cuda:0", steps=10, horizon=8)
    write_json(out / "protocol.json", seal({
        "gpu": gpu, "deadline": args.deadline.isoformat(), "plan": plan["sha256"],
        "states": [entry["state_id"] for entry, _ in states], "methods": METHODS,
        "seeds": SEEDS, "repeats": REPEATS, "sources": {name: file_hash(Path(__file__).with_name(name))
            for name in (Path(__file__).name, "experiment_native_n10_kv.py", "native_n10_video_refresh.py")},
    }))
    rows = []
    graph = ActionGraph()
    status = "loading"
    validated = False

    def progress():
        atomic_json(out / "status.json", {"status": status, "completed": len(rows), "expected": expected})

    try:
        progress()
        with alarm(args.deadline.timestamp()):
            server = research.build_server(checkpoint)
            model = research.model_identity(server, checkpoint)
            reference = read_sealed(source / "bank/state_bank/discovery/model.json")
            if model["assets"] != reference["assets"] or model["attention_mask_mode"] != "mutual":
                raise ValueError("Unexpected model assets or attention mask")
            write_json(out / "model.json", seal(model))
            replay = Replay(server.engine)
            status = "validating"
            progress()
            # A cold text encode in text-offload mode moves action weights off CUDA.
            # Warm every prompt before capture, then reject misses before the offload hook.
            for _, conditions in states:
                generate(replay, conditions, SEEDS[0], NATIVE, text="prime")

            def require_hot_text(_encoder, _inputs):
                raise RuntimeError("Text cache miss would invalidate captured action weight addresses")

            encoder = replay.architecture.video_backbone.text_encoder
            with encoder.register_forward_pre_hook(require_hot_text, prepend=True):
                validate(replay, states, graph, out)
                validated = True
                status = "timing"
                progress()
                for entry, conditions in states:
                    for row in timing_pairs(replay, entry, conditions, graph):
                        write_json(out / "pairs" / f"{len(rows):03}.json", seal(row))
                        rows.append(row)
                        progress()
        status = "complete"
    except TimeoutError:
        status = "budget_exhausted"
    except Exception:
        status = "failed"
        atomic_json(out / "error.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        methods = summarize(rows)
        atomic_json(out / "report.json", {
            "status": status, "completed": len(rows), "expected": expected, "validated": validated,
            "passed_2x": validated and len(rows) == expected and methods[GRAPH]["request_ms"]["p50_ratio"] >= 2,
            "graph_setup_ms": graph.setup_ms, "graph_setup_reserved_delta_bytes": graph.pool_reserved_bytes,
            "methods": methods,
        })
        progress()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    parsed = parser.parse_args()
    if parsed.deadline.tzinfo is None or parsed.deadline.timestamp() <= time.time():
        parser.error("--deadline must be in the future and include the timezone")
    run(parsed)
