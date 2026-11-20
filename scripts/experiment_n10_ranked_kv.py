"""Paired E2 ranking transfer at layer 29, step 9 of native N10."""

import argparse
from datetime import datetime
import fcntl
from itertools import combinations
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks.action_cache.next_experiment import load_selected
from benchmarks.libero import research
from scripts.experiment_native_n10_kv import (
    ROOT, Replay, action_metrics, alarm, atomic_json, choose_gpu, compare_trajectory,
    file_hash, generate, inputs, performance, read_sealed, sample_order, seal, write_json,
)
from scripts.research_e2_extensions import discovery_ranking

NATIVE = "native_n10"
FULL = "video_refresh_0"
RANDOM_NAMES = tuple(f"random_{i}" for i in range(5))
SEED = 20261119


def freeze(plan, source):
    rank = discovery_ranking(source / "base/diagnostics/e2_discovery")["29:9"]
    layout = read_sealed(source / "base/manifests/selection.json")["layouts"]["29"]
    groups = plan["target"]["token_groups"]
    prefix = list(range(layout["prefix_tokens"]))
    if sorted(row for group in groups for row in group) != list(range(len(prefix), layout["video_tokens"])):
        raise ValueError("Groups must partition the cached future video rows")
    if len(groups) != 8 or {len(group) for group in groups} != {30}:
        raise ValueError("Expected eight equal 30-row groups")
    low, high = sorted(rank["ascending"][:4]), sorted(rank["ascending"][4:])
    pool = [ids for ids in combinations(range(8), 4) if list(ids) not in (low, high)]
    choices = {"high": high, "low": low, **dict(zip(RANDOM_NAMES, random.Random(SEED).sample(pool, 5)))}
    rows = {name: prefix + sorted(row for group in ids for row in groups[group]) for name, ids in choices.items()}
    rows["uniform"] = prefix + sorted(row for group in groups for row in sorted(group)[::2])
    return {"rank": rank, "layout": layout, "groups": choices, "rows": rows}


def summarize(rows):
    if not rows:
        return {}
    timing = performance(rows, NATIVE)
    for value in timing.values():
        value["request_ms"]["p50_ratio"] = timing[NATIVE]["request_ms"]["p50"] / value["request_ms"]["p50"]
    comparisons = {}
    for control in ("random_mean", "uniform", "low", FULL):
        by_task = {}
        for row in rows:
            cases = row["cases"]
            controls = RANDOM_NAMES if control == "random_mean" else (control,)
            delta = cases["high"]["prefix_rmse"] - np.mean([cases[name]["prefix_rmse"] for name in controls])
            by_task.setdefault(tuple(row["task"]), []).append(delta)
        means = np.array([np.mean(values) for values in by_task.values()])
        boot = np.random.default_rng(SEED).choice(means, size=(10000, len(means))).mean(axis=1)
        comparisons[control] = {"delta": float(means.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist()}
    return {"methods": timing, "high_minus_control": comparisons}


def run(args):
    plan, protocol, source, split = inputs(args.reference_run)
    if research.source_identity()["source_sha256"] != protocol["research_source_sha256"]:
        raise ValueError("Model source differs from the sealed E2 reference")
    states = load_selected(source, split, plan, "calibration")
    frozen = freeze(plan, source)
    layout = frozen["layout"]
    expected = len(states) * 2
    selections = {name: (29, 9, rows, layout) for name, rows in frozen["rows"].items()}
    all_rows = (29, 9, list(range(layout["video_tokens"])), layout)
    gpu, _ = choose_gpu(args.gpu)
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, E2_MEMORY_MODE="text-offload")
    lock_path = ROOT / ".cache" / f"e2-reproduction-{gpu}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = args.output
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "protocol.json", seal({**frozen, "gpu": gpu, "profile": args.profile,
                   "plan": plan["sha256"], "states": [entry["state_id"] for entry, _ in states],
                   "seeds": [43, 44], "deadline": args.deadline.isoformat(),
                   "sources": {name: file_hash(Path(__file__).with_name(name)) for name in (
                       Path(__file__).name, "native_n10_video_refresh.py", "experiment_native_n10_kv.py")}}))
        status, rows = "loading", []
        try:
            atomic_json(out / "status.json", {"status": status})
            with alarm(args.deadline.timestamp()):
                options = SimpleNamespace(ckpt_dir=protocol["checkpoint_dir"], ckpt_name=protocol["checkpoint_name"],
                                          device="cuda:0", steps=10, horizon=8)
                server = research.build_server(options)
                model = research.model_identity(server, options)
                reference = read_sealed(source / "bank/state_bank/discovery/model.json")
                if model["assets"] != reference["assets"] or model["attention_mask_mode"] != "mutual":
                    raise ValueError("Unexpected model assets or attention mask")
                write_json(out / "model.json", seal(model))
                replay = Replay(server.engine)
                if args.profile:
                    profile_requests(replay, states, out)
                else:
                    atomic_json(out / "status.json", {"status": "validating", "expected": expected})
                    for _, conditions in states[:2]:
                        generate(replay, conditions, 43, NATIVE, text="prime")
                        actions, trace, _ = generate(replay, conditions, 43, FULL, capture=True)
                        actual, candidate, details = generate(replay, conditions, 43, "all_rows", capture=True,
                                                              selection=all_rows)
                        compare_trajectory(trace, candidate)
                        np.testing.assert_array_equal(actual, actions)
                        if details["video_refresh"]["selected_rows"] != [29, 9, layout["video_tokens"]]:
                            raise ValueError("Full-row control missed target")
                        for name, selection in selections.items():
                            _, candidate, details = generate(replay, conditions, 43, name, capture=True,
                                                              selection=selection)
                            if any(any(not torch.equal(a, b) for a, b in zip(x, y))
                                   for x, y in zip(trace[:9], candidate[:9])):
                                raise ValueError("Selection changed steps before the target")
                            compare_trajectory(trace, candidate, exact=False)
                            validate_selection(details)
                    atomic_json(out / "status.json", {"status": "timing", "completed": 0, "expected": expected})
                    for entry, conditions in states:
                        generate(replay, conditions, 43, NATIVE, text="prime")
                        for seed in (43, 44):
                            results = {}
                            for name in sample_order((NATIVE, FULL, *selections), entry, seed, 0):
                                actions, _, details = generate(replay, conditions, seed, name,
                                                               selection=selections.get(name))
                                if name in selections:
                                    validate_selection(details)
                                results[name] = actions, details
                            row = {"state": entry["state_id"], "task": [entry["suite"], entry["task_id"]],
                                   "seed": seed, "cases": {name: {**details, **action_metrics(results[NATIVE][0], a, 8)}
                                                          for name, (a, details) in results.items()}}
                            write_json(out / "pairs" / f"{len(rows):03}.json", seal(row))
                            rows.append(row)
                            atomic_json(out / "status.json", {"status": "timing", "completed": len(rows),
                                                             "expected": expected})
                    profile_requests(replay, states, out, trace=False)
            status = "complete"
        except TimeoutError:
            status = "budget_exhausted"
        except Exception as exc:
            atomic_json(out / "error.json", {"error": str(exc)})
            status = "failed"
            raise
        finally:
            result = {"status": status, "completed": len(rows), "expected": 0 if args.profile else expected}
            atomic_json(out / "report.json", {**result, **summarize(rows)})
            atomic_json(out / "status.json", result)


def validate_selection(details):
    stats = details["video_refresh"]
    if stats["action_forwards"] != 10 or stats["selected_rows"] != [29, 9, 240]:
        raise ValueError("Unexpected action forwards or selected row count")


def profile_requests(replay, states, out, trace=True):
    rows = []
    for entry, conditions in states[:2]:
        generate(replay, conditions, 43, NATIVE, text="prime")
        for rep in range(3):
            cases = {}
            for name in sample_order((NATIVE, FULL), entry, 43, rep):
                _, _, cases[name] = generate(replay, conditions, 43, name)
            rows.append(cases)
    atomic_json(out / "timing_recheck.json", rows)
    if trace:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            generate(replay, states[1][1], 43, FULL)
        prof.export_chrome_trace(str(out / "trace.json"))
        (out / "operators.txt").write_text(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    parser.add_argument("--profile", action="store_true")
    run(parser.parse_args())
