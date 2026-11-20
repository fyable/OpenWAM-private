"""Manual five-hour hot-request and group-reuse experiment."""

from collections import defaultdict
from contextlib import ExitStack, contextmanager
import copy
from datetime import datetime
import fcntl
from functools import partial
import os
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch

from openwam.research.protocol import digest, read_sealed, seal, token_groups, write_json
from scripts.research_e2_robustness import atomic_json

from .experiment import ROOT, choose_gpu, code_identity
from .next_runtime import FixedReuse, GroupReuse, replace_method
from .profiling import operator_profile
from .runtime import CacheConfig, cached_request

SEEDS = (43, 44)
REPETITIONS = 3
EXECUTION_HORIZON = 8
STAGE_MINUTES = {"warm": 75, "overhead": 150, "group": 285}
MIN_SPEEDUP = {"denoise": 1.10, "request": 1.05}
FIXED = {"fixed_2": 2, "fixed_3": 3}
WARM_CASES = ("native_n10", "native_n4", "refresh_all", *FIXED)
OVERHEAD_CASES = ("native_n10", *FIXED, *(name + "_optimized" for name in FIXED))
PAIR_SEED = 20261119


def task_key(entry):
    return entry["suite"], entry["task_id"]


def select_entries(entries, excluded_ids):
    excluded_episodes = {(task_key(e), e["episode_id"]) for e in entries if e["state_id"] in excluded_ids}
    by_task = defaultdict(dict)
    for entry in sorted(entries, key=lambda e: (task_key(e), e["episode_id"], e["request_index"], e["state_id"])):
        key = task_key(entry)
        if (key, entry["episode_id"]) not in excluded_episodes:
            by_task[key].setdefault(entry["episode_id"], entry["state_id"])
    if len(by_task) != 8 or any(len(episodes) < 2 for episodes in by_task.values()):
        raise ValueError("Need two eligible episodes for each of eight development tasks")
    return [state for episodes in by_task.values() for state in list(episodes.values())[:2]]


def freeze_plan(protocol, output, gpu_uuid, memory_mode, started):
    source = Path(protocol["e2_run"])
    selection = read_sealed(source / "base/manifests/selection.json")
    split = read_sealed(source / "manifests/split.json")
    if selection["sha256"] != protocol["selection_sha256"] or split["sha256"] != protocol["task_split_sha256"]:
        raise ValueError("E2 selection or split changed")
    states, banks = {}, {}
    for subset in ("discovery", "calibration"):
        bank = read_sealed(source / "bank/state_bank" / subset / "manifest.json")
        if bank["sha256"] != selection["bank_sha256_by_split"][subset]:
            raise ValueError("E2 bank differs from the sealed selection")
        excluded = set(selection["state_ids_by_split"][subset]) if subset == "calibration" else set()
        states[subset] = select_entries(bank["states"], excluded)
        banks[subset] = bank["sha256"]
    groups = token_groups(selection["layouts"]["29"])
    possible_pairs = [(i, j) for i in range(8) for j in range(i + 1, 8) if (i, j) != (1, 5)]
    plan = seal({"protocol_sha256": protocol["sha256"], "started_unix": started, "deadline_unix": started + 300 * 60,
                 "stage_minutes": STAGE_MINUTES, "gpu_uuid": gpu_uuid, "memory_mode": memory_mode,
                 "state_ids": states, "bank_sha256": banks, "seeds": list(SEEDS), "repetitions": REPETITIONS,
                 "execution_horizon": EXECUTION_HORIZON,
                 "target": {"layer": 29, "step": 9, "anchor_step": 8, "token_groups": groups},
                 "random_pairs": random.Random(PAIR_SEED).sample(possible_pairs, 5),
                 "e2_low_pair": [1, 5], "minimum_speedup": MIN_SPEEDUP})
    write_json(output / "next_plan.json", plan)
    return plan


class Stage:
    def __init__(self, root, plan, name, deadline):
        self.root, self.plan, self.name, self.deadline = root, plan, name, deadline
        self.path = root / name
        self.path.mkdir(exist_ok=False)
        self.rows = []
        self.status("loading")

    def status(self, status, **details):
        atomic_json(self.path / "status.json", {
            "status": status, "completed_pairs": len(self.rows), "deadline_unix": self.deadline,
            "updated_at": datetime.now().astimezone().isoformat(), **details,
        })

    def save(self, entry, seed, repetition, measurements, *, panel="main"):
        row = {"plan_sha256": self.plan["sha256"], "stage": self.name, "panel": panel,
               "state": entry["state_id"], "task": list(task_key(entry)),
               "seed": seed, "repetition": repetition, "cases": measurements}
        key = digest([panel, entry["state_id"], seed, repetition])
        atomic_json(self.path / "pairs" / f"{key}.json", seal(row))
        self.rows.append(row)
        self.status("running", last_state=row["state"], last_panel=panel)


class Replay:
    def __init__(self, engine):
        self.engine = engine
        self.architecture = engine.architecture
        self.device = self.architecture.device
        self.events = []
        self.encoder_calls = 0

    @contextmanager
    def measure(self):
        def count_encoder(_module, _inputs):
            self.encoder_calls += 1

        def wrap_forward(original):
            def measured(*args, **kwargs):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                result = original(*args, **kwargs)
                end.record()
                self.events.append((begin, end))
                return result
            return measured

        hook = self.architecture.video_backbone.text_encoder.register_forward_pre_hook(count_encoder)
        try:
            with replace_method(self.architecture, "forward", wrap_forward):
                yield
        finally:
            hook.remove()

    def run(self, conditions, seed, *, name="native_n10", text="warm", groups=None, chosen=None,
            horizon=None, profile=False):
        engine = self.engine
        engine.reset()
        engine._vace_cache.clear()
        if text == "cold":
            engine._prompt_embed_cache.clear()
        request = copy.deepcopy(conditions)
        steps = 4 if name == "native_n4" else 10
        request.update(seed=seed, denoise_steps=steps, denoise_mode="sync")
        with torch.inference_mode(), torch.random.fork_rng(devices=[torch.device(self.device).index or 0]):
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            with ExitStack() as stack:
                reuse, region_events = None, defaultdict(list)
                operators = None
                if groups is not None:
                    factory = partial(GroupReuse, groups=groups, chosen=chosen, horizon=horizon)
                    reuse = stack.enter_context(cached_request(engine, CacheConfig(policy="refresh"),
                                                              reuse_factory=factory))
                elif name == "refresh_all":
                    reuse = stack.enter_context(cached_request(engine, CacheConfig(policy="refresh")))
                elif name.removesuffix("_optimized") in FIXED:
                    interval = FIXED[name.removesuffix("_optimized")]
                    kwargs = {"reuse_factory": FixedReuse} if name.endswith("_optimized") else {}
                    reuse = stack.enter_context(cached_request(
                        engine, CacheConfig(interval=interval, max_age=interval), record_schedule=True, **kwargs))
                if profile:
                    operators = stack.enter_context(operator_profile(engine, self.device))
                    if reuse is not None:
                        for method in ("store", "assemble_reused_kv", "slice_reuse_mask", "capture_probe"):
                            stack.enter_context(replace_method(reuse, method, profile_method(region_events[method])))
                self.events.clear()
                self.encoder_calls = 0
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                start = time.perf_counter()
                result = engine.generate(request)
                torch.cuda.synchronize(self.device)
                wall_ms = (time.perf_counter() - start) * 1000
                stats = result_stats(reuse)
                peak = torch.cuda.max_memory_allocated(self.device)
                forward_ms = sum(begin.elapsed_time(end) for begin, end in self.events)

        if not np.isfinite(result["actions"]).all():
            raise FloatingPointError("Nonfinite action output")
        if len(self.events) != steps:
            raise ValueError(f"Expected {steps} joint forwards, got {len(self.events)}")
        if text != "prime" and self.encoder_calls != (1 if text == "cold" else 0):
            raise ValueError(f"{text} request unexpectedly called text encoder {self.encoder_calls} times")
        metrics = {"request_ms": wall_ms, "denoise_ms": forward_ms, "peak_bytes": peak,
                   "text_encoder_calls": self.encoder_calls, "cache": stats}
        if profile:
            metrics["profile"] = {**operators, "cache_region_ms": {
                name: sum(begin.elapsed_time(end) for begin, end in events) for name, events in region_events.items()
            }, "cache_region_calls": {name: len(events) for name, events in region_events.items()}}
        return result["actions"], metrics


def result_stats(reuse):
    if isinstance(reuse, GroupReuse):
        return {"groups": list(reuse.used_groups), "skipped_video_rows": reuse.skipped_rows,
                "anchor_seen": reuse.anchor_seen, "target_seen": reuse.target_seen}
    if reuse is None:
        return None
    result = reuse.report()
    if hasattr(reuse, "refresh_counts"):
        result["refresh_counts"] = reuse.refresh_counts
    return result


def profile_method(pairs):
    def wrap(original):
        def measured(*args, **kwargs):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            result = original(*args, **kwargs)
            end.record()
            pairs.append((begin, end))
            return result
        return measured
    return wrap


def action_metrics(reference, actual, horizon):
    reference, actual = np.asarray(reference), np.asarray(actual)
    if not (reference.shape == actual.shape and reference.ndim == 2 and reference.shape[1] == 10
            and 0 < horizon <= len(reference) and np.isfinite(reference).all() and np.isfinite(actual).all()):
        raise ValueError("Need finite matching EEF10 actions and a valid executed prefix")
    # LIBERO negates the native gripper column for every method, preserving sign disagreement.
    before, after = reference[:horizon], actual[:horizon]
    return {"prefix_rmse": float(np.sqrt(np.mean((after - before)**2))),
            "gripper_direction_disagreement": float(np.mean(np.sign(after[:, 9]) != np.sign(before[:, 9])))}


def check_parity(reference, candidate, label):
    if not np.allclose(reference, candidate, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"Action parity failed: {label}")


def sample_order(names, entry, seed, rep):
    return random.Random(digest([PAIR_SEED, entry["state_id"], seed, rep])).sample(names, len(names))


def run_speed(stage, replay, server, states):
    from benchmarks.libero.research import execution_horizon

    names = WARM_CASES if stage.name == "warm" else OVERHEAD_CASES
    entry, conditions = states[0]
    reference, _ = replay.run(conditions, SEEDS[0], text="prime")
    refreshed, _ = replay.run(conditions, SEEDS[0], name="refresh_all")
    check_parity(reference, refreshed, "refresh_all")
    for name in names:
        replay.run(conditions, SEEDS[0], name=name)
    seen_tasks = set()
    for entry, conditions in states:
        horizon = execution_horizon(server, conditions, None)
        replay.run(conditions, SEEDS[0], text="prime")
        key = task_key(entry)
        if key not in seen_tasks:
            if stage.name == "warm":
                cold_actions, cold = replay.run(conditions, SEEDS[0], text="cold")
                warm_actions, warm = replay.run(conditions, SEEDS[0])
                check_parity(cold_actions, warm_actions, "cold versus warm text embedding")
                stage.save(entry, SEEDS[0], 0, {"cold": cold, "warm": warm}, panel="text")
            else:
                profiles = {name: replay.run(conditions, SEEDS[0], name=name, profile=True)[1] for name in names}
                stage.save(entry, SEEDS[0], 0, profiles, panel="profile")
            seen_tasks.add(key)
        for seed in SEEDS:
            for rep in range(REPETITIONS):
                values = {name: replay.run(conditions, seed, name=name)
                          for name in sample_order(names, entry, seed, rep)}
                reference, _ = values["native_n10"]
                if stage.name == "warm":
                    check_parity(reference, values["refresh_all"][0], "refresh_all")
                else:
                    for name in FIXED:
                        a, old = values[name]
                        b, new = values[name + "_optimized"]
                        check_parity(a, b, name + " optimized")
                        if any(old["cache"][field] != new["cache"][field]
                               for field in ("reused_layers", "video_query_rows", "full_layers", "action_layers",
                                             "refresh_counts")):
                            raise RuntimeError("Fixed optimization changed the schedule's work counts")
                measurements = {name: {**details, **action_metrics(reference, actions, horizon)}
                                for name, (actions, details) in values.items()}
                stage.save(entry, seed, rep, measurements)


def run_group(stage, replay, server, discovery, calibration):
    from benchmarks.libero.research import execution_horizon

    groups = stage.plan["target"]["token_groups"]
    probe_cases = {f"group_{i}": (i,) for i in range(8)}

    def panel(states, choices, label):
        for entry, conditions in states:
            horizon = execution_horizon(server, conditions, None)
            replay.run(conditions, SEEDS[0], text="prime")
            for seed in SEEDS:
                reference, baseline = replay.run(conditions, seed)
                if entry == states[0][0] and seed == SEEDS[0]:
                    empty, details = replay.run(conditions, seed, groups=groups, chosen=(), horizon=horizon)
                    check_parity(reference, empty, "empty group intervention")
                    validate_group_rows(details, (), groups)
                measurements = {"native_n10": {**baseline, **action_metrics(reference, reference, horizon)}}
                for name in sample_order(tuple(choices), entry, seed, 0):
                    chosen = choices[name]
                    actual, details = replay.run(conditions, seed, groups=groups, chosen=chosen, horizon=horizon)
                    validate_group_rows(details, chosen, groups)
                    measurements[name] = {**details, **action_metrics(reference, actual, horizon)}
                stage.save(entry, seed, 0, measurements, panel=label)

    panel(discovery, probe_cases, "discovery")
    scores = {name: task_mean(stage.rows, lambda row: row["cases"][name]["prefix_rmse"]) for name in probe_cases}
    ranked = sorted(range(8), key=lambda i: (scores[f"group_{i}"], i))
    write_json(stage.path / "ranking.json", seal({"plan_sha256": stage.plan["sha256"], "split": "discovery",
                                                 "scores": scores, "ascending_groups": ranked}))
    choices = {"real_reuse": ranked[:2], "e2_low": stage.plan["e2_low_pair"], "attention": "attention"}
    choices.update({f"random_{i + 1}": pair for i, pair in enumerate(stage.plan["random_pairs"])})
    panel(calibration, choices, "calibration")


def validate_group_rows(details, chosen, groups):
    actual = details["cache"]
    expected_count = 2 if chosen == "attention" else len(chosen)
    ids = actual["groups"]
    if (not actual["anchor_seen"] or not actual["target_seen"] or len(set(ids)) != expected_count
            or actual["skipped_video_rows"] != len(groups[0]) * expected_count
            or (chosen != "attention" and set(ids) != set(chosen))):
        raise RuntimeError("Group experiment did not perform the requested reuse")


def task_mean(rows, metric):
    by_task = defaultdict(list)
    for row in rows:
        by_task[tuple(row["task"])].append(metric(row))
    return float(np.mean([np.mean(values) for values in by_task.values()]))


def performance(rows, reference):
    if not rows:
        return {}
    names = set.intersection(*(set(row["cases"]) for row in rows))
    result = {}
    for name in sorted(names):
        series = [row["cases"][name] for row in rows]
        result[name] = {"pairs": len(rows), "peak_bytes": max(m["peak_bytes"] for m in series)}
        for metric in ("request_ms", "denoise_ms"):
            values = [m[metric] for m in series]
            result[name][metric] = {"p50": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                                    "paired_speedup_p50": float(np.median([
                                        row["cases"][reference][metric] / row["cases"][name][metric] for row in rows
                                    ]))}
        if "prefix_rmse" in series[0]:
            result[name]["prefix_rmse_p50"] = float(np.median([m["prefix_rmse"] for m in series]))
            result[name]["prefix_rmse_p95"] = float(np.percentile([m["prefix_rmse"] for m in series], 95))
            result[name]["gripper_direction_disagreement"] = task_mean(
                rows, lambda row: row["cases"][name]["gripper_direction_disagreement"])
    return result


def stage_report(root, name, plan, status):
    path = root / name
    rows = [read_sealed(p) for p in sorted((path / "pairs").glob("*.json"))]
    if any(row["plan_sha256"] != plan["sha256"] or row["stage"] != name for row in rows):
        raise ValueError("Result provenance mismatch")
    panels = defaultdict(list)
    for row in rows:
        panels[row["panel"]].append(row)
    report = {"status": status, "stage": name, "plan_sha256": plan["sha256"],
              "pairs_by_panel": {panel: len(values) for panel, values in panels.items()}}
    if name in {"warm", "overhead"}:
        report["methods"] = performance(panels["main"], "native_n10")
    if name == "warm":
        report["text_cache"] = performance(panels["text"], "cold")
    if name == "overhead":
        report["profiling"] = profiles(panels["profile"])
        candidates = [method for method in report["methods"] if method != "native_n10"
                      and passes_speed_gate(report["methods"][method], plan["minimum_speedup"])]
        complete = len(panels["main"]) == len(plan["state_ids"]["discovery"]) * len(SEEDS) * REPETITIONS
        report["group_gate"] = bool(status == "complete" and complete and candidates)
        report["qualifying_candidates"] = candidates
    if name == "group":
        report["calibration"] = performance(panels["calibration"], "native_n10")
        if panels["calibration"]:
            report["comparison"] = group_comparison(panels["calibration"], status)
    atomic_json(path / "report.json", seal(report))
    return report


def passes_speed_gate(method, minimum):
    return (method["denoise_ms"]["paired_speedup_p50"] >= minimum["denoise"]
            and method["request_ms"]["paired_speedup_p50"] >= minimum["request"])


def profiles(rows):
    by_method = defaultdict(list)
    for row in rows:
        for name, measurement in row["cases"].items():
            by_method[name].append(measurement["profile"]["cache_region_ms"])
    return {name: {region: float(np.median([values.get(region, 0.) for values in entries]))
                   for region in sorted(set().union(*entries))} for name, entries in by_method.items()}


def group_comparison(rows, status):
    result = {}
    for control in ("attention", "e2_low", "random_mean"):
        control_names = [control] if control != "random_mean" else [f"random_{i}" for i in range(1, 6)]
        deltas = {}
        for field in ("prefix_rmse", "gripper_direction_disagreement"):
            deltas[field] = task_mean(rows, lambda row: row["cases"]["real_reuse"][field]
                                     - np.mean([row["cases"][name][field] for name in control_names]))
        result[control] = deltas
    attention = result["attention"]
    result["extend"] = bool(status == "complete" and attention["prefix_rmse"] < 0
                            and attention["gripper_direction_disagreement"] <= 0)
    return result


def load_selected(source, split, plan, subset):
    from benchmarks.libero.research import load_states

    bank, available = load_states(source / "bank/state_bank" / subset, split)
    if bank["sha256"] != plan["bank_sha256"][subset]:
        raise ValueError("Development bank changed after plan preparation")
    lookup = {entry["state_id"]: (entry, conditions) for entry, conditions in available}
    return [lookup[state] for state in plan["state_ids"][subset]]


@contextmanager
def alarm(deadline):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("Stage deadline already passed")

    def expired(_signum, _frame):
        raise TimeoutError("Stage deadline reached")

    handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


def run_stage(args):
    from benchmarks.libero import research

    root, name = args.output, args.stage
    protocol = read_sealed(root / "protocol/protocol.json")
    if protocol["source_files"] != code_identity():
        raise ValueError("Experiment code changed after protocol preparation")
    if research.source_identity()["source_sha256"] != protocol["research_source_sha256"]:
        raise ValueError("Model source changed since the E2 experiment")
    source = Path(protocol["e2_run"])
    plan_path = root / "next_plan.json"
    plan = read_sealed(plan_path) if plan_path.exists() else None
    if name != "warm" and plan is None:
        raise ValueError("Run the warm stage first")
    if plan is not None and plan["protocol_sha256"] != protocol["sha256"]:
        raise ValueError("Protocol changed after trial plan preparation")
    if name == "group":
        overhead = read_sealed(root / "overhead/report.json")
        if overhead["plan_sha256"] != plan["sha256"]:
            raise ValueError("Overhead report belongs to a different plan")
        if not overhead["group_gate"]:
            stage = Stage(root, plan, name, args.deadline.timestamp())
            stage.status("skipped", reason="Overhead stage did not meet both speed targets")
            stage_report(root, name, plan, "skipped")
            return
    gpu_uuid, total_mib = choose_gpu(args.gpu)
    if args.memory_mode == "resident" and total_mib < 28000:
        raise ValueError("Use text-offload on a 24 GB GPU")
    if plan is not None and (plan["gpu_uuid"], plan["memory_mode"]) != (gpu_uuid, args.memory_mode):
        raise ValueError("Use the same GPU and memory mode across stages")
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid, MUJOCO_EGL_DEVICE_ID=str(args.gpu),
                      E2_MEMORY_MODE=args.memory_mode)
    lock_path = ROOT / f".cache/e2-reproduction-{gpu_uuid}.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if plan is None:
            plan = freeze_plan(protocol, root, gpu_uuid, args.memory_mode, time.time())
        deadline = min(args.deadline.timestamp(), plan["started_unix"] + STAGE_MINUTES[name] * 60)
        stage = Stage(root, plan, name, deadline)
        status = "failed"
        try:
            with alarm(deadline):
                server_args = SimpleNamespace(ckpt_dir=protocol["checkpoint_dir"],
                                              ckpt_name=protocol["checkpoint_name"],
                                              device="cuda:0", steps=10, horizon=plan["execution_horizon"])
                server = research.build_server(server_args)
                model = research.model_identity(server, server_args)
                split = read_sealed(source / "manifests/split.json")
                if split["sha256"] != protocol["task_split_sha256"]:
                    raise ValueError("Task split changed")
                subsets = ("discovery", "calibration") if name == "group" else ("discovery",)
                for subset in subsets:
                    saved = read_sealed(source / "bank/state_bank" / subset / "model.json")
                    if model["assets"] != saved["assets"]:
                        raise ValueError("Model assets differ from the E2 state banks")
                if name != "warm":
                    previous = read_sealed(root / "warm/identity.json")
                    if previous["plan_sha256"] != plan["sha256"] or previous["model"]["sha256"] != model["sha256"]:
                        raise ValueError("Benchmark model identity changed after warm stage")
                write_json(stage.path / "identity.json", seal({"plan_sha256": plan["sha256"], "model": model}))
                discovery = load_selected(source, split, plan, "discovery")
                replay = Replay(server.engine)
                with replay.measure():
                    if name == "group":
                        calibration = load_selected(source, split, plan, "calibration")
                        run_group(stage, replay, server, discovery, calibration)
                    else:
                        run_speed(stage, replay, server, discovery)
            status = "complete"
            stage.status(status)
        except TimeoutError as error:
            status = "budget_exhausted"
            stage.status(status, reason=str(error))
        except BaseException as error:
            stage.status(status, reason=str(error))
            raise
        finally:
            stage_report(root, name, plan, status)


def summarize_next(root):
    root = Path(root)
    plan = read_sealed(root / "next_plan.json")
    reports = {}
    for name in STAGE_MINUTES:
        path = root / name / "report.json"
        if path.exists():
            report = read_sealed(path)
            if report["plan_sha256"] != plan["sha256"]:
                raise ValueError(f"{name} report belongs to a different plan")
            reports[name] = report
        else:
            reports[name] = {"status": "not_run"}
    if reports["overhead"].get("group_gate") is False and reports["group"]["status"] == "not_run":
        reports["group"] = {"status": "skipped", "reason": "Overhead speed gate did not pass"}
    result = {"plan_sha256": plan["sha256"], "gpu_uuid": plan["gpu_uuid"], "memory_mode": plan["memory_mode"],
              "reports": reports, "updated_at": datetime.now().astimezone().isoformat()}
    atomic_json(root / "next_report.json", seal(result))
    lines = ["# OpenWAM hot-request and group reuse experiments", "",
             f"GPU `{plan['gpu_uuid']}`, memory mode `{plan['memory_mode']}`.", "",
             "Request timing covers synchronized engine.generate. Denoising timing sums joint forwards.", "",
             "| Stage | Status | Complete paired records |",
             "|---|---|---:|"]
    for name, report in reports.items():
        lines.append(f"| {name} | {report['status']} | {sum(report.get('pairs_by_panel', {}).values())} |")
    for name in ("warm", "overhead"):
        methods = reports[name].get("methods", {})
        if methods:
            lines += ["", f"## {name.title()} timing", "",
                      "| Method | Request P50 / P95 (ms) | Joint forward P50 / P95 (ms) | Request speedup | Forward speedup | Peak allocated (GiB) | Prefix RMSE P95 |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for method, stats in methods.items():
                request, forward = stats["request_ms"], stats["denoise_ms"]
                lines.append(f"| {method} | {request['p50']:.2f} / {request['p95']:.2f} | "
                             f"{forward['p50']:.2f} / {forward['p95']:.2f} | "
                             f"{request['paired_speedup_p50']:.3f}× | {forward['paired_speedup_p50']:.3f}× | "
                             f"{stats['peak_bytes'] / 2**30:.2f} | {stats['prefix_rmse_p95']:.6g} |")
    text_cache = reports["warm"].get("text_cache", {})
    if text_cache:
        lines += ["", f"Text cache paired median speedup {text_cache['warm']['request_ms']['paired_speedup_p50']:.3f}×."]
    if "group_gate" in reports["overhead"]:
        passed = reports["overhead"]["group_gate"]
        qualifiers = ", ".join(reports["overhead"]["qualifying_candidates"])
        lines += ["", f"Group speed gate {'passed for ' + qualifiers if passed else 'did not pass'}."]
    comparison = reports["group"].get("comparison", {})
    if comparison:
        lines += ["", "## Group comparison on Calibration", "",
                  "Differences are real-reuse ordering minus the control, with tasks weighted equally.", "",
                  "| Control | Prefix RMSE difference | Gripper direction disagreement difference |",
                  "|---|---:|---:|"]
        for control in ("attention", "e2_low", "random_mean"):
            values = comparison[control]
            lines.append(f"| {control} | {values['prefix_rmse']:.6g} | "
                         f"{values['gripper_direction_disagreement']:.6g} |")
    (root / "next_report.md").write_text("\n".join(lines) + "\n")
    return result
