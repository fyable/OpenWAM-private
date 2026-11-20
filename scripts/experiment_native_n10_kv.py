"""Compare native N10 with per-request text/proprio projection reuse."""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime
import fcntl
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from benchmarks.action_cache.experiment import ROOT, choose_gpu
from benchmarks.action_cache.next_experiment import Replay, action_metrics, alarm, performance, sample_order
from benchmarks.action_cache.next_runtime import replace_method
from openwam.research.protocol import file_hash, read_sealed, seal, write_json
from scripts.native_n10_prefix import clean_prefix_kv
from scripts.native_n10_video_refresh import video_refresh
from scripts.research_e2_robustness import atomic_json

CASES = ("native_n10", "native_n10_context_kv")
PREFIX = "native_n10_context_kv_prefix"
PREFIX_PAD = "native_n10_context_kv_prefix_padded"
HORIZON = 8
STEPS = 10
RTOL, ATOL = 1e-4, 1e-5
KEY_STRIDES = {"video_refresh_0_stride2": 2, "video_refresh_0_stride4": 4,
               "video_refresh_0_9_stride2": 2}
REFRESH_SCHEDULES = {
    "video_refresh_0_stride2": (0,),
    "video_refresh_0_stride4": (0,),
    "video_refresh_0_9_stride2": (0, 9),
    "video_refresh_0_4_9": (0, 4, 9),
    "video_refresh_0_4": (0, 4),
    "video_refresh_0_7": (0, 7),
    "video_refresh_0_5": (0, 5),
    "video_refresh_0_9": (0, 9),
    "video_refresh_0": (0,),
    "video_refresh_0_reuse_buffer": (0,),
    "video_refresh_0_9_reuse_buffer": (0, 9),
    "video_refresh_all": tuple(range(STEPS)),
}


def memoized_projection(original, counts, family, snapshots=None, module=None):
    saved = None

    def forward(inputs):
        nonlocal saved
        if saved is None:
            saved = inputs, original(inputs)
            counts[family]["computed"] += 1
            if snapshots is not None:
                snapshots[module] = saved[1]
        elif inputs is not saved[0]:
            raise ValueError(f"{family} projection input changed during native denoising")
        else:
            counts[family]["reused"] += 1
        return saved[1]

    return forward


def shared_context(original):
    source = None
    cached = None
    static_keys = ("context", "context_mask", "seq_lens", "_proprio_sample_mask")

    def append(inputs, proprio):
        nonlocal source, cached
        current = (*[inputs.get(key) for key in static_keys], proprio)
        if source is None:
            source = current
            result = original(inputs, proprio)
            cached = {key: result[key] for key in ("context", "context_mask") if key in result}
            return result
        if any(before is not after for before, after in zip(source, current)):
            raise ValueError("Text or proprio conditioning changed within the request")
        result = dict(inputs, **cached)
        result.pop("_proprio_sample_mask", None)
        return result

    return append


@contextmanager
def context_kv(engine, *, diagnostic_replay=False, video_forwards=STEPS, graph_projections=None):
    architecture = engine.architecture
    if (architecture.training or not torch.is_inference_mode_enabled()
            or architecture._compiled_mot_run_joint_loop is not None
            or engine._joint_sampler is not None or engine._dit_cache is not None or engine._cfg_scale != 1):
        raise ValueError("Context K/V experiment requires native eager inference with CFG=1")
    counts = {}
    with ExitStack() as stack:
        stack.enter_context(replace_method(architecture, "_append_proprio_context_token", shared_context))
        for family, backend in (("video", architecture.video_backbone.dit),
                                ("action", architecture.action_backbone)):
            counts[family] = {"computed": 0, "reused": 0}
            modules = [backend.text_embedding]
            for block in backend.blocks:
                attention = block.cross_attn
                if getattr(attention, "has_image_input", False):
                    raise ValueError("This experiment targets text/proprio cross-attention")
                modules.extend((attention.k, attention.norm_k, attention.v))
            for module in modules:
                stack.enter_context(replace_method(
                    module, "forward", lambda original, name=family, owner=module: memoized_projection(
                        original, counts, name, graph_projections if name == "action" else None, owner)))
        yield counts
    for family, backend in (("video", architecture.video_backbone.dit), ("action", architecture.action_backbone)):
        projections = 1 + 3 * len(backend.blocks)
        forwards = video_forwards if family == "video" else (1 if graph_projections is not None else STEPS)
        diagnostic_hits = 3 * len(backend.blocks) if diagnostic_replay and family == "video" else 0
        if counts[family] != {"computed": projections, "reused": (forwards - 1) * projections + diagnostic_hits}:
            raise ValueError(f"Unexpected projection counts for {family}: {counts[family]}")


def generate(replay, conditions, seed, method, *, capture=False, diagnostic=None, text="warm", selection=None, action_graph=None):
    engine = replay.engine
    trajectory = []
    projection_counts = {}
    prefix_counts = {}
    refresh_counts = {}
    graph_projections = {} if action_graph is not None else None
    schedule = (0,) if selection is not None else REFRESH_SCHEDULES.get(method)

    def capture_forward(original):
        def forward(*args, **kwargs):
            predictions = original(*args, **kwargs)
            trajectory.append(tuple(value.detach().cpu() for value in predictions))
            return predictions
        return forward

    def cached_generate(original):
        def invoke(request):
            with ExitStack() as stack:
                counts = stack.enter_context(context_kv(
                    engine, diagnostic_replay=diagnostic is not None,
                    video_forwards=STEPS if schedule is None else len(schedule),
                    graph_projections=graph_projections))
                if method in (PREFIX, PREFIX_PAD):
                    reused_rows = stack.enter_context(clean_prefix_kv(
                        engine.architecture, steps=STEPS, verify_prefix=capture,
                        keep_attention_shape=method == PREFIX_PAD, diagnostic=diagnostic))
                result = original(request)
            projection_counts.update(counts)
            if method in (PREFIX, PREFIX_PAD):
                prefix_counts.update(reused_rows)
            return result
        return invoke

    with ExitStack() as stack:
        if schedule is not None:
            refresh_counts = stack.enter_context(video_refresh(
                engine.architecture, STEPS, schedule, inplace_kv=method.endswith("_reuse_buffer"),
                video_key_stride=KEY_STRIDES.get(method, 1), selection=selection,
                action_graph=action_graph, projections=graph_projections))
        stack.enter_context(replay.measure())
        if capture:
            stack.enter_context(replace_method(engine.architecture, "forward", capture_forward))
        if method != CASES[0]:
            stack.enter_context(replace_method(engine, "generate", cached_generate))
        actions, timing = replay.run(conditions, seed, text=text)
    step_ms = [begin.elapsed_time(end) for begin, end in replay.events]
    return actions, trajectory, {**timing, "step_ms": step_ms, "context_projections": projection_counts,
                                 "prefix_rows": prefix_counts, "video_refresh": refresh_counts}


def compare_trajectory(reference, candidate, *, exact=True):
    if len(reference) != STEPS or len(candidate) != STEPS:
        raise ValueError("Expected ten video/action predictions from both requests")
    errors = []
    for step, (before, after) in enumerate(zip(reference, candidate)):
        if len(before) != 2 or len(after) != 2:
            raise ValueError(f"Expected video and action predictions at step {step}")
        pair = []
        for modality, a, b in zip(("video", "action"), before, after):
            if exact:
                torch.testing.assert_close(b, a, rtol=RTOL, atol=ATOL,
                                           msg=lambda why: f"{modality} prediction at step {step}: {why}")
            elif a.shape != b.shape or not bool(torch.isfinite(b).all()):
                raise ValueError(f"Invalid {modality} prediction at step {step}")
            pair.append(float((b.float() - a.float()).abs().max()))
        errors.append(dict(step=step, video_max_abs=pair[0], action_max_abs=pair[1]))
    return errors


def inputs(reference_run):
    plan = read_sealed(reference_run / "next_plan.json")
    protocol = read_sealed(reference_run / "protocol/protocol.json")
    source = Path(protocol["e2_run"])
    split = read_sealed(source / "manifests/split.json")
    if (plan["protocol_sha256"] != protocol["sha256"] or split["sha256"] != protocol["task_split_sha256"]
            or len(plan["state_ids"]["discovery"]) != 16):
        raise ValueError("Expected the sealed four-GPU Discovery inputs")
    return plan, protocol, source, split


def evaluate(args):
    from benchmarks.libero import research
    from benchmarks.action_cache.next_experiment import load_selected

    plan, protocol, source, split = inputs(args.reference_run)
    states = load_selected(source, split, plan, "discovery")
    if args.smoke:
        states = [states[0], states[2]]
    seeds = (43,) if args.smoke else (43, 44)
    repeats = 1 if args.smoke else 3
    methods = (*CASES, PREFIX_PAD if args.keep_attention_shape else PREFIX) if args.prefix else CASES
    if args.video_refresh:
        methods = (*CASES, *REFRESH_SCHEDULES)
    if args.refresh_cases:
        methods = (*CASES, *args.refresh_cases)
    if args.preflight:
        print(f"{len(states)} observations × {len(seeds)} seeds × {repeats} paired timing repeats; {methods}")
        return
    if research.source_identity()["source_sha256"] != protocol["research_source_sha256"]:
        raise ValueError("Model source differs from the reference experiment")
    gpu, memory = choose_gpu(args.gpu)
    if args.memory_mode == "resident" and memory < 28000:
        raise ValueError("Use text-offload for a 24 GB GPU")
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, E2_MEMORY_MODE=args.memory_mode)
    lock_path = ROOT / ".cache" / f"e2-reproduction-{gpu}.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        expected = len(states) * len(seeds) * repeats
        rows, validation = [], []
        status = "failed"
        start = time.time()

        def progress(current, **details):
            atomic_json(args.output / "status.json", {"status": current, "completed": len(rows),
                        "expected": expected, "validated_method_checks": len(validation),
                        "validated_observation_seeds": len({(v["state"], v["seed"]) for v in validation}), **details})

        try:
            progress("loading")
            with alarm(args.deadline.timestamp()):
                options = SimpleNamespace(ckpt_dir=protocol["checkpoint_dir"], ckpt_name=protocol["checkpoint_name"],
                                          device="cuda:0", steps=STEPS, horizon=HORIZON)
                server = research.build_server(options)
                model = research.model_identity(server, options)
                reference_model = read_sealed(source / "bank/state_bank/discovery/model.json")
                if model["assets"] != reference_model["assets"] or model["attention_mask_mode"] != "mutual":
                    raise ValueError("Expected the same mutual Joint weights as the four-GPU replay")
                write_json(args.output / "identity.json", seal({"model": model, "gpu": gpu,
                           "launcher_sha256": file_hash(Path(__file__).resolve()), "reference_plan": plan["sha256"],
                           "states": [entry["state_id"] for entry, _ in states],
                           "seeds": list(seeds), "repeats": repeats, "smoke": args.smoke, "methods": list(methods),
                           "prefix_implementation_sha256": file_hash(Path(__file__).with_name("native_n10_prefix.py")),
                           "video_refresh_implementation_sha256": file_hash(
                               Path(__file__).with_name("native_n10_video_refresh.py"))}))
                replay = Replay(server.engine)
                progress("validating")
                for entry, conditions in states:
                    generate(replay, conditions, seeds[0], CASES[0], text="prime")
                    for seed in seeds:
                        a, a_trace, _ = generate(replay, conditions, seed, methods[0], capture=True)
                        for method in methods[1:]:
                            probe = [] if args.diagnose_prefix and method in (PREFIX, PREFIX_PAD) else None
                            b, b_trace, _ = generate(replay, conditions, seed, method, capture=True, diagnostic=probe)
                            if probe is not None:
                                atomic_json(args.output / "prefix_diagnostics.json", probe)
                            exact = method not in REFRESH_SCHEDULES or method == "video_refresh_all"
                            errors = compare_trajectory(a_trace, b_trace, exact=exact)
                            if exact:
                                np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
                            validation.append({"state": entry["state_id"], "seed": seed, "steps": errors,
                                               "method": method, "requires_native_parity": exact,
                                               **action_metrics(a, b, HORIZON)})
                            atomic_json(args.output / "validation.json", validation)
                        progress("validating")
                progress("timing")
                for entry, conditions in states:
                    generate(replay, conditions, seeds[0], CASES[0], text="prime")
                    for seed in seeds:
                        for rep in range(repeats):
                            cases = {name: generate(replay, conditions, seed, name)
                                     for name in sample_order(methods, entry, seed, rep)}
                            reference = cases[CASES[0]][0]
                            for candidate in methods[1:]:
                                if candidate not in REFRESH_SCHEDULES or candidate == "video_refresh_all":
                                    np.testing.assert_allclose(cases[candidate][0], reference, rtol=RTOL, atol=ATOL)
                            row = {"state": entry["state_id"], "task": [entry["suite"], entry["task_id"]],
                                   "seed": seed, "repetition": rep, "cases": {
                                       name: {**details, **action_metrics(reference, actions, HORIZON)}
                                       for name, (actions, _, details) in cases.items()}}
                            write_json(args.output / "pairs" / f"{len(rows):04}.json", seal(row))
                            rows.append(row)
                            progress("timing")
            status = "complete"
        except TimeoutError as exc:
            progress("budget_exhausted", error=str(exc))
            status = "budget_exhausted"
        except BaseException as exc:
            progress("failed", error=str(exc))
            raise
        finally:
            timing = performance(rows, CASES[0])
            two_x = [method for method in REFRESH_SCHEDULES if method in timing
                     and timing[method]["request_ms"]["paired_speedup_p50"] >= 2.0]
            report = {"status": status, "expected": expected, "completed": len(rows), "smoke": args.smoke,
                      "validated_observation_seeds": len({(v["state"], v["seed"]) for v in validation}),
                      "validated_method_checks": len(validation), "elapsed_seconds": time.time() - start,
                      "methods": timing}
            if args.video_refresh:
                report.update(video_refresh_schedules=REFRESH_SCHEDULES,
                              candidates_at_2x=two_x if status == "complete" else [],
                              quality_evidence="paired observation replay; closed loop unmeasured")
            atomic_json(args.output / "report.json", report)
            if status == "complete":
                progress(status)
            print(args.output / "report.json")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory-mode", choices=("text-offload", "resident"), default="text-offload")
    parser.add_argument("--deadline", type=datetime.fromisoformat)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prefix", action="store_true", help="Add request-local clean first-frame reuse")
    parser.add_argument("--video-refresh", action="store_true", help="Compare selective video refresh at native N10")
    parser.add_argument("--keep-attention-shape", action="store_true", help="Keep native mixed attention query shape")
    parser.add_argument("--diagnose-prefix", action="store_true", help="Record operator errors in the second forward")
    parser.add_argument("--preflight", action="store_true", help="Check and list input coverage without loading the model")
    parser.add_argument("--refresh-cases", choices=REFRESH_SCHEDULES, nargs="+")
    args = parser.parse_args()
    if args.refresh_cases and (not args.video_refresh or len(set(args.refresh_cases)) != len(args.refresh_cases)):
        parser.error("Select unique --refresh-cases with --video-refresh")
    if args.prefix and args.video_refresh:
        parser.error("Choose one of --prefix and --video-refresh")
    if (args.keep_attention_shape or args.diagnose_prefix) and not args.prefix:
        parser.error("Prefix controls require --prefix")
    if not args.preflight and (args.output is None or args.deadline is None or args.deadline.tzinfo is None
                               or args.deadline.timestamp() <= time.time()):
        parser.error("GPU runs need a new --output and a future --deadline with timezone")
    evaluate(args)


if __name__ == "__main__":
    main()
