"""Offline same-noise trajectory diagnostics; never used for deployment timing."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.libero.joint_experiment import build_engine, tensor_error, write_json


def assemble(source, output):
    from openwam.deploy.request_trace import _write_records

    sessions = sorted(Path(source).glob("session_*"))
    if len(sessions) != 1:
        raise ValueError("Expected exactly one capture session; do not silently concatenate retried sessions")
    paths = sorted(sessions[0].glob("request_*.npz"))
    if not paths:
        raise ValueError("No successful requests captured")
    _write_records(paths, output)


def episode_indices(starts, episode):
    boundaries = np.flatnonzero(starts).tolist() + [len(starts)]
    if not 0 <= episode < len(boundaries) - 1:
        raise ValueError(f"Episode {episode} unavailable")
    return range(boundaries[episode], boundaries[episode + 1])


def run(args):
    import torch
    from PIL import Image
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    from benchmarks.libero.replay_joint import load_observations
    from openwam.deploy import parallel_joint

    args.output.mkdir(parents=True, exist_ok=False)
    observations = load_observations(args.trace)
    indices = episode_indices(observations[3], args.episode)
    engine, checkpoint = build_engine(args)
    engine.cfg.inference.denoise_steps = 10
    sampler = parallel_joint.JointParaFlowSampler(cache_schedule=True, share_conditions=False,
                                                 action_tail_init=args.action_tail_init,
                                                 video_tail_init=args.video_tail_init)
    original_velocity = parallel_joint.joint_velocity
    reference = {}

    def metrics(actual, expected):
        return tensor_error(actual, expected)

    def checked(architecture, inputs, videos, actions, timesteps):
        prediction = original_velocity(architecture, inputs, videos, actions, timesteps)
        if len(actions) != 10:
            return prediction
        schedule = tuple(sampler.signature[2])
        grid = torch.tensor(schedule, device=actions.device, dtype=torch.float32)
        video_sigmas = grid[:, 0] / architecture.video_scheduler.num_train_timesteps
        action_sigmas = grid[:, 1] / architecture.action_scheduler.num_train_timesteps
        inactive_signature = sampler.signature[7]
        inactive = None if inactive_signature is None else torch.tensor(inactive_signature, device=actions.device)
        video_nodes, action_nodes = [videos[:1].float().clone()], [actions[:1].float().clone()]
        for stage, pair in enumerate(schedule[:-1]):
            video_velocity, action_velocity = original_velocity(
                architecture, inputs, video_nodes[-1], action_nodes[-1], [pair])
            next_video = video_nodes[-1] + video_velocity.float() * (video_sigmas[stage + 1] - video_sigmas[stage])
            next_action = action_nodes[-1] + action_velocity.float() * (action_sigmas[stage + 1] - action_sigmas[stage])
            next_video[:, :, :1] = inputs["first_frame_latents"]
            if inactive is not None:
                next_action[..., inactive] = action_sigmas[stage + 1] / action_sigmas[0] * actions[:1, ..., inactive]
            video_nodes.append(next_video)
            action_nodes.append(next_action)
        expected_video = torch.cat(video_nodes)
        expected_action = torch.cat(action_nodes)
        stages = []
        for stage in range(10):
            stages.append({"stage": stage,
                           "video": metrics(videos[stage], expected_video[stage]),
                           "video_overlap": metrics(videos[stage, :, 1:2], expected_video[stage, :, 1:2]),
                           "video_new_tail": metrics(videos[stage, :, 2:], expected_video[stage, :, 2:]),
                           "action": metrics(actions[stage], expected_action[stage]),
                           "action_overlap": metrics(actions[stage, :24], expected_action[stage, :24]),
                           "action_new_tail": metrics(actions[stage, 24:], expected_action[stage, 24:])})
        independent = [original_velocity(architecture, inputs, videos[index:index + 1],
                                        actions[index:index + 1], [timesteps[index]]) for index in range(10)]
        partition = {label: metrics(prediction[modality], torch.cat([result[modality] for result in independent]))
                     for modality, label in enumerate(("video", "action"))}
        reference.update(video=expected_video[-1:], action=expected_action[-1:], stages=stages, partition=partition)
        return prediction

    rows = []
    try:
        parallel_joint.joint_velocity = checked
        with (args.output / "requests.jsonl").open("x") as stream:
            for index in indices:
                condition = {"first_frame_image": [Image.fromarray(observations[0][index])],
                             "prompt": str(observations[1][index]), "proprio": observations[2][index], "seed": 42}
                engine._joint_sampler = None
                native = engine.generate(condition)
                reference.clear()
                engine._joint_sampler = sampler
                candidate = engine.generate(condition)
                row = {"index": index, "cold": sampler.last_stats["cold"],
                       "native_output": metrics(torch.as_tensor(candidate["actions"]), torch.as_tensor(native["actions"]))}
                if reference:
                    normalized = sampler.history[-1, 0].float().cpu().numpy()
                    expected = reference["action"][0].float().cpu().numpy()
                    normalizer = getattr(engine.architecture, "normalizer", None)
                    actual_raw = normalizer.unnormalize(normalized) if normalizer is not None else normalized
                    expected_raw = normalizer.unnormalize(expected) if normalizer is not None else expected
                    actual_env = np.stack([native_eef10_to_libero7d(value) for value in actual_raw[:8]])
                    expected_env = np.stack([native_eef10_to_libero7d(value) for value in expected_raw[:8]])
                    row.update(stages=reference["stages"], batch_partition=reference["partition"],
                               same_noise_normalized_endpoint=metrics(sampler.history[-1], reference["action"]),
                               same_noise_normalized_first8=metrics(torch.as_tensor(normalized[:8]), torch.as_tensor(expected[:8])),
                               same_noise_environment_first8=metrics(torch.as_tensor(actual_env), torch.as_tensor(expected_env)),
                               candidate_environment_first8=actual_env.tolist(),
                               reference_environment_first8=expected_env.tolist(),
                               same_noise_environment_action_delta=(actual_env - expected_env).tolist())
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    finally:
        parallel_joint.joint_velocity = original_velocity
    report = {"status": "completed", "timing_valid": False, "closed_loop": "not_run",
              "checkpoint": str(Path(args.ckpt_dir) / checkpoint), "trace": str(args.trace.resolve()),
              "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
              "core_identity": parallel_joint.core_identity(), "episode": args.episode,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": args.device, "torch": torch.__version__, "cuda": torch.version.cuda,
              "action_tail_init": args.action_tail_init, "video_tail_init": args.video_tail_init,
              "requests": len(rows), "warm_requests": sum(not row["cold"] for row in rows),
              "reference_boundary": "same initial noise and conditions, no state commit; Native outputs use their own native noise policy",
              "tolerances": {"rtol": .05, "atol": .05}, "equivalence": "not_claimed"}
    write_json(args.output / "diagnosis.json", report)
    lines = ["# Same-noise Joint trajectory diagnosis", "", "Offline only; extra reference forwards are excluded from deployment and all speed claims.",
             "Errors do not establish the cause of closed-loop failure. Full per-stage overlap/tail and batch-partition errors: requests.jsonl.",
             "", "| Request | Cold | Normalized first8 RMSE | Environment first8 RMSE |", "|---|---|---:|---:|"]
    for row in rows:
        norm = row.get("same_noise_normalized_first8", {}).get("rmse")
        env = row.get("same_noise_environment_first8", {}).get("rmse")
        lines.append(f"| {row['index']} | {row['cold']} | {norm} | {env} |")
    (args.output / "diagnosis.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("assemble", "replay"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ckpt-dir", default="/data1/wam-ckpts/OpenWAM-Alpha-Sim-LIBERO")
    parser.add_argument("--ckpt-name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--history-scope", default="phase_video_action", choices=("phase_video_action",))
    for modality in ("action", "video"):
        parser.add_argument(f"--{modality}-tail-init", choices=("stage_residual", "endpoint_bridge"), default="stage_residual")
    args = parser.parse_args()
    if args.mode == "assemble":
        if args.source is None:
            parser.error("assemble requires --source")
        assemble(args.source, args.output)
    else:
        if args.trace is None:
            parser.error("replay requires --trace")
        run(args)


if __name__ == "__main__":
    main()
