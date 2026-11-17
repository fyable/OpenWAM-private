import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from benchmarks.video_parallel.experiment import cases_for, parser, report
from benchmarks.video_parallel.runtime import capture_video, method_spec, tensor_error
from benchmarks.video_parallel.solver import SolverConfig, VideoGuidedSampler, residuals
from benchmarks.video_parallel.quality import clustered_interval, evaluate_manifest, video_metrics


class Euler:
    num_train_timesteps = 1000

    @staticmethod
    def flow_step(prediction, sigma, next_sigma, sample):
        return sample + prediction * (next_sigma - sigma)


def fixture(dtype=torch.float32):
    architecture = SimpleNamespace(video_scheduler=Euler(), action_scheduler=Euler())
    video = torch.ones(1, 2, 3, 4, 4, dtype=dtype)
    video[:, :, :1] = 3
    inputs = {"latents": video, "first_frame_latents": video[:, :, :1].clone()}
    actions = torch.ones(1, 4, 3, dtype=dtype)
    schedule = [(1000., 1000.), (750., 750.), (500., 500.), (250., 250.), (0., 0.)]
    return architecture, inputs, actions, schedule


def coupled(architecture, inputs, videos, actions, schedule):
    return videos * .2 + actions.mean(dim=(1, 2))[:, None, None, None, None] * .1, actions * .3 + videos.mean(dim=(1, 2, 3, 4))[:, None, None] * .1


def run(config, velocity=coupled, dtype=torch.float32, inactive=None):
    architecture, inputs, actions, schedule = fixture(dtype)
    sampler = VideoGuidedSampler(config, velocity)
    outputs = sampler.sample(architecture, inputs, actions, schedule, inactive, prompt="test", seed=42)
    return sampler, outputs


class VideoParallelTests(unittest.TestCase):
    def test_k1_matches_native_euler_arithmetic(self):
        for dtype in (torch.float32, torch.bfloat16):
            architecture, inputs, action, schedule = fixture(dtype)
            video = inputs["latents"].clone()
            for index in range(4):
                predicted_video, predicted_action = coupled(architecture, inputs, video, action, schedule[index:index + 1])
                video = video + predicted_video * -.25
                video[:, :, :1] = inputs["first_frame_latents"]
                action = architecture.action_scheduler.flow_step(predicted_action, schedule[index][1] / 1000,
                                                                schedule[index + 1][1] / 1000, action)
            _, outputs = run(SolverConfig(window=1), dtype=dtype)
            torch.testing.assert_close(outputs[0], video, atol=0, rtol=0)
            torch.testing.assert_close(outputs[1], action, atol=0, rtol=0)

    def test_picard_recovers_coupled_serial_reference(self):
        _, expected = run(SolverConfig(policy="serial", window=1))
        sampler, actual = run(SolverConfig(policy="fixed", window=4, rounds=4))
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
        self.assertEqual(sampler.last_stats["stage_evaluations"], 16)

    def test_zero_tolerance_preserves_serial_result(self):
        _, expected = run(SolverConfig(window=1))
        sampler, actual = run(SolverConfig(policy="residual", window=2, video_tolerance=0,
                                           local_tolerance=0, action_tolerance=0))
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)
        self.assertTrue(all(window["reason"] in ("serial", "residual_prefix", "residual_rejected")
                            for window in sampler.last_stats["windows"]))

    def test_clean_prefix_and_inactive_dimensions(self):
        sampler, result = run(SolverConfig(policy="fixed", window=4), inactive=torch.tensor([False, True, True]))
        self.assertTrue(torch.all(result[0][:, :, :1] == 3))
        self.assertTrue(torch.all(result[1][..., 1:] == 0))
        self.assertFalse(sampler.last_stats["history_reuse"])

    def test_nan_parallel_falls_back(self):
        def invalid_batch(architecture, inputs, videos, actions, schedule):
            predictions = coupled(architecture, inputs, videos, actions, schedule)
            return (predictions[0] * float("nan"), predictions[1]) if len(actions) > 1 else predictions

        sampler, actual = run(SolverConfig(policy="residual", window=2), invalid_batch)
        _, expected = run(SolverConfig(window=1))
        torch.testing.assert_close(actual[0], expected[0])
        self.assertGreater(sampler.last_stats["fallbacks"], 0)
        self.assertEqual(sampler.last_stats["windows"][0]["reason"], "nonfinite_parallel")

    def test_nonfinite_serial_is_rejected(self):
        def invalid(architecture, inputs, videos, actions, schedule):
            return videos * float("nan"), actions

        with self.assertRaises(FloatingPointError):
            run(SolverConfig(window=1), invalid)

    def test_adaptive_preserves_time_grid(self):
        sampler, result = run(SolverConfig(video_tolerance=100., local_tolerance=100., action_tolerance=100.))
        self.assertEqual(sum(window["accepted"] for window in sampler.last_stats["windows"]), 4)
        self.assertEqual(result[0].shape[2], 3)

    def test_prefix_acceptance_is_contiguous(self):
        sampler, _ = run(SolverConfig(policy="residual", window=4, video_tolerance=.02, local_tolerance=.02, action_tolerance=.02))
        position = 0
        for window in sampler.last_stats["windows"]:
            self.assertEqual(window["start"], position)
            position += window["accepted"]
        self.assertEqual(position, 4)

    def test_residual_excludes_prefix_and_inactive_actions(self):
        _, inputs, actions, _ = fixture()
        changed_video = inputs["latents"].clone()
        changed_video[:, :, :1] += 100
        changed_actions = actions.clone()
        changed_actions[..., 1:] += 100
        values = residuals(inputs["latents"], changed_video, actions, changed_actions, torch.tensor([True, False, False]), 4)
        self.assertTrue(all(value == 0 for value in values.values()))

    def test_local_residual_detects_spatial_change(self):
        video = torch.ones(1, 1, 3, 16, 16)
        changed = video.clone()
        changed[:, :, 1:, :4, :4] += 10
        actions = torch.ones(1, 2, 1)
        values = residuals(video, changed, actions, actions, None, 4)
        self.assertGreater(values["video_local_p95"], values["video_rms"])

    def test_reset_and_request_independence(self):
        architecture, inputs, actions, schedule = fixture()
        sampler = VideoGuidedSampler(SolverConfig(), coupled)
        first = sampler.sample(architecture, inputs, actions, schedule, None, prompt="one", seed=42)
        second = sampler.sample(architecture, inputs, actions, schedule, None, prompt="two", seed=123)
        torch.testing.assert_close(first[0], second[0])
        sampler.reset()
        self.assertIsNone(sampler.last_video)
        self.assertFalse(sampler.last_stats)

    def test_invalid_config(self):
        for kwargs in ({"window": 3}, {"rounds": 5}, {"video_tolerance": float("nan")}, {"action_tolerance": -1}, {"policy": "other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SolverConfig(**kwargs)

    def test_bad_schedule(self):
        architecture, inputs, actions, _ = fixture()
        with self.assertRaises(ValueError):
            VideoGuidedSampler(velocity=coupled).sample(architecture, inputs, actions, [(0, 0), (0, 0)], None, prompt="", seed=0)

    def test_methods_and_sweep_keep_native_unmodified(self):
        args = parser().parse_args(["sweep", "--methods", "native_n10,adaptive"])
        args.solver_overrides = {"action_guard": False}
        cases = cases_for(args)
        self.assertEqual(cases[0]["name"], "native_n10")
        self.assertEqual(cases[0]["overrides"], {})
        self.assertEqual(len(cases), 4)
        self.assertEqual(method_spec("native_n6"), (6, None))
        with self.assertRaises(ValueError):
            method_spec("native_n1")

    def test_capture_restores_after_failure(self):
        backbone = SimpleNamespace(decode_video=lambda value: value + 1)
        engine = SimpleNamespace(architecture=SimpleNamespace(video_backbone=backbone), _decode_video=False)
        original = backbone.decode_video
        with self.assertRaises(RuntimeError):
            with capture_video(engine) as captured:
                self.assertTrue(engine._decode_video)
                self.assertIsNone(backbone.decode_video(torch.ones(1)))
                self.assertEqual(captured["latents"].item(), 1)
                raise RuntimeError("fixture")
        self.assertFalse(engine._decode_video)
        self.assertIs(backbone.decode_video, original)

    def test_missing_results_are_not_completed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"cases": [{"name": "native_n10"}, {"name": "adaptive"}], "measure": 1,
                        "trace_provenance": "unverified_diagnostic", "decode_video": False}
            (root / "identity.json").write_text(json.dumps(identity))
            result = report(root)
            self.assertEqual(result["status"], "incomplete")
            self.assertIsNone(result["methods"]["adaptive"]["wall_ms"])
            self.assertIn("pending", (root / "comparison.md").read_text())

    def test_unchanged_thresholds_and_nan_rejection(self):
        result = tensor_error(torch.ones(4), torch.zeros(4))
        self.assertEqual(result["rtol"], .05)
        self.assertEqual(result["atol"], .05)
        self.assertFalse(result["passed"])
        self.assertFalse(tensor_error(torch.tensor([float("nan")]), torch.zeros(1))["finite"])

    def test_identical_video_metrics(self):
        import numpy as np

        pixels = np.zeros((2, 12, 12, 3), dtype=np.uint8)
        metrics = video_metrics(pixels, pixels)
        self.assertTrue(metrics["identical"])
        self.assertIsNone(metrics["psnr_db"])
        self.assertAlmostEqual(metrics["ssim"], 1.)
        self.assertEqual(metrics["temporal_difference_rmse"], 0.)

    def test_quality_requires_matching_rgb_shape_and_dtype(self):
        import numpy as np

        with self.assertRaises(ValueError):
            video_metrics(np.ones((2, 4, 4, 3)), np.ones((2, 4, 4, 3)))

    def test_real_future_requires_matched_action_conditions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"samples": [{"reference_kind": "matched_real_future"}]}))
            with self.assertRaisesRegex(ValueError, "matched conditioning"):
                evaluate_manifest(root / "manifest.json", root / "quality.json")

    def test_cluster_bootstrap_retains_scene_groups(self):
        interval = clustered_interval([{"scene": "a", "score": 1.}, {"scene": "a", "score": 1.},
                                       {"scene": "b", "score": 1.}], "score", bootstrap=10)
        self.assertEqual(interval["clusters"], 2)
        self.assertEqual(interval["ci95"], [1., 1.])

    def test_closed_loop_records_missing_methods_separately(self):
        from benchmarks.video_parallel.closed_loop import summarize

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "research_identity.json").write_text(json.dumps({"methods": ["native_n10", "adaptive"]}))
            result = summarize(root)
            self.assertEqual(result["methods"]["adaptive"]["status"], "pending")

    def test_validation_failure_remains_diagnostic_in_report(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "identity.json").write_text(json.dumps({"cases": [{"name": "native_n10"}], "measure": 1,
                                                          "trace_provenance": "verified", "decode_video": False}))
            (root / "validation.json").write_text(json.dumps({"status": "failed"}))
            self.assertTrue(report(root)["diagnostic_only"])

    def test_preflight_checks_historical_trace_without_gpu(self):
        import numpy as np
        from benchmarks.video_parallel.experiment import preflight

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint_step_1.safetensors").write_bytes(b"fixture")
            (root / "config.yaml").write_text("model:\n  architecture:\n    attention_mask_mode: mutual\n    state_dim: 3\ndataloader:\n  type: libero\n")
            (root / "normalization_stats.npy").write_bytes(b"fixture")
            trace = root / "observations.npz"
            np.savez(trace, images=np.zeros((3, 4, 4, 3), dtype=np.uint8), prompts=np.array(["a"] * 3),
                     states=np.ones((3, 3)), episode_starts=np.array([True, False, False]))
            args = parser().parse_args(["preflight", "--trace", str(trace), "--ckpt-dir", str(root),
                                       "--warmup", "0", "--measure", "1", "--output", str(root / "out"), "--min-free-gib", "0"])
            with self.assertRaisesRegex(ValueError, "provenance"):
                preflight(args)
            args.allow_unverified_trace = True
            identity = preflight(args)
            self.assertEqual(identity["trace_provenance"], "unverified_diagnostic")
            self.assertEqual(identity["gpu_validation"], "not_run")
            self.assertFalse((root / "out").exists())
            (root / "config.yaml").write_text("model:\n  architecture:\n    attention_mask_mode: mutual\n    state_dim: 80\ndataloader:\n  type: libero\n  unify_action: true\n  unify_action_map: ['0-2']\n")
            self.assertEqual(preflight(args)["benchmark"], "libero")
            (root / "config.yaml").write_text("model:\n  architecture:\n    attention_mask_mode: mutual\n    state_dim: 80\ndataloader:\n  type: libero\n  unify_action: true\n  unify_action_map: ['0-9']\n")
            with self.assertRaisesRegex(ValueError, "has 3 dimensions.*expects 10 raw dimensions"):
                preflight(args)

    def test_server_python_wrapper_only_replaces_deploy(self):
        import os
        import subprocess
        from benchmarks.video_parallel.experiment import ROOT

        result = subprocess.run([str(ROOT / "scripts/video_parallel_python.sh"), "-c", "print('forwarded')"],
                                env={**os.environ, "VIDEO_PARALLEL_PYTHON": str(ROOT / ".venv-openwam/bin/python")},
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "forwarded")

    def test_libero_runner_keeps_pinned_protocol_and_wrapper(self):
        from benchmarks.video_parallel import closed_loop

        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), methods=["native_n10", "adaptive"], overrides={},
                                   ckpt_dir=Path(directory), gpus="0", stage="screen")
            with patch.object(closed_loop, "campaign_identity", return_value={"checkpoint": "/fixture/checkpoint.safetensors", "methods": args.methods}), \
                    patch.object(closed_loop.subprocess, "run") as launch:
                closed_loop.libero(args)
            self.assertEqual(launch.call_count, 2)
            command = launch.call_args.args[0]
            self.assertIn("--diagnostic", command)
            self.assertIn("optimization.dit_cache.enabled=false", command)
            self.assertEqual(launch.call_args.kwargs["env"]["VIDEO_PARALLEL_METHOD"], "adaptive")


if __name__ == "__main__":
    unittest.main()
