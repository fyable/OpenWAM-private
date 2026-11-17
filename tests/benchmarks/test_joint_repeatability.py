import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from benchmarks.robotwin import joint_repeatability as repeat
from benchmarks.robotwin.joint_execution_replay import profile_indices_for, validation_gates
from benchmarks.utils.eval_manifest import seal_manifest
from openwam.deploy.request_trace import RequestTrace


class RepeatabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "output"
        self.entries = [{"episode": index, "seed": 100000 + index,
                         "instructions": {"unseen": f"instruction {index}", "seen": "seen"},
                         "extra_provenance": "preserved"} for index in range(3)]
        self.manifest = seal_manifest({"benchmark": "robotwin", "task": "adjust_bottle",
                                       "mode": "demo_clean", "entries": self.entries})
        self.protocol = {"tasks": ["adjust_bottle"], "episodes_per_task": 3, "checkpoint": "/weights/step.safetensors",
                         "checkpoint_size": 12, "checkpoint_mtime_ns": 34, "checkpoint_config": "config",
                         "normalization_stats": "norm"}
        shard = self.source / "shard_0"
        repeat.base.write_json(shard / "protocol.json", self.protocol)
        repeat.base.write_json(shard / "manifests/adjust_bottle.json", self.manifest)
        for backend, method in repeat.METHODS.items():
            result = self.result(self.manifest, [True, backend == "legacy", True])
            repeat.base.write_json(shard / method / "results/adjust_bottle.json", result)

    @staticmethod
    def result(manifest, outcomes):
        entries = manifest["entries"]
        return {"operation": "run_eval", "task": manifest["task"], "mode": "demo_clean",
                "instruction_type": "unseen", "episode_start": 0, "num_episodes": len(entries),
                "total_episodes": len(entries), "manifest_hash": manifest["manifest_hash"],
                "successes": sum(outcomes), "episodes": [
                    {"episode": index, "seed": entry["seed"], "instruction": entry["instructions"]["unseen"],
                     "success": success} for index, (entry, success) in enumerate(zip(entries, outcomes))]}

    def args(self, **changes):
        values = {"source": self.source, "output": self.output, "controls": 1, "repeats": 3,
                  "gpus": "0,1,2,3", "port": 8950, "trace_limit": 10000}
        return argparse.Namespace(**(values | changes))

    def save_run(self, suite, case, backend, index, success):
        directory = repeat.run_path(self.output, case, backend, index)
        repeat.base.write_json(directory / "result.json", self.result(case["manifest"], [success]))
        repeat.base.write_json(directory / "complete.json", {
            "suite_identity": suite["identity"], "case": case["id"], "backend": backend, "repeat": index,
            "result_sha256": repeat.base.digest(directory / "result.json"), "request_records": [], "capture_capped": False})

    def test_union_deduplicates_preserves_instruction_and_original_episode(self):
        cases, _, _ = repeat.select_cases(self.source, 1)
        self.assertEqual(len(cases), 2)
        difference, control = cases
        self.assertEqual(difference["source_episode"], 1)
        self.assertEqual(difference["manifest"]["entries"][0]["episode"], 0)
        self.assertEqual(difference["original_entry"], self.entries[1])
        self.assertEqual(difference["instruction"], "instruction 1")
        self.assertEqual(difference["manifest"]["entries"][0]["extra_provenance"], "preserved")
        self.assertEqual(control["source_episode"], 0)
        self.assertEqual(control["kind"], "control")

    def test_rejects_invalid_source_and_duplicate_gpus(self):
        with self.assertRaises(ValueError):
            repeat.make_suite(self.args(gpus="0,0"))
        with self.assertRaises(ValueError):
            repeat.make_suite(self.args(repeats=1))
        path = self.source / "shard_0/joint_prefix_n10/results/adjust_bottle.json"
        result = repeat.base.read_json(path)
        result["episodes"][0]["instruction"] = "wrong"
        repeat.base.write_json(path, result)
        with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
            repeat.select_cases(self.source, 1)

    def test_shards_and_seeded_order_are_stable(self):
        suite = repeat.make_suite(self.args())
        self.assertEqual([case["shard"] for case in suite["cases"]], [0, 1])
        self.assertEqual(suite, repeat.make_suite(self.args()))
        for index in range(3):
            order = repeat.method_order(suite, 0, index)
            self.assertEqual(set(order), set(repeat.METHODS))
            self.assertEqual(order, repeat.method_order(suite, 0, index))
        self.assertNotEqual(repeat.method_order(suite, 0, 0), repeat.method_order(suite, 0, 1))

    def test_summary_distinguishes_baseline_variability_and_paired_loss(self):
        suite = repeat.make_suite(self.args())
        repeat.base.write_json(self.output / "suite.json", suite)
        for case in suite["cases"]:
            for index in range(3):
                self.save_run(suite, case, "legacy", index, index != 1 if case["kind"] == "difference" else True)
                self.save_run(suite, case, "prepared", index, True)
                self.save_run(suite, case, "prefix", index, False)
        report = repeat.summarize(self.output)
        self.assertEqual(report["status"], "complete")
        first, second = report["cases"]
        self.assertEqual(first["legacy_self_comparison"], {"pairs": 3, "disagreements": 2})
        self.assertEqual(first["classification"]["prefix"]["label"], "baseline_variable")
        self.assertEqual(second["classification"]["prefix"]["label"], "repeatable_candidate_regression")
        self.assertEqual(report["paired"]["prepared"]["improvements"], 1)
        self.assertEqual(report["paired"]["prefix"]["regressions"], 5)

    def test_invalid_or_mutated_results_are_not_complete(self):
        suite = repeat.make_suite(self.args())
        case = suite["cases"][0]
        self.save_run(suite, case, "legacy", 0, True)
        self.assertIsNotNone(repeat.valid_run(self.output, suite, case, "legacy", 0))
        marker = repeat.run_path(self.output, case, "legacy", 0) / "complete.json"
        marker.write_text("[]")
        self.assertIsNone(repeat.valid_run(self.output, suite, case, "legacy", 0))
        self.save_run(suite, case, "legacy", 0, True)
        marker.with_name("result.json").write_text("{}")
        self.assertIsNone(repeat.valid_run(self.output, suite, case, "legacy", 0))

    def test_resume_skips_complete_runs_and_restarts_invalid_attempts(self):
        suite = repeat.make_suite(self.args(gpus="0", controls=0, repeats=2))
        repeat.base.write_json(self.output / "suite.json", suite)
        case = suite["cases"][0]
        repeat.immutable_json(repeat.manifest_path(self.output, case), case["manifest"])
        cfg = argparse.Namespace(root=self.output / "shard_0", method_configs={
            method: argparse.Namespace(server_environment={}) for method in repeat.METHODS.values()})

        @contextmanager
        def server(config, method):
            yield object()

        def client(config, task, directory, manifest, server):
            repeat.base.write_json(directory / "result.json", self.result(case["manifest"], [True]))

        with patch.object(repeat, "configured", return_value=cfg), patch.object(repeat.base, "policy_server", server), \
                patch.object(repeat.base, "run_client", side_effect=client) as run_client, \
                patch.object(repeat, "gpu_snapshot", return_value={}):
            repeat.shard_run(self.output, suite, 0)
            self.assertEqual(run_client.call_count, 6)
            repeat.shard_run(self.output, suite, 0)
            self.assertEqual(run_client.call_count, 6)
            path = repeat.run_path(self.output, case, "prepared", 1) / "result.json"
            path.write_text("{}")
            repeat.shard_run(self.output, suite, 0)
            self.assertEqual(run_client.call_count, 7)

    def test_failure_preserves_valid_runs_and_cleans_server(self):
        suite = repeat.make_suite(self.args(gpus="0", controls=0, repeats=2))
        repeat.base.write_json(self.output / "suite.json", suite)
        cfg = argparse.Namespace(root=self.output / "shard_0", method_configs={
            method: argparse.Namespace(server_environment={}) for method in repeat.METHODS.values()})
        cleaned = []

        @contextmanager
        def server(config, method):
            try:
                yield object()
            finally:
                cleaned.append(method)

        with patch.object(repeat, "configured", return_value=cfg), patch.object(repeat.base, "policy_server", server), \
                patch.object(repeat.base, "run_client", side_effect=RuntimeError("injected")), \
                patch.object(repeat, "gpu_snapshot", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                repeat.shard_run(self.output, suite, 0)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(repeat.base.read_json(self.output / "report.json")["status"], "incomplete")

    def test_plan_launcher_is_dry_run(self):
        env = dict(os.environ, SERVER_PYTHON=sys.executable)
        result = subprocess.run(["bash", "scripts/validate_joint_repeatability.sh", "plan", "--source", str(self.source),
                                 "--output", str(self.output), "--controls", "1", "--repeats", "3"],
                                env=env, cwd=repeat.base.ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)["episodes"], 18)
        self.assertFalse(self.output.exists())

    def test_immutable_identity_rejects_changed_resume(self):
        path = self.output / "suite.json"
        repeat.immutable_json(path, {"fixture": 1})
        repeat.immutable_json(path, {"fixture": 1})
        with self.assertRaisesRegex(ValueError, "identity changed"):
            repeat.immutable_json(path, {"fixture": 2})

    def test_source_identity_changes_when_result_changes(self):
        first = repeat.make_suite(self.args())
        path = self.source / "shard_0/joint_prepared_n10/results/adjust_bottle.json"
        result = repeat.base.read_json(path)
        result["episodes"][1]["success"] = True
        result["successes"] = 3
        repeat.base.write_json(path, result)
        second = repeat.make_suite(self.args())
        self.assertNotEqual(first["identity"], second["identity"])
        self.assertEqual(len(first["cases"]), len(second["cases"]))

    def test_reference_partition_failure_does_not_become_operator_failure(self):
        gates = validation_gates({"actions": {"passed": True}}, {
            "signature": {"video": {"passed": True}, "action": {"passed": True},
                          "partition": {"passed": False}, "passed": False}})
        self.assertEqual(gates, {"outputs_passed": True, "operator_passed": True, "batch_partition_passed": False})

    def test_tail_profile_uses_slowest_measured_prefix_requests(self):
        rows = [{"backend": "prefix", "index": 1, "wall_ms": 10, "measured": True},
                {"backend": "prefix", "index": 2, "wall_ms": 30, "measured": True},
                {"backend": "legacy", "index": 1, "wall_ms": 100, "measured": True},
                {"backend": "prefix", "index": 0, "wall_ms": 200, "measured": False}]
        self.assertEqual(profile_indices_for(rows, {1, 2}, 1, tail=True), {2})
        self.assertEqual(profile_indices_for(rows, {1, 2}, 1), {1})

    def test_replay_assembles_captures_archives_interrupted_temp_and_resumes(self):
        suite = repeat.make_suite(self.args(gpus="0", controls=0))
        repeat.base.write_json(self.output / "suite.json", suite)
        case = suite["cases"][0]
        self.save_run(suite, case, "legacy", 0, True)
        records = []
        for index in range(3):
            path = self.root / f"request_{index}.npz"
            np.savez(path, image=np.zeros((4, 4, 3), dtype=np.uint8), state=np.zeros(20),
                     prompt=np.array("instruction"), episode_start=np.array(index == 0))
            records.append(str(path))
        marker = repeat.run_path(self.output, case, "legacy", 0) / "complete.json"
        metadata = repeat.base.read_json(marker)
        metadata["request_records"] = records
        repeat.base.write_json(marker, metadata)
        shard = self.output / "shard_0"
        (shard / "observations.tmp").write_text("interrupted")
        cfg = SimpleNamespace(root=shard, checkpoint=Path("/weights"), weight=Path("/weights/step.safetensors"),
                              server_python=sys.executable, gpu="2", tasks=[case["task"]])

        def spawn(command, **kwargs):
            self.assertIn("--validate-every-request", command)
            self.assertIn("--profile-tail", command)
            self.assertIn("cuda:2", command)
            repeat.base.write_json(shard / "replay/timing.json", {"status": "complete", "methods": {}})
            return SimpleNamespace(poll=lambda: 0, wait=lambda timeout=None: 0, stdin=None)

        with patch.object(repeat, "configured", return_value=cfg), patch.object(repeat, "spawn_resource", side_effect=spawn) as launched:
            repeat.replay_shard(self.output, suite, 0, 0, 1, 0)
            repeat.replay_shard(self.output, suite, 0, 0, 1, 0)
            self.assertEqual(launched.call_count, 1)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                repeat.replay_shard(self.output, suite, 0, 0, 2, 0)
        self.assertEqual(len(list(shard.glob("observations_interrupted_*.tmp"))), 1)
        with np.load(shard / "observations.npz", allow_pickle=False) as data:
            self.assertEqual(len(data["images"]), 3)

    def test_tmux_launch_quotes_arguments_without_starting_gpu_jobs(self):
        executable = self.root / "tmux"
        arguments = self.root / "tmux_args"
        executable.write_text('#!/bin/bash\nif [[ "$1" == has-session ]]; then exit 1; fi\nprintf "%s\\n" "$@" > "$TMUX_ARGS"\n')
        executable.chmod(0o755)
        env = dict(os.environ, SERVER_PYTHON=sys.executable, PATH=f"{self.root}:{os.environ['PATH']}",
                   TMUX_ARGS=str(arguments), GPUS="4,5,6,7", REPEAT_ROOT=str(self.output))
        subprocess.run(["bash", "scripts/validate_joint_repeatability.sh", "tmux", "--source", str(self.source),
                        "--controls", "1", "--repeats", "3"], env=env, cwd=repeat.base.ROOT,
                       text=True, capture_output=True, check=True)
        text = arguments.read_text()
        self.assertIn("new-session", text)
        self.assertIn("GPUS=4", text)
        self.assertIn(" all ", text)
        self.assertFalse(self.output.exists())

    def test_action_diagnostics_distinguish_input_and_action_divergence(self):
        suite = repeat.make_suite(self.args(controls=0))
        repeat.base.write_json(self.output / "suite.json", suite)
        case = suite["cases"][0]
        for backend in repeat.METHODS:
            self.save_run(suite, case, backend, 0, True)
            records = []
            for index in range(2):
                path = self.root / f"{backend}_{index}.npz"
                np.savez(path, image=np.zeros((4, 4, 3), dtype=np.uint8), state=np.array([index if backend == "prefix" else 0]),
                         prompt=np.array("instruction"), episode_start=np.array(index == 0),
                         actions=np.full((32, 20), 1 if backend == "prefix" else 0, dtype=np.float32))
                records.append(str(path))
            marker = repeat.run_path(self.output, case, backend, 0) / "complete.json"
            metadata = repeat.base.read_json(marker)
            metadata["request_records"] = records
            repeat.base.write_json(marker, metadata)
        repeat.analyze_captures(self.output, suite)
        rows = repeat.base.read_json(self.output / "capture_diagnostics.json")["rows"]
        self.assertIsNone(rows[0]["first_action_difference"])
        self.assertEqual(rows[1]["first_input_difference"], 1)
        self.assertEqual(rows[1]["first_action_difference"]["request"], 0)
        self.assertTrue(rows[1]["first_action_difference"]["identical_observation"])


class TraceOutputTests(unittest.TestCase):
    def test_opt_in_output_capture_is_bounded_without_interrupting_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory) / "trace"
            trace = RequestTrace(Path(directory), warmup=0, measure=1, max_requests=3,
                                 record_all=True, include_outputs=True)
            conditions = {"first_frame_image": [np.zeros((4, 4, 3), dtype=np.uint8)],
                          "proprio": np.ones(20), "prompt": "fixture"}
            for index in range(5):
                trace.record(conditions, {"actions": np.full((32, 20), index, dtype=np.float32)})
            self.assertEqual(trace.count, 3)
            self.assertTrue((Path(directory) / "capped.json").exists())
            self.assertFalse((Path(directory) / "complete.json").exists())
            with np.load(Path(directory) / "request_000002.npz", allow_pickle=False) as data:
                self.assertTrue(np.all(data["actions"] == 2))

    def test_default_capture_boundary_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory) / "trace"
            trace = RequestTrace(Path(directory), warmup=0, measure=1)
            conditions = {"first_frame_image": [np.zeros((4, 4, 3), dtype=np.uint8)],
                          "proprio": np.ones(20), "prompt": "fixture"}
            for index in range(3):
                trace.record(conditions)
            self.assertTrue((Path(directory) / "complete.json").exists())
            with np.load(Path(directory) / "request_000000.npz", allow_pickle=False) as data:
                self.assertNotIn("actions", data.files)


if __name__ == "__main__":
    unittest.main()
