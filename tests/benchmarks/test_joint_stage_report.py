import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.utils import joint_stage_report as report


class StageReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.suite = {"source": str(self.source), "identity": "fixture", "repeats": 2, "gpus": ["4"],
                      "cases": [{"id": "case_a", "task": "task_a", "shard": 0},
                                {"id": "case_b", "task": "task_b", "shard": 0}]}
        (self.root / "suite.json").write_text(json.dumps(self.suite))
        methods = {method: {"episodes": 10, "expected_episodes": 10, "successes": 9,
                            "joint_request_timing": {"warm": {"p50_ms": 300.}}}
                   for method in report.repeat.METHODS.values()}
        report.repeat.base.write_json(self.source / "shard_0/comparison.json", {"methods": methods})
        self.outcomes = {"case_a": {"legacy": [True, True], "prepared": [False, False], "prefix": [True, None]},
                         "case_b": {"legacy": [False, True], "prepared": [True, None], "prefix": [False, True]}}

    def valid_run(self, root, suite, case, backend, index):
        value = self.outcomes[case["id"]][backend][index]
        return {"episodes": [{"success": value}]} if value is not None else None

    def collect(self):
        with patch.object(report.repeat, "valid_run", side_effect=self.valid_run):
            return report.collect(self.root)

    def test_common_samples_exclude_missing_but_include_failures(self):
        result = self.collect()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["totals"]["legacy"], {"completed": 4, "successes": 3})
        self.assertEqual(result["totals"]["prepared"], {"completed": 3, "successes": 1})
        self.assertEqual(result["common"], {"count": 2, "successes": {"legacy": 1, "prepared": 1, "prefix": 1}})
        self.assertEqual(result["paired"]["prepared"]["regressions"], 2)
        self.assertEqual(result["paired"]["prepared"]["improvements"], 1)
        self.assertEqual(result["paired"]["prepared"]["repeatable_regressions"], ["case_a"])
        self.assertEqual(result["baseline_repeat"], {"complete_cases": 2, "variable_cases": 1})

    def test_report_marks_pending_replay_and_states_boundaries(self):
        text = report.render(self.collect())
        self.assertIn("| 0 | — | — | — | — | — | 未完成 |", text)
        self.assertIn("不是原版 Native N10", text)
        self.assertIn("不包含视频 VAE 解码", text)
        self.assertIn("不合并或平均分片 P50", text)
        self.assertIn("case_a", text)
        self.assertEqual(report.rate(0, 0), "未完成")

    def test_replay_identity_mismatch_rejected(self):
        report.repeat.base.write_json(self.root / "shard_0/replay/timing.json", {
            "status": "complete", "trace": {"suite_identity": "other"}})
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.collect()

    def test_failed_parity_label_preserved_with_speedup(self):
        methods = {backend: {"wall_ms": {"count": 32, "p50": latency, "p95": latency + 10}}
                   for backend, latency in zip(report.repeat.METHODS, (360., 300., 280.))}
        report.repeat.base.write_json(self.root / "shard_0/replay/timing.json", {
            "status": "complete", "trace": {"suite_identity": "fixture"}, "methods": methods,
            "interpretation": "diagnostic_only_parity_failed"})
        text = report.render(self.collect())
        self.assertIn("1.200x | diagnostic_only_parity_failed", text)

    def test_launcher_rejects_unavailable_gpus_without_starting_workers(self):
        self.suite["gpus"] = ["4", "5", "6", "7"]
        for case in self.suite["cases"]:
            case["kind"] = "difference"
        (self.root / "suite.json").write_text(json.dumps(self.suite))
        binaries = self.root / "bin"
        binaries.mkdir()
        fake_smi = binaries / "nvidia-smi"
        fake_smi.write_text("#!/bin/sh\nexit 9\n")
        fake_smi.chmod(0o755)
        process = subprocess.run(["bash", "scripts/finish_joint_report.sh", "run"], cwd=report.repeat.base.ROOT,
                                 env={**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                                      "SERVER_PYTHON": sys.executable, "REPEAT_ROOT": str(self.root)},
                                 text=True, capture_output=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("experiment not started", process.stderr)
        self.assertFalse((self.root / "reporting").exists())

    def launch_with_busy_gpus(self, shared, free_mib=40960, mode="tmux"):
        self.suite["gpus"] = ["4", "5", "6", "7"]
        for case in self.suite["cases"]:
            case["kind"] = "difference"
        (self.root / "suite.json").write_text(json.dumps(self.suite))
        binaries = self.root / "bin"
        binaries.mkdir()
        smi = binaries / "nvidia-smi"
        smi.write_text("#!/bin/sh\ncase \"$1\" in\n"
                       "--query-gpu=index,uuid) printf '4, GPU-four\\n5, GPU-five\\n6, GPU-six\\n7, GPU-seven\\n';;\n"
                       "--query-compute-apps=gpu_uuid,pid) printf 'GPU-four, 99999\\n';;\n"
                       f"--query-gpu=index,memory.free) printf '4, {free_mib}\\n5, {free_mib}\\n6, {free_mib}\\n7, {free_mib}\\n';;\n"
                       "*) exit 2;;\nesac\n")
        tmux = binaries / "tmux"
        tmux.write_text('#!/bin/sh\nif [ "$1" = has-session ]; then exit 1; fi\nprintf \'%s\\n\' "$@" > "$CAPTURE"\n')
        timeout = binaries / "timeout"
        timeout.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$CAPTURE"\nexit 124\n')
        python = binaries / "python"
        python.write_text('#!/bin/sh\nif [ "$1" = -m ]; then exit 0; fi\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        for path in (smi, tmux, timeout, python):
            path.chmod(0o755)
        capture = self.root / "captured_args"
        process = subprocess.run(["bash", "scripts/finish_joint_report.sh", mode], cwd=report.repeat.base.ROOT,
                                 env={**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                                      "SERVER_PYTHON": str(python), "REPEAT_ROOT": str(self.root),
                                      "ALLOW_SHARED_GPU": str(shared), "MIN_FREE_GPU_GIB": "32", "CAPTURE": str(capture)},
                                 text=True, capture_output=True)
        return process, capture

    def test_busy_gpus_remain_rejected_by_default(self):
        process, capture = self.launch_with_busy_gpus(0)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Selected GPUs are busy", process.stderr)
        self.assertFalse(capture.exists())

    def test_shared_mode_propagated_to_tmux_worker(self):
        process, capture = self.launch_with_busy_gpus(1)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("ALLOW_SHARED_GPU=1", capture.read_text())
        self.assertIn("MIN_FREE_GPU_GIB=32", capture.read_text())

    def test_shared_mode_checks_free_memory(self):
        process, capture = self.launch_with_busy_gpus(1, free_mib=20480)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("requires 32 GiB", process.stderr)
        self.assertFalse(capture.exists())

    def test_shared_run_skips_replay_and_records_policy_on_timeout(self):
        process, capture = self.launch_with_busy_gpus(1, mode="run")
        self.assertEqual(process.returncode, 124, process.stderr)
        arguments = capture.read_text().splitlines()
        self.assertEqual(arguments[arguments.index("scripts/validate_joint_repeatability.sh") + 1], "run")
        self.assertIn("180m", arguments)
        policy = json.loads((self.root / "reporting/shared_gpu.json").read_text())
        self.assertEqual(policy["timing_replay"], "skipped_requires_exclusive_gpu")
        self.assertEqual(policy["foreign_contexts"], ["GPU-four, 99999"])
        self.assertIn("共享 GPU 续跑", report.render(self.collect()))


if __name__ == "__main__":
    unittest.main()
