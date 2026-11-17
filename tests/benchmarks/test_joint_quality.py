import json
import os
import subprocess
from pathlib import Path

import pytest

from benchmarks.libero.joint_quality import METHODS, summarize_quality


def test_quality_plan_does_not_launch_or_write(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "results"
    result = subprocess.run(["bash", "scripts/ablate_alpha_joint_quality.sh", "plan"], cwd=root,
                            env=dict(os.environ, COMPARE_ROOT=str(output)), capture_output=True, text=True, check=True)
    assert "20 episodes" in result.stdout
    assert not output.exists()


def test_quality_launcher_wires_all_four_arms(tmp_path):
    root = Path(__file__).resolve().parents[2]
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "bash"
    stub.write_text('#!/bin/sh\nprintf "ARM %s\\n" "$OUTPUT_DIR"\nprintf "%s\\n" "$*"\n')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", SERVER_PYTHON="/bin/echo",
               GPU="6", COMPARE_ROOT=str(tmp_path / "out"), SUITES="spatial", TASK_IDS="0,3")
    result = subprocess.run(["/bin/bash", "scripts/ablate_alpha_joint_quality.sh", "diagnostic"],
                            cwd=root, env=env, capture_output=True, text=True, check=True)
    commands = [line for line in result.stdout.splitlines() if line.startswith("scripts/eval_alpha_libero.sh")]
    assert len(commands) == 4
    for command, scope, shared in zip(commands, ["phase_video_action"] * 2 + ["action_only"] * 2,
                                      ["true", "false", "false", "true"]):
        assert f"inference.joint_history_scope={scope}" in command
        assert f"inference.joint_share_conditions={shared}" in command
        assert "inference.joint_cache_schedule=true" in command
        assert "--task-ids 0,3 --num-trials 5 --seed 42" in command
        assert "--denoise-steps 10 --inference-horizon 8" in command
    assert not (tmp_path / "out").exists()


def test_quality_summary_missing_and_completed(tmp_path):
    report = summarize_quality(tmp_path)
    assert all(row["status"] == "missing" for row in report["methods"].values())
    for method in METHODS:
        directory = tmp_path / method
        directory.mkdir()
        source = directory / "results.json"
        count = 4 if method.startswith("action_") else 2
        source.write_text(json.dumps({"trials": [
            {"trial": index, "success": index < count, "policy_steps": 80 if index < count else 600}
            for index in range(5)]}))
        (directory / "manifest.json").write_text(json.dumps({
            "diagnostic": True, "resume_signature": {"jobs_count": 1}, "trial_run": {"num_trials": 5}}))
        (directory / "summary.json").write_text(json.dumps({
            "overall": {"trials": 5, "successes": count, "success_rate": count / 5,
                        "tasks_complete": 1, "tasks_expected": 1},
            "suite_results": [], "missing_runs": [], "task_results": [
                {"suite": "libero_spatial", "task_id": 0, "result_paths": [str(source)]}]}))
    report = summarize_quality(tmp_path)
    assert all(row["status"] == "complete" for row in report["methods"].values())
    assert report["methods"]["action_shared_n10"]["screening"] == "candidate_for_task3_validation"
    assert len(report["methods"]["phase_shared_n10"]["trials"]) == 5
    assert report["historical_native"]["paired_current_run"] is False
    assert report["numerical_equivalence"] == "not_validated"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([str(root / ".venv-openwam/bin/python"), "-m", "benchmarks.libero.joint_quality", "--root", str(tmp_path)],
                   cwd=root, check=True, capture_output=True)
    assert "Policy steps" in (tmp_path / "comparison.md").read_text()


def test_quality_resume_records_history_scope(tmp_path):
    from benchmarks.libero.scheduler import _validate_resume_signature

    original = {"sampler": "joint-paraflow", "diagnostic": True,
                "joint_optimization_overrides": ["inference.joint_history_scope=phase_video_action"]}
    (tmp_path / "manifest.json").write_text(json.dumps({"resume_signature": original}))
    with pytest.raises(RuntimeError, match="joint_optimization_overrides"):
        _validate_resume_signature(tmp_path, {**original, "joint_optimization_overrides": ["inference.joint_history_scope=action_only"]})
