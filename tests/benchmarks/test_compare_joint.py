import json
import subprocess
from pathlib import Path

from benchmarks.libero.compare_joint import METHODS, summarize_root
from openwam.deploy.parallel_joint import core_identity


def test_local_identity_has_no_external_dependency():
    paths = core_identity()
    assert len(paths) == 2
    root = Path(__file__).resolve().parents[2] / "openwam" / "deploy"
    assert all(Path(path).parent == root for path in paths)


def test_comparison_plan_does_not_start_jobs(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(["bash", "scripts/compare_alpha_libero.sh", "plan"],
                            cwd=repo, capture_output=True, text=True, check=True)
    assert "6000 episodes" in result.stdout


def test_comparison_completeness(tmp_path):
    assert all(row["status"] == "missing" for row in summarize_root(tmp_path).values())
    for name in METHODS:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"diagnostic": True}))
        summary = {"overall": {"trials": 20, "successes": 19, "success_rate": .95,
                               "tasks_complete": 4, "tasks_expected": 4},
                   "suite_results": [], "missing_runs": []}
        (directory / "summary.json").write_text(json.dumps(summary))
    assert all(row["status"] == "complete" for row in summarize_root(tmp_path).values())
    summary["missing_runs"] = ["failed"]
    (directory / "summary.json").write_text(json.dumps(summary))
    assert summarize_root(tmp_path)[name]["status"] == "incomplete"
