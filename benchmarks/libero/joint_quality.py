"""Summarize small closed-loop Joint ablations without launching GPU work."""

import argparse
import json
from pathlib import Path

from benchmarks.libero.compare_joint import summarize_root


METHODS = ("phase_shared_n10", "phase_unshared_n10", "action_unshared_n10", "action_shared_n10")
TAIL_METHODS = ("tail_residual_n10", "tail_action_bridge_n10", "tail_video_bridge_n10", "tail_both_bridge_n10")
HISTORICAL = Path(__file__).resolve().parents[2] / "outputs/libero/compare_n10_20261031_205720/native_n10_e8/summary.json"


def summarize_quality(root, method_names=METHODS):
    methods = summarize_root(root, method_names)
    for name, method in methods.items():
        trials = []
        summary_path = root / name / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            for task in summary["task_results"]:
                seen = set()
                for source in task.get("result_paths", []):
                    payload = json.loads(Path(source).read_text())
                    for trial in payload["trials"]:
                        if trial["trial"] in seen:
                            raise ValueError(f"Duplicate trial in authoritative result paths: {source}")
                        seen.add(trial["trial"])
                        trials.append({"suite": task["suite"], "task_id": task["task_id"], **trial})
        method["trials"] = trials
        method["screening"] = "not_applicable"
        if name.startswith("action_") or name in TAIL_METHODS[1:]:
            screening_trials = [trial for trial in trials if trial["suite"] == "libero_spatial" and trial["task_id"] == 0]
            method["screening"] = ("not_run" if len(screening_trials) != 5 else
                                   "candidate_for_task3_validation" if sum(trial["success"] for trial in screening_trials) >= 4
                                   else "below_4_of_5_screening_target")
    historical = {"status": "not_available", "source": str(HISTORICAL), "paired_current_run": False}
    if HISTORICAL.is_file():
        historical.update(status="historical_reference_only", overall=json.loads(HISTORICAL.read_text())["overall"])
    return {"methods": methods, "historical_native": historical, "numerical_equivalence": "not_validated",
            "interpretation": "small_sample_quality_ablation_not_benchmark_success_rate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preset", choices=("history", "tail"), default="history")
    args = parser.parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    report = summarize_quality(args.root, TAIL_METHODS if args.preset == "tail" else METHODS)
    (args.root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Single-round Joint quality ablation", "",
             "All methods: N10/H32/E8, mutual attention, future video latent denoising, no pixel decode.",
             "One batch=10 forward per warm request; serial initialization is reported separately.",
             "", "| Method | Status | Successes / Trials | Success rate | Warm P50 ms | Cold requests | Warm requests |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, method in report["methods"].items():
        overall = method.get("overall")
        timing = method.get("joint_request_timing", {})
        warm, cold = timing.get("warm", {}), timing.get("cold", {})
        rate = f"{overall['success_rate']:.2%}" if overall and overall['trials'] else "—"
        counts = f"{overall['successes']} / {overall['trials']}" if overall else "—"
        latency = f"{warm['p50_ms']:.3f}" if warm else "not_run"
        lines.append(f"| {name} | {method['status']} | {counts} | {rate} | {latency} | {cold.get('count', 0)} | {warm.get('count', 0)} |")
    lines += ["", "## Per-trial behavior", "", "| Method | Suite | Task | Trial | Success | Policy steps |",
              "|---|---|---:|---:|---|---:|"]
    for name, method in report["methods"].items():
        for trial in method["trials"]:
            lines.append(f"| {name} | {trial['suite']} | {trial['task_id']} | {trial['trial']} | {trial['success']} | {trial['policy_steps']} |")
    lines += ["", "## Screening", ""]
    for name, method in report["methods"].items():
        lines.append(f"- {name}: {method['screening']}")
    lines += ["", "Historical Native reference (not paired with this run): " + json.dumps(report["historical_native"]),
              "", "At least 4/5 on spatial task 0 is only a screening target; validate on task 3 before further claims.",
              "Closed-loop timings are not paired speedups. Numeric equivalence and success-rate recovery are not presumed.",
              "Existing phase-history default is unchanged. Review preserved evaluation videos to identify first behavioral divergence."]
    (args.root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {args.root / 'comparison.md'}")
    if any(method["status"] != "complete" for method in report["methods"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
