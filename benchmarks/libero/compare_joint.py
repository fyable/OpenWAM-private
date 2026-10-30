"""Summarize existing comparison outputs without launching model inference."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np


METHODS = ("native_n10_e8", "native_n4_e8", "joint_n4_e8")


def summarize_root(root):
    report = {}
    for name in METHODS:
        directory = root / name
        summary_path = directory / "summary.json"
        manifest_path = directory / "manifest.json"
        if not summary_path.exists() or not manifest_path.exists():
            report[name] = {"status": "missing"}
            continue
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        overall = summary["overall"]
        diagnostic = manifest.get("diagnostic", False)
        expected = 20 if diagnostic else 2000
        complete = (not summary["missing_runs"] and overall["trials"] == expected
                    and overall["tasks_complete"] == overall["tasks_expected"])
        row = {"status": "complete" if complete else "incomplete", "diagnostic": diagnostic,
               "overall": overall, "suites": summary["suite_results"],
               "protocol": {key: manifest.get(key) for key in
                            ("checkpoint", "sampler", "denoise_steps", "inference_horizon", "effective_seed")}}
        requests = []
        for path in sorted((directory / "logs" / "servers").glob("*.log")):
            for line in path.open(errors="replace"):
                marker = "joint-paraflow request stats="
                if marker in line:
                    requests.append(ast.literal_eval(line.split(marker, 1)[1].strip()))
        row["joint_request_timing"] = {}
        for label, cold in (("cold", True), ("warm", False)):
            values = [item["full_request_ms"] for item in requests if item["cold"] == cold]
            if values:
                row["joint_request_timing"][label] = {
                    "count": len(values), "p50_ms": float(np.percentile(values, 50)),
                    "p95_ms": float(np.percentile(values, 95)),
                }
        report[name] = row
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    report = summarize_root(args.root)
    for name, row in report.items():
        overall = row.get("overall")
        result = "no results" if overall is None else (
            f"{overall['successes']}/{overall['trials']} ({overall['success_rate']:.2%})")
        print(f"{name:20s} {row['status']:10s} {result}")
    (args.root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Closed-loop timings are not paired speedup measurements; use replay for speed comparisons.")
    if any(row["status"] != "complete" for row in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
