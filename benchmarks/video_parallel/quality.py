"""Video fidelity metrics and scene-clustered confidence intervals."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .experiment import write_json


def video_metrics(candidate, reference):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if (candidate.shape != reference.shape or candidate.ndim != 4 or candidate.shape[-1] != 3
            or candidate.dtype != np.uint8 or reference.dtype != np.uint8 or min(candidate.shape[:3]) < 1):
        raise ValueError("Videos must be matching nonempty uint8 [T,H,W,3] RGB arrays")
    current = torch.from_numpy(candidate.copy()).permute(0, 3, 1, 2).float() / 255
    target = torch.from_numpy(reference.copy()).permute(0, 3, 1, 2).float() / 255
    mse = float((current - target).square().mean())
    window = min(11, current.shape[-2], current.shape[-1])
    if window % 2 == 0:
        window -= 1
    coordinates = torch.arange(window).float() - window // 2
    gaussian = torch.exp(-coordinates.square() / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, window, window).contiguous()

    def blur(value):
        return functional.conv2d(value, kernel, groups=3)

    mean_current, mean_target = blur(current), blur(target)
    variance_current = (blur(current.square()) - mean_current.square()).clamp_min(0)
    variance_target = (blur(target.square()) - mean_target.square()).clamp_min(0)
    covariance = blur(current * target) - mean_current * mean_target
    ssim = ((2 * mean_current * mean_target + .01**2) * (2 * covariance + .03**2)
            / ((mean_current.square() + mean_target.square() + .01**2)
               * (variance_current + variance_target + .03**2))).mean()
    temporal = None if len(current) < 2 else float(((current[1:] - current[:-1]) - (target[1:] - target[:-1])).square().mean().sqrt())
    return {"mse": mse, "psnr_db": -10 * math.log10(mse) if mse > 0 else None,
            "identical": mse == 0, "ssim": float(ssim), "temporal_difference_rmse": temporal,
            "lpips": "not_run", "interpretation": "Fidelity to supplied reference, not a physical correctness certificate"}


def clustered_interval(rows, field, bootstrap=2000, seed=42):
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    clusters = {}
    for row in rows:
        value = row[field]
        if value is not None:
            clusters.setdefault(row["scene"], []).append(float(value))
    if not clusters:
        return {"mean": None, "ci95": None, "clusters": 0}
    values = list(clusters.values())
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap):
        selected = generator.integers(0, len(values), size=len(values))
        estimates.append(float(np.mean([value for index in selected for value in values[index]])))
    return {"mean": float(np.mean([value for cluster in values for value in cluster])),
            "ci95": [float(value) for value in np.percentile(estimates, [2.5, 97.5])], "clusters": len(values)}


def evaluate_manifest(path, output):
    path, output = Path(path), Path(output)
    manifest = json.loads(path.read_text())
    rows = []
    keys = set()
    for entry in manifest["samples"]:
        if entry["reference_kind"] not in ("native_fidelity", "matched_real_future"):
            raise ValueError("Unknown reference kind")
        if entry["reference_kind"] == "matched_real_future" and entry.get("conditioning_actions_equal") is not True:
            raise ValueError("Real-future metrics require matched conditioning actions")
        key = (entry["method"], entry["scene"], entry["sample_id"])
        if key in keys:
            raise ValueError("Duplicate quality sample")
        keys.add(key)
        with np.load(path.parent / entry["path"], allow_pickle=False) as arrays:
            values = video_metrics(arrays["candidate_rgb"], arrays["reference_rgb"])
        rows.append({**{key: entry[key] for key in ("scene", "sample_id", "method", "reference_kind")}, **values})
    summary = {}
    for method, kind in sorted({(row["method"], row["reference_kind"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method and row["reference_kind"] == kind]
        summary[method + ":" + kind] = {name: clustered_interval(selected, name) for name in ("mse", "ssim", "temporal_difference_rmse")}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"rows": rows, "summary": summary, "physical_correctness": "not_established", "lpips": "not_run"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate_manifest(args.manifest, args.output)


if __name__ == "__main__":
    main()
