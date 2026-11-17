"""Bounded baseline V statistics for E2 selection and frozen scale calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openwam.research.interventions import Intervention
from openwam.research.protocol import read_sealed, seal, token_groups, write_json
from openwam.research.runner import baseline_context, paired_generate


def collect_features(args):
    from benchmarks.libero.research import build_server, load_states, model_identity, validate_t0

    split = read_sealed(args.split)
    selection = read_sealed(args.selection)
    if selection["task_split_sha256"] != split["sha256"] or selection["fitted_split"] != "discovery":
        raise ValueError("Feature selection must be frozen on Discovery")
    bank, states = load_states(args.bank, split)
    states = [(entry, condition) for entry, condition in states if entry["split"] == args.selected_split]
    selected_ids = selection.get("state_ids_by_split", {}).get(args.selected_split)
    if selected_ids is not None:
        states = [(entry, condition) for entry, condition in states if entry["state_id"] in selected_ids]
        if {entry["state_id"] for entry, _ in states} != set(selected_ids):
            raise ValueError("Frozen feature states are missing from bank")
    if not states:
        raise ValueError("No states in selected split")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    server = build_server(args)
    identity = model_identity(server, args)
    from openwam.research.exploratory import validate_bank_identity
    validate_bank_identity(args, bank, identity)
    validate_t0(read_sealed(args.t0), identity, bank, args.steps)
    write_json(root / "model.json", identity)
    records = []
    norms = []
    for entry, condition in states:
        for layer in selection["layer_ids"]:
            for seed in args.seeds:
                config = Intervention(layer_ids=(layer,), step_ids=(0,), gate=1.0)
                context = baseline_context(config, args.steps, capture=True)
                paired_generate(server.engine, condition, seed=seed, steps=args.steps, context=context)
                expected_layout = selection.get("layouts", {}).get(str(layer))
                if expected_layout is not None and expected_layout != context.layouts[layer]:
                    raise ValueError("Frozen representative token layout differs from the actual model")
                groups = token_groups(context.layouts[layer])
                prefix = context.layouts[layer]["prefix_tokens"]
                for step in range(args.steps):
                    current = context.values[(layer, step)]
                    previous = context.values.get((layer, step - 1))
                    diagnostics = next(row for row in context.records if row["layer_id"] == layer and row["step_id"] == step)
                    for group_id, tokens in enumerate(groups):
                        ids = [token - prefix for token in tokens]
                        features = current[:, :, ids]
                        norm = features.norm().item()
                        norms.append(norm)
                        change_norm = None if previous is None else (features - previous[:, :, ids]).norm().item()
                        records.append({"state_id": entry["state_id"], "episode_id": entry["episode_id"],
                                        "suite": entry["suite"], "task_id": entry["task_id"], "seed": seed,
                                        "layer_id": layer, "step_id": step, "group_id": group_id,
                                        "feature_norm": norm, "change_norm": change_norm,
                                        "sigma_video": diagnostics["sigma_video"],
                                        "sigma_action": diagnostics["sigma_action"], "layout": context.layouts[layer]})
    floor = max(float(np.median(norms)) * 1e-6, np.finfo(np.float32).tiny)
    if args.selected_split == "calibration":
        write_json(root / "calibration.json", seal({"fitted_split": "calibration", "task_split_sha256": split["sha256"],
                                                   "floor": floor, "rule": "median_group_norm * 1e-6, float32 tiny floor",
                                                   "selection_sha256": selection["sha256"], "model_sha256": identity["sha256"]}))
    write_json(root / "features.json", seal({"fitted_split": args.selected_split, "task_split_sha256": split["sha256"],
                                            "selection_sha256": selection["sha256"], "model_sha256": identity["sha256"],
                                            "bank_sha256": bank["sha256"], "records": records}))
