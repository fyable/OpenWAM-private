"""Versioned short-trajectory sampling and explicit source-transition records."""

from collections import defaultdict
from pathlib import Path

from openwam.research.protocol import read_sealed

PROTOCOL = "e2_progress_fallback_v1"
REVIEWED_FILES = {
    "benchmarks/libero/research.py", "openwam/research/features.py",
    "openwam/research/exploratory.py", "tests/test_research_exploratory.py",
}


def phase_of(state):
    return 0 if state["progress"] < 0.35 else (1 if state["progress"] < 0.65 else 2)


def select_states(states):
    tasks = defaultdict(list)
    for state in states:
        tasks[(state["suite"], state["task_id"])].append(state)
    if len(tasks) != 8:
        raise ValueError("Exploratory screening requires all eight development tasks")
    selected, coverage = [], []
    for task, candidates in sorted(tasks.items()):
        chosen = []
        available = {phase_of(state) for state in candidates}
        for phase, target in enumerate((0.2, 0.5, 0.8)):
            pool = [state for state in candidates if phase_of(state) == phase and state not in chosen]
            if pool:
                chosen.append(min(pool, key=lambda state: (abs(state["progress"] - target),
                                                           state["episode_id"], state["request_index"], state["state_id"])))
        while len(chosen) < 3:
            pool = [state for state in candidates if state not in chosen]
            if not pool:
                raise ValueError("Each task requires three distinct saved states")
            episodes = {state["episode_id"] for state in chosen}
            chosen.append(min(pool, key=lambda state: (state["episode_id"] in episodes,
                                                       abs(state["progress"] - 0.5), state["episode_id"],
                                                       state["request_index"], state["state_id"])))
        selected.extend(state["state_id"] for state in chosen)
        coverage.append({"suite": task[0], "task_id": task[1], "missing_phases": sorted({0, 1, 2} - available),
                         "states": [{"state_id": state["state_id"], "actual_phase": phase_of(state),
                                     "progress": state["progress"]} for state in chosen]})
    return selected, coverage


def validate_coverage(states, protocol=None):
    tasks = defaultdict(list)
    for state in states:
        tasks[(state["suite"], state["task_id"])].append(state)
    if protocol not in (None, PROTOCOL):
        raise ValueError("Unknown E2 sampling protocol")
    if len(tasks) != 8 or any(len(group) != 3 for group in tasks.values()):
        raise ValueError("E2 requires three states for each of eight tasks")
    if len({state["state_id"] for state in states}) != 24:
        raise ValueError("E2 states must be distinct")
    if protocol is None and any(sorted(map(phase_of, group)) != [0, 1, 2] for group in tasks.values()):
        raise ValueError("Strict E2 requires one state per progress-proxy stage")


def source_changes(old, current):
    before, after = old["source_files"], current["source_files"]
    changes = {name: {"before": before.get(name), "after": after.get(name)}
               for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)}
    if not changes.keys() <= REVIEWED_FILES:
        raise ValueError(f"Unreviewed source changes: {sorted(changes.keys() - REVIEWED_FILES)}")
    return changes


def validate_bank_identity(args, bank, identity):
    if bank["model_sha256"] == identity["sha256"]:
        return
    review_path = getattr(args, "source_review", None)
    if not review_path:
        raise ValueError("Bank identity differs; explicit source review and fresh T0 required")
    old = read_sealed(Path(args.bank) / "model.json")
    review = read_sealed(review_path)
    if (old["sha256"] != bank["model_sha256"] or review["old_model_sha256"] != old["sha256"]
        or review["bank_sha256"] != bank["sha256"]
        or review["new_source_sha256"] != identity["source_sha256"]
        or review["changes"] != source_changes(old, identity)):
        raise ValueError("Source-transition review does not match bank/current source")
    excluded = {"sha256", "source_sha256", "source_files"}
    if ({key: value for key, value in old.items() if key not in excluded}
        != {key: value for key, value in identity.items() if key not in excluded}):
        raise ValueError("Source review cannot authorize changed weights/config/runtime")
