import pytest
from types import SimpleNamespace

from openwam.research.exploratory import PROTOCOL, select_states, source_changes, validate_bank_identity, validate_coverage
from openwam.research.protocol import seal, write_json


def states():
    return [{"suite": "suite", "task_id": task, "episode_id": episode,
             "request_index": index, "progress": index / 3,
             "state_id": f"{task}_{episode}_{index}"}
            for task in range(8) for episode in range(2) for index in (1, 2)]


def test_short_episode_fallback_is_distinct_and_honest():
    entries = states()
    ids, coverage = select_states(entries)
    assert len(ids) == len(set(ids)) == 24
    assert all(row["missing_phases"] == [1] for row in coverage)
    assert all({state["actual_phase"] for state in row["states"]} == {0, 2} for row in coverage)
    chosen = [state for state in entries if state["state_id"] in ids]
    validate_coverage(chosen, PROTOCOL)
    with pytest.raises(ValueError, match="Strict"):
        validate_coverage(chosen)
    assert select_states(list(reversed(entries))) == (ids, coverage)


def test_reject_missing_tasks_and_unknown_protocol():
    with pytest.raises(ValueError):
        select_states(states()[:4])
    with pytest.raises(ValueError):
        validate_coverage(states(), "unknown")


def test_source_review_rejects_model_changes():
    old = {"source_files": {"openwam/model/architectures/base.py": "old"}}
    current = {"source_files": {"openwam/model/architectures/base.py": "new"}}
    with pytest.raises(ValueError, match="Unreviewed"):
        source_changes(old, current)
    assert source_changes({"source_files": {}}, {"source_files": {"openwam/research/exploratory.py": "new"}})


def test_transition_requires_exact_review_and_unchanged_runtime(tmp_path):
    old = seal({"source_files": {}, "source_sha256": "before", "weights": "fixed"})
    current = seal({"source_files": {"openwam/research/exploratory.py": "new"},
                    "source_sha256": "after", "weights": "fixed"})
    bank = {"model_sha256": old["sha256"], "sha256": "bank"}
    write_json(tmp_path / "model.json", old)
    review = seal({"old_model_sha256": old["sha256"], "bank_sha256": "bank",
                   "new_source_sha256": "after", "changes": source_changes(old, current)})
    write_json(tmp_path / "review.json", review)
    args = SimpleNamespace(bank=tmp_path, source_review=tmp_path / "review.json")
    validate_bank_identity(args, bank, current)
    with pytest.raises(ValueError, match="weights/config/runtime"):
        validate_bank_identity(args, bank, {**current, "weights": "changed"})
    with pytest.raises(ValueError, match="explicit source review"):
        validate_bank_identity(SimpleNamespace(bank=tmp_path), bank, current)
