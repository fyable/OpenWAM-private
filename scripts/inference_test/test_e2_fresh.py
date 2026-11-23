import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from openwam.research.protocol import file_hash, read_sealed, seal, write_json
from scripts.reproduce_e2_fresh import merge_collections, stop_descendants


def collection_shards(run):
    tasks = [dict(suite="suite", task_id=i, split="discovery") for i in range(2)]
    split = seal({"tasks": tasks})
    write_json(run / "manifests/split.json", split)
    model, environment = seal({"model": "same"}), seal({"environment": "same"})
    for i in range(2):
        root = run / f"parallel/collection_discovery/worker_{i}"
        write_json(root / "model.json", model)
        write_json(root / "environment.json", environment)
        entries = []
        for episode in range(10):
            state = f"s{i}_{episode}"
            (root / f"{state}.npz").write_bytes(b"observations")
            write_json(root / f"{state}.snapshot.json", {"snapshot": True})
            entry = {**tasks[i], "state_id": state, "episode_id": episode, "request_index": 0,
                "model_sha256": model["sha256"], "environment_sha256": environment["sha256"],
                "task_split_sha256": split["sha256"], "valid_run": True,
                "file": f"{state}.npz", "sha256": file_hash(root / f"{state}.npz"),
                "snapshot_file": f"{state}.snapshot.json", "snapshot_sha256": file_hash(root / f"{state}.snapshot.json")}
            entries.append(entry)
            write_json(root / "episodes" / f"suite_{i}_{episode}" / "episode.json",
                       {"valid_run": True, "selected_state_ids": [state]})
        write_json(root / "manifest.json", seal({"model_sha256": model["sha256"],
            "environment_sha256": environment["sha256"], "task_split_sha256": split["sha256"], "states": entries}))


def test_collection_merge_preserves_order_and_all_episodes(tmp_path):
    collection_shards(tmp_path)
    merge_collections(tmp_path, "discovery", 2)
    bank = read_sealed(tmp_path / "bank/state_bank/discovery/manifest.json")
    assert [row["state_id"] for row in bank["states"]] == [f"s{i}_{ep}" for i in range(2) for ep in range(10)]
    assert len(list((tmp_path / "bank/state_bank/discovery").glob("*.npz"))) == 20


@pytest.mark.parametrize("corruption", ["identity", "file", "assignment", "episode"])
def test_collection_merge_rejects_incompatible_or_corrupt_shards(tmp_path, corruption):
    collection_shards(tmp_path)
    root = tmp_path / "parallel/collection_discovery/worker_1"
    if corruption == "identity":
        (root / "model.json").unlink()
        write_json(root / "model.json", seal({"model": "other"}))
    elif corruption == "file":
        (root / "s1_0.npz").write_bytes(b"corrupt")
    elif corruption == "episode":
        (root / "episodes/suite_1_0/episode.json").unlink()
    else:
        path = root / "manifest.json"
        bank = read_sealed(path)
        bank.pop("sha256")
        bank["states"][0]["task_id"] = 0
        path.unlink()
        write_json(path, seal(bank))
    with pytest.raises((ValueError, FileNotFoundError)):
        merge_collections(tmp_path, "discovery", 2)


def test_deadline_cleanup_reaches_children_in_independent_sessions(tmp_path):
    # The coordinator, worker and worker's child each have separate sessions.
    marker = tmp_path / "child.pid"
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], start_new_session=True); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(60)"
    root = subprocess.Popen([sys.executable, "-c", code, str(marker)], start_new_session=True)
    try:
        for _ in range(100):
            if marker.exists() and marker.read_text():
                break
            time.sleep(0.02)
        child = int(marker.read_text())
        stop_descendants(root.pid, grace=0.2)
        for _ in range(100):
            path = Path(f"/proc/{child}/stat")
            if not path.exists() or path.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
            time.sleep(0.02)
        else:
            pytest.fail("Worker descendant survived deadline cleanup")
        assert root.poll() is None  # The caller can still write a final report.
    finally:
        root.terminate()
        root.wait(timeout=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA validation")
def test_text_staging_preserves_bf16_cuda_results_and_restores_devices():
    from openwam.deploy.research_offload import configure_text_offload

    class Architecture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.video_backbone = torch.nn.Module()
            self.video_backbone.dit = torch.nn.Linear(8, 8)
            self.video_backbone.text_encoder = torch.nn.Linear(8, 8)
            self.action_backbone = torch.nn.Linear(8, 8)

        def set_dtype_device(self, dtype, device):
            self.to(dtype=dtype, device=device)

    model = Architecture().eval()
    original = copy.deepcopy(model).to(device="cuda", dtype=torch.bfloat16)
    configure_text_offload(model, torch.bfloat16, torch.device("cuda"))
    inputs = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = original.action_backbone(original.video_backbone.dit(original.video_backbone.text_encoder(inputs)))
        for _ in range(2):
            actual = model.action_backbone(model.video_backbone.dit(model.video_backbone.text_encoder(inputs)))
            assert torch.equal(actual, expected)
            assert next(model.video_backbone.text_encoder.parameters()).device.type == "cpu"
            assert next(model.video_backbone.dit.parameters()).device.type == "cuda"
            assert next(model.action_backbone.parameters()).device.type == "cuda"
