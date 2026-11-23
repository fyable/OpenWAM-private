import json
import os
from pathlib import Path

from scripts import run_chunk_autonomous as campaign


def test_identity_rejects_nonexistent_and_identifies_self():
    assert campaign.process_identity(os.getpid()) is not None
    assert campaign.process_identity(999999999) is None


def test_stop_owned_never_signals_reused_pid(monkeypatch):
    monkeypatch.setattr(campaign, 'process_identity', lambda pid: 'new')
    monkeypatch.setattr(campaign.os, 'kill', lambda *args: (_ for _ in ()).throw(AssertionError('unsafe signal')))
    campaign.stop_owned(123, 'old')


def test_save_is_atomic_and_readable(tmp_path):
    path = tmp_path / 'state.json'
    campaign.save(path, {'deadline': 123})
    assert json.loads(path.read_text()) == {'deadline': 123}
    assert not path.with_suffix('.json.tmp').exists()


def test_resume_keeps_original_deadline(tmp_path, monkeypatch):
    import argparse
    monkeypatch.setattr(campaign.subprocess, 'check_output', lambda *args, **kwargs: '0, a\n2, b\n4, c\n6, d\n')
    campaign.save(tmp_path / 'state.json', {'deadline_s': 123, 'stages': {}, 'status': 'running'})
    args = argparse.Namespace(run=tmp_path, started_at=999, budget_hours=20, adopt_pid=999999999)
    manager = campaign.Campaign(args)
    assert manager.state['deadline_s'] == 123


def test_environment_does_not_enable_offload_or_change_gpus(tmp_path):
    manager = object.__new__(campaign.Campaign)
    manager.run = tmp_path
    manager.source = tmp_path / 'source'
    env = manager.environment({'groups': '0-1;2-3;4-5;6-7', 'frames': True})
    assert env['CUDA_VISIBLE_DEVICES'] == '0,2,4,6'
    assert env['OPENWAM_REQUIRE_GPU_RESIDENT'] == '1'
    assert 'E2_MEMORY_MODE' not in env
    assert env['OPENWAM_CHUNK_SELECTED_FRAMES'] == '1'
    assert env['OPENWAM_CHUNK_REUSE_NOISE'] == '0'
