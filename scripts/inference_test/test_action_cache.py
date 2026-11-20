import copy
from dataclasses import asdict
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from benchmarks.action_cache.runtime import CacheConfig, VideoReuse, cached_request, install
from benchmarks.action_cache.next_runtime import FixedReuse, GroupReuse, attention_group_scores
from benchmarks.action_cache.next_experiment import (
    MIN_SPEEDUP, action_metrics, passes_speed_gate, select_entries, stage_report, summarize_next,
)
from benchmarks.action_cache.evaluation import paired_interval, summarize
from openwam.research.protocol import seal, write_json


class Video(torch.nn.Module):
    dit_patch_size = (1, 2, 2)

    def __init__(self):
        super().__init__()
        self.pre_tokens, self.post_tokens = [], []

    def pre_attn_at_layer(self, layer, state):
        self.pre_tokens.append(state.hidden_states.shape[1])
        x = state.hidden_states
        return x * .3, x * .4, x * .5, x

    def post_attn_at_layer(self, layer, state, attention, before):
        self.post_tokens.append(state.hidden_states.shape[1])
        state.hidden_states = before + attention * .2 + .01 * (layer + 1)
        return state


class Action(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def pre_attn_at_layer(self, layer, state):
        self.calls += 1
        x = state.payload.x_action
        return x * .6, x * .7, x * .8, x

    def post_attn_at_layer(self, layer, state, attention, before):
        state.payload.x_action = before + attention * .1
        return state


class Driver:
    num_layers, attention_mask_mode = 2, "mutual"

    def __init__(self):
        self.vb, self.ab = Video().eval(), Action().eval()
        self.masks = []

    def _video_tokens_per_frame(self, state):
        return 2

    def _mixed_attention(self, q, k, v, mask):
        self.masks.append(None if mask is None else mask.clone())
        return torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            attn_mask=None if mask is None else mask).squeeze(1)

    def step(self, layer, vs, ac, attn_mask=None, **kwargs):
        qv, kv, vv, vp = self.vb.pre_attn_at_layer(layer, vs)
        qa, ka, va, ap = self.ab.pre_attn_at_layer(layer, ac)
        outputs = self._mixed_attention(torch.cat([qv, qa], 1), torch.cat([kv, ka], 1), torch.cat([vv, va], 1), attn_mask)
        v, a = outputs.split([6, 2], 1)
        return self.vb.post_attn_at_layer(layer, vs, v.contiguous(), vp), self.ab.post_attn_at_layer(layer, ac, a.contiguous(), ap)

    def run_joint_loop(self, vs, ac, **kwargs):
        mask = torch.ones(8, 8, dtype=torch.bool)
        # A nontrivial visibility mask exercises row slicing on reuse.
        mask[:2, 2:6] = False
        for layer in range(self.num_layers):
            vs, ac = self.step(layer, vs, ac, attn_mask=mask, **kwargs)
        return vs, ac


def engine():
    driver = Driver()
    arch = SimpleNamespace(mot_driver=driver, training=False, _compiled_mot_run_joint_loop=None)
    cfg = OmegaConf.create({"inference": {"denoise_steps": 10, "denoise_mode": "sync", "inference_mode": "sync"},
                           "optimization": {"compile": {"enabled": False}}})
    return SimpleNamespace(architecture=arch, _cfg_scale=1, _dit_cache=None, _joint_sampler=None, cfg=cfg)


def states(step=0):
    x = torch.arange(24).reshape(1, 6, 4).float() / 30 + step * .005
    vs = SimpleNamespace(hidden_states=x, rope_freqs=torch.ones(6, 1, 2), time_mod=torch.ones(1, 6, 6, 4),
                         grid_frames=3, prefix_kv_len=0, vace_hints=None)
    ac = SimpleNamespace(payload=SimpleNamespace(x_action=torch.arange(8).reshape(1, 2, 4).float() / 20))
    return vs, ac


def trajectory(model):
    outputs = []
    for step in range(10):
        vs, ac = model.architecture.mot_driver.run_joint_loop(*states(step))
        outputs.append((vs.hidden_states.clone(), ac.payload.x_action.clone()))
    return outputs


def test_refresh_all_matches_native_without_changing_attention_shapes():
    before = trajectory(engine())
    model = engine()
    with cached_request(model, CacheConfig(policy="refresh")) as reuse:
        after = trajectory(model)
        assert reuse.report()["full_layers"] == 20
        assert reuse.report()["reused_layers"] == 0
    for a, b in zip(before, after):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not reuse.cache
    assert "step" not in model.architecture.mot_driver.__dict__


def test_real_skip_reduces_future_video_work_but_keeps_every_action_and_prefix():
    model = engine()
    with cached_request(model, CacheConfig(interval=3, max_age=3)) as reuse:
        result = trajectory(model)
        stats = reuse.report()
        assert stats["reused_layers"] > 0
        assert stats["video_query_fraction"] < 1
        assert stats["action_layers"] == 20
        assert model.architecture.mot_driver.ab.calls == 20
        assert model.architecture.mot_driver.vb.pre_tokens.count(2) == stats["reused_layers"]
        assert model.architecture.mot_driver.vb.post_tokens.count(2) == stats["reused_layers"]
        masks = model.architecture.mot_driver.masks
        reduced = next(m for m in masks if m.shape == (4, 8))
        full = masks[0]
        assert torch.equal(reduced, torch.cat([full[:2], full[6:]], 0))
        assert all(torch.isfinite(t).all() for output in result for t in output)


def test_cache_lifetime_is_one_request_and_restores_after_error():
    model = engine()
    results = []
    for _ in range(2):
        with cached_request(model, CacheConfig()) as reuse:
            assert not reuse.cache
            results.append(trajectory(model))
    for a, b in zip(*results):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    with pytest.raises(RuntimeError, match="test failure"):
        with cached_request(model, CacheConfig()):
            model.architecture.mot_driver.run_joint_loop(*states())
            raise RuntimeError("test failure")
    assert "step" not in model.architecture.mot_driver.__dict__
    assert "_action_cache_active" not in model.architecture.mot_driver.__dict__


def test_probe_uses_candidate_history_only_and_measures_targeted_step():
    model = engine()
    with cached_request(model, CacheConfig(policy="probe", layers=(1,), reuse_steps=(4,))) as reuse:
        trajectory(model)
        assert reuse.stats["reused_layers"] == 1
        assert reuse.stats["full_layers"] == 19


@pytest.mark.parametrize("interval", [2, 3])
def test_fixed_optimization_preserves_schedule_and_actions_without_unused_copies(interval):
    config = CacheConfig(interval=interval, max_age=interval)
    with cached_request(engine_instance := engine(), config, record_schedule=True) as original:
        baseline = trajectory(engine_instance)
        old_schedule = original.refresh_counts
    with cached_request(engine_instance, config, reuse_factory=FixedReuse, record_schedule=True) as optimized:
        actual = trajectory(engine_instance)
        assert optimized.refresh_counts == old_schedule
        assert all(entry["probe"] is None for entry in optimized.cache.values())
        assert all(entry["step"] < 9 for entry in optimized.cache.values())
    for before, after in zip(baseline, actual):
        assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_group_reuse_skips_only_requested_rows_on_last_step_and_keeps_old_kv_columns():
    class GroupDriver(Driver):
        num_layers = num_heads = 1

        def _mixed_attention(self, q, k, v, mask):
            self.attention_inputs.append((q.clone(), k.clone(), v.clone()))
            return super()._mixed_attention(q, k, v, mask)

        def step(self, layer, vs, ac, attn_mask=None, **kwargs):
            qv, kv, vv, vp = self.vb.pre_attn_at_layer(layer, vs)
            qa, ka, va, ap = self.ab.pre_attn_at_layer(layer, ac)
            joint = self._mixed_attention(torch.cat([qv, qa], 1), torch.cat([kv, ka], 1),
                                          torch.cat([vv, va], 1), attn_mask)
            v, a = joint.split([10, 2], 1)
            return self.vb.post_attn_at_layer(layer, vs, v, vp), self.ab.post_attn_at_layer(layer, ac, a, ap)

        def run_joint_loop(self, vs, ac, **kwargs):
            mask = torch.ones(12, 12, dtype=torch.bool)
            mask[:2, 2:10] = False
            return self.step(0, vs, ac, attn_mask=mask)

    def execute(model):
        driver = model.architecture.mot_driver
        for step in range(10):
            vs, ac = states(step)
            vs.hidden_states = torch.arange(40).reshape(1, 10, 4).float() / 30 + step * .005
            vs.rope_freqs = torch.ones(10, 1, 2)
            vs.time_mod = torch.ones(1, 10, 6, 4)
            vs, ac = driver.run_joint_loop(vs, ac)
        return vs.hidden_states, ac.payload.x_action

    model = engine()
    driver = model.architecture.mot_driver = GroupDriver()
    driver.attention_inputs = []
    native = execute(model)
    native_inputs = driver.attention_inputs.copy()
    groups = [[i] for i in range(2, 10)]
    for chosen in ((), (1, 5), "attention"):
        driver.attention_inputs.clear()
        factory = partial(GroupReuse, groups=groups, chosen=chosen, horizon=2, target_layer=0)
        with cached_request(model, CacheConfig(policy="refresh"), reuse_factory=factory) as reuse:
            actual = execute(model)
            assert reuse.anchor_seen and reuse.target_seen
            if not chosen:
                assert all(torch.equal(before, after) for before, after in zip(native, actual))
                assert reuse.skipped_rows == 0
                continue
            skipped = [groups[i][0] for i in reuse.used_groups]
            fresh = [i for i in range(12) if i not in skipped]
            assert len(skipped) == 2 and reuse.skipped_rows == 2
            if chosen != "attention":
                assert reuse.used_groups == chosen
            assert len(driver.vb.pre_tokens) == len(driver.vb.post_tokens)
            assert driver.vb.pre_tokens[-1] == driver.vb.post_tokens[-1] == 8
            q, k, v = driver.attention_inputs[-1]
            assert q.shape[1] == 10
            assert torch.equal(driver.masks[-1], driver.masks[-2][fresh])
            for observed, current, previous in zip((k, v), native_inputs[-1][1:], native_inputs[-2][1:]):
                assert torch.equal(observed[:, fresh], current[:, fresh])
                assert torch.equal(observed[:, skipped], previous[:, skipped])
    assert "_mixed_attention" not in driver.__dict__


def test_group_attention_uses_original_mask_and_all_visible_keys():
    q = torch.ones(1, 4, 1)
    k = torch.tensor([[[0.], [1.], [3.], [2.]]])
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[3, 2] = False
    scores = attention_group_scores(q, k, mask, heads=1, video_tokens=3, horizon=1, groups=[[1], [2]])
    assert scores[0] == pytest.approx(float(torch.tensor([0., 1., 2.]).softmax(0)[1]))
    assert scores[1] == 0


def test_next_experiment_excludes_entire_e2_episodes_and_measures_gripper_sign():
    entries = [dict(suite="s", task_id=task, episode_id=episode, request_index=req,
                    state_id=f"{task}:{episode}:{req}")
               for task in range(8) for episode in range(3) for req in (0, 1)]
    excluded = {f"{task}:0:1" for task in range(8)}
    assert select_entries(entries, excluded) == [
        f"{task}:{ep}:0" for task in range(8) for ep in (1, 2)]
    a, b = np.zeros((3, 10)), np.zeros((3, 10))
    a[:, 9], b[:, 9] = [-.8, .8, .8], [-.2, -.3, -.3]
    assert action_metrics(a, b, 2)["gripper_direction_disagreement"] == .5


def test_next_speed_gate_requires_both_metrics_and_reports_why_group_was_skipped(tmp_path):
    def case(denoise, request):
        return {"denoise_ms": {"paired_speedup_p50": denoise}, "request_ms": {"paired_speedup_p50": request}}

    assert passes_speed_gate(case(1.10, 1.05), MIN_SPEEDUP)
    assert not passes_speed_gate(case(1.40, 1.04), MIN_SPEEDUP)
    assert not passes_speed_gate(case(1.09, 1.25), MIN_SPEEDUP)
    plan = seal({"state_ids": {"discovery": ["state"]}, "gpu_uuid": "GPU-0",
                 "memory_mode": "text-offload", "minimum_speedup": MIN_SPEEDUP})
    write_json(tmp_path / "next_plan.json", plan)
    report = stage_report(tmp_path, "overhead", plan, "budget_exhausted")
    assert report["group_gate"] is False
    assert summarize_next(tmp_path)["reports"]["group"]["status"] == "skipped"


@pytest.mark.parametrize("config", [{"max_age": 0}, {"probes": 0}, {"threshold": float("nan")},
    {"layers": [0, 0]}, {"reuse_steps": [-1]}, {"policy": "probe"}, {"policy": "other"}])
def test_invalid_configuration_rejected(config):
    with pytest.raises(ValueError):
        CacheConfig(**config)


def test_risk_requires_measurements_and_enforces_maximum_age():
    model = engine()
    with pytest.raises(ValueError, match="measured Discovery"):
        with cached_request(model, CacheConfig(policy="adaptive")):
            pass
    scores = {f"{layer}:{stage}": 1. for layer in range(2) for stage in range(3)}
    with cached_request(model, CacheConfig(policy="adaptive", threshold=100, max_age=2), sensitivities=scores) as reuse:
        trajectory(model)
        assert reuse.stats["full_layers"] == 12  # steps 0,2,4,6,8 and pinned last step 9
        assert reuse.stats["reused_layers"] == 8


def test_rejects_mixed_execution_modes_and_incomplete_schedule():
    model = engine()
    model._dit_cache = object()
    with pytest.raises(ValueError):
        with cached_request(model, CacheConfig()):
            pass
    model._dit_cache = None
    with pytest.raises(ValueError, match="exactly ten"):
        with cached_request(model, CacheConfig()):
            model.architecture.mot_driver.run_joint_loop(*states())
    with pytest.raises(RuntimeError, match="Concurrent"):
        with cached_request(model, CacheConfig()):
            with cached_request(model, CacheConfig()):
                pass


def test_hierarchical_interval_keeps_policy_seeds_paired():
    data = {(f"suite", task, seed, initial): .1 for task in range(4) for seed in (43, 44, 45) for initial in range(5)}
    report = paired_interval(data, replicates=200)
    assert report["tasks"] == 4 and report["episodes"] == 60
    assert report["mean"] == pytest.approx(.1)
    assert report["ci95"] == pytest.approx([.1, .1])


def closed_loop_files(tmp_path):
    jobs = []
    for name, outcomes in (("native_n10", [True, False]), ("fixed_2", [True, True])):
        output = tmp_path / name
        result = output / "results.json"
        write_json(result, {"suite": "suite", "task_id": 0, "manifest_hash": "paired",
            "trials": [{"trial": i, "init_state_index": i, "success": value} for i, value in enumerate(outcomes)]})
        write_json(output / "summary.json", {"task_results": [{"suite": "suite", "task_id": 0,
            "complete": True, "result_paths": [str(result)]}]})
        jobs.append(dict(id=name, method=name, output=str(output), suite="suite", task_ids=[0], policy_seed=43))
    manifest = tmp_path / "manifest.json"
    write_json(manifest, seal({"jobs": jobs, "tasks": [{"suite": "suite", "task_id": 0}],
        "seeds": [43], "trials_per_task": 2, "cases": [{"name": "native_n10"}, {"name": "fixed_2"}],
        "stage": "screen", "bootstrap_seed": 42, "noninferiority_margin": .02}))
    return manifest


def test_closed_loop_does_not_treat_missing_trials_as_failures_or_claim_dev_noninferiority(tmp_path):
    manifest = closed_loop_files(tmp_path)
    report = summarize(manifest, tmp_path / "report.json")
    assert report["status"] == "complete"
    assert report["methods"]["fixed_2"]["improvements"] == 1
    assert report["methods"]["fixed_2"]["noninferior"] is None
    (tmp_path / "fixed_2/summary.json").unlink()
    report = summarize(manifest, tmp_path / "report.json")
    assert report["status"] == "incomplete"
    assert report["methods"]["fixed_2"]["success_rate"] is None


def test_closed_loop_rejects_different_initial_state_manifest(tmp_path):
    import json
    manifest = closed_loop_files(tmp_path)
    path = tmp_path / "fixed_2/results.json"
    record = json.loads(path.read_text())
    record["manifest_hash"] = "different"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="initial-state manifest mismatch"):
        summarize(manifest, tmp_path / "report.json")


def test_server_adapter_pins_policy_seed_without_changing_environment_or_carrying_cache():
    model = engine()
    received = []
    def generate(conditions):
        received.append(conditions)
        trajectory(model)
        return {"actions": np.ones((32, 10))}
    model.generate = generate
    install(model, CacheConfig(), policy_seed=44)
    first = model.generate({"seed": 42})
    second = model.generate({"seed": 43})
    assert [row["seed"] for row in received] == [44, 44]
    assert first["action_cache_stats"] == second["action_cache_stats"]
    assert first["action_cache_stats"]["steps"] == 10


def probe_files(tmp_path):
    from benchmarks.action_cache.experiment import code_identity
    write_json(tmp_path / "identity.json", seal({"subset": "discovery", "phase": "probe",
        "protocol_sha256": "protocol", "task_split_sha256": "split", "source_model_sha256": "model",
        "model": {"sha256": "measured_model"}, "source_files": code_identity()}))
    write_json(tmp_path / "status.json", {"status": "complete"})
    for task in range(8):
        for step in (1, 4, 7):
            write_json(tmp_path / "results" / f"{task}_{step}.json", seal({
                "protocol_sha256": "protocol", "suite": "suite", "task_id": task,
                "config": {"cache": asdict(CacheConfig(policy="probe", layers=(5,), reuse_steps=(step,)))},
                "cache_stats": {"reused_layers": 1}, "metrics": {"prefix_native_rmse": .01}}))


def test_fit_uses_only_measured_discovery_cells_and_never_interpolates(tmp_path):
    from benchmarks.action_cache.experiment import fit
    from openwam.research.protocol import read_sealed
    probe_files(tmp_path)
    fit(tmp_path, tmp_path / "sensitivity.json")
    record = read_sealed(tmp_path / "sensitivity.json")
    assert record["sensitivities"] == {"5:0": 1., "5:1": 1., "5:2": 1.}
    assert record["fitted_split"] == "discovery"


def test_fit_rejects_incomplete_task_coverage(tmp_path):
    from benchmarks.action_cache.experiment import fit
    probe_files(tmp_path)
    (tmp_path / "results/7_1.json").unlink()
    with pytest.raises(ValueError, match="eight Discovery tasks"):
        fit(tmp_path, tmp_path / "sensitivity.json")


def test_fit_refuses_calibration_leakage(tmp_path):
    from benchmarks.action_cache.experiment import fit
    probe_files(tmp_path)
    (tmp_path / "identity.json").unlink()
    write_json(tmp_path / "identity.json", seal({"subset": "calibration", "phase": "probe"}))
    with pytest.raises(ValueError, match="Discovery"):
        fit(tmp_path, tmp_path / "sensitivity.json")


def test_development_replay_rejects_heldout(tmp_path):
    from benchmarks.action_cache.experiment import selected_states
    write_json(tmp_path / "manifests/split.json", seal({"tasks": []}))
    from openwam.research.protocol import read_sealed
    split = read_sealed(tmp_path / "manifests/split.json")
    with pytest.raises(ValueError, match="heldout"):
        selected_states({"e2_run": str(tmp_path), "task_split_sha256": split["sha256"]}, "heldout")
