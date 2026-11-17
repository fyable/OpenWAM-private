from dataclasses import replace

import pytest
import torch
import torch.nn.functional as functional

from openwam.research.interventions import Intervention, ResearchContext, intervene_attention


def inputs():
    generator = torch.Generator().manual_seed(17)
    tensors = [torch.randn(1, 9, 8, generator=generator) for _ in range(3)]
    mask = torch.ones(9, 9, dtype=torch.bool)
    mask[:2, 6:] = False
    return *tensors, mask


def original_attention(query, keys, values, mask):
    def heads(value):
        return value.reshape(1, 9, 2, 4).transpose(1, 2)
    return functional.scaled_dot_product_attention(heads(query), heads(keys), heads(values),
                                                  attn_mask=mask).transpose(1, 2).reshape(1, 9, 8)


def apply(config, tensors=None, context=None, index=1):
    query, keys, values, mask = tensors if tensors is not None else inputs()
    context = context or ResearchContext(config)
    result = intervene_attention(query, keys, values, mask, heads=2, video_tokens=6,
                                 tokens_per_frame=2, grid_frames=3, layer=0,
                                 step=context.step(index, 0.8, 0.7, 1, True, True))
    return result, context


@pytest.mark.parametrize("direction", [("action", "future_video"), ("action", "video"), ("future_video", "action")])
@pytest.mark.parametrize("gate", [0.0, 0.5, 0.75, 1.0])
def test_gate_matches_explicit_formula_and_direction_isolation(direction, gate):
    query, keys, values, mask = inputs()
    config = Intervention((0,), (1,), query_modality=direction[0], source_modality=direction[1], gate=gate, chunk_size=1)
    result, context = apply(config)
    logits = query.reshape(1, 9, 2, 4).transpose(1, 2) @ keys.reshape(1, 9, 2, 4).transpose(1, 2).transpose(-1, -2) / 2
    probabilities = torch.softmax(logits.masked_fill(~mask, -torch.inf), -1)
    queries = slice(6, 9) if direction[0] == "action" else slice(2, 6)
    sources = slice(6, 9) if direction[1] == "action" else slice(2 if direction[1] == "future_video" else 0, 6)
    changed = probabilities.clone()
    changed[:, :, queries, sources] *= gate
    expected = (changed @ values.reshape(1, 9, 2, 4).transpose(1, 2)).transpose(1, 2).reshape(1, 9, 8)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=2e-7)
    original = original_attention(query, keys, values, mask)
    if direction[0] == "action":
        torch.testing.assert_close(result[:, :6], original[:, :6], rtol=0, atol=0)
    else:
        torch.testing.assert_close(result[:, :2], original[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(result[:, 6:], original[:, 6:], rtol=0, atol=0)
    assert context.records[0]["sigma_action"] == 0.7


def test_masked_rows_finite_and_mask_preserved():
    query, keys, values, mask = inputs()
    mask[7] = False
    result, _ = apply(Intervention((0,), (1,), gate=0), (query, keys, values, mask))
    assert torch.isfinite(result).all()
    assert torch.equal(result[:, 7], torch.zeros_like(result[:, 7]))


def test_noise_positive_negative_and_global_rng_isolation():
    config = Intervention((0,), (1,), kind="value_noise", floor=1e-6, epsilon=0.03, token_ids=(2, 3))
    tensors = inputs()
    state = torch.get_rng_state().clone()
    positive, _ = apply(config, tensors)
    negative, _ = apply(replace(config, sign=-1), tensors)
    assert torch.equal(state, torch.get_rng_state())
    baseline = original_attention(*tensors)
    torch.testing.assert_close(positive + negative, 2 * baseline, rtol=1e-5, atol=2e-7)
    torch.testing.assert_close(positive[:, :6], baseline[:, :6], rtol=0, atol=0)
    assert not torch.equal(positive[:, 6:], baseline[:, 6:])


def test_stale_capture_single_request_and_shape_validation():
    config = Intervention((0,), (1,), gate=1)
    capture = ResearchContext(config, capture_values=True)
    apply(config, context=capture, index=0)
    stale_config = replace(config, kind="value_stale", floor=1e-6)
    current = list(inputs())
    current[2] = current[2] + 0.1
    stale = ResearchContext(stale_config, stale_values=capture.values)
    result, _ = apply(stale_config, current, stale)
    assert torch.isfinite(result).all()
    with pytest.raises(ValueError, match="Missing matching"):
        apply(stale_config)


@pytest.mark.parametrize("options", [dict(gate=float("nan")), dict(layer_ids=()), dict(step_ids=(-1,)),
                                    dict(token_ids=(0, 0)), dict(query_modality="unknown"),
                                    dict(kind="value_noise"), dict(chunk_size=0)])
def test_bad_configuration_rejected(options):
    with pytest.raises(ValueError):
        Intervention(**({"layer_ids": (0,), "step_ids": (1,)} | options))


def test_prefix_and_rectangular_kv_rejected():
    with pytest.raises(ValueError, match="protected prefix"):
        apply(Intervention((0,), (1,), token_ids=(0,)))
    query, keys, values, mask = inputs()
    with pytest.raises(ValueError, match="matching Q/K/V"):
        apply(Intervention((0,), (1,)), (query, keys[:, :-1], values[:, :-1], mask))


def test_context_single_use_and_bounds():
    context = ResearchContext(Intervention((2,), (3,)))
    with pytest.raises(ValueError, match="exceed"):
        context.begin(3, 2)
    context.begin(10, 4)
    with pytest.raises(ValueError, match="single-use"):
        context.begin(10, 4)
