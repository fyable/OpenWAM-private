import pytest
import torch
from torch import nn

from openwam.deploy.joint_conditions import shared_stage_conditions


class Conditions(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embedding = nn.Linear(8, 8)
        self.cross_attn = nn.Module()
        self.cross_attn.k = nn.Linear(8, 8)
        self.cross_attn.v = nn.Linear(8, 8)

    def forward(self, value):
        context = self.text_embedding(value)
        return self.cross_attn.k(context), self.cross_attn.v(context)


@pytest.mark.parametrize("shared", [True, False])
def test_shared_conditions_preserve_predictions_and_restore(shared):
    model = Conditions().eval()
    value = torch.randn(4, 7, 8)
    if shared:
        value = value[:1].expand_as(value).clone()
    with torch.no_grad():
        expected = model(value)
        with shared_stage_conditions(model) as stats:
            actual = model(value)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)
    assert stats["shared_calls"] == (3 if shared else 0)
    assert "forward" not in model.text_embedding.__dict__


def test_training_and_grad_paths_not_shared():
    model = Conditions()
    value = torch.randn(1, 7, 8).expand(4, -1, -1).requires_grad_()
    with shared_stage_conditions(model) as stats:
        model(value)[0].sum().backward()
    assert stats["shared_calls"] == 0
    assert value.grad is not None


def test_exception_restores_instance_forward():
    model = Conditions().eval()
    original = model.text_embedding.forward
    model.text_embedding.forward = original
    with pytest.raises(RuntimeError):
        with shared_stage_conditions(model):
            raise RuntimeError("failure")
    assert model.text_embedding.forward is original


def test_changed_conditions_are_not_reused_across_requests():
    model = Conditions().eval()
    with torch.no_grad():
        for _ in range(3):
            value = torch.randn(1, 7, 8).expand(4, -1, -1).clone()
            expected = model(value)
            with shared_stage_conditions(model):
                actual = model(value)
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_bfloat16_cuda_shared_conditions():
    model = Conditions().eval().to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        value = torch.randn(1, 513, 8, device="cuda", dtype=torch.bfloat16).expand(4, -1, -1).clone()
        expected = model(value)
        with shared_stage_conditions(model) as stats:
            actual = model(value)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0.05, atol=0.05)
    assert stats["shared_calls"] == 3
