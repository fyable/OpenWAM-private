import copy

import numpy as np
import pytest
import torch

pytest.importorskip("modelscope")

from omegaconf import OmegaConf

from openwam.deploy.engine import JointInferenceEngine
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.research.interventions import Intervention, ResearchContext
from tests.test_action_normalization import _CaptureDeployArchitecture, _TinyGenerateArchitecture
from tests.test_mot_driver import _make_action_dit, _make_states
from tests.test_openwam_trainer import _MockVideoBackbone


class CaptureArchitecture(_CaptureDeployArchitecture):
    def __init__(self):
        super().__init__(None)
        self.requests = []

    def generate(self, **kwargs):
        self.requests.append(kwargs)
        return {"video": None, "actions": np.zeros((1, 10), dtype=np.float32)}


def engine_config():
    return OmegaConf.create({"inference": {"denoise_steps": 2, "inference_mode": "sync", "denoise_mode": "sync"},
                             "optimization": {"decode_video": False, "compile": {"enabled": False}}})


def test_engine_explicit_routing_and_absent_default():
    architecture = CaptureArchitecture()
    engine = JointInferenceEngine(engine_config(), architecture=architecture)
    engine.generate({})
    assert "research_context" not in architecture.requests[-1]
    context = ResearchContext(Intervention((0,), (0,)))
    engine.generate({"research_context": context})
    assert architecture.requests[-1]["research_context"] is context
    engine.generate({"research_context": {"layer_ids": [0], "step_ids": [0], "gate": 0}})
    assert isinstance(architecture.requests[-1]["research_context"], ResearchContext)


@pytest.mark.parametrize("unsupported", ["cache", "compile", "async_execution", "async_denoise", "cfg", "sampler"])
def test_engine_rejects_incompatible_research(unsupported):
    engine = JointInferenceEngine(engine_config(), architecture=CaptureArchitecture())
    conditions = {"research_context": ResearchContext(Intervention((0,), (0,)))}
    if unsupported == "cache":
        engine._dit_cache = object()
    elif unsupported == "compile":
        engine.cfg.optimization.compile.enabled = True
    elif unsupported == "async_execution":
        engine.cfg.inference.inference_mode = "async"
    elif unsupported == "async_denoise":
        conditions["denoise_mode"] = "async"
    elif unsupported == "cfg":
        engine._cfg_scale = 2
    else:
        engine._joint_sampler = object()
    with pytest.raises(ValueError, match="Research requires"):
        engine.generate(conditions)


class TinyResearchArchitecture(DualSystemSelfAttnArchitecture):
    def __init__(self):
        _TinyGenerateArchitecture.__init__(self, None)
        self.video_backbone.num_layers = 2
        self._compiled_mot_run_joint_loop = None
        self.received = []
        self.video_backbone.preprocess_input_for_inference = lambda **kwargs: {
            "latents": torch.zeros(1, 1, 3, 1, 1), "first_frame_latents": torch.full((1, 1, 1, 1, 1), 7.0)}

    def forward(self, actions, action_time, **kwargs):
        self.received.append(kwargs)
        assert torch.equal(kwargs["latents"][:, :, :1], kwargs["first_frame_latents"])
        return torch.ones_like(kwargs["latents"]), torch.ones_like(actions) * 0.1


def test_generate_step_context_and_default_path_equivalence():
    architecture = TinyResearchArchitecture()
    schedule = [(1000, 1000), (500, 700), (0, 0)]
    context = ResearchContext(Intervention((0,), (0, 1)))
    raw = architecture.generate(schedule, "task", num_frames=4, decode_video=False, seed=4)
    assert all("research_step" not in request for request in architecture.received)
    architecture.received.clear()
    result = architecture.generate(schedule, "task", num_frames=4, decode_video=False, seed=4, research_context=context)
    np.testing.assert_array_equal(raw["actions"], result["actions"])
    assert [request["research_step"].index for request in architecture.received] == [0, 1]
    assert architecture.received[1]["research_step"].sigma_video == 0.5
    assert architecture.received[1]["research_step"].sigma_action == 0.7
    assert len(context.predictions) == 2
    assert context.normalized_actions.shape == (3, 20)


def test_generate_rejects_non_joint_architecture():
    architecture = _TinyGenerateArchitecture(None)
    with pytest.raises(ValueError, match="joint_self_attn"):
        architecture.generate([(1000, 1000), (0, 0)], "", decode_video=False,
                              research_context=ResearchContext(Intervention((0,), (0,))))


def test_driver_single_layer_direction_and_gate_one():
    video = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    action = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    video.eval()
    action.eval()
    driver = DualSystemMoTDriver(video, action, attention_mask_mode="mutual")
    with torch.no_grad():
        vstate, astate = _make_states(video, action, B=1, s_video=6, s_action=3)
        mask = driver._build_attention_mask(6, 3, 1, device=torch.device("cpu"))
        reference_v, reference_a = driver.step(0, copy.deepcopy(vstate), copy.deepcopy(astate), mask)
        context = ResearchContext(Intervention((0,), (0,), gate=0))
        changed_v, changed_a = driver.step(0, copy.deepcopy(vstate), copy.deepcopy(astate), mask,
                                          research_step=context.step(0, 1, 1, 1, True, True))
        torch.testing.assert_close(reference_v.hidden_states, changed_v.hidden_states, rtol=0, atol=0)
        assert not torch.equal(reference_a.payload.x_action, changed_a.payload.x_action)
        context = ResearchContext(Intervention((0,), (0,), gate=1))
        same_v, same_a = driver.step(0, copy.deepcopy(vstate), copy.deepcopy(astate), mask,
                                    research_step=context.step(0, 1, 1, 1, True, True))
        torch.testing.assert_close(reference_v.hidden_states, same_v.hidden_states)
        torch.testing.assert_close(reference_a.payload.x_action, same_a.payload.x_action)
