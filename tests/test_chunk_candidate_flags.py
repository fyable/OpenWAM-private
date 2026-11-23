from types import SimpleNamespace

import numpy as np
import torch

from openwam.deploy.chunk_pipeline import Chunk, ChunkWorker
from openwam.model.video_backbone import wan_backbone
from openwam.model.video_backbone.wan.preprocess import vae_output_to_video
from tests.test_chunk_pipeline import make_request
from openwam.deploy.chunk_graph import signature, update


def test_selected_frames_keep_full_temporal_decode_and_exact_pixels(monkeypatch):
    video = torch.linspace(-1.1, 1.1, 3 * 9 * 4 * 4).reshape(1, 3, 9, 4, 4)
    latent = torch.zeros(1, 16, 3, 1, 1)
    observed = []

    def decode(value, **kwargs):
        observed.append(value)
        return video

    monkeypatch.setattr(wan_backbone.wan_encode, 'decode_latents', decode)
    backbone = SimpleNamespace(video_encoder=None, vae=None, device='cpu')
    actual = wan_backbone.WanBase.decode_selected_video_frames(backbone, latent, indices=(2, 4, 6))
    reference = vae_output_to_video(video)
    assert observed == [latent]
    for frame, index in zip(actual, (2, 4, 6)):
        np.testing.assert_array_equal(np.asarray(frame), np.asarray(reference[index]))


def test_selected_frames_external_encoder_uses_original_conversion():
    backbone = SimpleNamespace(video_encoder=object(), decode_video=lambda *args, **kwargs: list(range(9)))
    result = wan_backbone.WanBase.decode_selected_video_frames(backbone, None, indices=(2, 4, 6))
    assert result == [2, 4, 6]


def test_chunk_selected_frame_offsets(monkeypatch):
    monkeypatch.setenv('OPENWAM_CHUNK_SELECTED_FRAMES', '1')
    model = SimpleNamespace(video_backbone=SimpleNamespace(decode_selected_video_frames=lambda *args, **kwargs: [2, 4, 6]))
    assert ChunkWorker(model).decode(Chunk(8, 1), make_request()) == {16: 2, 24: 4, 32: 6}


def test_graph_signature_checks_structure_not_tensor_content():
    first = {'value': torch.ones(2), 'mode': 'fixed'}
    second = {'value': torch.zeros(2), 'mode': 'fixed'}
    assert signature(first) == signature(second)
    assert signature(first) != signature(dict(second, value=torch.zeros(3)))
    update(first, second)
    assert torch.equal(first['value'], second['value'])


def test_reuse_noise_preserves_cold_start_and_existing_storage(monkeypatch):
    monkeypatch.setenv('OPENWAM_CHUNK_REUSE_NOISE', '1')
    calls = []
    reference = torch.ones(1, 2, 1, 2, 2)
    monkeypatch.setattr(wan_backbone, 'check_resize_height_width', lambda height, width, frames, **kwargs: (height, width, frames))
    monkeypatch.setattr(wan_backbone.wan_encode, 'encode_text_for_inference', lambda *args, **kwargs: (torch.zeros(1), torch.ones(1)))
    monkeypatch.setattr(wan_backbone.wan_encode, 'preprocess_video', lambda *args, **kwargs: reference)
    monkeypatch.setattr(wan_backbone.wan_encode, 'encode_video', lambda *args, **kwargs: reference)

    def noise(**kwargs):
        calls.append(kwargs['seed'])
        return torch.zeros(1, 2, 3, 2, 2)

    monkeypatch.setattr(wan_backbone.wan_conditioning, 'build_deploy_noise', noise)
    monkeypatch.setattr(wan_backbone.wan_conditioning, 'build_vace_context_for_deploy', lambda *args, **kwargs: None)
    backbone = SimpleNamespace(
        scheduler=SimpleNamespace(set_timesteps=lambda **kwargs: None),
        _height_division_factor=8, _width_division_factor=8,
        _time_division_factor=4, _time_division_remainder=1,
        _tokenizer=None, text_encoder=None, device='cpu', dtype=torch.float32,
        _latent_spec=None, vae=None, _dit=SimpleNamespace(has_image_input=False),
        _is_ti2v=True, _has_vace=False, video_encoder=None)
    existing = torch.ones(1, 2, 3, 2, 2)
    kwargs = dict(prompt='test', first_frame_image=[100], height=16, width=16, num_frames=9)
    result = wan_backbone.WanBase.preprocess_input_for_inference(backbone, **kwargs, initial_latents=existing)
    assert result['latents'] is existing and result['noise'] is existing
    assert calls == []
    cold = wan_backbone.WanBase.preprocess_input_for_inference(backbone, **kwargs)
    assert calls == [42]
    assert cold['latents'].sum() == 0
    wan_backbone.WanBase.preprocess_input_for_inference(backbone, **kwargs, initial_latents=existing, seed=None)
    assert calls == [42, None]


def test_cached_timestep_tensors_reused_without_changing_outputs(monkeypatch):
    monkeypatch.setenv('OPENWAM_CHUNK_TIMESTEP_CACHE', '1')
    times = []

    def forward(action, action_time, **inputs):
        times.append((action_time, inputs['timestep']))
        return torch.ones_like(inputs['latents']), torch.ones_like(action)

    model = SimpleNamespace(device='cpu', dtype=torch.float32,
                            video_scheduler=SimpleNamespace(num_train_timesteps=10),
                            action_scheduler=SimpleNamespace(num_train_timesteps=10,
                                flow_step=lambda flow, current, following, action: action + flow * (following - current)),
                            forward=forward)
    worker = ChunkWorker(model)
    outputs = []
    for _ in range(2):
        chunk = Chunk(0, 0, video=torch.zeros(1, 2, 3, 2, 2), action=torch.zeros(1, 32, 7))
        worker.step(chunk, make_request(), ({'first_frame_latents': torch.ones(1, 2, 1, 2, 2)}, None))
        outputs.append(chunk.video)
    assert times[0][0] is times[1][0] and times[0][1] is times[1][1]
    torch.testing.assert_close(outputs[0], outputs[1])
