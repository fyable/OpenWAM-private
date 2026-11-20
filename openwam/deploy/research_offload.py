"""Opt-in research-only weight staging. All module computation stays on CUDA."""

from __future__ import annotations

import torch


def configure_text_offload(architecture, dtype, device):
    if device.type != "cuda":
        raise ValueError("Text offload requires CUDA computation")
    video = architecture.video_backbone
    encoder = getattr(video, "text_encoder", None)
    if encoder is None or getattr(video, "dit", None) is None:
        raise ValueError("Text offload requires a Wan text encoder and denoiser")
    # Keep the encoder out of the initial recursive CUDA move. It is the largest
    # module not needed by the denoising loop, and still runs in bf16 on CUDA.
    del video._modules["text_encoder"]
    try:
        encoder.to(dtype=dtype, device="cpu")
        architecture.set_dtype_device(dtype, device)
    finally:
        video.text_encoder = encoder
    denoisers = [video.dit, architecture.action_backbone]

    def before(module, inputs):
        for denoiser in denoisers:
            denoiser.to(device="cpu")
        torch.cuda.empty_cache()
        module.to(device=device)

    def after(module, inputs, output):
        module.to(device="cpu")
        torch.cuda.empty_cache()
        for denoiser in denoisers:
            denoiser.to(device=device)
        return output

    encoder.register_forward_pre_hook(before)
    encoder.register_forward_hook(after, always_call=True)
