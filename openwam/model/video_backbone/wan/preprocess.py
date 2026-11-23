"""Stateless pixel/latent preprocessing for the Wan backbone.

Pure functions lifted verbatim from the former ``BasePipeline`` helpers so the
backbone owns its deploy preprocessing without a pipeline object. ``dtype`` /
``device`` (and the division factors for :func:`check_resize_height_width`) are
passed explicitly instead of read off ``self`` — same numerics, no hidden state.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import reduce, repeat
from PIL import Image

from openwam.chunk_diagnostics import context, profiled, region


@profiled('image_preprocess')
def preprocess_image(image, *, dtype, device, pattern="B C H W", min_value=-1, max_value=1):
    """PIL.Image -> tensor in ``[min_value, max_value]`` on ``(dtype, device)``."""
    with region('image_numpy_tensor', gpu=False):
        image = torch.Tensor(np.array(image, dtype=np.float32))
    with region('image_h2d_call'):
        image = image.to(dtype=dtype, device=device)
    image = image * ((max_value - min_value) / 255) + min_value
    image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
    return image


def preprocess_video(video, *, dtype, device, pattern="B C T H W", min_value=-1, max_value=1):
    """List[PIL.Image] -> stacked tensor along the ``T`` axis of ``pattern``."""
    video = [
        preprocess_image(image, dtype=dtype, device=device, min_value=min_value, max_value=max_value) for image in video
    ]
    video = torch.stack(video, dim=pattern.index("T") // 2)
    return video


@profiled('decoded_frame')
def _vae_output_to_image(vae_output, pattern="B C H W", min_value=-1, max_value=1):
    if pattern != "H W C":
        vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
    with region('pixel_scale_clamp'):
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
    with region('frame_d2h_cast_call'):
        image = image.to(device="cpu", dtype=torch.uint8)
    with region('frame_numpy_pil', gpu=False):
        image = Image.fromarray(image.numpy())
    return image


def vae_output_to_video(vae_output, pattern="B C T H W", min_value=-1, max_value=1):
    """VAE pixel tensor -> list[PIL.Image]."""
    if pattern != "T H W C":
        vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
    frames = []
    for frame_index, image in enumerate(vae_output):
        with context(frame_index=frame_index, selected_for_future=frame_index in (2, 4, 6)):
            frames.append(_vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value))
    return frames


def check_resize_height_width(
    height,
    width,
    num_frames=None,
    *,
    height_division_factor,
    width_division_factor,
    time_division_factor,
    time_division_remainder,
):
    """Round ``height``/``width``/``num_frames`` up to the backbone's model grid.

    ``time_division_factor == 1`` (per-frame encoders) skips the temporal check,
    which would otherwise bump ``num_frames`` by 1 every call (any ``N % 1 == 0``).
    """
    if height % height_division_factor != 0:
        height = (height + height_division_factor - 1) // height_division_factor * height_division_factor
        print(f"height % {height_division_factor} != 0. We round it up to {height}.")
    if width % width_division_factor != 0:
        width = (width + width_division_factor - 1) // width_division_factor * width_division_factor
        print(f"width % {width_division_factor} != 0. We round it up to {width}.")
    if num_frames is None:
        return height, width
    if time_division_factor > 1 and (num_frames % time_division_factor != time_division_remainder):
        num_frames = (
            num_frames + time_division_factor - 1
        ) // time_division_factor * time_division_factor + time_division_remainder
        print(f"num_frames % {time_division_factor} != {time_division_remainder}. We round it up to {num_frames}.")
    return height, width, num_frames


@profiled('noise_total')
def generate_noise(shape, *, dtype, device, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32):
    """Seeded Gaussian noise built on ``rand_device`` then moved to ``(dtype, device)``."""
    generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
    with region('noise_generate', gpu=rand_device != 'cpu'):
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
    with region('noise_h2d_call'):
        return noise.to(dtype=dtype, device=device)
