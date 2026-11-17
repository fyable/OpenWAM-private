"""Separate instrumented pass; never add inclusive regions to obtain totals."""

from contextlib import contextmanager

import torch


@contextmanager
def boundaries(engine, device):
    architecture = engine.architecture
    video, action, driver = architecture.video_backbone, architecture.action_backbone, architecture.mot_driver
    targets = [(architecture, "forward", "joint_forward_inclusive"),
               (getattr(video, "vae", None), "encode", "vae_encode"),
               (getattr(video, "vae", None), "decode", "vae_decode")]
    for name in ("prepare", "pre_attn_at_layer", "post_attn_at_layer", "finalize"):
        targets.append((video, name, "video_" + name))
    for name in ("prepare_state", "pre_attn_at_layer", "post_attn_at_layer", "extract_prediction"):
        targets.append((action, name, "action_" + name))
    targets.append((driver, "_mixed_attention", "shared_mixed_attention"))
    originals, events, result = [], [], {"regions_ms": {}, "calls": {}}
    try:
        for owner, name, label in targets:
            if owner is None or not callable(getattr(owner, name, None)):
                continue
            original = getattr(owner, name)
            originals.append((owner, name, name in owner.__dict__, owner.__dict__.get(name)))

            def measured(*args, _original=original, _label=label, **kwargs):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(torch.cuda.current_stream(device))
                with torch.profiler.record_function("video_parallel." + _label):
                    output = _original(*args, **kwargs)
                end.record(torch.cuda.current_stream(device))
                events.append((_label, begin, end))
                return output

            setattr(owner, name, measured)
        yield result
        torch.cuda.synchronize(device)
        for label, begin, end in events:
            result["regions_ms"][label] = result["regions_ms"].get(label, 0.) + begin.elapsed_time(end)
            result["calls"][label] = result["calls"].get(label, 0) + 1
        result["interpretation"] = "Separate instrumented pass; inclusive joint_forward overlaps other regions. Missing regions are unmeasured, not zero."
    finally:
        for owner, name, present, previous in reversed(originals):
            if present:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
