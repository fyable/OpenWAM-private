"""Separate operator-level profiling; overlapping regions are never summed."""

from contextlib import contextmanager

import torch


@contextmanager
def operator_profile(engine, device):
    from benchmarks.video_parallel.profiling import boundaries
    video = engine.architecture.video_backbone
    targets = []
    for index, block in enumerate(video.dit.blocks):
        for name in ("q", "k", "v", "o"):
            targets.append((getattr(block.self_attn, name), f"video_attention_{name}", index))
        targets.append((block.ffn, "video_mlp", index))
        targets.append((block.cross_attn, "video_text_cross_attention", index))
    events, handles = [], []
    result = {"operator_ms": {}, "operator_calls": {}, "layer_operator_ms": {}}
    for module, label, layer in targets:
        started = []
        def before(owner, args, _stack=started):
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(device))
            _stack.append(event)
        def after(owner, args, output, _stack=started, _label=label, _layer=layer):
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(device))
            events.append((_label, _layer, _stack.pop(), event))
        handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
    try:
        with boundaries(engine, device) as inclusive:
            yield result
        result["inclusive_regions"] = inclusive
        torch.cuda.synchronize(device)
        for label, layer, begin, end in events:
            elapsed = begin.elapsed_time(end)
            result["operator_ms"][label] = result["operator_ms"].get(label, 0.) + elapsed
            result["operator_calls"][label] = result["operator_calls"].get(label, 0) + 1
            key = f"{layer}:{label}"
            result["layer_operator_ms"][key] = result["layer_operator_ms"].get(key, 0.) + elapsed
        result["interpretation"] = "Instrumented native operator attribution only; do not sum inclusive and operator regions or substitute for uninstrumented request timing"
    finally:
        for handle in handles:
            handle.remove()
