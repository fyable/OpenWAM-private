"""Request-scoped sharing of identical stage conditioning projections."""

from contextlib import contextmanager

import torch


@contextmanager
def shared_stage_conditions(architecture):
    originals = []
    stats = {"shared_calls": 0, "fallback_calls": 0}
    targets = []
    for name, module in architecture.named_modules():
        if name.rsplit(".", 1)[-1] == "text_embedding":
            targets.append((module, True))
        elif name.rsplit(".", 1)[-1] == "cross_attn":
            for child_name in ("k", "v", "norm_k", "k_img", "v_img", "norm_k_img"):
                child = getattr(module, child_name, None)
                if isinstance(child, torch.nn.Module):
                    targets.append((child, False))
    seen = set()
    for module, verify in targets:
        if id(module) in seen:
            continue
        seen.add(id(module))
        present = "forward" in module.__dict__
        previous = module.__dict__.get("forward")
        original = module.forward

        def forward(value, *args, _original=original, _verify=verify, _module=module, **kwargs):
            eligible = (not _module.training and not torch.is_grad_enabled()
                        and torch.is_tensor(value) and value.ndim == 3 and value.shape[0] > 1
                        and not args and not kwargs)
            shared = eligible and (value.stride(0) == 0 or (
                _verify and torch.equal(value, value[:1].expand_as(value))))
            if shared:
                stats["shared_calls"] += 1
                result = _original(value[:1])
                return result.expand(value.shape[0], *result.shape[1:])
            stats["fallback_calls"] += 1
            return _original(value, *args, **kwargs)

        originals.append((module, present, previous))
        module.forward = forward
    try:
        yield stats
    finally:
        for module, present, previous in reversed(originals):
            if present:
                module.forward = previous
            else:
                del module.forward
