"""Fixed-shape pure forward CUDA graph; failures reject the candidate."""

import torch


def signature(value):
    if isinstance(value, torch.Tensor):
        return ('tensor', tuple(value.shape), value.dtype, str(value.device), tuple(value.stride()))
    if isinstance(value, dict):
        return tuple((key, signature(item)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(signature(item) for item in value))
    return (type(value).__name__, value)


def clone(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cuda':
            raise ValueError('Forward CUDA graph accepts only GPU tensor inputs')
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    if isinstance(value, list):
        return [clone(item) for item in value]
    return value


def update(target, source):
    if isinstance(target, torch.Tensor):
        target.copy_(source)
    elif isinstance(target, dict):
        for key in target:
            update(target[key], source[key])
    elif isinstance(target, (list, tuple)):
        for destination, value in zip(target, source):
            update(destination, value)


class ForwardGraph:
    def __init__(self, model, action, action_time, inputs):
        self.key = signature((action, action_time, inputs))
        self.arguments = clone((action, action_time, inputs))
        self.model = model
        with torch.cuda.device(model.device):
            dit = model.video_backbone.dit
            dit.freqs = tuple(value.to(model.device) for value in dit.freqs)
            if torch.cuda.mem_get_info()[0] < 8 * 1024 ** 3:
                raise RuntimeError('Insufficient GPU headroom for graph candidate; no CPU fallback')
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.output = self.call()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self.call()

    def call(self):
        action, action_time, inputs = self.arguments
        return self.model.forward(action, action_time, **inputs)

    def replay(self, action, action_time, inputs):
        if signature((action, action_time, inputs)) != self.key:
            raise ValueError('CUDA graph input signature changed')
        update(self.arguments, (action, action_time, inputs))
        self.graph.replay()
        return self.output
