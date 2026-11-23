"""Optional fixed-shape graph for the full causal VAE decoder core."""

import torch


class DecodeGraph:
    def __init__(self, model, latent, scale):
        self.model = model
        self.latent = latent.clone()
        self.scale = [value.to(device=latent.device, dtype=latent.dtype).clone() for value in scale]
        self.key = (tuple(latent.shape), latent.dtype, latent.device)
        if torch.cuda.mem_get_info(latent.device)[0] < 8 * 1024 ** 3:
            raise RuntimeError('Insufficient VAE graph GPU headroom; no CPU fallback')
        stream = torch.cuda.Stream(device=latent.device)
        stream.wait_stream(torch.cuda.current_stream(latent.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.output = model.decode(self.latent, self.scale)
        torch.cuda.current_stream(latent.device).wait_stream(stream)
        torch.cuda.synchronize(latent.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = model.decode(self.latent, self.scale)

    def replay(self, latent):
        if (tuple(latent.shape), latent.dtype, latent.device) != self.key:
            raise ValueError('VAE graph shape changed')
        self.latent.copy_(latent)
        self.graph.replay()
        return self.output
