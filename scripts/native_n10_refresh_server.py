"""Enable a fixed N10 video refresh schedule in a LIBERO evaluation server."""

import logging
import os
import time

import torch

from openwam.deploy import server
from scripts.experiment_n10_cuda_graph import ActionGraph
from scripts.experiment_native_n10_kv import STEPS, context_kv
from scripts.native_n10_video_refresh import video_refresh


def main():
    schedule = tuple(int(step) for step in os.environ["N10_REFRESH_STEPS"].split(",")
                     if step)
    seed = int(os.environ.get("N10_POLICY_SEED", "43"))
    use_graph = os.environ.get("N10_ACTION_GRAPH") == "1"
    if use_graph and schedule != (0,):
        raise ValueError("Action Graph requires video refresh at step zero")
    build_server = server.build_server_from_config

    def build(*args, **kwargs):
        policy = build_server(*args, **kwargs)
        engine = policy.engine
        generate = engine.generate
        graph = ActionGraph() if use_graph else None
        last_reported_graph = None

        if use_graph:
            def invalidate_graph(_encoder, _inputs):
                nonlocal graph, last_reported_graph
                if graph.graph is not None:
                    torch.cuda.synchronize(engine.architecture.device)
                    logging.getLogger(__name__).info("N10 graph invalidated by text cache miss")
                    last_reported_graph = None
                    graph = ActionGraph()

            engine.architecture.video_backbone.text_encoder.register_forward_pre_hook(
                invalidate_graph, prepend=True)

        @torch.inference_mode()
        def generate_request(conditions):
            request = {**conditions, "seed": seed, "denoise_steps": STEPS, "denoise_mode": "sync"}
            nonlocal last_reported_graph
            start = time.perf_counter()
            if not schedule:
                result = generate(request)
            elif not use_graph:
                with video_refresh(engine.architecture, STEPS, schedule):
                    with context_kv(engine, video_forwards=len(schedule)):
                        result = generate(request)
            else:
                # Resolve the graph after text encoding, which can relocate weights.
                class CurrentGraph:
                    def run(self, *args, **kwargs):
                        return graph.run(*args, **kwargs)

                projections = {}
                with video_refresh(engine.architecture, STEPS, schedule,
                                   action_graph=CurrentGraph(), projections=projections) as stats:
                    with context_kv(engine, video_forwards=1, graph_projections=projections):
                        result = generate(request)
                if stats.get("graph_replays") != 9:
                    raise RuntimeError(f"Unexpected closed-loop graph schedule: {stats}")
            if os.environ.get("N10_RECORD_LATENCY") == "1":
                capture_ms = graph.setup_ms if graph is not None and graph is not last_reported_graph else None
                last_reported_graph = graph
                logging.getLogger(__name__).info("N10 request_ms=%.3f graph_setup_ms=%s",
                                                 (time.perf_counter() - start) * 1000, capture_ms)
            return result

        engine.generate = generate_request
        return policy

    server.build_server_from_config = build
    try:
        server.main()
    finally:
        server.build_server_from_config = build_server


if __name__ == "__main__":
    main()
