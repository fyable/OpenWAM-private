"""Opt-in policy server, preserving all canonical deployment CLI arguments."""

import json
import logging
import os


def main():
    from openwam.deploy import server
    from .runtime import configure_engine, method_spec

    original = server.build_server_from_config
    method = os.environ.get("VIDEO_PARALLEL_METHOD", "adaptive")
    method_spec(method)
    overrides = json.loads(os.environ.get("VIDEO_PARALLEL_OVERRIDES", "{}"))

    def build(*args, **kwargs):
        policy = original(*args, **kwargs)
        configure_engine(policy.engine, method, overrides)
        logging.getLogger(__name__).info("Research sampler=%s overrides=%s; residual acceptance is not equivalence", method, overrides)
        return policy

    server.build_server_from_config = build
    try:
        server.main()
    finally:
        server.build_server_from_config = original


if __name__ == "__main__":
    main()
