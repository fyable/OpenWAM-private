"""Deployment adapter for a frozen cache experiment, never enabled by default."""

import json
import os


def main():
    from openwam.deploy import server
    from openwam.research.protocol import read_sealed
    from .runtime import CacheConfig, install

    case = json.loads(os.environ["ACTION_CACHE_CASE"])
    seed = int(os.environ["ACTION_CACHE_POLICY_SEED"])
    if seed not in (43, 44, 45):
        raise ValueError("Use predeclared policy seeds 43/44/45")
    original = server.build_server_from_config
    def build(*args, **kwargs):
        policy = original(*args, **kwargs)
        sensitivities = None
        path = os.environ.get("ACTION_CACHE_SENSITIVITY")
        if path:
            record = read_sealed(path)
            if record["fitted_split"] != "discovery":
                raise ValueError("Sensitivity must be fitted on Discovery")
            sensitivities = record["sensitivities"]
        if case["cache"] is not None:
            install(policy.engine, CacheConfig(**case["cache"]), sensitivities=sensitivities, policy_seed=seed)
        else:
            generate = policy.engine.generate
            def seeded(conditions):
                return generate({**conditions, "seed": seed, "denoise_steps": case["steps"]})
            policy.engine.generate = seeded
        return policy
    server.build_server_from_config = build
    try:
        server.main()
    finally:
        server.build_server_from_config = original


if __name__ == "__main__":
    main()
