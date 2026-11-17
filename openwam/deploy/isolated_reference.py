"""Opt-in current-checkpoint isolated ablation using read-only ParaFlow operators."""

import hashlib
import importlib
import importlib.util
import logging
import sys
import time
from pathlib import Path

import torch


LOGGER = logging.getLogger(__name__)
DEFAULT_PARAFLOW = "/home/zhijian/paraflow-main"
DEFAULT_NATIVE = "/data2/jikangye/research/runs/paraflow/hardware-ablation-20261017/wam-port/native/openwam"
_REFERENCE = None


def load_reference(paraflow_root=DEFAULT_PARAFLOW, native_root=DEFAULT_NATIVE):
    global _REFERENCE
    roots = (Path(paraflow_root).resolve(), Path(native_root).resolve())
    if _REFERENCE is not None:
        if _REFERENCE[0] != roots:
            raise ValueError("Cannot mix reference source roots in one process")
        return _REFERENCE[1:]
    source = roots[0] / "src"
    paths = [roots[1] / "openwam/deploy" / name for name in ("denoising_common.py", "history_denoising.py")]
    for path in [source / "paraflow/backends/openwam.py", source / "paraflow/backends/openwam_first_frame.py", *paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, module in list(sys.modules.items()):
        if name == "paraflow" or name.startswith("paraflow."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(source):
                raise RuntimeError(f"Foreign ParaFlow import already loaded: {origin}")
    aliases = {}
    previous_path = list(sys.path)
    previous_modules = set(sys.modules)
    try:
        sys.path.insert(0, str(source))
        for path in paths:
            name = f"openwam.deploy.{path.stem}"
            aliases[name] = sys.modules.get(name)
            if aliases[name] is not None:
                raise RuntimeError(f"Refusing to replace existing module: {name}")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        backend = importlib.import_module("paraflow.backends.openwam")
        first_frame = importlib.import_module("paraflow.backends.openwam_first_frame")
        importlib.import_module("paraflow.infer")
        stages = sys.modules["openwam.deploy.denoising_common"].JointStages
        owned = [module for name, module in sys.modules.items() if name == "paraflow" or name.startswith("paraflow.")]
        identity_paths = {Path(module.__file__).resolve() for module in owned if getattr(module, "__file__", None)}
        identity_paths.update(paths)
        identity_paths.add(Path(__file__).resolve())
        for root in roots:
            for name in ("LICENSE", "NOTICE"):
                if (root / name).is_file():
                    identity_paths.add(root / name)
        identity = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(identity_paths)}
        _REFERENCE = (roots, backend, first_frame.FirstFrameStages, stages, identity)
    except Exception:
        for name in set(sys.modules) - previous_modules:
            if name == "paraflow" or name.startswith("paraflow."):
                sys.modules.pop(name, None)
        raise
    finally:
        sys.path[:] = previous_path
        for name, previous in aliases.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return _REFERENCE[1:]


def configure_isolated(architecture):
    if type(architecture).__name__ != "DualSystemSelfAttnArchitecture":
        raise ValueError("Isolated reference requires DualSystemSelfAttnArchitecture")
    if getattr(architecture, "_compiled_mot_run_joint_loop", None) is not None:
        raise ValueError("Disable compile before changing experimental attention semantics")
    original = architecture.mot_driver.attention_mask_mode
    architecture._mot_driver_kwargs = dict(architecture._mot_driver_kwargs,
                                           attention_mask_mode="isolated", video_attention_mask_mode="first_frame_causal")
    architecture.build_mot_driver()
    if architecture.mot_driver.attention_mask_mode != "isolated" or architecture.video_backbone.video_attention_mask_mode != "first_frame_causal":
        raise RuntimeError("Actual isolated masks do not match requested experimental semantics")
    return original


class IsolatedReferenceSampler:
    def __init__(self, *, method, paraflow_root=DEFAULT_PARAFLOW, native_root=DEFAULT_NATIVE,
                 allow_partition_failure=False):
        if method not in ("native", "paraflow"):
            raise ValueError(method)
        self.method = method
        self.backend, self.first_frame, self.full_stages, self.identity = load_reference(paraflow_root, native_root)
        self.allow_partition_failure = allow_partition_failure
        self.checkpoint_semantics = "mutual_checkpoint_experimental_isolated_override"
        self.validation = None
        self.batch_validation = None
        self.reset()

    def reset(self):
        self.owner = None
        self.signature = None
        self.last_stats = None

    def validate_operator(self, architecture, inputs, noise, schedule, inactive):
        from benchmarks.libero.joint_experiment import tensor_error

        first = self.first_frame(architecture, inputs, schedule, inactive)
        full = self.full_stages(architecture, inputs, schedule, inactive)
        diagnostics = []
        for index in range(len(schedule) - 1):
            indices = torch.tensor([index], device=noise.device)
            actual = first.full(inputs["latents"], noise, indices)[1]
            expected = full.full(inputs["latents"], noise, indices)[1]
            diagnostics.append(tensor_error(actual, expected))
        finite = all(row["finite"] for row in diagnostics)
        passed = all(row["passed"] for row in diagnostics)
        self.validation = {"status": "passed" if passed else "failed", "rtol": .05, "atol": .05,
                           "boundary": "frozen identical candidates, first-frame versus full isolated action velocity",
                           "stages": diagnostics}
        LOGGER.info("isolated reference operator validation=%s", self.validation)
        if not finite or (not passed and not self.allow_partition_failure):
            raise RuntimeError("First-frame/full-isolated action parity failed; inspect server validation log")

    @torch.no_grad()
    def sample(self, architecture, inputs, action_noise, schedule, inactive, *, prompt, seed):
        snapshot = dict(self.__dict__)
        owner_state = None if self.owner is None else dict(self.owner.__dict__)
        generators = [] if self.owner is None else [getattr(self.owner, name, None)
                                                   for name in ("_noise_generator", "_video_noise_generator")]
        rng = [(generator, generator.get_state()) for generator in generators if generator is not None]
        original = self.first_frame.full

        def checked(operator, videos, actions, indices=None, capture=()):
            from benchmarks.libero.joint_experiment import tensor_error

            result = original(operator, videos, actions, indices, capture)
            if len(actions) > 1 and self.batch_validation is None:
                count = len(operator.block_calls)
                try:
                    stage_indices = torch.arange(len(actions), device=actions.device) if indices is None else indices
                    independent = [original(operator, videos[index:index + 1], actions[index:index + 1],
                                            stage_indices[index:index + 1], capture)[1]
                                   for index in range(len(actions))]
                    diagnostic = tensor_error(result[1], torch.cat(independent))
                    self.batch_validation = {"status": "passed" if diagnostic["passed"] else "failed",
                                             "rtol": .05, "atol": .05, "action": diagnostic}
                    LOGGER.info("isolated batch partition validation=%s", self.batch_validation)
                    if not diagnostic["finite"] or (not diagnostic["passed"] and not self.allow_partition_failure):
                        raise RuntimeError("Isolated stage batch parity failed; inspect server validation log")
                finally:
                    del operator.block_calls[count:]
            return result

        self.first_frame.full = checked
        try:
            return self._sample(architecture, inputs, action_noise, schedule, inactive, prompt=prompt, seed=seed)
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            if self.owner is not None and owner_state is not None:
                self.owner.__dict__.clear()
                self.owner.__dict__.update(owner_state)
            for generator, state in rng:
                generator.set_state(state)
            raise
        finally:
            self.first_frame.full = original

    def _sample(self, architecture, inputs, action_noise, schedule, inactive, *, prompt, seed):
        if architecture.mot_driver.attention_mask_mode != "isolated":
            raise ValueError("Isolated sampler requires actual isolated attention")
        if action_noise.shape[:2] != (1, 32):
            raise ValueError("Isolated LIBERO experiment requires H32 and one episode")
        signature = (prompt, seed, tuple(map(tuple, schedule)), tuple(action_noise.shape),
                     str(action_noise.device), str(action_noise.dtype), tuple(inputs["latents"].shape),
                     None if inactive is None else tuple(inactive.tolist()))
        if signature != self.signature:
            self.reset()
            self.signature = signature
            active_dim = action_noise.shape[-1] if inactive is None else int((~inactive).sum())
            config = self.backend.BackendConfig(
                steps=len(schedule) - 1, advance=8, active_horizon=32, active_dim=active_dim,
                method=self.method, update="affine", noise_coupling="none" if self.method == "native" else "shift",
                route="sequential" if self.method == "native" else "paraflow", visual_conditioning="first_frame",
                history_scope="action_only", accumulation_dtype="float32")
            self.owner = self.backend.ParaFlowDenoiser(config, 0, 0, 0).eval()
            if self.validation is None:
                self.validate_operator(architecture, inputs, action_noise, schedule, inactive)
        from paraflow.infer.engine import environment_switch

        if environment_switch() is not None:
            raise ValueError("Disable ParaFlow acceleration environment switches for this controlled ablation")
        generator = self.owner.action_noise_generator(seed, action_noise.device)
        noise = torch.randn(action_noise.shape, device=action_noise.device, dtype=action_noise.dtype, generator=generator)
        started = time.perf_counter()
        result = self.owner.sample(architecture, inputs, noise, schedule, inactive)
        if action_noise.is_cuda:
            torch.cuda.synchronize(action_noise.device)
        self.last_stats = dict(self.owner.last_stats, sampler=f"isolated-{self.method}",
                               operator_validation=self.validation["status"],
                               batch_partition_validation=self.batch_validation,
                               full_request_ms=(time.perf_counter() - started) * 1000,
                               checkpoint_semantics=self.checkpoint_semantics,
                               future_video_generation=False)
        LOGGER.info("isolated-reference request stats=%s", self.last_stats)
        return result
