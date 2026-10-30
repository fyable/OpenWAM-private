"""Validate CPU-side simulator dependencies without requiring GPU rendering."""

import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    egl = Path(sys.prefix) / "lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    if not os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") and not os.environ.get("__EGL_VENDOR_LIBRARY_DIRS") and egl.is_file():
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl)
    report = {"gpu_rendering": "not_run", "packages": {}, "assets": {}}
    for name in ("torch", "sapien", "mplib", "warp-lang", "numpy", "websockets", "PyYAML"):
        report["packages"][name] = importlib.metadata.version(name)
    for name in ("torch", "sapien", "mplib", "warp", "websockets", "yaml", "transforms3d", "trimesh", "open3d"):
        module = importlib.import_module(name)
        print(f"{name}: {module.__file__}")
    import torch

    for name in ("geom_cu", "kinematics_fused_cu", "lbfgs_step_cu", "line_search_cu", "tensor_step_cu"):
        importlib.import_module(f"curobo.curobolib.{name}")
    report["curobo_extensions"] = "all_five_loaded"
    repo = ROOT / "third_party/RoboTwin"
    for relative in ("script/eval_policy.py", "assets/embodiments/aloha-agilex", "assets/objects",
                     "assets/background_texture", "task_config/demo_clean.yml"):
        path = repo / relative
        if not path.exists():
            raise FileNotFoundError(path)
        report["assets"][relative] = str(path.resolve())
    import yaml

    for name in ("curobo_left.yml", "curobo_right.yml"):
        config = yaml.safe_load((repo / "assets/embodiments/aloha-agilex" / name).read_text())
        for key in ("urdf_path", "collision_spheres"):
            path = Path(config["robot_cfg"]["kinematics"][key])
            if not path.is_file():
                raise FileNotFoundError(path)
    report["torch_cuda_version"] = torch.version.cuda
    report["status"] = "cpu_dependencies_passed"
    output = ROOT / "outputs/robotwin/environment_check.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"CPU dependency checks passed. GPU simulation remains unverified. Report: {output}")


if __name__ == "__main__":
    main()
