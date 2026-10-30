"""Bind the copied simulator environment exclusively to local RoboTwin code."""

import json
import shlex
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".venv-robotwin"
REPO = ROOT / "third_party/RoboTwin"
PACKAGES = ENV / "lib/python3.10/site-packages"


def main():
    assets = REPO / "assets"
    source_assets = assets.resolve()
    if assets.is_symlink():
        assets.unlink()
        assets.mkdir()
        for source in source_assets.iterdir():
            if source.name != "embodiments":
                (assets / source.name).symlink_to(source)
        embodiments = assets / "embodiments"
        embodiments.mkdir()
        for source in (source_assets / "embodiments").iterdir():
            if source.name == "aloha-agilex":
                shutil.copytree(source, embodiments / source.name, copy_function=shutil.copyfile)
            else:
                (embodiments / source.name).symlink_to(source)
    robot = assets / "embodiments/aloha-agilex"
    if robot.exists():
        import yaml

        for name in ("curobo_left.yml", "curobo_right.yml"):
            path = robot / name
            config = yaml.safe_load(path.read_text())
            kinematics = config["robot_cfg"]["kinematics"]
            for key in ("urdf_path", "collision_spheres"):
                original = str(kinematics[key])
                suffix = original.split("aloha-agilex/", 1)[-1]
                local = robot / suffix
                if not local.is_file():
                    raise FileNotFoundError(local)
                kinematics[key] = str(local)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
    for path in PACKAGES.glob("*curobo*.pth"):
        path.unlink()
    (PACKAGES / "openwam_robotwin.pth").write_text(str(REPO / "envs/curobo/src") + "\n")
    patches = {
        PACKAGES / "sapien/wrapper/urdf_loader.py": [
            ('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")'),
            ('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")'),
            ('urdf_file[:-4] + "srdf"', 'urdf_file[:-4] + ".srdf"'),
        ],
        PACKAGES / "mplib/planner.py": [
            ("if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
             "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"),
        ],
    }
    for path, replacements in patches.items():
        text = path.read_text()
        for before, after in replacements:
            text = text.replace(before, after)
        path.write_text(text)
    output = ROOT / "outputs/robotwin"
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache/robotwin"
    cache.mkdir(parents=True, exist_ok=True)
    values = {
        "ROBOTWIN_PATH": str(REPO), "ROBOTWIN_PYTHON": str(ENV / "bin/python"),
        "SERVER_PYTHON": str(ROOT / ".venv-openwam/bin/python"),
        "CKPT_DIR": "/data1/wam-ckpts/OpenWAM-Alpha-Sim-RoboTwin-Full",
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
        "WARP_CACHE_PATH": str(cache / "warp"), "XDG_CACHE_HOME": str(cache),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    (output / "environment.sh").write_text("\n".join(
        f'if [ -z "${{{key}:-}}" ]; then export {key}={shlex.quote(value)}; fi'
        for key, value in values.items()) + "\n")
    (output / "environment.json").write_text(json.dumps(values, indent=2) + "\n")
    print(f"Configured local Python: {sys.executable}")


if __name__ == "__main__":
    main()
