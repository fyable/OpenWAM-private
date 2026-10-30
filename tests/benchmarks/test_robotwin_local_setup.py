import importlib.util
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_local_configure_is_idempotent_and_keeps_shared_assets_unchanged(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("configure_robotwin_local", ROOT / "scripts/configure_robotwin_local.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    repository = tmp_path / "third_party/RoboTwin"
    repository.mkdir(parents=True)
    source = tmp_path / "shared-assets"
    robot = source / "embodiments/aloha-agilex"
    (robot / "urdf").mkdir(parents=True)
    (robot / "urdf/robot.urdf").write_text("robot")
    (robot / "collision.yml").write_text("spheres: []")
    config = {"robot_cfg": {"kinematics": {
        "urdf_path": "/old/assets/embodiments/aloha-agilex/urdf/robot.urdf",
        "collision_spheres": "/old/assets/embodiments/aloha-agilex/collision.yml",
    }}}
    for name in ("curobo_left.yml", "curobo_right.yml"):
        (robot / name).write_text(yaml.safe_dump(config))
    (source / "objects").mkdir()
    (repository / "assets").symlink_to(source)
    packages = tmp_path / ".venv-robotwin/lib/python3.10/site-packages"
    (packages / "sapien/wrapper").mkdir(parents=True)
    urdf_loader = packages / "sapien/wrapper/urdf_loader.py"
    urdf_loader.write_text('open(urdf_file, "r")\nurdf_file[:-4] + "srdf"\n')
    (packages / "mplib").mkdir()
    (packages / "mplib/planner.py").write_text(
        "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:\n"
    )
    (packages / "__editable__.nvidia_curobo.pth").write_text("/other-user/curobo")
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    monkeypatch.setattr(setup, "ENV", tmp_path / ".venv-robotwin")
    monkeypatch.setattr(setup, "REPO", repository)
    monkeypatch.setattr(setup, "PACKAGES", packages)
    before = (robot / "curobo_left.yml").read_bytes()
    setup.main()
    setup.main()
    assert (robot / "curobo_left.yml").read_bytes() == before
    assert not (repository / "assets").is_symlink()
    assert (repository / "assets/objects").resolve() == source / "objects"
    local_config = yaml.safe_load((repository / "assets/embodiments/aloha-agilex/curobo_left.yml").read_text())
    assert Path(local_config["robot_cfg"]["kinematics"]["urdf_path"]).is_file()
    assert str(tmp_path) in local_config["robot_cfg"]["kinematics"]["collision_spheres"]
    assert not (packages / "__editable__.nvidia_curobo.pth").exists()
    assert (packages / "openwam_robotwin.pth").read_text().strip() == str(repository / "envs/curobo/src")
    assert 'encoding="utf-8"' in urdf_loader.read_text()
    env = dict(os.environ, ROBOTWIN_PATH="/explicit/override", CKPT_DIR="/explicit/weights")
    env.pop("ROBOTWIN_PYTHON", None)
    result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s\\n%s\\n%s" "$ROBOTWIN_PATH" "$CKPT_DIR" "$ROBOTWIN_PYTHON"',
                             "bash", str(tmp_path / "outputs/robotwin/environment.sh")],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == ["/explicit/override", "/explicit/weights",
                                         str(tmp_path / ".venv-robotwin/bin/python")]
