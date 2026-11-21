"""OpenYAM adapter for the public Sandbox MuJoCo package and YAM Dream loop."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from local_rooms import desk_texture, themed_xml


CACHE = Path(tempfile.gettempdir()) / "sweat-dream-sandbox-scene"
HERE = Path(__file__).resolve().parent


class SandboxEnv:
    """A dual OpenYAM scene with the same 7-value-per-side control seam as YAM."""

    object_bodies = {name: f"{name}/object" for name in
                     ("apple", "orange", "banana", "water_bottle", "bowl", "plate")}
    fingertip_bodies = ("left_left_finger", "left_right_finger",
                        "right_left_finger", "right_right_finger")

    def __init__(self, room="yam_tabletop"):
        if shutil.which("node") is None:
            raise RuntimeError("OpenYAM scene setup requires Node.js on PATH")
        cache = CACHE / "openyam"
        base = cache / "scenes" / "fruit_table" / "scene.source.xml"
        if not base.exists():
            subprocess.run(["node", str(HERE / "sandbox_scene.mjs"), "openyam", str(cache)],
                           check=True, timeout=60, capture_output=True, text=True)
            base.write_bytes((cache / "scenes" / "fruit_table" / "scene.xml").read_bytes())
        self._package_dir = tempfile.TemporaryDirectory(prefix="sweat-dream-openyam-")
        folder = Path(self._package_dir.name)
        shutil.copytree(cache, folder, dirs_exist_ok=True)
        base = folder / "scenes" / "fruit_table" / "scene.source.xml"
        scene = folder / "scenes" / "fruit_table" / "scene.xml"
        if room == "yam_tabletop":
            scene.write_bytes(base.read_bytes())
        else:
            try:
                texture = desk_texture(room)
            except (OSError, ValueError) as error:
                texture = None
                print(f"Room desk texture unavailable; using catalog color: {error}", flush=True)
            scene.write_text(themed_xml(base.read_text(), room, texture))
        self.model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        result = {}
        for side in ("left", "right"):
            joints = [self.data.joint(f"{side}_joint{i}").qpos[0] for i in range(1, 7)]
            opening = sum(self.data.joint(f"{side}_{finger}_finger").qpos[0]
                          for finger in ("left", "right")) / .095
            result[side] = np.r_[joints, np.clip(opening, 0, 1)]
        return result

    def step(self, left, right):
        for side, action in (("left", left), ("right", right)):
            row = np.asarray(action, dtype=float)
            if row.shape != (7,) or not np.isfinite(row).all() or not 0 <= row[6] <= 1:
                raise ValueError("OpenYAM action needs six finite joints and a normalized gripper")
            for i, value in enumerate(row[:6], 1):
                joint = f"{side}_joint{i}"
                limits = self.model.joint(joint).range
                self.data.actuator(joint).ctrl[0] = np.clip(value, limits[0], limits[1])
            self.data.actuator(f"{side}_gripper").ctrl[0] = row[6] * .0475
        mujoco.mj_step(self.model, self.data, nstep=25)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def images(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=360, width=640)
        images = {}
        for role, name in (("top", "overhead"), ("left", "left_wrist_cam"),
                           ("right", "right_wrist_cam")):
            self.renderer.update_scene(self.data, camera=name)
            images[role] = self.renderer.render().copy()
        return images

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._package_dir.cleanup()
