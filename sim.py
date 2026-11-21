"""Run one Dream-1 task in the Dream Labs simulator (headless) and save its trajectory.

    python sim.py "put apple into the bowl" --seconds 60

Every run is saved in runs/<id>/:
  meta.json        task, time limit, start and end, whether it was stopped
  trajectory.json  every control step (20 per simulated second): where each object, the bowl and the plate are,
                   and whether a gripper fingertip touches each object
  last.jpg         the cameras at the end of the run

app.py starts this for every run. Environment variables connect it to the website:
  RUN_DIR     the folder to save the run in (default: runs/<new id>)
  LIVE_FRAME  path of a JPEG that is replaced with the latest camera view (top camera + both wrist cameras)
  STOP_FILE   when this file appears, the run stops and the Dream session is released
"""

import argparse
import importlib.util
import json
import os
import secrets
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# The simulator ships inside the dreamlabs package (dream_examples/yam_mujoco) and imports its modules by name.
spec = importlib.util.find_spec("dream_examples")
if spec is None:
    sys.exit("The simulator is missing: run  pip install dreamlabs")
sys.path.insert(0, str(Path(next(iter(spec.submodule_search_locations))) / "yam_mujoco"))

import dream  # noqa: E402
from dream import credentials as dream_credentials  # noqa: E402
import dream_control  # noqa: E402
import yam_env  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import simulator_settings as sim_settings

HERE = Path(__file__).resolve().parent
LIVE_FRAME = os.environ.get("LIVE_FRAME")
STOP_FILE = os.environ.get("STOP_FILE")
OBJECTS = ["apple", "orange", "banana", "water_bottle"]
TARGETS = ["bowl", "plate"]
FINGERTIPS = [f"{side}_{part}" for side in ("left", "right") for part in ("tip_left", "tip_right", "gripper")]


def new_run_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def camera_view(images):
    """The top camera on the left, the two wrist cameras stacked on the right, as one image."""
    canvas = Image.new("RGB", (720, 270))
    canvas.paste(Image.fromarray(images["top"]).resize((480, 270)), (0, 0))
    canvas.paste(Image.fromarray(images["left"]).resize((240, 135)), (480, 0))
    canvas.paste(Image.fromarray(images["right"]).resize((240, 135)), (480, 135))
    return canvas


def save_jpeg(image, path):
    """Write a JPEG so that readers never see half a file."""
    image.save(str(path) + ".tmp", format="JPEG", quality=80)
    os.replace(str(path) + ".tmp", path)


class Trajectory:
    """Positions of the objects and targets, and fingertip contact, after every control step."""

    def __init__(self):
        self.t = []
        self.positions = {name: [] for name in OBJECTS + TARGETS}
        self.touching = {name: [] for name in OBJECTS}
        self.ids = None

    def record(self, env):
        m, d = env.model, env.data
        if self.ids is None:
            self.ids = {name: m.body(name).id for name in OBJECTS + TARGETS}
            self.tips = {m.body(name).id for name in FINGERTIPS}
            self.object_of = {self.ids[name]: name for name in OBJECTS}
        touching = set()
        for contact in d.contact[: d.ncon]:
            a, b = m.geom_bodyid[contact.geom1], m.geom_bodyid[contact.geom2]
            if a in self.tips and b in self.object_of:
                touching.add(self.object_of[b])
            elif b in self.tips and a in self.object_of:
                touching.add(self.object_of[a])
        self.t.append(round(float(d.time), 3))
        for name, body in self.ids.items():
            self.positions[name].append([round(float(v), 4) for v in d.xpos[body]])
        for name in OBJECTS:
            self.touching[name].append(int(name in touching))

    def to_json(self):
        return {"t": self.t, "positions": self.positions, "touching": self.touching}


TRAJECTORY = Trajectory()
STATS = {"model_responses": 0, "model_actions": 0}

# Record the scene after every control step.
original_step = yam_env.YamEnv.step


def step(self, left, right):
    result = original_step(self, left, right)
    TRAJECTORY.record(self)
    if STATS["model_responses"]:
        STATS["model_actions"] += 1
    return result


# Render the cameras only on the steps the model sees (1 in 4); the others reuse the last images. Much faster.
original_capture = dream_control.capture
last_images = {}


def capture(env, windows, keep=True):
    if keep or not last_images:
        last_images.update(original_capture(env, windows, keep))
        if LIVE_FRAME:
            save_jpeg(camera_view(last_images), LIVE_FRAME)
    return dict(last_images)


# A connection the server dropped while idle is reopened by asking again (up to twice).
original_get_action = dream.DreamClient.get_action


def get_action(self, *args, **kwargs):
    for attempt in range(3):
        try:
            reply = original_get_action(self, *args, **kwargs)
            STATS["model_responses"] += 1
            return reply
        except dream.DreamProtocolError:
            if attempt == 2:
                raise
            print("  Connection dropped; asking again.", flush=True)
            time.sleep(0.2)


yam_env.YamEnv.step = step
dream_control.capture = capture
dream.DreamClient.get_action = get_action


def apply_lighting(env, lighting):
    """Change rendering only; physics, cameras, and object geometry stay fixed."""
    if lighting == "standard":
        return
    palette = {
        "warm": ([0.32, 0.25, 0.18], [0.52, 0.39, 0.28],
                 [0.72, 0.56, 0.42], [0.54, 0.43, 0.34], [0.83, 0.76, 0.69, 1]),
        "cool": ([0.18, 0.22, 0.30], [0.30, 0.39, 0.55],
                 [0.42, 0.52, 0.68], [0.35, 0.46, 0.62], [0.68, 0.75, 0.84, 1]),
    }
    ambient, diffuse, key_light, fill_light, floor = palette[lighting]
    env.model.vis.headlight.ambient[:] = ambient
    env.model.vis.headlight.diffuse[:] = diffuse
    env.model.light_diffuse[0] = key_light
    env.model.light_diffuse[1] = fill_light
    for geom_id in range(env.model.ngeom):
        if env.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            env.model.geom_rgba[geom_id] = floor


def main():
    parser = argparse.ArgumentParser(description="Run one Dream-1 task in the simulator")
    parser.add_argument("task", nargs="?", default="put apple into the bowl")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, help="Seed for reproducible object variation")
    parser.add_argument("--layout", choices=sim_settings.LAYOUTS, default="fixed")
    parser.add_argument("--lighting", choices=sim_settings.LIGHTING, default="standard")
    args = parser.parse_args()
    if args.layout == "varied" and args.seed is None:
        parser.error("--layout varied requires --seed")
    if args.layout == "fixed" and args.seed is not None:
        parser.error("--seed requires --layout varied")

    # Match the key detected by the web app. The bundled sim otherwise gives a
    # separate, possibly stale /login key priority over ~/.dreamlabs/credentials.
    key = dream_credentials.load().api_key
    if key:
        os.environ["DREAM_API_KEY"] = key

    run_dir = Path(os.environ.get("RUN_DIR") or HERE / "runs" / new_run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"id": run_dir.name, "task": args.task, "seconds": args.seconds, "seed": args.seed,
            "robot": "yam", "room": "yam_tabletop", "layout": args.layout, "lighting": args.lighting,
            "started_at": datetime.now().isoformat(timespec="seconds"), "stopped_by_user": False,
            "error": None, "error_type": None}

    env = yam_env.YamEnv()
    apply_lighting(env, args.lighting)
    if args.layout == "varied":
        rng = np.random.default_rng(args.seed)
        for body, joint in (("apple", "apple_free"), ("orange", "orange_free"),
                            ("banana", "banana_free"), ("water_bottle", "bottle_free")):
            qpos = env.data.joint(joint).qpos
            qpos[:2] += rng.uniform(-0.02, 0.02, 2)
            if body != "water_bottle":
                yaw = np.deg2rad(rng.uniform(-25, 25)) / 2
                rotated = np.empty(4)
                mujoco.mju_mulQuat(rotated, np.array([np.cos(yaw), 0, 0, np.sin(yaw)]), qpos[3:7].copy())
                qpos[3:7] = rotated
        mujoco.mj_forward(env.model, env.data)

    stop = threading.Event()
    if STOP_FILE:
        def watch():
            while not stop.is_set():
                if os.path.exists(STOP_FILE):
                    print("  Stopped by user.", flush=True)
                    meta["stopped_by_user"] = True
                    stop.set()
                time.sleep(0.3)

        threading.Thread(target=watch, daemon=True).start()

    print(f"Starting the simulator: {args.task} ({args.seconds:g} s)", flush=True)
    try:
        dream_control.main(["--task", args.task, "--seconds", str(args.seconds), "--headless", "--quiet"],
                           shared_env=env, stop_event=stop)
    except Exception as error:
        meta["error"] = str(error)
        meta["error_type"] = type(error).__name__
        print(f"Stopped: {error}", flush=True)
    finally:
        stop.set()
        meta["ended_at"] = datetime.now().isoformat(timespec="seconds")
        meta["sim_seconds"] = TRAJECTORY.t[-1] if TRAJECTORY.t else 0
        meta.update(STATS)
        (run_dir / "trajectory.json").write_text(json.dumps(TRAJECTORY.to_json(), separators=(",", ":")))
        if last_images:
            save_jpeg(camera_view(last_images), run_dir / "last.jpg")
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        env.close()
        print(f"Saved to runs/{run_dir.name}", flush=True)
    sys.exit(1 if meta["error"] else 0)


if __name__ == "__main__":
    main()
