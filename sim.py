"""Run one Dream-1 task in the Dream Labs simulator (headless) and save its trajectory.

    python sim.py "put apple into the bowl" --seconds 60                  a new random seed (printed)
    python sim.py "put apple into the bowl" --seed 42                     the start scene of seed 42 (scene.py)
    python sim.py "put apple into the bowl" --seed 42 --ood-random light,table --ood '{"blur_px": 2}'
    python sim.py --preview start.jpg --seed 42                           the start scene as a JPEG (no API call)

The start scene comes from the seed (scene.py: object layout, and the scene controls of ood.py marked random); see
SCENE_SEED.md. A custom layout from the page (SCENE_JSON) replaces the seed's layout and implies --default-scene.

Every run is saved in runs/<id>/:
  meta.json        task, time limit, start and end, whether it was stopped
  trajectory.json  every control step (20 per simulated second): where each object, the bowl and the plate are,
                   and whether a gripper fingertip touches each object
  last.jpg         the cameras at the end of the run

app.py starts this for every run. Environment variables connect it to the website:
  RUN_DIR     the folder to save the run in (default: runs/<new id>)
  SCENE_JSON  a custom layout: which objects are on the table and where: {"items": {"apple": [x, y], ...}} in
              metres (world frame); objects left out are removed. Without it, the seed's layout is used.
  BATCH_ID    set by a batch: stored in meta.json
  JUDGE_PAIRS  for a typed prompt: the object -> target pairs it is judged on, as JSON
               [{"object": "apple", "target": "bowl"}, ...] (meta.json "judge")
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
import xml.etree.ElementTree as ET
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
from PIL import Image  # noqa: E402

import ood  # noqa: E402  scene controls: camera, arm angles, blur, light, table
import scene  # noqa: E402  the seed's start scene; imported before build_xml is patched below (it keeps the original)

HERE = Path(__file__).resolve().parent
LIVE_FRAME = os.environ.get("LIVE_FRAME")
STOP_FILE = os.environ.get("STOP_FILE")
OBJECTS = ["apple", "orange", "banana", "water_bottle"]
TARGETS = ["bowl", "plate"]
SCENE = json.loads(os.environ.get("SCENE_JSON") or "{}")
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


# Build the scene the user set up: remove the objects left out, and move the others (their height stays as it is).
original_build_xml = yam_env.build_xml


def build_xml():
    xml = original_build_xml()
    items = SCENE.get("items")
    if not items:
        return xml
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    for body in list(world.findall("body")):
        name = body.get("name")
        if name not in OBJECTS + TARGETS:
            continue
        if name not in items:
            world.remove(body)
        else:
            x, y = items[name]
            z = body.get("pos").split()[2]
            body.set("pos", f"{float(x):.4f} {float(y):.4f} {z}")
    return ET.tostring(root, encoding="unicode")


yam_env.build_xml = build_xml


class Trajectory:
    """Positions of the objects and targets, and fingertip contact, after every control step."""

    def __init__(self):
        self.t = []
        self.positions = {}
        self.touching = {}
        self.ids = None

    def record(self, env):
        m, d = env.model, env.data
        if self.ids is None:
            self.ids = {}
            for name in OBJECTS + TARGETS:  # only the ones on the table
                try:
                    self.ids[name] = m.body(name).id
                except KeyError:
                    continue
                self.positions[name] = []
                if name in OBJECTS:
                    self.touching[name] = []
            self.tips = {m.body(name).id for name in FINGERTIPS}
            self.object_of = {self.ids[name]: name for name in OBJECTS if name in self.ids}
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
        for name in self.touching:
            self.touching[name].append(int(name in touching))

    def to_json(self):
        return {"t": self.t, "positions": self.positions, "touching": self.touching}


TRAJECTORY = Trajectory()
STATS = {"model_responses": 0, "model_actions": 0}

# Record the scene after every control step.
original_step = yam_env.YamEnv.step


RECORDING = {"on": False}  # off while the scene is built and settled, on for the run itself


def step(self, left, right):
    result = original_step(self, left, right)
    if RECORDING["on"]:
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


def main():
    parser = argparse.ArgumentParser(description="Run one Dream-1 task in the simulator")
    parser.add_argument("task", nargs="?", default="put apple into the bowl")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--ood", type=json.loads, default={}, help="Scene settings (ood.py) as a JSON object")
    parser.add_argument("--preview", type=Path, help="Render the initial camera view to a JPEG and exit")
    scene.add_arguments(parser)  # --seed, --default-scene, --split, --setting, --ood-random
    args = parser.parse_args()
    try:
        ood.validate(args.ood)
    except ValueError as error:
        parser.error(str(error))
    if SCENE:  # a custom layout replaces the seed's: random scene controls need the seed, so they cannot be used
        if args.seed is not None or args.ood_random:
            parser.error("A custom layout (SCENE_JSON) cannot be used with --seed or --ood-random.")
        args.default_scene = True
    if not args.default_scene:
        args.seed = scene.pick_seed(args)  # the seed the scene is built from, kept in meta.json

    if args.preview:
        env, _changes = scene.build_env(args)
        try:
            state = env.state()
            env.step(state["left"], state["right"])
            save_jpeg(camera_view(env.images()), args.preview)
        finally:
            env.close()
        return

    # The key: DREAM_API_KEY first (the website passes the key typed on the page), then ~/.dreamlabs/credentials
    # (batch.py and the command line). The bundled sim would otherwise prefer a separate, possibly stale /login key.
    key = dream_credentials.load().api_key
    if key:
        os.environ["DREAM_API_KEY"] = key

    run_dir = Path(os.environ.get("RUN_DIR") or HERE / "runs" / new_run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"id": run_dir.name, "task": args.task, "seconds": args.seconds,
            "seed": args.seed, "split": scene.split_of(args.seed) if args.seed is not None else None,
            "setting": args.setting, "scene_version": scene.SCENE_VERSION,
            "layout": "custom" if SCENE else "default" if args.seed is None else "seed",
            "ood_random": args.ood_random, "ood": None,
            "started_at": datetime.now().isoformat(timespec="seconds"), "stopped_by_user": False,
            "error": None, "error_type": None, "scene": SCENE or None,
            "batch_id": os.environ.get("BATCH_ID"),
            "judge": {"pairs": json.loads(os.environ["JUDGE_PAIRS"])} if os.environ.get("JUDGE_PAIRS") else None}

    def save(env=None):
        meta["ended_at"] = datetime.now().isoformat(timespec="seconds")
        meta["sim_seconds"] = TRAJECTORY.t[-1] if TRAJECTORY.t else 0
        meta.update(STATS)
        (run_dir / "trajectory.json").write_text(json.dumps(TRAJECTORY.to_json(), separators=(",", ":")))
        if last_images:
            save_jpeg(camera_view(last_images), run_dir / "last.jpg")
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if env is not None:
            env.close()
        print(f"Saved to runs/{run_dir.name}", flush=True)

    try:
        env, changes = scene.build_env(args)  # the seed's layout (or the custom one) and the scene controls
    except SystemExit:  # build_env printed why it stopped (a bad seed, a pose that overlaps, ...)
        meta["error"], meta["error_type"] = "The start scene could not be built (see the log).", "SceneError"
        save()
        sys.exit(1)
    meta["ood"] = changes or None

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
    RECORDING["on"] = True
    try:
        dream_control.main(["--task", args.task, "--seconds", str(args.seconds), "--headless", "--quiet"],
                           shared_env=env, stop_event=stop)
    except Exception as error:
        meta["error"] = str(error)
        meta["error_type"] = type(error).__name__
        print(f"Stopped: {error}", flush=True)
    finally:
        stop.set()
        save(env)
    sys.exit(1 if meta["error"] else 0)


if __name__ == "__main__":
    main()
