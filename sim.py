"""Run one Dream-1 task in the Dream Labs simulator (headless) and save its trajectory.

    python sim.py "put apple into the bowl" --seconds 60                  a new random seed (printed)
    python sim.py "put apple into the bowl" --seed 42                     the start scene of seed 42 (scene.py)
    python sim.py "put apple into the bowl" --seed 42 --ood-random light,table --ood '{"blur_px": 2}'
    python sim.py --preview start.jpg --seed 42                           the start scene as a JPEG (no API call)

Run modes (KSY: monitor.py, recovery.py, teleop.py, recorder.py):
    python sim.py "put apple into the bowl" --collect        data collection: Dream-1, then the recovery script when it
                                                             fails; no human. If the script gives up, the run ends there
    python sim.py "put apple into the bowl" --recover human  test: a failure asks a human at once (T takes over / gives back)
    python sim.py "put apple into the bowl" --teleop         human only: no Dream-1 until T (no API key or credits)
    python sim.py --resume <run id> --resume-step 120        restart a recorded run at step 120, exactly as it was, with a
                                                             human in control
    python sim.py "put apple into the bowl" --plain          Dream-1 only: no monitor, script or human control
    python sim.py "put apple into the bowl" --script-only    rule-based only: the recovery script does the whole task
                                                             from the first step (no Dream-1, no API key)
Data collection options (faster successes): --stall-s 6 (hand over after 6 s without progress), --quick-fail (also a
missed grasp or an object pushed away), --script-goal finish (the script finishes the task, not only picks it up),
--script-speed 2 (the script moves twice as fast as Dream-1), --end-on-success (end the run when the task is done).
    (no mode flag: Dream-1, then the recovery script, then a human if the script gives up)
OpenYAM and custom layouts without all six items always run as --plain (the monitor and teleop need the YAM table).

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
  BATCH_SEED  the batch's seed (each episode's own seed is derived from it: seed_web.episode_seed): meta.json
  ORIGINAL    set by an "original" batch (none of our method: Dream-1 alone, default scene and settings): meta.json
  ORIGINAL_OF the batch an original batch was run for (Run as original): stored in meta.json
  JUDGE_SPEC  what the run is judged on (judge.py), as JSON, e.g. {"type": "pick_place", "pairs": [{"object":
              "apple", "target": "bowl"}]} or {"type": "lift", "object": "any"} (meta.json "judge"); without it
              the task's words give it (JUDGE_PAIRS, the older pairs-only form, is still read)
  LIVE_FRAME  path of a JPEG that is replaced with the latest camera view (top camera + both wrist cameras)
  STOP_FILE   when this file appears, the run stops and the Dream session is released
  TELEOP_STATE  path of a JSON file replaced with the control state (mode, arm, gripper, monitor, recovery script)
Keyboard and mouse input arrives on stdin, one JSON object per line: {"held": ["w"]}, {"press": "t"}, {"click": [u, v],
"camera": "top"}, ... (teleop.py). Runs in a KSY mode are also recorded to episodes/<run id>/ (recorder.py: every
step's source, inputs, cameras and full physics state, for resuming); --no-record skips that.
"""

import argparse
import math
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

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import judge  # noqa: E402  goal scenes (the monitor works towards them)
import ood  # noqa: E402  scene controls: camera, arm angles, blur, light, table
import scene  # noqa: E402  the seed's start scene; imported before build_xml is patched below (it keeps the original)
import simulator_settings as sim_settings  # noqa: E402  robot and room choices
from local_rooms import desk_texture, themed_xml  # noqa: E402  Sandbox room styling for the YAM table
from sandbox_env import SandboxEnv  # noqa: E402  OpenYAM on the Sandbox table (needs Node.js to fetch it)
from monitor import Monitor  # noqa: E402  failure monitor (KSY)
from recorder import BODIES, EPISODES_DIR, PHYSICS, EpisodeRecorder  # noqa: E402  episode recorder (KSY)
from recovery import Recovery  # noqa: E402  recovery script (KSY)
import teleop  # noqa: E402
from teleop import STEP_S, Teleop, camera_position, select  # noqa: E402  human control (KSY)

HERE = Path(__file__).resolve().parent
LIVE_FRAME = os.environ.get("LIVE_FRAME")
STOP_FILE = os.environ.get("STOP_FILE")
TELEOP_STATE = os.environ.get("TELEOP_STATE")
TELEOP = Teleop()
MONITOR = None  # the failure monitor, set in main() once the task is known (KSY modes)
RECOVERY = None  # the recovery script; None with --recover human
RECORDER = None  # the episode recorder (episodes/); None with --no-record or --plain
COLLECT = False  # --collect: Dream-1, then the recovery script; no human input (also --script-only)
SCRIPT_ONLY = False  # --script-only: the recovery script from the first step, no Dream-1
KSY = False  # whether the run goes through teleop / monitor / recovery (not --plain, OpenYAM or a partial table)
DREAM_STARTED = [False]  # Dream-1 was asked for a real chunk: before that, the loop only fills its camera window
HOLD_ROWS = 20  # while a human (or the script) has the arms, Dream-1 is not called: the loop gets 1 s of hold rows
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
            bodies = getattr(env, "object_bodies", {name: name for name in OBJECTS + TARGETS})  # OpenYAM: "apple/object"
            fingertips = getattr(env, "fingertip_bodies", FINGERTIPS)
            self.ids = {}
            for name in OBJECTS + TARGETS:  # only the ones on the table
                try:
                    self.ids[name] = m.body(bodies[name]).id
                except KeyError:
                    continue
                self.positions[name] = []
                if name in OBJECTS:
                    self.touching[name] = []
            self.tips = {m.body(name).id for name in fingertips}
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
        if keep and RECORDER is not None:
            RECORDER.on_frame(last_images)  # the frames the model sees, 5 per second
    return dict(last_images)


MODEL = {"id": dream_control.BETA_MODEL}  # the checkpoint this run asks (--model); set in main()
STOP_EVENT = threading.Event()  # set when the run is stopped (Stop, the stop file, or the end of the run)
WARMING_LIMIT_S = 600  # how long a run waits for a model that is starting (DA-2000 pool_warming) before giving up


class PausableClock:
    """dream_control's clock (its run deadline and its round-trip times): the time module, minus the seconds spent
    waiting for a model to start, so a fine-tune that is not running yet does not use up the run's time."""

    paused_s = 0.0

    def __getattr__(self, name):
        return getattr(time, name)

    def monotonic(self):
        return time.monotonic() - self.paused_s


DREAM_CLOCK = PausableClock()
dream_control.time = DREAM_CLOCK


def wait_for_model(error, waited):
    """Wait while the model starts (the first call to a model that is not running, such as a fine-tune, is refused
    with DA-2000 pool_warming), off the run's clock.
    @param error The dream.DreamServerError. @param waited The seconds waited so far in this run.
    @return The seconds waited this time. @throws error When it is another refusal, or the model took too long."""
    if getattr(error, "reason", None) != "pool_warming" or waited >= WARMING_LIMIT_S:
        raise error
    pause = min(max(float(getattr(error, "retry_after_s", None) or 5), 1.0), 30.0)
    print(f"  The model {MODEL['id']} is starting; asking again in {pause:.0f} s "
          f"(waited {waited:.0f} s, not counted in the run's time).", flush=True)
    started = time.monotonic()
    STOP_EVENT.wait(pause)
    if STOP_EVENT.is_set():
        raise error
    spent = time.monotonic() - started
    DREAM_CLOCK.paused_s += spent
    return spent


# A connection the server dropped while idle is reopened by asking again (up to twice).
original_get_action = dream.DreamClient.get_action


def get_action(self, *args, **kwargs):
    # While a human (or the recovery script) has the arms, Dream-1 is not asked: no credits, no waiting, and the loop
    # keeps running at 20 Hz. The rows returned here are replaced by their actions anyway; a hand-back completes when
    # Dream-1 is asked again (a fresh chunk).
    if KSY and not TELEOP.chunk_started():
        left, right = TELEOP.last_action
        return {"q_l_chunk": np.tile(left, (HOLD_ROWS, 1)), "q_r_chunk": np.tile(right, (HOLD_ROWS, 1))}
    DREAM_STARTED[0] = True
    if RECORDER is not None and RECORDER.model_id is None:
        RECORDER.model_id = MODEL["id"]
    dropped, waited = 0, 0.0
    while True:
        try:
            reply = original_get_action(self, *args, **kwargs)
            STATS["model_responses"] += 1
            if waited:  # app.py reads this line: the page stops showing "the model is starting"
                print(f"  The model {MODEL['id']} is ready (it took {waited:.0f} s to start).", flush=True)
            return reply
        except dream.DreamProtocolError:
            dropped += 1
            if dropped == 3:
                raise
            print("  Connection dropped; asking again.", flush=True)
            time.sleep(0.2)
        except dream.DreamServerError as error:
            waited += wait_for_model(error, waited)


yam_env.YamEnv.step = step
dream_control.capture = capture

original_sandbox_step = SandboxEnv.step


def sandbox_step(self, left, right):
    result = original_sandbox_step(self, left, right)
    if RECORDING["on"]:
        TRAJECTORY.record(self)
    if STATS["model_responses"]:
        STATS["model_actions"] += 1
    return result


SandboxEnv.step = sandbox_step

## OpenYAM's Sandbox table is in its own frame: the judge reads heights relative to the bowl and plate, the floor
## level, and the workspace's middle and directions (see workspace_frame), instead of the YAM table's numbers.
OPENYAM_HEIGHT_BANDS = {"bowl": [0.01, 0.12], "plate": [0.0, 0.09]}
OPENYAM_OFF_TABLE_Z = -0.1


def theme_yam_room(room):
    """Style the YAM table as a Sandbox room (desk, background, light); the objects and physics stay the same."""
    if room == "yam_tabletop":
        return
    try:
        texture = desk_texture(room)
    except (OSError, ValueError) as error:
        texture = None
        print(f"Room desk texture unavailable; using the room's colour: {error}", flush=True)
    base = yam_env.build_xml
    yam_env.build_xml = lambda: themed_xml(base(), room, texture)


def build_openyam(args):
    """OpenYAM on the Sandbox table. The table has no seed scenes: a seed shifts each object by up to 2 cm and turns
    the fruit by up to 25 degrees (reproducibly)."""
    env = SandboxEnv(args.room)
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        for body in OBJECTS:
            try:
                qpos = env.data.joint(f"{body}/free").qpos
            except KeyError:
                continue
            qpos[:2] += rng.uniform(-0.02, 0.02, 2)
            if body != "water_bottle":
                yaw = np.deg2rad(rng.uniform(-25, 25)) / 2
                turned = np.empty(4)
                mujoco.mju_mulQuat(turned, np.array([np.cos(yaw), 0, 0, np.sin(yaw)]), qpos[3:7].copy())
                qpos[3:7] = turned
        mujoco.mj_forward(env.model, env.data)
    print(f"Scene: OpenYAM, Sandbox table, room {args.room}"
          + (f", seed {args.seed} (objects shifted up to 2 cm)" if args.seed is not None else ""), flush=True)
    return env


def workspace_frame(env, camera):
    """The workspace for judging areas: the middle of the items on the table, and the directions the camera sees
    as left and as front (towards the bottom of its image)."""
    m, d = env.model, env.data
    xmat = d.cam_xmat[m.camera(camera).id].reshape(3, 3)
    left, front = -xmat[:2, 0], -xmat[:2, 1]
    bodies = getattr(env, "object_bodies", {name: name for name in OBJECTS + TARGETS})
    points = []
    for name in OBJECTS + TARGETS:
        try:
            points.append(d.body(bodies[name]).xpos[:2].copy())
        except KeyError:
            continue
    center = np.mean(points, axis=0)
    unit = lambda v: [round(float(x), 4) for x in v / (np.linalg.norm(v) or 1)]
    return {"center": [round(float(x), 4) for x in center], "left": unit(left), "front": unit(front)}
dream.DreamClient.get_action = get_action


## A fine-tune can overshoot the gripper a little (1.07 is fully open); dream_control stops the run on any gripper
## value more than 0.05 outside 0..1, as a sign of a wrong response contract. Values up to GRIPPER_SLACK outside are
## clipped to 0..1 here first; anything further out still stops the run (a real contract error).
GRIPPER_SLACK = 0.25
GRIPPER_CLIPPED = {"rows": 0, "max": 0.0}
original_validate_chunk = dream_control.validate_chunk


def clip_gripper(actions):
    """@return The actions with each chunk's gripper column clipped to 0..1 where it is at most GRIPPER_SLACK outside
    (a copy; the reply itself is not changed), and the largest correction made."""
    clipped, largest = dict(actions), 0.0
    for key in ("q_l_chunk", "q_r_chunk"):
        try:
            rows = np.array(actions[key], dtype=float, copy=True)
        except (KeyError, TypeError, ValueError):
            continue  # validate_chunk says what is wrong
        if rows.ndim != 2 or rows.shape[1] != 7 or not np.isfinite(rows[:, 6]).all():
            continue
        gripper = rows[:, 6]
        fixable = (gripper >= -GRIPPER_SLACK) & (gripper <= 1 + GRIPPER_SLACK)
        bounded = np.where(fixable, np.clip(gripper, 0.0, 1.0), gripper)
        changed = np.abs(bounded - gripper)
        if changed.any():
            GRIPPER_CLIPPED["rows"] += int(np.count_nonzero(changed))
            largest = max(largest, float(changed.max()))
            rows[:, 6] = bounded
            clipped[key] = rows
    return clipped, largest


def validate_chunk(env, actions, *args, **kwargs):
    clipped, largest = clip_gripper(actions)
    if largest > 0.05 and largest > GRIPPER_CLIPPED["max"]:  # said once, and again only for a larger overshoot
        GRIPPER_CLIPPED["max"] = largest
        print(f"  Gripper value {largest:.3f} outside 0..1, clipped (the model overshoots open / closed).", flush=True)
    return original_validate_chunk(env, clipped, *args, **kwargs)


dream_control.validate_chunk = validate_chunk


def read_keys(env, stop):
    """Keyboard and mouse input from app.py, one JSON object per line on stdin (KSY)."""
    for line in sys.stdin:
        if stop.is_set():
            return
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if COLLECT:
            continue  # data collection: the model and the recovery script only (a human recovers afterwards)
        if RECORDER is not None:
            RECORDER.event("input", **message)  # every key, click, drag and button, for the record
        if "held" in message:
            TELEOP.set_held(message["held"])
        if message.get("press"):
            TELEOP.press(message["press"], env)
        if message.get("click"):
            TELEOP.queue_click(*message["click"], camera=message.get("camera", "top"))
        if message.get("drag"):
            TELEOP.queue_drag(*message["drag"])
        if message.get("wheel"):
            TELEOP.wheel(message["wheel"])


def write_state(stop):
    """The control state for the page (TELEOP_STATE), five times a second."""
    while not stop.is_set():
        try:
            with open(TELEOP_STATE + ".tmp", "w", encoding="utf-8") as file:
                json.dump({**TELEOP.status(), "monitor": MONITOR.status() if MONITOR else None, "collecting": COLLECT,
                           "recording": {"id": RECORDER.id, "steps": len(RECORDER.rows["t"])} if RECORDER else None,
                           "recovery": RECOVERY.status() if RECOVERY else None}, file)
            os.replace(TELEOP_STATE + ".tmp", TELEOP_STATE)
        except OSError:
            pass
        time.sleep(0.2)


def keyboard_only(env, seconds, stop):
    """No Dream-1: the human drives the arms from the start, 20 Hz, cameras at 10 Hz (KSY).
    @return The seconds left when the human handed the arms to Dream-1 (key T), else None."""
    TELEOP.start_human(env)
    deadline = time.monotonic() + seconds
    steps = 0
    while not stop.is_set() and time.monotonic() < deadline:
        if TELEOP.mode == "returning":
            return deadline - time.monotonic()
        tick = time.monotonic()
        env.step(None, None)  # the keyboard replaces the action (see controlled_step in main)
        if steps % 2 == 0:
            images = env.images()
            last_images.update(images)  # the run's last picture (runs/<id>/last.jpg)
            if LIVE_FRAME:
                save_jpeg(camera_view(images), LIVE_FRAME)
            if steps % 4 == 0 and RECORDER is not None:
                RECORDER.on_frame(images)  # 5 per second, as in Dream-1 runs
        steps += 1
        time.sleep(max(0, STEP_S - (time.monotonic() - tick)))
    return None


def all_done():
    """Every object of the task is at its goal (the monitor's stage is the watched object's only)."""
    return MONITOR.stage == "success" and {obj for obj, _entry in MONITOR.plan} <= set(MONITOR.done)


def script_only(env, seconds, stop):
    """Rule-based only (--script-only): the recovery script (recovery.py) does the task from the first step, object
    by object, as it finishes a task after a failure in data collection; Dream-1 is never asked (no key, no API use).
    Real time: one step every 50 ms of wall time, as Dream-1's chunks are played and the arms of a real robot move.
    Cameras every 4 steps (5 per second), as the frames Dream-1 sees. The run ends when the task is done (--end-on-success), the
    script gives up, or the time is up."""
    TELEOP.dream_available = False  # T and clicks hand nothing to Dream-1
    steps, limit = 0, int(round(seconds / STEP_S))
    while not stop.is_set() and steps < limit:
        if RECOVERY is not None and not RECOVERY.active and TELEOP.mode != "paused" and not all_done():
            # The next object (the first, or the next of a goal scene): the script takes the arms for it.
            TELEOP.chunk_started()  # "returning" after the last object: back to "dream", which the script takes over
            MONITOR._fail("scripted", "success")  # its goal: this object at its place
            if RECORDER is not None:
                RECORDER.event("script_start", object=MONITOR.object)
            RECOVERY.start(env, "scripted", "success")
        tick = time.monotonic()
        env.step(None, None)  # the script replaces the action (TELEOP.filter in controlled_step)
        if steps % 4 == 0:
            images = env.images()
            last_images.update(images)
            if LIVE_FRAME:
                save_jpeg(camera_view(images), LIVE_FRAME)
            if RECORDER is not None:
                RECORDER.on_frame(images)
        steps += 1
        time.sleep(max(0, STEP_S - (time.monotonic() - tick)))


GOAL_DROP_M = 0.06  # a dragged object is let go this far above where it will rest, then physics settles it
GOAL_SETTLE_STEPS = 40  # 2 s of simulated time, arms holding still


def projection(env, camera="top"):
    """The camera's 3x4 projection from world points to the picture (u, v from 0 to 1 of env.images()'s crop),
    fitted (DLT) to points the renderer's own picking finds on a grid of the picture, so it matches the picture
    exactly, calibrated intrinsics included. @return (P as a list of lists, mean reprojection error in pixels)."""
    pairs = []
    for u in np.linspace(0.03, 0.97, 16):
        for v in np.linspace(0.05, 0.95, 10):
            _body, point = select(env, u, v, camera)
            if point is not None:
                pairs.append((u, v, *point))
    rows = []
    for u, v, x, y, z in pairs:
        rows.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        rows.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    p = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 4)
    p /= p[2, :3] @ np.array(pairs[0][2:]) + p[2, 3]  # positive depth in front of the camera
    errors = []
    for u, v, x, y, z in pairs:
        h = p @ np.array([x, y, z, 1.0])
        errors.append(math.hypot((h[0] / h[2] - u) * 640, (h[1] / h[2] - v) * 360))
    return p.tolist(), float(np.mean(errors))


def goal_preview(env, goal, out):
    """The goal popup's picture: the start scene with each dragged object let go just above its goal spot, settled
    by the physics (it lands in the bowl, on the plate, on the table), as the top camera sees it.
    @param goal {object: [x, y]} (world metres), or {} for the start scene itself.
    @param out Folder for goal.jpg and goal.json (positions after settling, the camera projection, the seed)."""
    state = env.state()
    for _ in range(3):
        env.step(state["left"], state["right"])
    m, d = env.model, env.data
    for name, (x, y) in goal.items():
        joint = m.body(name).jntadr[0]
        q, dof = m.jnt_qposadr[joint], m.jnt_dofadr[joint]
        d.qpos[q:q + 2] = [x, y]
        d.qpos[q + 2] += GOAL_DROP_M
        d.qvel[dof:dof + 6] = 0
    mujoco.mj_forward(m, d)
    if goal:
        for _ in range(GOAL_SETTLE_STEPS):
            env.step(state["left"], state["right"])
    images = env.images()
    save_jpeg(Image.fromarray(images["top"]), Path(out) / "goal.jpg")
    matrix, error = projection(env)
    positions = {name: [round(float(v), 4) for v in d.body(name).xpos] for name in OBJECTS + TARGETS}
    # What each dragged object ended up as: in the bowl, on the plate, or on the table (its spot).
    import judge
    relations = {name: judge.relation_of(positions[name], {t: positions[t] for t in TARGETS}) for name in goal}
    (Path(out) / "goal.json").write_text(json.dumps({"positions": positions, "relations": relations,
                                                     "projection": matrix,
                                                     "reprojection_px": round(error, 2), "size": [640, 360]}))


def read_meta(folder):
    try:
        return json.loads((Path(folder) / "meta.json").read_text())
    except (OSError, ValueError):
        return None


def record_plain(env, args, meta):
    """Model-only runs (--plain, the Test page): record the episode as the other modes do (episodes/<id>: steps,
    the three camera videos, events), with no failure monitor, recovery script or human control. Every step is the
    model's once its first chunk arrives, and "hold" before that."""
    global RECORDER
    RECORDER = EpisodeRecorder(env, args.task, mode="dream",
                               extra={"recover": None, "time_limit_s": args.seconds, "resumed_from": None,
                                      "run": meta["id"], "seed": meta["seed"], "room": meta["room"], "plain": True})
    RECORDER.id, RECORDER.dir = meta["id"], EPISODES_DIR / meta["id"]  # one id: runs/<id> and episodes/<id>
    step = env.step
    no_hands, no_grips = np.full((2, 3), np.nan), np.full(2, np.nan)  # no human targets in a model-only run

    def recorded_step(left, right):
        RECORDER.before_step()
        result = step(left, right)
        RECORDER.after_step(left, right, "dream" if DREAM_STARTED[0] else "hold", no_hands, no_grips, None)
        return result

    env.step = recorded_step
    print(f"  Recording: episodes/{RECORDER.id}", flush=True)


def start_ksy(env, args, meta, stop, source_dir):
    """KSY's control loop on the built scene: every action goes through the human controller (teleop.py), which
    passes Dream-1's on or replaces them; every step is watched by the failure monitor (monitor.py), which stops
    Dream-1 when it fails: the recovery script (recovery.py) tries first, then the arms are held for a human (or
    at once with --recover human). The recorder (recorder.py) keeps every step, under the run's id."""
    global MONITOR, RECOVERY, RECORDER
    resume = None
    if source_dir is not None:
        # The recorded run, restored exactly: its start (for the home pose and the monitor's resting heights), then
        # the full physics state before the chosen step.
        recorded = np.load(source_dir / "steps.npz")
        step_k = max(0, min(int(args.resume_step), len(recorded["t"]) - 1))
        resume = {"episode": source_dir.name, "step": step_k, "t": float(recorded["t"][step_k - 1]) if step_k else 0.0}
        mujoco.mj_setState(env.model, env.data, recorded["physics"][0], PHYSICS)
        mujoco.mj_forward(env.model, env.data)
    TELEOP.set_home(env)
    # The goal scene, when the run has one (JUDGE_SPEC from the page): the monitor and the recovery script work
    # towards it, object by object; else the task's words.
    goal, tolerance = None, None
    try:
        spec = judge.normalize_spec(json.loads(os.environ.get("JUDGE_SPEC") or "null"))
        if spec and spec["type"] == "goal":
            goal = spec
            saved = json.loads((HERE / "runs" / "rules.json").read_text()) if (HERE / "runs" / "rules.json").exists() else {}
            tolerance = judge.normalize_rules(saved)["goal"]["tolerance_cm"]
    except (ValueError, OSError, TypeError, KeyError):
        pass
    quick = COLLECT  # data collection may use the faster failure signals; other modes keep the monitor's defaults
    MONITOR = Monitor(args.task, env, goal=goal, tolerance_cm=tolerance,
                      stall_s=args.stall_s if quick and args.stall_s else 12.0,
                      quick=quick and args.quick_fail, finish=quick and args.script_goal == "finish")
    if goal:
        print("  Goal scene: " + "; ".join(f"{o} {e.get('in') or e.get('near') or 'spot'}" for o, e in goal["objects"].items()),
              flush=True)
    if resume:
        mujoco.mj_setState(env.model, env.data, recorded["physics"][resume["step"]], PHYSICS)
        mujoco.mj_forward(env.model, env.data)
        if resume["step"] > 0:  # the arms hold what they were last commanded, not where gravity has them
            last = recorded["action"][resume["step"] - 1]
            TELEOP.last_action = (last[:7].copy(), last[7:].copy())
    if not args.no_record:
        RECORDER = EpisodeRecorder(env, args.task, mode="resumed" if resume else "human-only" if args.teleop
                                   else "script-only" if SCRIPT_ONLY else "dream",
                                   extra={"recover": args.recover, "time_limit_s": args.seconds, "resumed_from": resume,
                                          "run": meta["id"], "seed": meta["seed"], "room": meta["room"]})
        RECORDER.id, RECORDER.dir = meta["id"], EPISODES_DIR / meta["id"]  # one id: runs/<id> and episodes/<id>
        if resume:
            RECORDER.load_prefix(source_dir, resume["step"])  # the new episode starts with the original's steps
        TELEOP.on_control = lambda mode, reason: RECORDER.event("control", mode=mode, reason=reason)
        TELEOP.on_grip_hold = lambda side, what, why: RECORDER.event("grip_hold", side=side, what=what, why=why)
        print(f"  Recording: episodes/{RECORDER.id}", flush=True)
    if args.recover == "script":
        RECOVERY = Recovery(TELEOP, MONITOR)
    if not MONITOR.understood():
        print("  The failure monitor does not understand this task (pick and place only); it will not ask for help.",
              flush=True)
    step = env.step
    script_result = [None]
    previous_mode = [TELEOP.mode]
    frames = [0]

    def controlled_step(left, right):
        click = TELEOP.take_click()  # mouse input needs the renderer, so it is looked up here, on the main thread
        if click:
            u, v, camera = click
            body, point = select(env, u, v, camera)
            if body:
                TELEOP.clicked(env, body, point, camera)
        drag = TELEOP.take_drag()
        if drag:
            body, point = select(env, *drag)
            if body:
                TELEOP.dragged(env, camera_position(env), point)
        mode_before = TELEOP.mode
        if RECORDER is not None:
            RECORDER.before_step()
        sent = TELEOP.filter(env, left, right)
        result = step(*sent)
        mode = TELEOP.mode
        frames[0] += 1
        # The model's frames are 5 per second; while a human steers, show the picture twice as often.
        if mode != "dream" and TELEOP.dream_available and LIVE_FRAME and frames[0] % 2 == 0:
            save_jpeg(camera_view(env.images()), LIVE_FRAME)
        if previous_mode[0] in ("human", "script") and mode in ("returning", "dream"):
            MONITOR.recovered()  # the human or the script gave the arms back: watch Dream-1 again from here
        previous_mode[0] = mode
        # Failures are looked for only once Dream-1 has started acting (its first chunk), so the time without
        # progress does not include its start-up.
        event = MONITOR.update(env, watching=(mode == "dream" and TELEOP.dream_available and DREAM_STARTED[0]))
        if event:
            print(f"  Monitor: {event}" + (f" · recovery goal: {MONITOR.status()['goal']}" if MONITOR.goal else ""),
                  flush=True)
            if event != "off_table":
                if RECOVERY is not None and MONITOR.goal:
                    RECOVERY.start(env, event, MONITOR.goal)
                else:
                    TELEOP.pause(env, event)
        elif RECOVERY is not None:
            RECOVERY.tick(env)
        if RECOVERY is not None and RECOVERY.result != script_result[0]:
            script_result[0] = RECOVERY.result
            if RECORDER is not None and RECOVERY.result:
                RECORDER.event("script", result=RECOVERY.result)
        if args.end_on_success and COLLECT and not stop.is_set() and all_done():
            # The task is done (placed, released, at rest): end here, before anything can undo it.
            print("  Ending the run: the task is done.", flush=True)
            meta["ended_early"] = "task done"
            if RECORDER is not None:
                RECORDER.event("ended", reason="task done")
            stop.set()
        if COLLECT and not stop.is_set() and (TELEOP.mode == "paused" or event == "off_table"):
            # Data collection asks no human: a failure the script could not recover ends the run here. It is a failure
            # to recover by hand later, from the Recordings panel (it resumes before the model failed).
            why = "the object fell off the table" if event == "off_table" else "the recovery script gave up"
            print(f"  Ending the run: {why}. Recover it later from the Recordings panel.", flush=True)
            meta["ended_early"] = why
            if RECORDER is not None:
                RECORDER.event("ended", reason=why)
            stop.set()
        if RECORDER is not None:
            if event:
                RECORDER.event("monitor", event=event, goal=MONITOR.status()["goal"])
            source = ("human" if mode_before == "human" else "script" if mode_before == "script"
                      else "dream" if mode_before == "dream" and DREAM_STARTED[0] and TELEOP.dream_available
                      else "hold")
            hands, grips = TELEOP.record_view()
            RECORDER.after_step(*sent, source, hands, grips, MONITOR.stage)
        return result

    env.step = controlled_step
    if not sys.stdin.isatty():
        threading.Thread(target=read_keys, args=(env, stop), daemon=True).start()
    if TELEOP_STATE:
        threading.Thread(target=write_state, args=(stop,), daemon=True).start()
    if resume:
        print(f"  Resuming {resume['episode']} at step {resume['step']} ({resume['t']:.2f} s): you have the arms.",
              flush=True)


def resume_options(args):
    """--resume: the original run's start options (runs/<id>/meta.json), so the scene is built the same way before
    its recorded physics state is restored. @return The recorded episode folder."""
    global SCENE
    source = EPISODES_DIR / Path(args.resume).name
    if not (source / "steps.npz").exists():
        sys.exit(f"Stopped: no recording episodes/{source.name} to resume.")
    original = json.loads((HERE / "runs" / source.name / "meta.json").read_text()) \
        if (HERE / "runs" / source.name / "meta.json").exists() else {}
    recorded = json.loads((source / "meta.json").read_text()) if (source / "meta.json").exists() else {}
    args.task = original.get("task") or recorded.get("task") or args.task
    args.teleop = True  # a resumed run starts with a human in control (T hands it to Dream-1)
    if original.get("robot", "yam") != "yam":
        sys.exit("Stopped: only YAM runs can be resumed.")
    args.room = original.get("room", args.room)
    args.setting = original.get("setting", args.setting)
    args.ood, args.ood_random = original.get("ood") or {}, []  # the values the original drew, as they were
    SCENE = original.get("scene") or {}
    args.seed = original.get("seed") if not SCENE else None
    args.default_scene = args.seed is None
    if not os.environ.get("JUDGE_SPEC") and original.get("judge"):
        os.environ["JUDGE_SPEC"] = json.dumps(original["judge"])
    return source


def main():
    parser = argparse.ArgumentParser(description="Run one Dream-1 task in the simulator")
    parser.add_argument("task", nargs="?", default="Put the apple into the bowl")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--ood", type=json.loads, default={}, help="Scene settings (ood.py) as a JSON object")
    parser.add_argument("--preview", type=Path, help="Render the initial camera view to a JPEG and exit")
    parser.add_argument("--goal-preview", type=Path, metavar="FOLDER",
                        help="The goal popup: render the start scene with the objects of --goal settled at their goal "
                             "spots (top camera, positions, camera projection) into FOLDER and exit")
    parser.add_argument("--goal", type=json.loads, default={}, help='Goal spots {"apple": [x, y], ...} (world metres)')
    parser.add_argument("--robot", choices=("yam", "openyam"), default="yam", help="The robot (simulator_settings.py)")
    parser.add_argument("--room", choices=sim_settings.ROOMS, default="yam_tabletop", help="The room's styling")
    parser.add_argument("--teleop", action="store_true", help="human only: no Dream-1 until T")
    parser.add_argument("--collect", action="store_true",
                        help="data collection: Dream-1, then the recovery script on a failure; no human input")
    parser.add_argument("--recover", choices=("script", "human"), default="script",
                        help="when Dream-1 fails: the recovery script first (script), or a human at once (human)")
    parser.add_argument("--resume", metavar="RUN", help="restart a recorded run (with --resume-step), a human in control")
    parser.add_argument("--resume-step", type=int, default=0, help="the step of --resume to start from")
    parser.add_argument("--no-record", action="store_true", help="do not record the run to episodes/")
    parser.add_argument("--plain", action="store_true", help="Dream-1 only: no monitor, recovery script or human control")
    parser.add_argument("--script-only", action="store_true",
                        help="rule-based only: the recovery script does the whole task from the first step; no Dream-1")
    parser.add_argument("--stall-s", type=float, help="data collection: seconds without progress before the script takes over")
    parser.add_argument("--quick-fail", action="store_true", help="data collection: a missed grasp or a pushed-away object fails at once")
    parser.add_argument("--script-goal", choices=("carry", "finish"), default="carry",
                        help="data collection: the script picks the object up and gives the arms back (carry), or finishes the task")
    parser.add_argument("--script-speed", type=float, default=1.0, help="data collection: the script's speed, x Dream-1's pace (1 to 3)")
    parser.add_argument("--end-on-success", action="store_true", help="end the run once the monitor sees the task done")
    parser.add_argument("--model", type=dream_control.resolve_model,
                        default=os.environ.get("DREAM_MODEL") or dream_control.BETA_MODEL,
                        help="the checkpoint Dream-1 runs: beta, or a model id such as a fine-tune (dream-1-ft:...);"
                             " default DREAM_MODEL, else dreamlabs/dream-1-beta")
    scene.add_arguments(parser)  # --seed, --default-scene, --split, --setting, --ood-random
    args = parser.parse_args()
    MODEL["id"] = args.model
    source_dir = resume_options(args) if args.resume else None
    openyam = args.robot == "openyam"
    if openyam and (SCENE or args.ood or args.ood_random):
        parser.error("Scene settings, random settings and custom layouts are for YAM; OpenYAM uses the Sandbox table.")
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
    # KSY's monitor, recovery script, human control and recorder need the YAM table with all six items.
    global KSY, COLLECT, SCRIPT_ONLY, MONITOR, RECOVERY, RECORDER
    partial = bool(SCENE) and set(SCENE.get("items", {})) != set(BODIES)
    KSY = not (args.plain or openyam or partial)
    if not KSY and (args.teleop or args.resume):
        parser.error("Human control and resuming need YAM with all six items on the table.")
    if not KSY and (args.collect or args.recover == "human") and not args.plain:
        print("  Monitor, recovery script and human control are for YAM with all six items: running Dream-1 only.",
              flush=True)
    if args.script_only and (not KSY or args.teleop or args.resume):
        parser.error("Rule-based only runs need YAM with all six items on the table, and start fresh (no human).")
    SCRIPT_ONLY = KSY and args.script_only
    COLLECT = KSY and (args.collect or SCRIPT_ONLY) and not args.teleop and not args.resume
    if COLLECT:
        args.recover = "script"
        speed = max(1.0, min(3.0, args.script_speed))
        for name in ("SCRIPT_SPEED_M_S", "SCRIPT_ACCEL_M_S2", "SCRIPT_MAX_JOINT_RAD_S", "SCRIPT_JOINT_ACCEL_RAD_S2"):
            setattr(teleop, name, getattr(teleop, name) * speed)
        teleop.SCRIPT_GRIP_SECONDS /= speed
    mode = ("plain" if not KSY else "resumed" if args.resume else "human" if args.teleop else "script" if SCRIPT_ONLY
            else "collect" if COLLECT
            else "test" if args.recover == "human" else "assist")
    if not openyam:
        theme_yam_room(args.room)

    def build():
        """The run's simulator and the scene settings in effect."""
        return (build_openyam(args), {}) if openyam else scene.build_env(args)

    if args.goal_preview:
        if openyam:
            parser.error("The goal scene is for YAM.")
        goal = {name: [float(v) for v in spot][:2] for name, spot in (args.goal or {}).items() if name in OBJECTS}
        env, _changes = build()
        try:
            goal_preview(env, goal, args.goal_preview)
        finally:
            env.close()
        print(f"Goal preview: seed {args.seed}", flush=True)
        return

    if args.preview:
        env, _changes = build()
        try:
            state = env.state()
            env.step(state["left"], state["right"])
            save_jpeg(camera_view(env.images()), args.preview)
            # Where each item starts (x, y in metres): the page's custom layout starts from the seed's.
            layout = {}
            for name in OBJECTS + TARGETS:
                try:
                    layout[name] = [round(float(v), 4) for v in env.data.body(name).xpos[:2]]
                except KeyError:  # left out of a custom layout, or not in this robot's scene
                    pass
            print(f"Scene layout: {json.dumps(layout, sort_keys=True)}", flush=True)
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
            "robot": args.robot, "room": args.room,
            "layout": "sandbox" if openyam else "custom" if SCENE else "default" if args.seed is None else "seed",
            "mode": mode, "resumed_from": {"run": Path(args.resume).name, "step": args.resume_step} if args.resume else None,
            # A human ends human, test and resumed runs with Stop: that is their normal end, not a cut-short run.
            "stop_ends_run": mode in ("human", "test", "resumed"),
            "collection": {"stall_s": args.stall_s, "quick_fail": args.quick_fail, "script_goal": args.script_goal,
                           "script_speed": args.script_speed, "end_on_success": args.end_on_success} if COLLECT else None,
            "ood_random": args.ood_random, "ood": None,
            "started_at": datetime.now().isoformat(timespec="seconds"), "stopped_by_user": False,
            "error": None, "error_type": None, "scene": SCENE or None,
            "model": None if args.script_only else args.model,
            "batch_id": os.environ.get("BATCH_ID"), "original": os.environ.get("ORIGINAL") == "1",
            "batch_seed": int(os.environ["BATCH_SEED"]) if os.environ.get("BATCH_SEED") else None,
            "original_of": os.environ.get("ORIGINAL_OF") or None,
            "testset": os.environ.get("TESTSET") or None,  # the test set this start came from (app.py batches)
            "judge": (json.loads(os.environ["JUDGE_SPEC"]) if os.environ.get("JUDGE_SPEC")
                      else {"pairs": json.loads(os.environ["JUDGE_PAIRS"])} if os.environ.get("JUDGE_PAIRS") else None)}

    def save(env=None):
        if RECORDER is not None and not RECORDER.closed:
            saved = RECORDER.close(MONITOR.status() if MONITOR else None, None if not meta["error"] else meta["error"])
            if saved:
                meta["episode"] = RECORDER.id
                meta["steps_by_source"] = (read_meta(saved) or {}).get("steps_by_source")
                print(f"  Recorded: episodes/{saved.name} ({len(RECORDER.rows['t'])} steps)", flush=True)
        meta["ended_at"] = datetime.now().isoformat(timespec="seconds")
        meta["sim_seconds"] = TRAJECTORY.t[-1] if TRAJECTORY.t else 0
        meta.update(STATS)
        meta["gripper_clipped"] = dict(GRIPPER_CLIPPED) if GRIPPER_CLIPPED["rows"] else None
        (run_dir / "trajectory.json").write_text(json.dumps(TRAJECTORY.to_json(), separators=(",", ":")))
        if last_images:
            save_jpeg(camera_view(last_images), run_dir / "last.jpg")
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if env is not None:
            env.close()
        print(f"Saved to runs/{run_dir.name}", flush=True)

    try:
        env, changes = build()  # the seed's layout (or the custom one) and the scene controls
    except SystemExit:  # build_env printed why it stopped (a bad seed, a pose that overlaps, ...)
        meta["error"], meta["error_type"] = "The start scene could not be built (see the log).", "SceneError"
        save()
        sys.exit(1)
    except Exception as error:  # noqa: BLE001  OpenYAM: Node.js missing, the Sandbox package not reachable, ...
        meta["error"], meta["error_type"] = f"The simulator could not be built: {error}", type(error).__name__
        print(f"Stopped: {meta['error']}", flush=True)
        save()
        sys.exit(1)
    meta["ood"] = changes or None
    if openyam:
        meta["relative_height_bands"] = OPENYAM_HEIGHT_BANDS
        meta["off_table_z"] = OPENYAM_OFF_TABLE_Z
        meta["workspace"] = workspace_frame(env, "overhead")

    stop = STOP_EVENT
    if STOP_FILE:
        def watch():
            while not stop.is_set():
                if os.path.exists(STOP_FILE):
                    print("  Stopped by user.", flush=True)
                    meta["stopped_by_user"] = True
                    stop.set()
                time.sleep(0.3)

        threading.Thread(target=watch, daemon=True).start()
    if KSY:
        start_ksy(env, args, meta, stop, source_dir)
    elif args.plain and not openyam and not partial and not args.no_record:
        record_plain(env, args, meta)

    print(f"Starting the simulator: {args.task} ({args.seconds:g} s)"
          + {"human": " · you have the arms (T hands them to Dream-1)", "resumed": " · you have the arms",
             "collect": " · collecting: Dream-1, then the recovery script; no human",
             "script": " · rule-based only: the recovery script from the start; no Dream-1",
             "test": " · press T to take over"}.get(mode, ""), flush=True)
    RECORDING["on"] = True
    try:
        if SCRIPT_ONLY:
            if not MONITOR.understood():
                raise RuntimeError("The rule-based script needs a pick-and-place task or a goal scene it understands.")
            script_only(env, args.seconds, stop)
        elif args.teleop:
            left = keyboard_only(env, args.seconds, stop)
            if left is not None and left > 1 and not stop.is_set():
                # Hand-off: Dream-1 continues from the scene exactly as the human left it, for the time that is left.
                print(f"  Handing the arms to Dream-1 ({left:.0f} s left).", flush=True)
                dream_control.main(["--task", args.task, "--seconds", f"{left:.1f}", "--headless", "--quiet",
                                    "--model", args.model],
                                   shared_env=env, stop_event=stop)
        else:
            dream_control.main(["--task", args.task, "--seconds", str(args.seconds), "--headless", "--quiet",
                                "--model", args.model],
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
