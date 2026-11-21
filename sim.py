"""Run one Dream-1 task in the Dream Labs simulator (headless).

    python sim.py "put apple into the bowl" --seconds 60

    python sim.py "put apple into the bowl" --teleop     # keyboard only, no Dream-1 (no API key or credits)

    python sim.py "put apple into the bowl" --recover human   # on a failure ask a human at once (default: the
                                                              # recovery script tries first, see recovery.py)

    python sim.py --resume 20261121-175222-5395 --resume-step 120   # restart a recorded run at step 120, exactly as
                                                                     # it was, with a human in control (T: Dream-1)

    python sim.py "put apple into the bowl" --collect   # data collection: Dream-1, then the recovery script when it
                                                        # fails; no human. If the script gives up, the run ends there
                                                        # (a recorded failure, for a human to resume later)

Every run is recorded to episodes/<id>/ (recorder.py): states, actions, who controlled each step (Dream-1, human,
script or hold), the human's inputs, the cameras, and the full physics state before every step. --no-record skips it.

app.py starts this for every run. Environment variables connect it to the website:
  LIVE_FRAME    path of a JPEG that is replaced with the latest camera view (top camera + both wrist cameras)
  STOP_FILE     when this file appears, the run stops and the Dream session is released
  TELEOP_STATE  path of a JSON file replaced with the keyboard control state (mode, arm, gripper, hand target)
Keyboard input arrives on stdin, one JSON object per line: {"held": ["w", "a"]} or {"press": "t"} (see teleop.py).
"""

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# The simulator ships inside the dreamlabs package (dream_examples/yam_mujoco) and imports its modules by name.
spec = importlib.util.find_spec("dream_examples")
if spec is None:
    sys.exit("The simulator is missing: run  pip install dreamlabs")
sys.path.insert(0, str(Path(next(iter(spec.submodule_search_locations))) / "yam_mujoco"))

import dream  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import dream_control  # noqa: E402
from PIL import Image  # noqa: E402
from yam_env import YamEnv  # noqa: E402

from monitor import Monitor  # noqa: E402
from recorder import EPISODES_DIR, PHYSICS, EpisodeRecorder  # noqa: E402
from recovery import Recovery  # noqa: E402
from teleop import STEP_S, Teleop, camera_position, select  # noqa: E402

LIVE_FRAME = os.environ.get("LIVE_FRAME")
STOP_FILE = os.environ.get("STOP_FILE")
TELEOP_STATE = os.environ.get("TELEOP_STATE")
TELEOP = Teleop()
MONITOR = None  # set in main() once the task is known
RECOVERY = None  # the recovery script (recovery.py); None with --recover human
RECORDER = None  # the episode recorder (recorder.py); None with --no-record
COLLECT = False  # --collect: data collection, model then recovery script, no human input
DREAM_STARTED = [False]  # Dream-1 was asked for a real chunk: before that, the loop only fills its camera window
HOLD_ROWS = 20  # while a human has the arms, Dream-1 is not called: the loop gets 1 s of "hold" rows at a time


def save_live_frame(images, path):
    """The top camera on the left, the two wrist cameras stacked on the right, as one JPEG (replaced atomically)."""
    canvas = Image.new("RGB", (720, 270))
    canvas.paste(Image.fromarray(images["top"]).resize((480, 270)), (0, 0))
    canvas.paste(Image.fromarray(images["left"]).resize((240, 135)), (480, 0))
    canvas.paste(Image.fromarray(images["right"]).resize((240, 135)), (480, 135))
    canvas.save(path + ".tmp", format="JPEG", quality=80)
    os.replace(path + ".tmp", path)


# Render the cameras only on the steps the model sees (1 in 4); the others reuse the last images. Much faster.
original_capture = dream_control.capture
last_images = {}


def capture(env, windows, keep=True):
    if keep or not last_images:
        last_images.update(original_capture(env, windows, keep))
        if LIVE_FRAME:
            save_live_frame(last_images, LIVE_FRAME)
        if keep and RECORDER is not None:
            RECORDER.on_frame(last_images)  # the frames the model sees, 5 per second
    return dict(last_images)


# A connection the server dropped while idle is reopened by asking again (up to twice).
original_get_action = dream.DreamClient.get_action


def get_action(self, *args, **kwargs):
    # While a human has the arms (or the monitor is waiting for one), Dream-1 is not asked: no credits, no waiting,
    # and the loop keeps running at 20 Hz so the human's actions reach the simulator. The rows returned here are
    # replaced by the human's actions anyway. A hand-back completes when Dream-1 is asked again (a fresh chunk).
    if not TELEOP.chunk_started():
        left, right = TELEOP.last_action
        return {"q_l_chunk": np.tile(left, (HOLD_ROWS, 1)), "q_r_chunk": np.tile(right, (HOLD_ROWS, 1))}
    DREAM_STARTED[0] = True
    if RECORDER is not None and RECORDER.model_id is None:
        RECORDER.model_id = getattr(dream_control, "BETA_MODEL", "dream-1")
    for attempt in range(3):
        try:
            return original_get_action(self, *args, **kwargs)
        except dream.DreamProtocolError:
            if attempt == 2:
                raise
            print("  Connection dropped; asking again.", flush=True)
            time.sleep(0.2)


dream_control.capture = capture
dream.DreamClient.get_action = get_action


def read_keys(env, stop):
    """Keyboard input from app.py, one JSON object per line on stdin."""
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
    """No Dream-1: the human drives the arms from the start, 20 Hz, cameras at 10 Hz.
    @return The seconds left when the human handed the arms to Dream-1 (key T), else None."""
    TELEOP.start_human(env)
    deadline = time.monotonic() + seconds
    steps = 0
    while not stop.is_set() and time.monotonic() < deadline:
        if TELEOP.mode == "returning":
            return deadline - time.monotonic()
        tick = time.monotonic()
        env.step(None, None)  # the keyboard replaces the action (see the env.step wrapper in main)
        if steps % 2 == 0 and (LIVE_FRAME or RECORDER is not None):
            images = env.images()
            if LIVE_FRAME:
                save_live_frame(images, LIVE_FRAME)
            if steps % 4 == 0 and RECORDER is not None:
                RECORDER.on_frame(images)  # 5 per second, as in Dream-1 runs
        steps += 1
        time.sleep(max(0, STEP_S - (time.monotonic() - tick)))


def main():
    parser = argparse.ArgumentParser(description="Run one Dream-1 task in the simulator")
    parser.add_argument("task", nargs="?", default="put apple into the bowl")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--teleop", action="store_true", help="keyboard only, no Dream-1")
    parser.add_argument("--no-record", action="store_true", help="do not save the run to episodes/")
    parser.add_argument("--collect", action="store_true",
                        help="data collection: Dream-1, then the recovery script on a failure; no human input. When "
                             "the script gives up (or the object falls off the table) the run ends there")
    parser.add_argument("--resume", metavar="EPISODE", help="start from a recorded episode (with --resume-step); "
                        "a human has the arms, T hands them to Dream-1")
    parser.add_argument("--resume-step", type=int, default=0, help="the step of --resume to start from")
    parser.add_argument("--recover", choices=("script", "human"), default="script",
                        help="when Dream-1 fails: the recovery script tries first, a human if it gives up (script), "
                             "or a human right away (human)")
    args = parser.parse_args()

    # Every action goes through the human controller, which passes Dream-1's on or replaces them, and every step
    # is watched by the failure monitor, which stops Dream-1 when it fails: the recovery script tries first, and the
    # arms are held for a human if it gives up (or at once, with --recover human).
    global MONITOR, RECOVERY, RECORDER, COLLECT
    COLLECT = args.collect and not args.teleop and not args.resume
    if COLLECT:
        args.recover = "script"
    env = YamEnv()
    resume = None
    if args.resume:
        # A recorded episode, restored exactly: its start (for the home pose and the monitor's resting heights),
        # then the full physics state before the chosen step. The run continues with a human in control.
        source_dir = EPISODES_DIR / Path(args.resume).name
        recorded = np.load(source_dir / "steps.npz")
        meta = json.loads((source_dir / "meta.json").read_text())
        step_k = max(0, min(int(args.resume_step), len(recorded["t"]) - 1))
        args.task, args.teleop = meta.get("task", args.task), True
        resume = {"episode": source_dir.name, "step": step_k, "t": float(recorded["t"][step_k - 1]) if step_k else 0.0}
        mujoco.mj_setState(env.model, env.data, recorded["physics"][0], PHYSICS)
        mujoco.mj_forward(env.model, env.data)
    TELEOP.set_home(env)
    MONITOR = Monitor(args.task, env)
    if resume:
        mujoco.mj_setState(env.model, env.data, recorded["physics"][resume["step"]], PHYSICS)
        mujoco.mj_forward(env.model, env.data)
        if resume["step"] > 0:  # the arms hold what they were last commanded, not where gravity has them
            last = recorded["action"][resume["step"] - 1]
            TELEOP.last_action = (last[:7].copy(), last[7:].copy())
    if not args.no_record:
        RECORDER = EpisodeRecorder(env, args.task,
                                   mode="resumed" if resume else "human-only" if args.teleop else "dream",
                                   extra={"recover": args.recover, "time_limit_s": args.seconds,
                                          "resumed_from": resume})
        if resume:
            RECORDER.load_prefix(source_dir, resume["step"])  # the new episode starts with the original's steps
        TELEOP.on_control = lambda mode, reason: RECORDER.event("control", mode=mode, reason=reason)
        # Steps between a grip_hold start and end: the gripper column of the action is the hold, not Dream-1's command.
        TELEOP.on_grip_hold = lambda side, what, why: RECORDER.event("grip_hold", side=side, what=what, why=why)
        print(f"  Recording: episodes/{RECORDER.id}", flush=True)
    if args.recover == "script":
        RECOVERY = Recovery(TELEOP, MONITOR)
    if not MONITOR.understood():
        print("  The failure monitor does not understand this task; it will not ask for help.", flush=True)
    step = env.step
    stop = threading.Event()
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
            save_live_frame(env.images(), LIVE_FRAME)
        if previous_mode[0] in ("human", "script") and mode in ("returning", "dream"):
            MONITOR.recovered()  # the human or the script gave the arms back: watch Dream-1 again from here
        previous_mode[0] = mode
        event = MONITOR.update(env, watching=(mode == "dream" and TELEOP.dream_available))
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
        if COLLECT and not stop.is_set() and (TELEOP.mode == "paused" or event == "off_table"):
            # Data collection asks no human: a failure the script could not recover ends the run here. The recording
            # is a failure to recover by hand later, from the Recordings panel (it resumes before the model failed).
            why = "the object fell off the table" if event == "off_table" else "the recovery script gave up"
            print(f"  Ending the run: {why}. Recover it later from the Recordings panel.", flush=True)
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

    if STOP_FILE:
        def watch():
            while not stop.is_set():
                if os.path.exists(STOP_FILE):
                    print("  Stopped by user.", flush=True)
                    stop.set()
                time.sleep(0.3)

        threading.Thread(target=watch, daemon=True).start()
    if not sys.stdin.isatty():
        threading.Thread(target=read_keys, args=(env, stop), daemon=True).start()
    if TELEOP_STATE:
        threading.Thread(target=write_state, args=(stop,), daemon=True).start()

    if resume:
        print(f"  Resuming {resume['episode']} at step {resume['step']} ({resume['t']:.2f} s): you have the arms.",
              flush=True)
    print(f"Starting the simulator: {args.task} ({args.seconds:g} s)"
          + (" · keyboard only" if args.teleop else " · collecting: Dream-1, then the recovery script; no human"
             if COLLECT else " · press T to take over"), flush=True)
    try:
        if args.teleop:
            left = keyboard_only(env, args.seconds, stop)
            if left is not None and left > 1 and not stop.is_set():
                # Hand-off: Dream-1 continues from the scene exactly as the human left it, for the time that is left.
                print(f"  Handing the arms to Dream-1 ({left:.0f} s left).", flush=True)
                dream_control.main(["--task", args.task, "--seconds", f"{left:.1f}", "--headless", "--quiet"],
                                   shared_env=env, stop_event=stop)
        else:
            dream_control.main(["--task", args.task, "--seconds", str(args.seconds), "--headless", "--quiet"],
                               shared_env=env, stop_event=stop)
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        print(f"Stopped: {error}", flush=True)
        failure = error
        sys.exit(1)
    finally:
        stop.set()
        if RECORDER is not None:
            saved = RECORDER.close(MONITOR.status(), locals().get("failure"))
            if saved:
                print(f"  Saved: episodes/{saved.name} ({len(RECORDER.rows['t'])} steps)", flush=True)
        env.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
