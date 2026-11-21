"""Run one Dream-1 task in the Dream Labs simulator (headless).

    python sim.py "put apple into the bowl" --seconds 60

app.py starts this for every run. Two environment variables connect it to the website:
  LIVE_FRAME  path of a JPEG that is replaced with the latest camera view (top camera + both wrist cameras)
  STOP_FILE   when this file appears, the run stops and the Dream session is released
"""

import argparse
import importlib.util
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
import dream_control  # noqa: E402
from PIL import Image  # noqa: E402

import scene  # noqa: E402  random start scene from a seed (--seed, --split): see scene.py

LIVE_FRAME = os.environ.get("LIVE_FRAME")
STOP_FILE = os.environ.get("STOP_FILE")


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
    return dict(last_images)


# A connection the server dropped while idle is reopened by asking again (up to twice).
original_get_action = dream.DreamClient.get_action


def get_action(self, *args, **kwargs):
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


def main():
    parser = argparse.ArgumentParser(description="Run one Dream-1 task in the simulator")
    parser.add_argument("task", nargs="?", default="put apple into the bowl")
    parser.add_argument("--seconds", type=float, default=60)
    scene.add_arguments(parser)
    args = parser.parse_args()
    env = scene.prepare_env(args)

    stop = threading.Event()
    if STOP_FILE:
        def watch():
            while not stop.is_set():
                if os.path.exists(STOP_FILE):
                    print("  Stopped by user.", flush=True)
                    stop.set()
                time.sleep(0.3)

        threading.Thread(target=watch, daemon=True).start()

    print(f"Starting the simulator: {args.task} ({args.seconds:g} s)", flush=True)
    try:
        dream_control.main(["--task", args.task, "--seconds", str(args.seconds), "--headless", "--quiet"],
                           shared_env=env, stop_event=stop)
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        print(f"Stopped: {error}", flush=True)
        sys.exit(1)
    finally:
        stop.set()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
