"""A small website for running Dream-1 in the Dream Labs simulator.

    python app.py              then open http://localhost:8000

Press Run on the page: sim.py runs the task in its own process, the page shows the cameras live, and the run's
trajectory is saved in runs/<id>/. The start scene comes from a seed (scene.py; see SCENE_SEED.md) with the scene
settings of ood.py (camera, arm angles, blur, light, table; any of them drawn from the seed when marked random), or
from a custom layout set on the page. /api/preview renders a start scene without the Dream API.

The page judges every saved run against the success rules set on it (target distance, release, time standing
still); the rules are kept in runs/rules.json and each run's label in runs/<id>/label.json. A batch runs many
episodes in the background (no camera view), one seed each (seed, seed + 1, ... or the fixed eval seeds), and
labels each one with the rules as they were when the batch started.

Run modes (KSY): data collection (Dream-1, then the recovery script when it fails; no human), test (Dream-1 and a
human take turns with T), human only, and Dream-1 only. The page sends keys, clicks, drags and the wheel to sim.py
(stdin), which moves the arms by hand (teleop.py); its control state comes back through TELEOP_STATE. Runs in those
modes are also recorded to episodes/<run id>/ (recorder.py), and any recording can be resumed at any step with a
human in control (Recordings). Standard library only, plus numpy, PIL and PyAV (installed with dreamlabs) to show a
recording's frames.
"""

import argparse
import json
import ast
import os
import urllib.parse
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import judge
import lerobot_export  # the Dream Labs dataset format (LeRobot v2.1) for saved datasets
import ood
import seed_web
import simulator_settings as sim_settings  # the seed (scene.py): seed field, random scene controls

HERE = Path(__file__).resolve().parent
PAGE = HERE / "static" / "index.html"
RUNS = HERE / "runs"
RULES = RUNS / "rules.json"
SETTINGS = RUNS / "simulator_settings.json"  # the robot and room picked on the page (simulator_settings.py)
DEFAULT_RULES = judge.DEFAULT_RULES
## What can be on the table, and the area objects may be placed in (metres, world frame; reachable by both arms).
OBJECTS = ["apple", "orange", "banana", "water_bottle"]
TARGETS = ["bowl", "plate"]
AREA = {"x": (-0.30, 0.30), "y": (-0.22, 0.18)}
MAX_BATCH = 500
FULL_TABLE = {"items": {name: None for name in OBJECTS + TARGETS}}  # a seed's scene has all six items
ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")


def live_frame(run_id):
    """Where a batch episode writes its latest camera view (the single run uses LIVE)."""
    return WORK / f"live-{run_id}.jpg"
WORK = Path(tempfile.mkdtemp(prefix="dream-web-"))
LIVE = WORK / "live.jpg"
STOP = WORK / "stop"
TELEOP_STATE = WORK / "teleop.json"
EPISODES = HERE / "episodes"  # KSY's recordings (recorder.py), one per run id
DATASETS = HERE / "datasets"  # datasets saved from the page: chosen runs, their recordings and a manifest
MODES = ("collect", "script", "test", "human", "plain")  # script: rule-based only (sim.py --script-only), no Dream-1
NO_KEY_MODES = ("human", "script")  # runs that do not call Dream-1 (a human-only run calls it only after T)
## Data collection: how fast a failure is handed to the recovery script, and what it does (sim.py's options).
COLLECTION = {"stall_s": 6.0, "quick_fail": True, "script_goal": "finish", "script_speed": 1.0, "end_on_success": True}


def collection_args(given):
    """sim.py's data collection options from the page's settings (defaults: COLLECTION).
    @return (the settings, the arguments). @throws ValueError"""
    c = {**COLLECTION, **(given or {})}
    try:
        stall, speed = float(c["stall_s"]), float(c["script_speed"])
    except (TypeError, ValueError) as error:
        raise ValueError("Collection settings: numbers expected.") from error
    if not 2 <= stall <= 30:
        raise ValueError("Hand over after 2 to 30 s without progress.")
    if not 1 <= speed <= 3:
        raise ValueError("The script's speed is 1 to 3 times Dream-1's.")
    if c["script_goal"] not in ("carry", "finish"):
        raise ValueError("The script's goal is carry or finish.")
    c = {"stall_s": stall, "quick_fail": bool(c["quick_fail"]), "script_goal": c["script_goal"],
         "script_speed": speed, "end_on_success": bool(c["end_on_success"])}
    args = ["--stall-s", f"{stall:g}", "--script-goal", c["script_goal"], "--script-speed", f"{speed:g}"]
    args += (["--quick-fail"] if c["quick_fail"] else []) + (["--end-on-success"] if c["end_on_success"] else [])
    return c, args
PREVIEW_LOCK = threading.Lock()
## Tasks without a bowl or plate are run but not judged (the judge needs an object and a target).
EXAMPLES = [
    "Move an object to the center of the workspace",
    "Move an object to the left side of the workspace",
    "Move an object to the right side of the workspace",
]


def eval_seeds():
    """@return scene.EVAL_SEEDS, the 50 fixed evaluation seeds, read from scene.py without importing it (app.py
    stays light: scene.py needs numpy, shapely and MuJoCo)."""
    tree = ast.parse((HERE / "scene.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "EVAL_SEEDS" for t in node.targets):
            return list(ast.literal_eval(node.value))
    return []


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(path.with_suffix(".tmp"), path)


def current_rules():
    """Every task type's rules (runs/rules.json; types or values not saved keep their defaults)."""
    try:
        return judge.normalize_rules(read_json(RULES, {}) or {})
    except (TypeError, ValueError):
        return judge.normalize_rules({})


def current_settings():
    """The robot and room picked on the page (runs/simulator_settings.json)."""
    try:
        return sim_settings.normalize(read_json(SETTINGS, {}) or {})
    except ValueError:
        return dict(sim_settings.DEFAULT)


def robot_args(changes, seeding, scene):
    """sim.py's --robot / --room for the picked settings, checked against what can run here.
    @return The arguments. @throws ValueError When the robot runs only in the Dream Labs Sandbox, or when OpenYAM
    is asked for YAM-only things (scene settings, settings marked random, a custom layout)."""
    settings = current_settings()
    if not sim_settings.local_supported(settings):
        raise ValueError(f"{sim_settings.ROBOTS[settings['robot']]} runs only in the Dream Labs Sandbox "
                         f"({sim_settings.catalog()['sandbox_url']}). Choose YAM or OpenYAM to run here.")
    if settings["robot"] == "openyam" and (changes or scene or "--ood-random" in seeding):
        raise ValueError("Scene settings, random settings and custom layouts are for YAM; OpenYAM uses the Sandbox table.")
    return ["--robot", settings["robot"], "--room", settings["room"]]


def run_dir(run_id):
    """The folder of a saved run, or None for an unknown or malformed id."""
    if not ID_RE.match(run_id or ""):
        return None
    d = RUNS / run_id
    return d if (d / "meta.json").exists() else None


## The API key typed on the page: the only key the website uses. Kept in this process's memory only (never written
## to disk or logged) and handed to each run as DREAM_API_KEY. Clear on the page removes it.
PAGE_KEY = {"value": None}
KEY_RE = re.compile(r"^dl_[A-Za-z0-9_-]{16,500}$")  # Dream API keys: dl_live_..., letters, digits, _ and -


def key_problem(key):
    """Why a value typed into the API key box is not a Dream API key (None when it looks like one). Catches pastes
    of something else, such as a command-line option, before a run or batch is refused with UNAUTHORIZED."""
    if not key.startswith("dl_"):
        return "That is not a Dream API key: keys start with dl_ (for example dl_live_…)."
    if any(c.isspace() for c in key):
        return "That is not a Dream API key: it has spaces or line breaks in it."
    if not KEY_RE.match(key):
        return "That is not a Dream API key: it is too short or has characters a key does not have."
    return None


def key_status():
    """@return {"source": "page" or None, "hint": the key's last 4 characters}."""
    if PAGE_KEY["value"]:
        return {"source": "page", "hint": PAGE_KEY["value"][-4:]}
    return {"source": None, "hint": None}


def has_api_key():
    return key_status()["source"] is not None


class Runner:
    """One run at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.task = None
        self.started = None
        self.ended = None
        self.run_id = None
        self.log = deque(maxlen=30)

    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, task, seconds, changes=None, seeding=(), scene=None, live=True, extra_env=None, robot=(),
              mode="plain", record=True, resume=None, collection=()):
        """Start one run of sim.py.
        @param changes Scene settings (ood.validate output).
        @param seeding sim.py's seed arguments (seed_web.seeding_args): --seed, --ood-random.
        @param scene A custom layout (parse_scene), which replaces the seed's; seeding is then not used.
        @param live Whether to write the camera view for the page (batches do not).
        @param extra_env More environment variables for sim.py (BATCH_ID, JUDGE_SPEC).
        @param mode "collect", "script", "test", "human" or "plain" (see sim.py). @param record Record to episodes/ (KSY modes).
        @param resume (run id, step): restart that recording there with a human in control.
        """
        with self.lock:
            if self.running():
                raise RuntimeError("A run is already going.")
            if not has_api_key() and mode not in NO_KEY_MODES and not resume:
                raise RuntimeError("Enter your API key first.")
            self.last_start = dict(task=task, seconds=seconds, changes=changes, seeding=seeding, scene=scene,
                                   live=live, extra_env=extra_env, robot=robot, mode=mode, record=record, resume=resume,
                                   collection=collection)
            TELEOP_STATE.unlink(missing_ok=True)
            STOP.unlink(missing_ok=True)
            LIVE.unlink(missing_ok=True)
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore",
                   "STOP_FILE": str(STOP), "RUN_DIR": str(RUNS / run_id), "TELEOP_STATE": str(TELEOP_STATE),
                   **(extra_env or {})}
            if live:
                env["LIVE_FRAME"] = str(LIVE)
            if PAGE_KEY["value"]:
                env["DREAM_API_KEY"] = PAGE_KEY["value"]
            command = [sys.executable, str(HERE / "sim.py"), task, "--seconds", str(seconds), *robot]
            if changes:
                command.extend(["--ood", json.dumps(changes)])
            if resume:  # sim.py builds the original's scene from its runs/<id>/meta.json
                command = [sys.executable, str(HERE / "sim.py"), task, "--seconds", str(seconds),
                           "--resume", resume[0], "--resume-step", str(int(resume[1]))]
            elif scene:
                env["SCENE_JSON"] = json.dumps(scene)  # sim.py then builds the default scene with this layout
            else:
                command.extend(seeding)
            if not resume:
                command += {"collect": ["--collect", *collection], "script": ["--script-only", *collection],
                            "test": ["--recover", "human"],
                            "human": ["--teleop"], "plain": ["--plain"]}[mode]
            if not record:
                command.append("--no-record")
            self.process = subprocess.Popen(
                command, cwd=str(HERE), env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            self.task, self.started, self.ended, self.run_id = task, time.time(), None, run_id
            self.mode = "resumed" if resume else mode
            self.log.clear()
            threading.Thread(target=self._read, args=(self.process,), daemon=True).start()

    def _read(self, process):
        for line in process.stdout:
            line = line.strip()
            # Sub-0.1 s "RTT" lines are the hold rows sim.py gives the loop while a human has the arms (no API call).
            hold = re.search(r"Dream · (\d+\.\d+)s RTT", line)
            if hold and float(hold.group(1)) < 0.1:
                continue
            if line and "Warning" not in line and "warnings.warn" not in line:
                self.log.append(line)
        process.wait()
        self.ended = time.time()

    def stop(self):
        if self.running():
            STOP.touch()

    def stop_and_wait(self, timeout=30):
        """Stop the current run, if any, and wait until it has saved its recording and exited."""
        process = self.process
        if process is None or process.poll() is not None:
            return
        self.stop()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def restart(self):
        """Reset: stop the run (a Dream run first releases its session) and start the same one again from its start."""
        if not getattr(self, "last_start", None):
            raise RuntimeError("Nothing to reset: press Run first.")
        self.stop_and_wait()
        self.start(**self.last_start)

    def send_keys(self, message):
        """Pass keyboard and mouse input to the running sim.py (one JSON line on its stdin)."""
        with self.lock:
            if not self.running():
                return False
            try:
                self.process.stdin.write(json.dumps(message) + "\n")
                self.process.stdin.flush()
                return True
            except (OSError, ValueError):
                return False

    def teleop_state(self):
        try:
            return json.loads(TELEOP_STATE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def status(self):
        running = self.running()
        return {
            "running": running,
            "task": self.task,
            "mode": getattr(self, "mode", None),
            "teleop": self.teleop_state() if running else None,
            "elapsed_s": round((self.ended or time.time()) - self.started) if self.started else None,
            "exit_code": None if running or self.process is None else self.process.returncode,
            "run_id": self.run_id,
            "log": list(self.log),
            "key": key_status(),
            "examples": EXAMPLES,
            "batch": BATCH.status(),
        }


RUNNER = Runner()


def parse_scene(scene):
    """Check a scene from the page: {"items": {name: [x, y]}} with at least one object and one bowl or plate.
    @return The scene with positions clamped to AREA, or None for the simulator's own scene.
    @throws ValueError When it names an unknown item or has no object or no target.
    """
    if not scene:
        return None
    items = {}
    for name, position in (scene.get("items") or {}).items():
        if name not in OBJECTS + TARGETS:
            raise ValueError(f"Unknown item: {name}")
        x, y = (float(v) for v in position)
        items[name] = [round(min(max(x, AREA["x"][0]), AREA["x"][1]), 4), round(min(max(y, AREA["y"][0]), AREA["y"][1]), 4)]
    if not any(name in items for name in OBJECTS):
        raise ValueError("Put at least one object on the table.")
    if not any(name in items for name in TARGETS):
        raise ValueError("Put a bowl or a plate on the table.")
    return {"items": items}


LABELS = ("success", "failure", "invalid")


def list_episodes():
    """Recordings (episodes/), newest first: what the page needs to pick one and a step to resume from (KSY)."""
    rows = []
    for folder in sorted(EPISODES.glob("*"), reverse=True) if EPISODES.is_dir() else []:
        if not ID_RE.match(folder.name) or not (folder / "steps.npz").exists():
            continue
        meta, outcome = read_json(folder / "meta.json", {}), read_json(folder / "outcome.json", {})
        events = read_json(folder / "events.json", []) or []
        rows.append({
            "id": folder.name, "task": meta.get("task"), "started_as": meta.get("started_as"),
            "steps": meta.get("steps"), "sim_s": meta.get("sim_s"), "label": outcome.get("label"),
            "final_stage": outcome.get("final_stage"),
            "failures": [{"event": f.get("event"), "step": f.get("step")} for f in outcome.get("failures_seen") or []],
            "steps_by_source": meta.get("steps_by_source"), "resumed_from": meta.get("resumed_from"),
            "ended": next((e.get("reason") for e in events if e.get("kind") == "ended"), None),
            "script": [e.get("result") for e in events if e.get("kind") == "script" and e.get("result")],
        })
    return rows


FRAMES = {}  # episode id -> (frame_t, step times, top camera frames), the last few looked at


def episode_frame(episode, step):
    """The top camera picture at a step of a recording (the last frame at or before it), as JPEG."""
    import io

    import numpy as np
    from PIL import Image

    if episode not in FRAMES:
        import av

        folder = EPISODES / episode
        data = np.load(folder / "steps.npz")
        with av.open(str(folder / "top.mp4")) as container:
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        if len(FRAMES) >= 3:
            FRAMES.pop(next(iter(FRAMES)))
        FRAMES[episode] = (data["frame_t"], data["t"], frames)
    frame_t, t, frames = FRAMES[episode]
    step = max(0, min(int(step), len(t) - 1))
    before = float(t[step - 1]) if step > 0 else 0.0  # simulated time at the start of the step
    index = max(0, min(int(np.searchsorted(frame_t, before + 1e-9)) - 1, len(frames) - 1))
    buffer = io.BytesIO()
    Image.fromarray(frames[index]).save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def delete_runs(run_ids):
    """Delete finished runs: runs/<id> (trajectory, labels) and its recording episodes/<id>. The run going now is
    never deleted. @return (deleted ids, skipped ids)."""
    deleted, skipped = [], []
    for run_id in run_ids:
        busy = (RUNNER.running() and RUNNER.run_id == run_id) or run_id in BATCH.active_ids()
        folders = [folder / run_id for folder in (RUNS, EPISODES)]
        if not ID_RE.match(run_id) or busy or not any(f.exists() for f in folders):
            skipped.append(run_id)
            continue
        for folder in folders:
            if folder.is_dir():
                shutil.rmtree(folder)
        FRAMES.pop(run_id, None)
        for key in [k for k in VERDICTS if k[0] == run_id]:
            VERDICTS.pop(key, None)
        deleted.append(run_id)
    return deleted, skipped


SETTING_KEYS = ("mode", "robot", "room", "seed", "split", "setting", "scene_version", "layout", "ood", "ood_random",
                "scene", "collection", "seconds", "batch_id", "resumed_from")


def run_settings(meta):
    """How a run was set up (what the page shows, and what a dataset's manifest keeps)."""
    return {key: meta.get(key) for key in SETTING_KEYS}


def link_tree(source, target):
    """Copy a folder as hard links (no extra disk, and it survives deleting the run); plain copies where links fail."""
    def link(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    shutil.copytree(source, target, copy_function=link)


def save_dataset(run_ids, name, dream_format=False, location=None):
    """A dataset of the chosen runs: datasets/<time>-<name>/ with runs/<id>/ (trajectory, labels, meta), episodes/<id>/
    (recording: steps, physics, cameras) and manifest.json (each run's label, task, judge spec and settings).
    @param dream_format Also write the recordings in the Dream Labs format (LeRobot v2.1, lerobot_export.py) to
        dream_format/, with embodiment.json, and run the Dream Labs checks on it.
    @param location The folder on this computer to save it in (a new <time>-<name>/ inside it); None: datasets/.
    @return (folder, manifest). @throws ValueError when no chosen run can be saved."""
    rules = current_rules()
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(name or "")).strip("-")[:40] or "dataset"
    folder = save_location(location) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug}"
    if folder.exists():
        raise ValueError(f"{folder} already exists.")
    entries, skipped = [], []
    busy = BATCH.active_ids() | ({RUNNER.run_id} if RUNNER.running() else set())
    for run_id in dict.fromkeys(run_ids):
        d = run_dir(run_id) if isinstance(run_id, str) else None
        if not d or run_id in busy:
            skipped.append(run_id)
            continue
        meta = read_json(d / "meta.json", {})
        verdict, hand = run_verdict(d, rules), read_json(d / "label.json") or {}
        by_hand = hand.get("source") == "human"
        link_tree(d, folder / "runs" / run_id)
        if (EPISODES / run_id).is_dir():
            link_tree(EPISODES / run_id, folder / "episodes" / run_id)
        entries.append({
            "id": run_id, "task": meta.get("task"),
            "label": hand["label"] if by_hand else verdict["label"],
            "label_source": "human" if by_hand else "rules",
            "reason": hand.get("reason") if by_hand else verdict["reason"],
            "rules_verdict": {"label": verdict["label"], "reason": verdict["reason"]},
            "judge": verdict.get("spec"), "success_time_s": verdict.get("success_time_s"),
            "sim_seconds": meta.get("sim_seconds"), "steps_by_source": meta.get("steps_by_source"),
            "recorded": (EPISODES / run_id).is_dir(), "settings": run_settings(meta),
        })
    if not entries:
        raise ValueError("None of the chosen runs can be saved (still going, or gone).")
    counts = {label: sum(e["label"] == label for e in entries) for label in LABELS}
    manifest = {"name": slug, "created_at": datetime.now().isoformat(timespec="seconds"), "rules": rules,
                "count": len(entries), "labels": counts, "skipped": skipped, "runs": entries,
                "layout": {"runs/<id>/": "meta.json, trajectory.json (object poses, contacts per step), label.json, last.jpg",
                           "episodes/<id>/": "steps.npz (states, actions, who sent each step, physics), events.json, "
                                             "top/left/right.mp4, meta.json, outcome.json (recorded runs only)"}}
    if dream_format:
        manifest["dream_format"] = export_dream_format(folder, entries)
        manifest["layout"]["dream_format/"] = ("the recordings in the Dream Labs format (LeRobot v2.1): upload with "
                                               "client.upload_dataset(..., embodiment), embodiment.json holds its config")
    write_json(folder / "manifest.json", manifest)
    return folder, manifest


def export_dream_format(folder, entries):
    """The dataset's recordings as one LeRobot v2.1 dataset in folder/dream_format/, checked with the Dream Labs checks.
    Each manifest entry gets its episode_index there. @return What was written, skipped and the checks' report."""
    recorded = [e for e in entries if e["recorded"]]
    result = lerobot_export.export([(folder / "episodes" / e["id"], e["task"]) for e in recorded],
                                   folder / "dream_format", robot_type=(recorded[0]["settings"].get("robot") or "yam")
                                   if recorded else "yam")
    by_source = {w["source"]: w["episode_index"] for w in result["episodes"]}
    for e in entries:
        e["dream_episode"] = by_source.get(e["id"])
    skipped = result["skipped"] + [{"source": e["id"], "reason": "not recorded (plain run)"}
                                   for e in entries if not e["recorded"]]
    write_json(folder / "embodiment.json", {
        "note": "client.add_embodiment(<name>, config); then client.upload_dataset(<folder>/dream_format, <dataset "
                "name>, <embodiment name>). State and action: right arm first, joints in rad, gripper 0 closed .. 1 "
                "open; 5 frames per second.",
        "config": lerobot_export.EMBODIMENT})
    return {"folder": "dream_format/", "format": "LeRobot v2.1", "fps": lerobot_export.FPS,
            "episodes": len(result["episodes"]), "frames": result["frames"], "skipped": skipped,
            "check": lerobot_export.validate(folder / "dream_format") if result["episodes"] else
            {"ok": False, "error": "No recorded run to convert."}}


def save_location(location):
    """The folder a dataset is saved in. @throws ValueError when it is not an existing folder this server can write."""
    if not location:
        return DATASETS
    base = Path(str(location)).expanduser()
    if not base.is_absolute():
        raise ValueError("Choose a full path for the save location (e.g. /Users/me/data).")
    base = base.resolve()
    if not base.is_dir():
        raise ValueError(f"{base} is not a folder.")
    if not os.access(base, os.W_OK):
        raise ValueError(f"Cannot write in {base}.")
    return base


def shown_path(folder):
    """A saved dataset's folder as the page shows it: datasets/<name> in this project, else its full path."""
    try:
        return str(folder.relative_to(HERE))
    except ValueError:
        return str(folder)


def folder_listing(path):
    """The folders inside one folder on this computer, for choosing where to save a dataset (hidden ones left out).
    @return {path, parent, folders, writable, places}. @throws ValueError when it is not a readable folder."""
    DATASETS.mkdir(exist_ok=True)
    folder = Path(str(path or DATASETS)).expanduser()
    if not folder.is_absolute():
        raise ValueError("Use a full path.")
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder.")
    try:
        names = sorted((c.name for c in folder.iterdir() if c.is_dir() and not c.name.startswith(".")), key=str.lower)
    except OSError as error:
        raise ValueError(f"Cannot open {folder}: {error.strerror or error}.")
    home = Path.home()
    places = [{"name": "Project datasets", "path": str(DATASETS)}, {"name": "Home", "path": str(home)}]
    places += [{"name": n, "path": str(home / n)} for n in ("Desktop", "Documents", "Downloads") if (home / n).is_dir()]
    return {"path": str(folder), "parent": str(folder.parent) if folder.parent != folder else None,
            "folders": names[:500], "more": max(0, len(names) - 500), "writable": os.access(folder, os.W_OK),
            "places": places}


def list_datasets():
    rows = []
    for folder in sorted(DATASETS.glob("*"), reverse=True) if DATASETS.is_dir() else []:
        manifest = read_json(folder / "manifest.json")
        if manifest:
            rows.append({"folder": f"datasets/{folder.name}", "count": manifest.get("count"),
                         "labels": manifest.get("labels"), "created_at": manifest.get("created_at")})
    return rows


def write_label(d, verdict, rules, **extra):
    """Save the rules' verdict as the run's label (source "rules", or "batch" when a batch saves it)."""
    write_json(d / "label.json", {"label": verdict["label"], "reason": verdict["reason"],
                                  "success_time_s": verdict.get("success_time_s"), "rules": rules,
                                  "source": "rules", "saved_at": datetime.now().isoformat(timespec="seconds"), **extra})


def is_hand_label(d):
    """Whether the run's label was set by a person (rules and bulk saves then leave it alone)."""
    return (read_json(d / "label.json") or {}).get("source") == "human"


def write_hand_label(d, label, note, rules):
    """A person's label for a run: it wins over the rules. The rules' verdict at that moment is kept beside it, so
    disagreements can be found later."""
    verdict = run_verdict(d, rules)
    meta = read_json(d / "meta.json", {})
    write_json(d / "label.json", {"label": label, "reason": note or "set by hand", "source": "human",
                                  "rules_verdict": {"label": verdict["label"], "reason": verdict["reason"]},
                                  "rules": rules, "batch_id": meta.get("batch_id"),
                                  "saved_at": datetime.now().isoformat(timespec="seconds")})


MAX_WORKERS = 4  # episodes at once; 3 at once ran like one at a time on an M3 (8 cores), see README
NO_CAPACITY_TRIES = 5  # Dream-1 "at max capacity" (DA-2000 NO_CAPACITY): the episode is tried again this many times


BATCH_ID_RE = re.compile(r"^b[0-9]{8}-[0-9]{6}$")


def original_episodes(batch_id):
    """The baseline of a finished batch: as many episodes, with each run's task and judge spec and the batch's time
    limit, but none of our method: every episode starts from the simulator's own default scene (the same object
    layout each time, no seed), with default scene settings, and Dream-1 runs alone (the caller's "plain" mode).
    @return (tasks, seeds: None per episode = the default scene, seconds, changes {}, random_names [], split)
    @throws ValueError When the batch is unknown or has no runs."""
    if not BATCH_ID_RE.match(batch_id):
        raise ValueError("Unknown batch.")
    metas = [read_json(d / "meta.json", {}) for d in RUNS.iterdir() if run_dir(d.name)] if RUNS.is_dir() else []
    metas = sorted((m for m in metas if m.get("batch_id") == batch_id and m.get("seed") is not None),
                   key=lambda m: m["seed"])
    if not metas:
        raise ValueError("That batch has no runs with a seed.")
    tasks = parse_task_list([{"prompt": m["task"], "judge": m.get("judge")} for m in metas], FULL_TABLE)
    seconds = min(300, max(10, int(float(metas[0].get("seconds") or 60))))
    return tasks, [None] * len(metas), seconds, {}, [], "default"


class Batch:
    """Many episodes in the background, `workers` at a time: no camera view, one seed each (its start scene, and the
    scene settings marked random drawn from it), and a label from the rules as they were when the batch started.
    Episode i always gets the i-th seed and task, however many run at once. Stops starting new episodes on a run
    error (such as a rejected API key) or after three invalid episodes in a row; Stop batch stops them all."""

    def __init__(self):
        self.thread = None
        self.stop_requested = False
        self.info = None
        self.lock = threading.Lock()
        self.active = {}  # run id -> (episode index, process, stop file)
        self.next = 0

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def active_ids(self):
        with self.lock:
            return set(self.active)

    def start(self, tasks, seeds, seconds, changes, random_names, split, robot=(), mode="plain", workers=1,
              collection=None, original_of=None, original=False, batch_seed=None):
        """@param seeds One seed per episode. @param changes, random_names The scene settings, and the ones drawn
        from each seed. @param split "train" or "eval" (shown on the page). @param workers Episodes at once."""
        if self.running() or RUNNER.running():
            raise RuntimeError("A run is already going.")
        if not has_api_key() and mode not in NO_KEY_MODES:
            raise RuntimeError("Enter your API key first.")
        workers = max(1, min(MAX_WORKERS, int(workers), len(seeds)))
        self.stop_requested, self.next, self.active = False, 0, {}
        self.info = {"id": "b" + datetime.now().strftime("%Y%m%d-%H%M%S"), "tasks": tasks, "total": len(seeds),
                     "seconds": seconds, "seeds": seeds, "seed": seeds[0], "split": split, "ood": changes,
                     "ood_random": random_names, "rules": current_rules(), "robot": list(robot), "mode": mode,
                     "workers": workers, "collection": collection, "original_of": original_of, "original": original or bool(original_of), "batch_seed": batch_seed, "done": 0, "counts": {"success": 0, "failure": 0, "invalid": 0},
                     "started": time.time(), "ended": None, "current": None, "active": [], "stopped_reason": None,
                     "durations": [], "invalid_in_a_row": 0}
        self.thread = threading.Thread(target=self._supervise, daemon=True)
        self.thread.start()

    def stop(self):
        if self.running():
            self.stop_requested = True
            with self.lock:
                for _index, _process, stop_file in self.active.values():
                    stop_file.touch()

    def _supervise(self):
        info = self.info
        threads = []
        try:
            for w in range(info["workers"]):
                if w:
                    time.sleep(1.0)  # spread the Dream session starts a little
                thread = threading.Thread(target=self._worker, daemon=True)
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
            if self.stop_requested:
                info["stopped_reason"] = "Stopped by you."
        except Exception as error:  # noqa: BLE001  (shown on the page)
            info["stopped_reason"] = f"Batch error: {error}"
        finally:
            info["current"], info["active"] = None, []
            info["ended"] = time.time()

    def _spawn(self, i, run_id, stop_file):
        """Start episode i's sim.py (no camera view or control state; its output goes to a log file)."""
        info = self.info
        task, seed = info["tasks"][i % len(info["tasks"])], info["seeds"][i]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore",
               "STOP_FILE": str(stop_file), "RUN_DIR": str(RUNS / run_id), "BATCH_ID": info["id"],
               "JUDGE_SPEC": json.dumps(task["judge"]),
               "LIVE_FRAME": str(live_frame(run_id)),  # its camera view, for the page (/api/live.jpg?run=...)
               **({"ORIGINAL": "1"} if info.get("original") else {}),
               **({"BATCH_SEED": str(info["batch_seed"])} if info.get("batch_seed") is not None else {}),
               **({"ORIGINAL_OF": info["original_of"]} if info.get("original_of") else {})}
        if PAGE_KEY["value"]:
            env["DREAM_API_KEY"] = PAGE_KEY["value"]
        command = [sys.executable, str(HERE / "sim.py"), task["prompt"], "--seconds", str(info["seconds"]),
                   *info["robot"], *(["--seed", str(seed)] if seed is not None else ["--default-scene"]),
                   {"collect": "--collect", "script": "--script-only", "plain": "--plain"}[info["mode"]]]
        if info["mode"] in ("collect", "script"):
            command += collection_args(info["collection"])[1]
        if info["ood"]:
            command += ["--ood", json.dumps(info["ood"])]
        if info["ood_random"]:
            command += ["--ood-random", ",".join(info["ood_random"])]
        log = open(WORK / f"{run_id}.log", "w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=str(HERE), env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT)
        log.close()
        return process, task["prompt"], seed

    def _worker(self):
        info = self.info
        retry = None  # (episode index, tries so far): Dream-1 was at capacity, so the same episode runs again
        while True:
            with self.lock:
                if self.stop_requested or info["stopped_reason"]:
                    return
                if retry is None and self.next >= info["total"]:
                    return
                if retry is None:
                    i, tries = self.next, 0
                    self.next += 1
                else:
                    i, tries = retry
                run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
                stop_file = WORK / f"stop-{run_id}"
            retry = None
            started = time.time()
            process, prompt, seed = self._spawn(i, run_id, stop_file)
            with self.lock:
                self.active[run_id] = (i, process, stop_file)
                self._show_active(prompt_of={run_id: (prompt, seed)})
            while process.poll() is None:  # a run that hangs is stopped, then killed
                waited = time.time() - started
                if waited > info["seconds"] + 180:
                    stop_file.touch()
                if waited > info["seconds"] + 240:
                    process.kill()
                time.sleep(0.5)
            d = run_dir(run_id)
            meta = read_json(d / "meta.json", {}) if d else {}
            if "NO_CAPACITY" in str(meta.get("error") or "") and tries + 1 < NO_CAPACITY_TRIES and not self.stop_requested:
                # Dream-1 is busy (shared by everyone): not this episode's fault. Drop the attempt, wait, try again.
                with self.lock:
                    self.active.pop(run_id, None)
                    self._show_active()
                    info["capacity_retries"] = info.get("capacity_retries", 0) + 1
                delete_runs([run_id])
                stop_file.unlink(missing_ok=True)
                time.sleep(5 * (tries + 1))
                retry = (i, tries + 1)
                continue
            verdict = (judge.evaluate(meta, read_json(d / "trajectory.json", {}), info["rules"]) if d
                       else judge.invalid("no_result"))
            if d:
                write_label(d, verdict, info["rules"], batch_id=info["id"], source="batch")  # a new run: no hand label yet
            with self.lock:
                self.active.pop(run_id, None)
                self._show_active()
                info["counts"][verdict["label"]] += 1
                info["done"] += 1
                info["durations"].append(time.time() - started)
                info["invalid_in_a_row"] = info["invalid_in_a_row"] + 1 if verdict["label"] == "invalid" else 0
                if verdict["reason"] == "execution_error" and not self.stop_requested and not info["stopped_reason"]:
                    info["stopped_reason"] = f"Run error: {str(meta.get('error') or '')[:160]}"
                elif info["invalid_in_a_row"] >= 3 and not info["stopped_reason"]:
                    info["stopped_reason"] = "Three invalid episodes in a row."
            stop_file.unlink(missing_ok=True)
            live_frame(run_id).unlink(missing_ok=True)

    def _show_active(self, prompt_of=None):
        """The episodes going now, for the page (called with the lock held)."""
        info = self.info
        names = getattr(self, "_names", {})
        names.update(prompt_of or {})
        self._names = names
        active = sorted((index, run_id) for run_id, (index, _p, _s) in self.active.items())
        info["active"] = [{"index": index + 1, "task": names.get(run_id, ("", None))[0],
                           "seed": names.get(run_id, ("", None))[1], "run": run_id} for index, run_id in active]
        info["current"] = info["active"][0] if info["active"] else None

    def status(self):
        if not self.info:
            return None
        info = {k: v for k, v in self.info.items() if k not in ("durations", "seeds", "invalid_in_a_row")}
        durations = self.info["durations"]
        remaining = info["total"] - info["done"]
        info["running"] = self.running()
        info["elapsed_s"] = round((info["ended"] or time.time()) - info["started"])
        info["eta_s"] = (round(sum(durations) / len(durations) * remaining / info["workers"])
                         if durations and self.running() else None)
        return info


BATCH = Batch()

## Verdicts by (run id, rules, meta and trajectory modification times), so the run list stays fast with many runs.
VERDICTS = {}


def run_verdict(d, rules):
    key = (d.name, json.dumps(rules, sort_keys=True),
           *((d / name).stat().st_mtime if (d / name).exists() else 0 for name in ("meta.json", "trajectory.json")))
    if key not in VERDICTS:
        VERDICTS[key] = judge.evaluate(read_json(d / "meta.json", {}), read_json(d / "trajectory.json", {}), rules)
    return VERDICTS[key]


def brief(verdict):
    """A verdict without its per-step series (the run list does not draw charts)."""
    return {k: v for k, v in verdict.items() if k not in ("dist", "touch", "t", "series")}


def spec_items(spec):
    """The scene items a judge spec needs on the table ("any" object needs none in particular)."""
    if spec["type"] == "pick_place":
        return {name for pair in spec["pairs"] for name in (pair["object"], pair["target"])}
    if spec["type"] == "goal":  # the objects, and the bowl / plate they go into
        return set(spec["objects"]) | {e.get("in") or e.get("near") for e in spec["objects"].values()} - {None}
    return set() if spec["object"] == "any" else {spec["object"]}


def task_spec(prompt, given, scene):
    """The spec a task is judged on: the one given (from the page's "judge as"), else the one its words give.
    @param given A spec, older {"pairs": [...]} / {"object", "target"}, or None.
    @return The spec, or None when the task is not judged.
    @throws ValueError When a given spec is not valid or needs an item that is not on the table.
    """
    spec = judge.normalize_spec(given) if given else judge.classify(prompt)
    if given and spec is None:
        raise ValueError(f"Pick what to judge '{prompt}' on (each object once, at most {judge.MAX_PAIRS} pairs).")
    if spec:
        missing = sorted(name for name in spec_items(spec) if name not in scene["items"])
        if missing:
            raise ValueError(f"Not on the table for '{prompt}': {', '.join(missing)}")
    return spec


def parse_task_list(tasks, scene):
    """Check a batch's tasks: each a prompt, with the spec it is judged on ("judge", or older "pairs" /
    "object" + "target"), or a prompt alone whose words give the spec. Every batch task must be judged.
    @return [{"prompt", "judge": spec}].
    @throws ValueError
    """
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Pick at least one task.")
    clean = []
    for task in tasks:
        given = task if isinstance(task, dict) else {"prompt": task}
        prompt = re.sub(r"\s+", " ", str(given.get("prompt") or "")).strip()
        if not prompt or len(prompt) > 300:
            raise ValueError("A prompt must be 1 to 300 characters.")
        old = {"pairs": given["pairs"]} if isinstance(given.get("pairs"), list) else \
            ({"object": given.get("object"), "target": given.get("target")} if given.get("target") else None)
        spec = task_spec(prompt, given.get("judge") or old, scene)
        if spec is None:
            raise ValueError(f"Pick what to judge '{prompt}' on.")
        clean.append({"prompt": prompt, "judge": spec})
    return clean


def spec_text(spec):
    """A spec in words, for the page."""
    if not spec:
        return None
    name = lambda n: "any object" if n == "any" else n.replace("water_", "")
    if spec["type"] == "pick_place":
        return "Pick & place: " + ", ".join(f"{name(p['object'])} → {p['target']}" for p in spec["pairs"])
    if spec["type"] == "move_to_area":
        return f"Move to area: {name(spec['object'])} → {spec['area']}"
    return f"{spec['type'].replace('_', ' ').capitalize()}: {name(spec['object'])}"


def run_options(body):
    """The start-scene options of a run or preview request.
    @return (changes, seeding, scene): the scene settings, sim.py's seed arguments, and a custom layout (or None;
    a custom layout replaces the seed's, so the seed and the random settings are not used with it).
    @throws ValueError
    """
    changes = ood.validate(body.get("ood") or {})
    try:
        scene = parse_scene(body.get("scene"))
    except (TypeError, AttributeError) as error:
        raise ValueError(f"Scene: {error}") from error
    seeding = () if scene else seed_web.seeding_args(body, ood.FIELDS)
    return changes, seeding, scene, robot_args(changes, seeding, scene)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, content_type, status=200, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=200):
        self.send(json.dumps(data).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if path in ("/ood.js", "/seed.js"):
            return self.send(PAGE.with_name(path[1:]).read_bytes(), "text/javascript; charset=utf-8")
        if path == "/api/ood":
            return self.send_json(ood.GROUPS)
        if path == "/api/status":
            return self.send_json(RUNNER.status())
        if path == "/api/datasets":
            return self.send_json({"datasets": list_datasets()})
        if path == "/api/folders":  # ?path=/abs/folder: the folders in it, to choose where to save a dataset
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                return self.send_json(folder_listing(query.get("path", [""])[0]))
            except ValueError as error:
                return self.send_json({"error": str(error)}, 400)
        if path == "/api/episodes":
            return self.send_json({"episodes": list_episodes()})
        m = re.fullmatch(r"/api/episodes/([^/]+)/(top|left|right)\.mp4", path)
        if m and ID_RE.match(m.group(1)):  # a recording's camera video, for the page's player (byte ranges for seeking)
            try:
                data = (EPISODES / m.group(1) / f"{m.group(2)}.mp4").read_bytes()
            except OSError:
                return self.send_json({"error": "no video for that recording"}, 404)
            r = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
            if r and (r.group(1) or r.group(2)):
                start = int(r.group(1)) if r.group(1) else max(0, len(data) - int(r.group(2)))
                end = min(int(r.group(2)), len(data) - 1) if r.group(1) and r.group(2) else len(data) - 1
                if start > end:
                    return self.send(b"", "video/mp4", 416, headers={"Content-Range": f"bytes */{len(data)}"})
                return self.send(data[start:end + 1], "video/mp4", 206,
                                 headers={"Content-Range": f"bytes {start}-{end}/{len(data)}", "Accept-Ranges": "bytes"})
            return self.send(data, "video/mp4", headers={"Accept-Ranges": "bytes"})
        m = re.fullmatch(r"/api/episodes/([^/]+)/frame\.jpg", path)
        if m and ID_RE.match(m.group(1)):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                return self.send(episode_frame(m.group(1), int(query.get("step", ["0"])[0])), "image/jpeg")
            except (OSError, ValueError, KeyError, IndexError):
                return self.send_json({"error": "no picture for that recording"}, 404)
        if path == "/api/live.jpg":
            run = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("run", [""])[0]  # a batch episode view
            if run and not ID_RE.match(run):
                return self.send_json({"error": "bad run id"}, 400)
            try:
                return self.send((live_frame(run) if run else LIVE).read_bytes(), "image/jpeg")
            except OSError:
                return self.send_json({"error": "no frame yet"}, 404)
        if path == "/api/rules":
            return self.send_json(current_rules())
        if path == "/api/simulator-settings":
            return self.send_json({"settings": current_settings(), "catalog": sim_settings.catalog()})
        if path == "/api/task-types":  # each type's rules: default, minimum, maximum (the page builds its controls)
            return self.send_json({"rules": judge.RULES, "areas": judge.AREAS})
        if path == "/api/classify":  # ?task=...: what a task would be judged on
            task = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("task", [""])[0]
            spec = judge.classify(task)
            return self.send_json({"spec": spec, "text": spec_text(spec)})
        if path == "/api/runs":
            runs = []
            rules = current_rules()
            if RUNS.is_dir():
                for d in sorted(RUNS.iterdir(), reverse=True):
                    if run_dir(d.name):
                        runs.append({**read_json(d / "meta.json", {}), "id": d.name, "label": read_json(d / "label.json"),
                                     "verdict": brief(run_verdict(d, rules))})
            return self.send_json({"runs": runs})
        m = re.match(r"^/api/runs/([^/]+)/verdict$", path)
        if m and run_dir(m.group(1)):
            return self.send_json(run_verdict(RUNS / m.group(1), current_rules()))
        m = re.match(r"^/api/runs/([^/]+)/(trajectory\.json|last\.jpg)$", path)
        if m and run_dir(m.group(1)) and (RUNS / m.group(1) / m.group(2)).exists():
            kind = "application/json" if m.group(2).endswith(".json") else "image/jpeg"
            return self.send((RUNS / m.group(1) / m.group(2)).read_bytes(), kind)
        self.send_json({"error": "not found"}, 404)

    def preview(self, body):
        """The start scene as a JPEG (sim.py --preview: no Dream API), with the values the seed drew in a header."""
        try:
            changes, seeding, scene, robot = run_options(body)
        except ValueError as error:
            return self.send_json({"error": str(error)}, 400)
        if RUNNER.running() or BATCH.running() or not PREVIEW_LOCK.acquire(blocking=False):
            return self.send_json({"error": "The simulator is busy."}, 409)
        try:
            env = {**os.environ, "PYTHONWARNINGS": "ignore", **({"SCENE_JSON": json.dumps(scene)} if scene else {})}
            with tempfile.TemporaryDirectory(prefix="dream-preview-") as folder:
                frame = Path(folder) / "initial.jpg"
                result = subprocess.run(
                    [sys.executable, str(HERE / "sim.py"), "--preview", str(frame), "--ood", json.dumps(changes),
                     *seeding, *robot], cwd=HERE, env=env, capture_output=True, text=True, timeout=120)
                if result.returncode:
                    # sim.py says why on stdout ("Stopped: ..."); stderr may end with an unrelated MuJoCo warning.
                    said = [line[len("Stopped:"):].strip() for line in result.stdout.splitlines() if line.startswith("Stopped:")]
                    error = said or (result.stderr or result.stdout).strip().splitlines()
                    return self.send_json({"error": error[-1] if error else "Camera rendering failed."}, 422)
                return self.send(frame.read_bytes(), "image/jpeg", headers=seed_web.drawn_header(result.stdout))
        except (OSError, subprocess.TimeoutExpired):
            return self.send_json({"error": "Could not render the initial camera view."}, 500)
        finally:
            PREVIEW_LOCK.release()

    def goal_preview(self, body):
        """The goal popup: the start scene (or, with "goal" {object: [x, y]}, the scene with those objects let go
        at their spots and settled) as the top camera sees it, with every item's position and the camera projection.
        @return JSON {image (data URL), positions, projection (3x4, picture u v 0..1), size}."""
        import base64

        try:
            changes, seeding, scene, robot = run_options(body)
            goal = body.get("goal") or {}
            if not isinstance(goal, dict) or any(name not in OBJECTS or len(spot) != 2 for name, spot in goal.items()):
                raise ValueError("The goal must be {object: [x, y]}.")
            goal = {name: [float(v) for v in spot] for name, spot in goal.items()}
        except (ValueError, TypeError) as error:
            return self.send_json({"error": str(error)}, 400)
        if robot and "openyam" in robot:
            return self.send_json({"error": "The goal scene is for YAM."}, 400)
        if RUNNER.running() or BATCH.running() or not PREVIEW_LOCK.acquire(blocking=False):
            return self.send_json({"error": "The simulator is busy."}, 409)
        try:
            env = {**os.environ, "PYTHONWARNINGS": "ignore", **({"SCENE_JSON": json.dumps(scene)} if scene else {})}
            with tempfile.TemporaryDirectory(prefix="dream-goal-") as folder:
                result = subprocess.run(
                    [sys.executable, str(HERE / "sim.py"), "--goal-preview", folder, "--goal", json.dumps(goal),
                     "--ood", json.dumps(changes), *seeding, *robot], cwd=HERE, env=env, capture_output=True,
                    text=True, timeout=120)
                if result.returncode:
                    error = (result.stderr or result.stdout).strip().splitlines()
                    return self.send_json({"error": error[-1] if error else "Rendering the goal failed."}, 422)
                info = json.loads((Path(folder) / "goal.json").read_text())
                image = base64.b64encode((Path(folder) / "goal.jpg").read_bytes()).decode()
                return self.send_json({**info, "image": "data:image/jpeg;base64," + image})
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return self.send_json({"error": "Could not render the goal scene."}, 500)
        finally:
            PREVIEW_LOCK.release()

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self.send_json({"error": "Invalid request."}, 400)
        if not isinstance(body, dict):
            return self.send_json({"error": "Expected a JSON object."}, 400)
        if self.path == "/api/preview":
            return self.preview(body)
        if self.path == "/api/goal-preview":
            return self.goal_preview(body)
        if self.path == "/api/run":
            task = re.sub(r"\s+", " ", str(body.get("task") or "")).strip()
            if not task:
                return self.send_json({"error": "Describe what to pick up and where to place it."}, 400)
            try:
                changes, seeding, scene, robot = run_options(body)
                # Judged on the spec sent, or the one the task's words give; a task with neither runs, not judged.
                spec = task_spec(task, body.get("judge"), scene or FULL_TABLE)
            except ValueError as error:
                return self.send_json({"error": str(error)}, 400)
            if BATCH.running():
                return self.send_json({"error": "A batch is running."}, 409)
            mode = body.get("mode") or "collect"
            if mode not in MODES:
                return self.send_json({"error": "Unknown run mode."}, 400)
            if mode in ("human", "script") and ("openyam" in robot
                                                or (scene and set(scene["items"]) != set(OBJECTS + TARGETS))):
                return self.send_json({"error": ("Human control" if mode == "human" else "The rule-based script")
                                       + " needs YAM with all six items on the table."}, 400)
            try:
                _settings, collection = collection_args(body.get("collection"))
            except ValueError as error:
                return self.send_json({"error": str(error)}, 400)
            try:
                seconds = min(900 if mode in ("human", "test") else 300, max(10, int(float(body.get("seconds") or 60))))
                RUNNER.start(task, seconds, changes, seeding, scene,
                             extra_env={"JUDGE_SPEC": json.dumps(spec)} if spec else None, robot=robot,
                             mode=mode, record=body.get("record", True) is not False, collection=collection)
            except (ValueError, TypeError):
                return self.send_json({"error": "The time limit must be a number."}, 400)
            except RuntimeError as error:
                return self.send_json({"error": str(error)}, 409)
            return self.send_json(RUNNER.status())
        if self.path == "/api/simulator-settings":
            try:
                settings = sim_settings.normalize(body)
            except ValueError as error:
                return self.send_json({"error": str(error)}, 400)
            write_json(SETTINGS, settings)
            return self.send_json({"settings": settings, "catalog": sim_settings.catalog()})
        if self.path == "/api/rules":
            try:
                rules = judge.normalize_rules(body)
            except ValueError as error:
                return self.send_json({"error": str(error)}, 400)
            write_json(RULES, rules)
            return self.send_json(rules)
        m = re.match(r"^/api/runs/([^/]+)/label$", self.path)
        if m:  # {"label": "success" | "failure" | "invalid", "note": ...} by hand, or {"label": null}: back to the rules
            d = run_dir(m.group(1))
            if not d:
                return self.send_json({"error": "No such run."}, 404)
            label, note = body.get("label"), re.sub(r"\s+", " ", str(body.get("note") or "")).strip()[:300]
            rules = current_rules()
            if label is None:
                meta = read_json(d / "meta.json", {})
                write_label(d, run_verdict(d, rules), rules, batch_id=meta.get("batch_id"))
            elif label in LABELS:
                write_hand_label(d, label, note, rules)
            else:
                return self.send_json({"error": "The label must be success, failure or invalid."}, 400)
            return self.send_json({"label": read_json(d / "label.json")})
        if self.path == "/api/datasets":  # {"run_ids": [...], "name": "..."}: save those runs as a dataset
            run_ids = body.get("run_ids")
            if not isinstance(run_ids, list) or not run_ids or any(not isinstance(r, str) for r in run_ids):
                return self.send_json({"error": "Pick at least one run."}, 400)
            try:
                folder, manifest = save_dataset(run_ids, body.get("name"), bool(body.get("dream_format")),
                                                body.get("location") or None)
            except ValueError as error:
                return self.send_json({"error": str(error)}, 400)
            return self.send_json({"folder": shown_path(folder), "count": manifest["count"],
                                   "labels": manifest["labels"], "skipped": manifest["skipped"],
                                   "dream_format": manifest.get("dream_format")})
        if self.path == "/api/runs/delete":  # {"run_ids": [...]}: finished runs and their recordings, for good
            run_ids = body.get("run_ids")
            if not isinstance(run_ids, list) or not run_ids or any(not isinstance(r, str) for r in run_ids):
                return self.send_json({"error": "run_ids must be a list of run IDs."}, 400)
            deleted, skipped = delete_runs(run_ids)
            return self.send_json({"deleted": deleted, "skipped": skipped})
        if self.path == "/api/labels":
            saved = kept = 0
            rules = current_rules()
            run_ids = body.get("run_ids")
            if not isinstance(run_ids, list) or any(not isinstance(run_id, str) for run_id in run_ids):
                return self.send_json({"error": "run_ids must be a list of run IDs."}, 400)
            for run_id in run_ids:
                d = run_dir(run_id)
                if d and is_hand_label(d):
                    kept += 1  # a person's label is never replaced by the rules
                elif d:
                    meta = read_json(d / "meta.json", {})
                    write_label(d, run_verdict(d, rules), rules, batch_id=meta.get("batch_id"))
                    saved += 1
            return self.send_json({"saved": saved, "kept": kept})
        if self.path == "/api/key":
            key = str(body.get("key") or "").strip()
            problem = key and key_problem(key)
            if problem:  # the key in use (if any) stays as it was
                return self.send_json({"error": problem}, 400)
            PAGE_KEY["value"] = key or None
            return self.send_json(key_status())
        if self.path == "/api/batch" and body.get("original_of"):
            # The baseline: the same episodes as a finished batch (each one's seed, so the same object layout, task,
            # judge, scene settings and time limit), run by Dream-1 alone: no failure monitor, recovery script or human.
            try:
                tasks, seeds, seconds, changes, random_names, split = original_episodes(str(body["original_of"]))
                robot = robot_args(changes, ["--ood-random"] if random_names else [], None)
                workers = int(body.get("workers") or 1)
                if not 1 <= workers <= MAX_WORKERS:
                    raise ValueError(f"Run 1 to {MAX_WORKERS} episodes at once.")
                BATCH.start(tasks, seeds, seconds, changes, random_names, split, robot, "plain", workers, None,
                            original_of=str(body["original_of"]))
            except (ValueError, TypeError, AttributeError) as error:
                return self.send_json({"error": str(error)}, 400)
            except RuntimeError as error:
                return self.send_json({"error": str(error)}, 409)
            return self.send_json(RUNNER.status())
        if self.path == "/api/batch":
            try:
                tasks = parse_task_list(body.get("tasks"), FULL_TABLE)  # a seed's scene has all six items
                count = int(body.get("count") or 0)
                if not 1 <= count <= MAX_BATCH:
                    raise ValueError(f"Episodes must be 1 to {MAX_BATCH}.")
                seconds = min(300, max(10, int(float(body.get("seconds") or 60))))
                changes = ood.validate(body.get("ood") or {})
                seed_web.seeding_args({"ood_random": body.get("ood_random")}, ood.FIELDS)  # checks the names
                random_names = list(body.get("ood_random") or [])
                robot = robot_args(changes, ["--ood-random"] if random_names else [], None)
                split = body.get("split") or "train"
                if split == "eval":  # the fixed evaluation seeds, in order
                    seeds = eval_seeds()[:count]
                    if len(seeds) < count:
                        raise ValueError(f"There are {len(seeds)} eval seeds.")
                elif split == "train":
                    first = str(body.get("seed", "")).strip()
                    first = int(first) if first else secrets.randbelow(1_000_000)
                    if not 0 <= first < 1_000_000:
                        raise ValueError("Train seeds are 0 to 999999.")
                    seeds = [seed_web.episode_seed(first, i) for i in range(count)]  # one seed is the whole batch
                else:
                    raise ValueError("Seeds must be 'train' or 'eval'.")
                mode = body.get("mode") or "collect"
                if mode not in ("collect", "script", "plain", "original"):
                    raise ValueError("A batch runs in data collection, rule-based only, Dream-1 only or original mode.")
                if mode == "script" and "openyam" in robot:
                    raise ValueError("The rule-based script needs YAM.")
                original = mode == "original"
                if original:  # none of our method: the default scene every episode, default settings, Dream-1 alone
                    seeds, changes, random_names, split, mode = [None] * count, {}, [], "default", "plain"
                    robot = robot_args(changes, [], None)
                workers = int(body.get("workers") or 1)
                if not 1 <= workers <= MAX_WORKERS:
                    raise ValueError(f"Run 1 to {MAX_WORKERS} episodes at once.")
                collection, _args = collection_args(body.get("collection"))
                BATCH.start(tasks, seeds, seconds, changes, random_names, split, robot, mode, workers, collection,
                            original=original, batch_seed=first if split == "train" else None)
            except (ValueError, TypeError, AttributeError) as error:
                return self.send_json({"error": str(error)}, 400)
            except RuntimeError as error:
                return self.send_json({"error": str(error)}, 409)
            return self.send_json(RUNNER.status())
        if self.path == "/api/keys":  # keyboard and mouse for the running sim.py (teleop.py)
            message = {}
            if isinstance(body.get("held"), list):
                message["held"] = [str(key)[:8] for key in body["held"]][:12]
            if body.get("press"):
                message["press"] = str(body["press"])[:8]  # e.g. space, tab, v, t, home, lower, lift, grip
            for name in ("click", "drag"):
                point = body.get(name)
                if isinstance(point, list) and len(point) == 2:
                    try:
                        message[name] = [min(1.0, max(0.0, float(v))) for v in point]
                    except (TypeError, ValueError):
                        pass
            if message.get("click") and body.get("camera") in ("top", "left", "right"):
                message["camera"] = body["camera"]
            if body.get("wheel"):
                try:
                    message["wheel"] = max(-5.0, min(5.0, float(body["wheel"])))
                except (TypeError, ValueError):
                    pass
            return self.send_json({"sent": RUNNER.send_keys(message)})
        if self.path == "/api/resume":
            episode = str(body.get("episode") or "")
            if not ID_RE.match(episode) or not (EPISODES / episode / "steps.npz").exists():
                return self.send_json({"error": "No such recording."}, 404)
            if BATCH.running():
                return self.send_json({"error": "A batch is running."}, 409)
            task = (read_json(RUNS / episode / "meta.json", {}) or read_json(EPISODES / episode / "meta.json", {})
                    or {}).get("task") or "put apple into the bowl"
            RUNNER.stop_and_wait()  # Resume replaces a run that is going (its recording is saved first)
            try:
                seconds = min(900, max(10, int(float(body.get("seconds") or 60))))
                RUNNER.start(task, seconds, record=body.get("record", True) is not False,
                             resume=(episode, int(body.get("step") or 0)))
            except (ValueError, TypeError):
                return self.send_json({"error": "The step and the time limit must be numbers."}, 400)
            except RuntimeError as error:
                return self.send_json({"error": str(error)}, 409)
            return self.send_json(RUNNER.status())
        if self.path == "/api/reset":
            if BATCH.running():
                return self.send_json({"error": "A batch is running."}, 409)
            try:
                RUNNER.restart()
            except RuntimeError as error:
                return self.send_json({"error": str(error)}, 409)
            return self.send_json(RUNNER.status())
        if self.path == "/api/stop":
            BATCH.stop()
            RUNNER.stop()
            return self.send_json(RUNNER.status())
        self.send_json({"error": "not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description="Website for running Dream-1 in the simulator")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"Open http://localhost:{args.port}  (Ctrl-C to quit)", flush=True)
    if not has_api_key():
        print("No API key yet: enter it at the top of the page.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        BATCH.stop()
        RUNNER.stop()


if __name__ == "__main__":
    main()
