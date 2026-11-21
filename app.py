"""A small website for running Dream-1 in the Dream Labs simulator.

    python app.py              then open http://localhost:8000

Press Run on the page: sim.py runs the task in its own process, the page shows the cameras live, and the run's
trajectory is saved in runs/<id>/. The page judges every saved run against the success rules set on it (target
distance, release, time standing still); the rules are kept in runs/rules.json and each run's label in
runs/<id>/label.json. Standard library only.
"""

import argparse
import json
import os
import re
import secrets
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
import simulator_settings as sim_settings

HERE = Path(__file__).resolve().parent
PAGE = HERE / "static" / "index.html"
RUNS = HERE / "runs"
RULES = RUNS / "rules.json"
SETTINGS = RUNS / "simulator_settings.json"
DEFAULT_RULES = judge.DEFAULT_RULES
## What can be on the table, and the area objects may be placed in (metres, world frame; reachable by both arms).
OBJECTS = ["apple", "orange", "banana", "water_bottle"]
TARGETS = ["bowl", "plate"]
AREA = {"x": (-0.30, 0.30), "y": (-0.22, 0.18)}
ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
WORK = Path(tempfile.mkdtemp(prefix="dream-web-"))
LIVE = WORK / "live.jpg"
STOP = WORK / "stop"
EXAMPLES = ["put apple into the bowl", "put orange into the bowl", "put banana on the plate"]


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
    saved = read_json(RULES, {})
    try:
        return judge.normalize_rules({**DEFAULT_RULES, **saved})
    except (TypeError, ValueError):
        return dict(DEFAULT_RULES)


def current_settings():
    saved = read_json(SETTINGS, {})
    try:
        return sim_settings.normalize(saved)
    except ValueError:
        return dict(sim_settings.DEFAULT)


def run_dir(run_id):
    """The folder of a saved run, or None for an unknown or malformed id."""
    if not ID_RE.match(run_id or ""):
        return None
    d = RUNS / run_id
    return d if (d / "meta.json").exists() else None


## The API key typed on the page: the only key the website uses. Kept in this process's memory only (never written
## to disk or logged) and handed to each run as DREAM_API_KEY. Clear on the page removes it.
PAGE_KEY = {"value": None}


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

    def start(self, task, seconds, settings, scene=None):
        with self.lock:
            if self.running():
                raise RuntimeError("A run is already going.")
            if not has_api_key():
                raise RuntimeError("Enter your API key first.")
            STOP.unlink(missing_ok=True)
            LIVE.unlink(missing_ok=True)
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore",
                   "LIVE_FRAME": str(LIVE), "STOP_FILE": str(STOP), "RUN_DIR": str(RUNS / run_id)}
            if PAGE_KEY["value"]:
                env["DREAM_API_KEY"] = PAGE_KEY["value"]
            if scene:
                env["SCENE_JSON"] = json.dumps(scene)
            command = [sys.executable, str(HERE / "sim.py"), task, "--seconds", str(seconds),
                       "--lighting", settings["lighting"], "--layout", settings["layout"],
                       "--room", settings["room"], "--robot", settings["robot"]]
            if settings["layout"] == "varied":
                command += ["--seed", str(settings["seed"] if settings["seed"] is not None
                                          else secrets.randbelow(2**32))]
            self.process = subprocess.Popen(
                command, cwd=str(HERE), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            self.task, self.started, self.ended, self.run_id = task, time.time(), None, run_id
            self.log.clear()
            threading.Thread(target=self._read, args=(self.process,), daemon=True).start()

    def _read(self, process):
        for line in process.stdout:
            line = line.strip()
            if line and "Warning" not in line and "warnings.warn" not in line:
                self.log.append(line)
        process.wait()
        self.ended = time.time()

    def stop(self):
        if self.running():
            STOP.touch()

    def status(self):
        running = self.running()
        return {
            "running": running,
            "task": self.task,
            "elapsed_s": round((self.ended or time.time()) - self.started) if self.started else None,
            "exit_code": None if running or self.process is None else self.process.returncode,
            "run_id": self.run_id,
            "log": list(self.log),
            "key": key_status(),
            "examples": EXAMPLES,
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


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=200):
        self.send(json.dumps(data).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/status":
            return self.send_json(RUNNER.status())
        if path == "/api/live.jpg":
            try:
                return self.send(LIVE.read_bytes(), "image/jpeg")
            except OSError:
                return self.send_json({"error": "no frame yet"}, 404)
        if path == "/api/rules":
            return self.send_json(current_rules())
        if path == "/api/simulator-settings":
            return self.send_json({"settings": current_settings(), "catalog": sim_settings.catalog()})
        if path == "/api/runs":
            runs = []
            rules = current_rules()
            if RUNS.is_dir():
                for d in sorted(RUNS.iterdir(), reverse=True):
                    if run_dir(d.name):
                        meta = read_json(d / "meta.json", {})
                        trajectory = read_json(d / "trajectory.json", {})
                        runs.append({**meta, "id": d.name, "label": read_json(d / "label.json"),
                                     "verdict": judge.evaluate(meta, trajectory, rules)})
            return self.send_json({"runs": runs})
        m = re.match(r"^/api/runs/([^/]+)/(trajectory\.json|last\.jpg)$", path)
        if m and run_dir(m.group(1)) and (RUNS / m.group(1) / m.group(2)).exists():
            kind = "application/json" if m.group(2).endswith(".json") else "image/jpeg"
            return self.send((RUNS / m.group(1) / m.group(2)).read_bytes(), kind)
        m = re.match(r"^/api/runs/([^/]+)/frames/(\d{5})\.jpg$", path)
        if m and run_dir(m.group(1)):
            frame = RUNS / m.group(1) / "frames" / f"{m.group(2)}.jpg"
            if frame.is_file():
                return self.send(frame.read_bytes(), "image/jpeg")
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self.send_json({"error": "Invalid request."}, 400)
        if self.path == "/api/run":
            task = re.sub(r"\s+", " ", str(body.get("task") or "")).strip()
            if not task:
                return self.send_json({"error": "Type a task, such as: put apple into the bowl"}, 400)
            if judge.parse_task(task) is None:
                return self.send_json({"error": "Automatic judging supports putting apple, orange, banana, or bottle in a bowl or on a plate."}, 400)
            settings = current_settings()
            if not sim_settings.local_supported(settings):
                return self.send_json({"error": "The selected robot has no local control adapter. Choose YAM or OpenYAM."}, 400)
            if settings["robot"] == "openyam" and body.get("scene"):
                return self.send_json({"error": "Custom object placement is available with YAM; OpenYAM uses its fixed Sandbox table."}, 400)
            try:
                scene = parse_scene(body.get("scene"))
            except (ValueError, TypeError, AttributeError) as error:
                return self.send_json({"error": f"Scene: {error}"}, 400)
            missing = [name for name in judge.parse_task(task) if scene and name not in scene["items"]]
            if missing:
                return self.send_json({"error": f"Not on the table: {', '.join(missing)}"}, 400)
            try:
                seconds = min(300, max(10, int(float(body.get("seconds") or 60))))
                RUNNER.start(task, seconds, settings, scene)
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
        if self.path == "/api/labels":
            saved = 0
            rules = current_rules()
            run_ids = body.get("run_ids")
            if not isinstance(run_ids, list) or any(not isinstance(run_id, str) for run_id in run_ids):
                return self.send_json({"error": "run_ids must be a list of run IDs."}, 400)
            for run_id in run_ids:
                d = run_dir(run_id)
                if d:
                    verdict = judge.evaluate(read_json(d / "meta.json", {}),
                                             read_json(d / "trajectory.json", {}), rules)
                    write_json(d / "label.json", {"label": verdict["label"], "reason": verdict["reason"],
                        "success_time_s": verdict.get("success_time_s"), "rules": rules,
                        "saved_at": datetime.now().isoformat(timespec="seconds")})
                    saved += 1
            return self.send_json({"saved": saved})
        if self.path == "/api/key":
            key = str(body.get("key") or "").strip()
            if key and (len(key) > 512 or any(c.isspace() for c in key)):
                return self.send_json({"error": "That does not look like an API key."}, 400)
            PAGE_KEY["value"] = key or None
            return self.send_json(key_status())
        if self.path == "/api/stop":
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
        RUNNER.stop()


if __name__ == "__main__":
    main()
