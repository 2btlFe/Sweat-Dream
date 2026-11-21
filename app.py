"""A small website for running Dream-1 in the Dream Labs simulator.

    python app.py              then open http://localhost:8000

Press Run on the page: sim.py runs the task in its own process, the page shows the cameras live, and the run's
trajectory is saved in runs/<id>/. The page judges every saved run against the success rules set on it (target
distance, release, time standing still); the rules are kept in runs/rules.json and each run's label in
runs/<id>/label.json. A batch runs many episodes in the background (no camera view), each from a seeded start
layout, and labels each one with the rules as they were when the batch started. Standard library only.
"""

import argparse
import json
import os
import random
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

HERE = Path(__file__).resolve().parent
PAGE = HERE / "static" / "index.html"
RUNS = HERE / "runs"
RULES = RUNS / "rules.json"
DEFAULT_RULES = judge.DEFAULT_RULES
## What can be on the table, and the area objects may be placed in (metres, world frame; reachable by both arms).
OBJECTS = ["apple", "orange", "banana", "water_bottle"]
TARGETS = ["bowl", "plate"]
AREA = {"x": (-0.30, 0.30), "y": (-0.22, 0.18)}
## Footprint radius of each item (metres), for random layouts that keep items apart; the page's map uses the same.
RADII = {"apple": 0.040, "orange": 0.033, "banana": 0.050, "water_bottle": 0.030, "bowl": 0.083, "plate": 0.072}
MAX_BATCH = 500
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

    def start(self, task, seconds, scene=None, seed=None, live=True, extra_env=None):
        """Start one run of sim.py.
        @param scene The table setup, as parse_scene returns it, or None for the default scene.
        @param seed Seed for the small start shifts (random when None).
        @param live Whether to write the camera view for the page (batches do not).
        @param extra_env More environment variables for sim.py (BATCH_ID, LAYOUT).
        """
        with self.lock:
            if self.running():
                raise RuntimeError("A run is already going.")
            if not has_api_key():
                raise RuntimeError("Enter your API key first.")
            STOP.unlink(missing_ok=True)
            LIVE.unlink(missing_ok=True)
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore",
                   "STOP_FILE": str(STOP), "RUN_DIR": str(RUNS / run_id), **(extra_env or {})}
            if live:
                env["LIVE_FRAME"] = str(LIVE)
            if PAGE_KEY["value"]:
                env["DREAM_API_KEY"] = PAGE_KEY["value"]
            if scene:
                env["SCENE_JSON"] = json.dumps(scene)
            self.process = subprocess.Popen(
                [sys.executable, str(HERE / "sim.py"), task, "--seconds", str(seconds),
                 "--seed", str(secrets.randbelow(1_000_000_000) if seed is None else seed)], cwd=str(HERE), env=env,
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


def random_layout(names, rng):
    """Place the named items at random in AREA, at least 2 cm apart.
    @param names Items on the table.
    @param rng A random.Random (seeded, so a layout can be made again).
    @return A scene, as parse_scene returns it.
    @throws RuntimeError When the items do not fit (should not happen with six items).
    """
    placed = {}
    for name in sorted(names, key=lambda n: -RADII[n]):  # big ones first
        for _ in range(500):
            x, y = rng.uniform(*AREA["x"]), rng.uniform(*AREA["y"])
            if all((x - px) ** 2 + (y - py) ** 2 > (RADII[name] + RADII[other] + 0.02) ** 2
                   for other, (px, py) in placed.items()):
                placed[name] = [round(x, 4), round(y, 4)]
                break
        else:
            raise RuntimeError("Could not place the items apart; try fewer items.")
    return {"items": placed}


def write_label(d, verdict, rules, **extra):
    write_json(d / "label.json", {"label": verdict["label"], "reason": verdict["reason"],
                                  "success_time_s": verdict.get("success_time_s"), "rules": rules,
                                  "saved_at": datetime.now().isoformat(timespec="seconds"), **extra})


class Batch:
    """Many episodes one after another in the background: no camera view, a seeded start layout each, and a label
    from the rules as they were when the batch started. Stops on a run error (such as a rejected API key), after
    three invalid episodes in a row, or when asked."""

    def __init__(self):
        self.thread = None
        self.stop_requested = False
        self.info = None

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, tasks, count, seconds, layout, scene, seed):
        if self.running() or RUNNER.running():
            raise RuntimeError("A run is already going.")
        if not has_api_key():
            raise RuntimeError("Enter your API key first.")
        self.stop_requested = False
        self.info = {"id": "b" + datetime.now().strftime("%Y%m%d-%H%M%S"), "tasks": tasks, "total": count,
                     "seconds": seconds, "layout": layout, "seed": seed, "rules": current_rules(),
                     "done": 0, "counts": {"success": 0, "failure": 0, "invalid": 0},
                     "started": time.time(), "ended": None, "current": None, "stopped_reason": None, "durations": []}
        self.thread = threading.Thread(target=self._loop, args=(scene,), daemon=True)
        self.thread.start()

    def stop(self):
        if self.running():
            self.stop_requested = True
            RUNNER.stop()

    def _loop(self, scene):
        info = self.info
        invalid_in_a_row = 0
        try:
            for i in range(info["total"]):
                if self.stop_requested:
                    break
                task, seed = info["tasks"][i % len(info["tasks"])], info["seed"] + i
                prompt = task["prompt"]
                episode_scene = scene if info["layout"] == "shift" else random_layout(scene["items"], random.Random(seed))
                info["current"] = {"index": i + 1, "task": prompt, "seed": seed}
                started = time.time()
                RUNNER.start(prompt, info["seconds"], episode_scene, seed=seed, live=False,
                             extra_env={"BATCH_ID": info["id"], "LAYOUT": info["layout"],
                                        "JUDGE_PAIRS": json.dumps(task["pairs"])})
                run_id, process = RUNNER.run_id, RUNNER.process
                while process.poll() is None:  # a run that hangs is stopped, then killed
                    waited = time.time() - started
                    if waited > info["seconds"] + 180:
                        RUNNER.stop()
                    if waited > info["seconds"] + 240:
                        process.kill()
                    time.sleep(0.5)
                d = run_dir(run_id)
                meta = read_json(d / "meta.json", {}) if d else {}
                verdict = (judge.evaluate(meta, read_json(d / "trajectory.json", {}), info["rules"]) if d
                           else judge.invalid("no_result"))
                if d:
                    write_label(d, verdict, info["rules"], batch_id=info["id"], source="batch")
                info["counts"][verdict["label"]] += 1
                info["done"] += 1
                info["durations"].append(time.time() - started)
                invalid_in_a_row = invalid_in_a_row + 1 if verdict["label"] == "invalid" else 0
                if verdict["reason"] == "execution_error" and not self.stop_requested:
                    info["stopped_reason"] = f"Run error: {str(meta.get('error') or '')[:160]}"
                    break
                if invalid_in_a_row >= 3:
                    info["stopped_reason"] = "Three invalid episodes in a row."
                    break
            if self.stop_requested:
                info["stopped_reason"] = "Stopped by you."
        except Exception as error:  # noqa: BLE001  (shown on the page)
            info["stopped_reason"] = f"Batch error: {error}"
        finally:
            info["current"] = None
            info["ended"] = time.time()

    def status(self):
        if not self.info:
            return None
        info = {k: v for k, v in self.info.items() if k != "durations"}
        durations = self.info["durations"]
        remaining = info["total"] - info["done"]
        info["running"] = self.running()
        info["elapsed_s"] = round((info["ended"] or time.time()) - info["started"])
        info["eta_s"] = round(sum(durations) / len(durations) * remaining) if durations and self.running() else None
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


def parse_task_list(tasks, scene):
    """Check a batch's tasks. Each is a prompt the judge understands ("put apple into the bowl"), or any prompt with
    the object -> target pairs it is judged on: {"prompt": ..., "pairs": [{"object": "apple", "target": "bowl"}, ...]}
    (or one {"object", "target"}). Every object at most once, up to judge.MAX_PAIRS pairs, all on the table.
    @return [{"prompt", "pairs": [{"object", "target"}]}].
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
        if isinstance(given.get("pairs"), list):
            pairs = [(p.get("object"), p.get("target")) for p in given["pairs"] if isinstance(p, dict)]
        elif given.get("object") or given.get("target"):
            pairs = [(given.get("object"), given.get("target"))]
        else:
            parsed = judge.parse_task(prompt)
            pairs = [parsed] if parsed else []
        objects = [o for o, _ in pairs]
        if not pairs or any(o not in OBJECTS or g not in TARGETS for o, g in pairs):
            raise ValueError(f"Pick the objects and targets to judge '{prompt}' on.")
        if len(pairs) > judge.MAX_PAIRS or len(set(objects)) != len(objects):
            raise ValueError(f"'{prompt}': each object once, at most {judge.MAX_PAIRS}.")
        missing = sorted({name for pair in pairs for name in pair if name not in scene["items"]})
        if missing:
            raise ValueError(f"Not on the table for '{prompt}': {', '.join(missing)}")
        clean.append({"prompt": prompt, "pairs": [{"object": o, "target": g} for o, g in pairs]})
    return clean


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
            try:
                scene = parse_scene(body.get("scene"))
            except (ValueError, TypeError, AttributeError) as error:
                return self.send_json({"error": f"Scene: {error}"}, 400)
            missing = [name for name in judge.parse_task(task) if scene and name not in scene["items"]]
            if missing:
                return self.send_json({"error": f"Not on the table: {', '.join(missing)}"}, 400)
            if BATCH.running():
                return self.send_json({"error": "A batch is running."}, 409)
            try:
                seconds = min(300, max(10, int(float(body.get("seconds") or 60))))
                RUNNER.start(task, seconds, scene)
            except (ValueError, TypeError):
                return self.send_json({"error": "The time limit must be a number."}, 400)
            except RuntimeError as error:
                return self.send_json({"error": str(error)}, 409)
            return self.send_json(RUNNER.status())
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
                    meta = read_json(d / "meta.json", {})
                    write_label(d, run_verdict(d, rules), rules, batch_id=meta.get("batch_id"))
                    saved += 1
            return self.send_json({"saved": saved})
        if self.path == "/api/key":
            key = str(body.get("key") or "").strip()
            if key and (len(key) > 512 or any(c.isspace() for c in key)):
                return self.send_json({"error": "That does not look like an API key."}, 400)
            PAGE_KEY["value"] = key or None
            return self.send_json(key_status())
        if self.path == "/api/batch":
            try:
                scene = parse_scene(body.get("scene"))
                if scene is None:
                    raise ValueError("Scene: pick what is on the table.")
                tasks = parse_task_list(body.get("tasks"), scene)
                count = int(body.get("count") or 0)
                if not 1 <= count <= MAX_BATCH:
                    raise ValueError(f"Episodes must be 1 to {MAX_BATCH}.")
                seconds = min(300, max(10, int(float(body.get("seconds") or 60))))
                layout = body.get("layout")
                if layout not in ("shift", "random"):
                    raise ValueError("Start positions must be 'shift' or 'random'.")
                seed = int(body["seed"]) if str(body.get("seed", "")).strip() else secrets.randbelow(1_000_000)
                BATCH.start(tasks, count, seconds, layout, scene, seed)
            except (ValueError, TypeError, AttributeError) as error:
                return self.send_json({"error": str(error)}, 400)
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
