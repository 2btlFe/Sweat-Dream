"""A small website for running Dream-1 in the Dream Labs simulator.

    python app.py              then open http://localhost:8000

Press Run on the page: sim.py runs the task in its own process, and the page shows the cameras live.
The page also sends the keyboard (keys held, single presses) to sim.py, which moves the arms by hand (teleop.py).
Recorded runs (episodes/) are listed on the page; any one can be resumed at any step with a human in control.
Standard library only, plus numpy and PyAV (installed with dreamlabs) for reading recordings.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "static" / "index.html"
WORK = Path(tempfile.mkdtemp(prefix="dream-web-"))
LIVE = WORK / "live.jpg"
STOP = WORK / "stop"
TELEOP_STATE = WORK / "teleop.json"
EXAMPLES = ["put apple into the bowl", "put orange into the bowl", "put banana on the plate"]
EPISODES = HERE / "episodes"
EPISODE_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def list_episodes():
    """Recorded runs, newest first: what the page needs to pick one and a step to resume from."""
    rows = []
    for folder in sorted(EPISODES.glob("*"), reverse=True):
        if not EPISODE_ID.match(folder.name) or not (folder / "steps.npz").exists():
            continue
        meta, outcome = read_json(folder / "meta.json", {}), read_json(folder / "outcome.json", {})
        failures = outcome.get("failures_seen") or []
        events = read_json(folder / "events.json", []) or []
        ended = next((e.get("reason") for e in events if e.get("kind") == "ended"), None)
        script = [e.get("result") for e in events if e.get("kind") == "script" and e.get("result")]
        rows.append({
            "id": folder.name,
            "task": meta.get("task"),
            "started_as": meta.get("started_as"),
            "steps": meta.get("steps"),
            "sim_s": meta.get("sim_s"),
            "label": outcome.get("label"),
            "final_stage": outcome.get("final_stage"),
            "failures": [{"event": f.get("event"), "step": f.get("step")} for f in failures],
            "steps_by_source": meta.get("steps_by_source"),
            "resumed_from": meta.get("resumed_from"),
            "ended": ended,  # why a collection run ended early (the recovery script gave up, ...)
            "script": script,  # what the recovery script did each time it was called
        })
    return rows


FRAMES = {}  # episode id -> (frame_t, step times, top camera frames), the last few looked at


def episode_frame(episode, step):
    """The top camera picture at a step of a recorded episode (the last frame at or before it), as JPEG."""
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


def has_api_key():
    """The places the simulator reads the key from: the environment, the sim's /login file, ~/.dreamlabs/credentials."""
    if os.environ.get("DREAM_API_KEY"):
        return True
    try:
        login = Path.home() / ".config" / "dream-yam-sim" / "credentials.json"
        if json.loads(login.read_text(encoding="utf-8")).get("api_key"):
            return True
    except (OSError, ValueError, AttributeError):
        pass
    try:
        return "DREAM_API_KEY=" in (Path.home() / ".dreamlabs" / "credentials").read_text(encoding="utf-8")
    except OSError:
        return False


class Runner:
    """One run at a time."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.task = None
        self.teleop_only = False
        self.started = None
        self.ended = None
        self.log = deque(maxlen=30)

    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, task, seconds, teleop_only=False, record=True, resume=None, collect=False):
        """@param resume (episode id, step) to restart a recorded run there with a human in control, or None."""
        with self.lock:
            if self.running():
                raise RuntimeError("A run is already going.")
            STOP.unlink(missing_ok=True)
            LIVE.unlink(missing_ok=True)
            TELEOP_STATE.unlink(missing_ok=True)
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore",
                   "LIVE_FRAME": str(LIVE), "STOP_FILE": str(STOP), "TELEOP_STATE": str(TELEOP_STATE)}
            command = [sys.executable, str(HERE / "sim.py"), task, "--seconds", str(seconds)]
            if teleop_only:
                command.append("--teleop")
            if not record:
                command.append("--no-record")
            if resume:
                command += ["--resume", resume[0], "--resume-step", str(int(resume[1]))]
                teleop_only, collect = True, False
            if collect and not teleop_only:
                command.append("--collect")
            self.process = subprocess.Popen(
                command, cwd=str(HERE), env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            self.task, self.teleop_only, self.started, self.ended = task, teleop_only, time.time(), None
            self.seconds, self.record, self.resume, self.collect = seconds, record, resume, collect
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

    def stop_and_wait(self, timeout=20):
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
        """Reset: stop the current run (a Dream run first releases its session) and start the same one again,
        so the scene is back at its start."""
        if self.task is None:
            raise RuntimeError("Nothing to reset: press Run first.")
        process = self.process
        self.stop()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        self.start(self.task, self.seconds, self.teleop_only, self.record, self.resume, self.collect)

    def send_keys(self, message):
        """Pass keyboard input to the running sim.py (one JSON line on its stdin)."""
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
            "teleop_only": self.teleop_only,
            "teleop": self.teleop_state() if running else None,
            "elapsed_s": round((self.ended or time.time()) - self.started) if self.started else None,
            "exit_code": None if running or self.process is None else self.process.returncode,
            "log": list(self.log),
            "api_key": has_api_key(),
            "examples": EXAMPLES,
        }


RUNNER = Runner()


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
        if path == "/api/episodes":
            return self.send_json({"episodes": list_episodes()})
        match = re.fullmatch(r"/api/episodes/([^/]+)/frame\.jpg", path)
        if match and EPISODE_ID.match(match.group(1)):
            query = dict(p.split("=", 1) for p in (self.path.split("?", 1)[1:] or [""])[0].split("&") if "=" in p)
            try:
                return self.send(episode_frame(match.group(1), int(query.get("step", 0))), "image/jpeg")
            except (OSError, ValueError, KeyError, IndexError):
                return self.send_json({"error": "no picture for that episode"}, 404)
        if path == "/api/live.jpg":
            try:
                return self.send(LIVE.read_bytes(), "image/jpeg")
            except OSError:
                return self.send_json({"error": "no frame yet"}, 404)
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
            try:
                seconds = min(900, max(10, int(float(body.get("seconds") or 60))))
                RUNNER.start(task, seconds, teleop_only=bool(body.get("teleop_only")),
                             record=body.get("record", True) is not False,
                             collect=body.get("collect", True) is not False)
            except (ValueError, TypeError):
                return self.send_json({"error": "The time limit must be a number."}, 400)
            except RuntimeError as error:
                return self.send_json({"error": str(error)}, 409)
            return self.send_json(RUNNER.status())
        if self.path == "/api/keys":
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
            if not EPISODE_ID.match(episode) or not (EPISODES / episode / "steps.npz").exists():
                return self.send_json({"error": "No such recording."}, 404)
            task = (read_json(EPISODES / episode / "meta.json", {}) or {}).get("task") or "put apple into the bowl"
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
            try:
                RUNNER.restart()
            except RuntimeError as error:
                return self.send_json({"error": str(error)}, 409)
            return self.send_json(RUNNER.status())
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
        print("No API key yet: put DREAM_API_KEY=... in ~/.dreamlabs/credentials (see README).", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        RUNNER.stop()


if __name__ == "__main__":
    main()
