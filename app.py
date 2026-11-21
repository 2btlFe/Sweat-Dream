"""A small website for running Dream-1 in the Dream Labs simulator.

    python app.py              then open http://localhost:8000

Press Run on the page: sim.py runs the task in its own process, and the page shows the cameras live.
Standard library only.
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

import ood

HERE = Path(__file__).resolve().parent
PAGE = HERE / "static" / "index.html"
WORK = Path(tempfile.mkdtemp(prefix="dream-web-"))
LIVE = WORK / "live.jpg"
STOP = WORK / "stop"
PREVIEW_LOCK = threading.Lock()
EXAMPLES = [
    "Move an object to the center of the workspace",
    "Move an object to the left side of the workspace",
    "Move an object to the right side of the workspace",
]


def has_api_key():
    if os.environ.get("DREAM_API_KEY"):
        return True
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
        self.started = None
        self.ended = None
        self.log = deque(maxlen=30)

    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, task, seconds, changes):
        with self.lock:
            if self.running():
                raise RuntimeError("A run is already going.")
            STOP.unlink(missing_ok=True)
            LIVE.unlink(missing_ok=True)
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore",
                   "LIVE_FRAME": str(LIVE), "STOP_FILE": str(STOP)}
            command = [sys.executable, str(HERE / "sim.py"), task, "--seconds", str(seconds)]
            if changes:
                command.extend(["--ood", json.dumps(changes)])
            self.process = subprocess.Popen(
                command, cwd=str(HERE), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            self.task, self.started, self.ended = task, time.time(), None
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
        if path == "/ood.js":
            script = PAGE.with_name("ood.js")
            return self.send(script.read_bytes(), "text/javascript; charset=utf-8")
        if path == "/api/ood":
            return self.send_json(ood.GROUPS)
        if path == "/api/status":
            return self.send_json(RUNNER.status())
        if path == "/api/live.jpg":
            try:
                return self.send(LIVE.read_bytes(), "image/jpeg")
            except OSError:
                return self.send_json({"error": "no frame yet"}, 404)
        self.send_json({"error": "not found"}, 404)

    def preview(self, body):
        try:
            changes = ood.validate(body.get("ood", {}))
        except ValueError as error:
            return self.send_json({"error": str(error)}, 400)
        if RUNNER.running() or not PREVIEW_LOCK.acquire(blocking=False):
            return self.send_json({"error": "The simulator is busy."}, 409)
        try:
            with tempfile.TemporaryDirectory(prefix="dream-preview-") as folder:
                frame = Path(folder) / "initial.jpg"
                result = subprocess.run(
                    [sys.executable, str(HERE / "sim.py"), "--preview", str(frame), "--ood", json.dumps(changes)],
                    cwd=HERE, capture_output=True, text=True, timeout=25,
                )
                if result.returncode:
                    error = (result.stderr or result.stdout).strip().splitlines()
                    return self.send_json({"error": error[-1] if error else "Camera rendering failed."}, 422)
                return self.send(frame.read_bytes(), "image/jpeg")
        except (OSError, subprocess.TimeoutExpired):
            return self.send_json({"error": "Could not render the initial camera view."}, 500)
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
        if self.path == "/api/run":
            task = re.sub(r"\s+", " ", str(body.get("task") or "")).strip()
            if not task:
                return self.send_json({"error": "Describe what to pick up and where to place it."}, 400)
            try:
                seconds = min(300, max(10, int(float(body.get("seconds") or 60))))
            except (ValueError, TypeError):
                return self.send_json({"error": "The time limit must be a number."}, 400)
            try:
                changes = ood.validate(body.get("ood", {}))
                RUNNER.start(task, seconds, changes)
            except ValueError as error:
                return self.send_json({"error": str(error)}, 400)
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
