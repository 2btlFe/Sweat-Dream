"""Labeling server: review recorded episodes in the browser and save human labels.

Usage: python label_server.py [--port 8765] [--labeler NAME]      then open http://localhost:8765

Standard library only. Human labels are saved next to each episode as episodes/<id>/label.json:
    {"label": "success" | "failure" | "uncertain", "reason": ..., "failure_time_s": ..., "notes": ...,
     "labeler": ..., "labeled_at": ...}
The simulator judgement (outcome.json) is never changed here; the page shows both so they can be compared.
"""

import argparse
import json
import mimetypes
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import summary

HERE = Path(__file__).resolve().parent
EPISODES = summary.EPISODES_DIR
STATIC = HERE / "static"
ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
LABELS = {"success", "failure", "uncertain"}
REASONS = ["placed_outside_target", "dropped_in_transit", "wrong_object_moved", "touched_but_not_moved",
           "object_not_moved", "still_holding", "still_holding_over_target", "not_settled_in_target",
           "left_target_after_placing", "object_fell_off_table", "unsafe_motion", "other"]
VIDEO_FILES = {"top.mp4", "left.mp4", "right.mp4"}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def episode_dir(episode_id):
    """The folder of a valid episode id, or None (ids are checked so no path outside episodes/ is reachable)."""
    if not ID_RE.match(episode_id or ""):
        return None
    d = EPISODES / episode_id
    return d if (d / "meta.json").exists() else None


def episode_summary(d):
    meta = read_json(d / "meta.json", {})
    outcome = read_json(d / "outcome.json", {})
    label = read_json(d / "label.json")
    verifier = read_json(d / "verifier.json")
    return {
        "id": d.name, "task": meta.get("task"), "source": meta.get("source"), "seed": meta.get("seed"),
        "batch_id": meta.get("batch_id"), "started_at": meta.get("started_at"), "sim_s": meta.get("sim_s"),
        "sim_label": outcome.get("label"), "sim_reason": outcome.get("reason"),
        "human_label": label.get("label") if label else None,
        "verifier_label": verifier.get("label") if verifier else None,
        "verifier_confidence": verifier.get("confidence") if verifier else None,
    }


def episode_detail(d):
    meta = read_json(d / "meta.json", {})
    frame_t = []
    try:
        with np.load(d / "steps.npz") as z:
            frame_t = z["frame_t"].tolist()
    except (OSError, KeyError, ValueError):
        pass
    return {
        **episode_summary(d), "meta": meta, "outcome": read_json(d / "outcome.json", {}),
        "label": read_json(d / "label.json"), "verifier": read_json(d / "verifier.json"),
        "videos": sorted(f.name for f in d.iterdir() if f.name in VIDEO_FILES),
        "frame_t0": frame_t[0] if frame_t else 0.0, "frame_count": len(frame_t), "fps": meta.get("fps", 5),
    }


def stats(items):
    """Agreement between human labels, the simulator judge and the verifier."""
    labeled = [e for e in items if e["human_label"] in ("success", "failure")]
    agree = sum(e["human_label"] == e["sim_label"] for e in labeled)
    confusion = {f"human_{h}/sim_{s}": sum(e["human_label"] == h and e["sim_label"] == s for e in labeled)
                 for h in ("success", "failure") for s in ("success", "failure")}
    with_verifier = [e for e in items if e["verifier_label"] in ("success", "failure")
                     and e["sim_label"] in ("success", "failure")]
    return {
        "episodes": len(items),
        "human_labeled": sum(e["human_label"] is not None for e in items),
        "uncertain": sum(e["human_label"] == "uncertain" for e in items),
        "human_vs_sim_agreement": round(agree / len(labeled), 3) if labeled else None,
        "human_vs_sim_n": len(labeled),
        "confusion": confusion,
        "verifier_vs_sim_agreement": round(sum(e["verifier_label"] == e["sim_label"] for e in with_verifier)
                                           / len(with_verifier), 3) if with_verifier else None,
        "verifier_vs_sim_n": len(with_verifier),
    }


class Handler(BaseHTTPRequestHandler):
    labeler = "anonymous"

    def log_message(self, fmt, *args):  # quiet: one line per request is noise during review
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        """Serve a file, honouring a single Range request (browsers need 206 responses to seek in a video)."""
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError):
                    return
                remaining -= len(chunk)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self.send_file(STATIC / "index.html")
        if path == "/api/episodes":
            items = [episode_summary(d) for d in sorted(EPISODES.iterdir(), reverse=True) if episode_dir(d.name)]
            return self.send_json({"episodes": items, "stats": stats(items), "reasons": REASONS,
                                   "labeler": self.labeler})
        m = re.match(r"^/api/episodes/([^/]+)$", path)
        if m:
            d = episode_dir(m.group(1))
            return self.send_json(episode_detail(d)) if d else self.send_json({"error": "not found"}, 404)
        m = re.match(r"^/media/([^/]+)/([a-z]+\.mp4)$", path)
        if m:
            d = episode_dir(m.group(1))
            if d and m.group(2) in VIDEO_FILES and (d / m.group(2)).exists():
                return self.send_file(d / m.group(2))
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        m = re.match(r"^/api/episodes/([^/]+)/label$", self.path)
        d = episode_dir(m.group(1)) if m else None
        if not d:
            return self.send_json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self.send_json({"error": "invalid JSON"}, 400)
        if body.get("clear"):
            (d / "label.json").unlink(missing_ok=True)
            return self.send_json({"ok": True, "label": None})
        label = body.get("label")
        reason = body.get("reason") or None
        t = body.get("failure_time_s")
        if label not in LABELS:
            return self.send_json({"error": f"label must be one of {sorted(LABELS)}"}, 400)
        if reason is not None and reason not in REASONS:
            return self.send_json({"error": "unknown reason"}, 400)
        if t is not None:
            try:
                t = round(float(t), 2)
            except (TypeError, ValueError):
                return self.send_json({"error": "failure_time_s must be a number"}, 400)
        record = {
            "label": label,
            "reason": reason if label == "failure" else None,
            "failure_time_s": t if label == "failure" else None,
            "notes": str(body.get("notes") or "")[:2000],
            "labeler": str(body.get("labeler") or self.labeler)[:80],
            "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (d / "label.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        self.send_json({"ok": True, "label": record})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--labeler", default="anonymous")
    args = parser.parse_args()
    Handler.labeler = args.labeler
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Labeling UI: http://localhost:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
