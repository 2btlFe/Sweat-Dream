"""Episode recorder: every run is saved, whoever controls the arms (Dream-1, a human, or nobody holding still).

Every control step (50 ms of simulated time) it records the arms' state, the action sent to the simulator, who sent it,
the human's hand targets and gripper commands, where every object is and whether a fingertip touches it, and the full
MuJoCo state *before* the step (so any moment can be restored exactly, e.g. to take over from a failure). Every frame
the model sees (one step in four) it keeps the three camera images. Every human input and every control change is
logged with its step. On close it writes one folder per run:

    episodes/<episode_id>/
        meta.json      task, model, who controlled how much, timings, versions, the human/Dream hand-overs
        steps.npz      t (N,)                    simulated time after the step, s
                       state (N,14)              arms before the step: left 6 joints + gripper, right 6 + gripper
                       action (N,14)             what was sent to env.step (same layout)
                       source (N,)               who sent it: 0 dream, 1 human, 2 hold (bootstrap, paused, returning),
                                                 3 script (the recovery script, recovery.py)
                       hand_target (N,2,3)       the human controller's hand targets (left, right), metres
                       grip_command (N,2)        the human controller's gripper commands, 0 closed .. 1 open
                       objects (N,6,3)           positions of BODIES after the step
                       objects_quat (N,6,4)      orientations (w, x, y, z)
                       held (N,6)                whether a fingertip touches each body
                       stage (N,)                failure monitor stage: 0 reach, 1 carry, 2 placed, 3 success
                       physics (N,S)             mj_getState(INTEGRATION) before the step: mj_setState + action[k]
                                                 reproduces step k exactly
                       frame_t (F,)              simulated time of each camera frame
                       bodies, sources, stages   the names behind the codes above
        events.json    [{step, t, kind, ...}]   human inputs (keys, clicks, buttons), control changes, monitor events
        top.mp4 left.mp4 right.mp4   the cameras at 5 fps of simulated time
        outcome.json   the failure monitor's verdict: final stage, success, failures it saw

steps.npz and meta.json follow the layout of the team's earlier recorder (commit 98d9797) and add to it.
"""

from __future__ import annotations

import importlib.metadata
import json
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

EPISODES_DIR = Path(__file__).resolve().parent / "episodes"
CAMERAS = ("top", "left", "right")
FPS = 5  # camera frames: one every 0.2 s of simulated time (the frames the model sees)
BODIES = ["apple", "orange", "banana", "water_bottle", "bowl", "plate"]
SOURCES = ["dream", "human", "hold", "script"]
STAGES = ["reach", "carry", "placed", "success"]
PHYSICS = mujoco.mjtState.mjSTATE_INTEGRATION  # everything mj_step reads: qpos, qvel, act, ctrl, warm start, time


def read_video(path):
    """Every frame of a video as HxWx3 uint8 RGB."""
    import av

    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def new_episode_id():
    """A sortable, unique episode id: UTC time plus a short random suffix."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


class EpisodeRecorder:
    """Collects one run and writes it to disk (thread-safe: inputs arrive from the stdin thread)."""

    def __init__(self, env, task, *, mode, out_dir=EPISODES_DIR, extra=None):
        """
        @param env The YamEnv being recorded.
        @param task Instruction text.
        @param mode How the run started: "dream" or "human-only".
        """
        self.env, self.task, self.mode = env, task, mode
        self.id = new_episode_id()
        self.dir = Path(out_dir) / self.id
        self.extra = dict(extra or {})
        self.lock = threading.Lock()
        self.rows = {k: [] for k in ("t", "state", "action", "source", "hand_target", "grip_command", "objects",
                                     "objects_quat", "held", "stage", "physics")}
        self.frames = {name: [] for name in CAMERAS}
        self.frame_t = []
        self.events = []
        self.model_id = None
        self.started_wall = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        m = env.model
        self.body_ids = [m.body(name).id for name in BODIES]
        self.tip_ids = {m.body(f"{side}_{part}").id for side in ("left", "right")
                        for part in ("tip_left", "tip_right", "gripper")}
        self.physics_size = mujoco.mj_stateSize(m, PHYSICS)
        self._before = None
        self.closed = False

    # -- resuming ----------------------------------------------------------------------------------------------

    def load_prefix(self, source_dir, step):
        """Start this episode with the first `step` steps of a recorded one (rows, events, camera frames), so the new
        episode is one whole trajectory: the original up to `step`, then what happens from there.
        @param source_dir The recorded episode folder.
        @param step The step the new run starts from (the recording's physics[step] is restored by the caller).
        """
        data = np.load(Path(source_dir) / "steps.npz")
        sources = [str(s) for s in data["sources"]] if "sources" in data.files else ["dream", "human", "hold"]
        with self.lock:
            for key in self.rows:
                if key == "source":  # re-code by name: an older recording may list fewer sources
                    self.rows[key] = [SOURCES.index(sources[int(c)]) for c in data["source"][:step]]
                elif key in data.files:
                    self.rows[key] = list(data[key][:step])
                elif key == "hand_target":
                    self.rows[key] = [np.full((2, 3), np.nan)] * step
                elif key == "grip_command":
                    self.rows[key] = [np.full(2, np.nan)] * step
                else:
                    self.rows[key] = [-1] * step
            cut = float(data["t"][step - 1]) if step > 0 else 0.0
            try:
                events = json.loads((Path(source_dir) / "events.json").read_text())
            except (OSError, ValueError):
                events = []
            self.events = [e for e in events if e.get("step", 0) < step]
            keep = [i for i, t in enumerate(data["frame_t"]) if t <= cut + 1e-9]
        for name in CAMERAS:
            video = Path(source_dir) / f"{name}.mp4"
            if video.exists() and keep:
                frames = read_video(video)
                with self.lock:
                    self.frames[name] = [frames[i] for i in keep if i < len(frames)]
        with self.lock:
            self.frame_t = [float(data["frame_t"][i]) for i in keep][: len(self.frames["top"]) or None]
            self.events.append({"step": step, "t": round(cut, 3), "kind": "resumed",
                                "from_episode": Path(source_dir).name, "from_step": step})

    # -- recording ---------------------------------------------------------------------------------------------

    def before_step(self):
        """Call right before env.step: the arms' state and the full physics state the step starts from."""
        s = self.env.state()
        physics = np.empty(self.physics_size)
        mujoco.mj_getState(self.env.model, self.env.data, physics, PHYSICS)
        self._before = (np.r_[s["left"], s["right"]], physics)

    def after_step(self, left, right, source, hand_target, grip_command, stage):
        """Call right after env.step with what was sent and who sent it."""
        if self._before is None:
            return
        d = self.env.data
        state, physics = self._before
        self._before = None
        with self.lock:
            r = self.rows
            r["t"].append(float(d.time))
            r["state"].append(state)
            r["action"].append(np.r_[np.asarray(left, float), np.asarray(right, float)])
            r["source"].append(SOURCES.index(source))
            r["hand_target"].append(hand_target)
            r["grip_command"].append(grip_command)
            r["objects"].append(np.array([d.xpos[b] for b in self.body_ids]))
            r["objects_quat"].append(np.array([d.xquat[b] for b in self.body_ids]))
            r["held"].append(self._held())
            r["stage"].append(STAGES.index(stage) if stage in STAGES else -1)
            r["physics"].append(physics)

    def on_frame(self, images):
        """Keep one set of camera images."""
        if not images:
            return
        with self.lock:
            self.frame_t.append(float(self.env.data.time))
            for name in CAMERAS:
                self.frames[name].append(np.asarray(images[name]))

    def event(self, kind, **details):
        """A human input, a control change or a monitor event, at the current step."""
        with self.lock:
            self.events.append({"step": len(self.rows["t"]), "t": round(float(self.env.data.time), 3), "kind": kind,
                                **details})

    def _held(self):
        m, d = self.env.model, self.env.data
        held = np.zeros(len(self.body_ids), dtype=bool)
        index = {b: i for i, b in enumerate(self.body_ids)}
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in self.tip_ids and b2 in index:
                held[index[b2]] = True
            elif b2 in self.tip_ids and b1 in index:
                held[index[b1]] = True
        return held

    # -- writing -----------------------------------------------------------------------------------------------

    def _write_video(self, path, frames):
        """H.264 MP4 (browser-playable), falling back to MPEG-4 Part 2."""
        import av

        h, w = frames[0].shape[:2]
        for codec in ("libx264", "h264", "mpeg4"):
            try:
                with av.open(str(path), mode="w") as container:
                    stream = container.add_stream(codec, rate=FPS)
                    stream.width, stream.height, stream.pix_fmt = w - w % 2, h - h % 2, "yuv420p"
                    for rgb in frames:
                        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[: h - h % 2, : w - w % 2]),
                                                           format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                return codec
            except Exception:
                path.unlink(missing_ok=True)
        return None

    def close(self, monitor_status=None, error=None):
        """Write the episode folder. @return The folder, or None when nothing was recorded."""
        with self.lock:
            if self.closed:
                return None
            self.closed = True
            r = {k: list(v) for k, v in self.rows.items()}
            frames = {k: list(v) for k, v in self.frames.items()}
            frame_t, events = list(self.frame_t), list(self.events)
        n = len(r["t"])
        if n == 0:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        source = np.array(r["source"], dtype=np.int8)
        np.savez_compressed(
            self.dir / "steps.npz",
            t=np.array(r["t"]), state=np.array(r["state"]), action=np.array(r["action"]), source=source,
            hand_target=np.array(r["hand_target"], dtype=float).reshape(n, 2, 3),
            grip_command=np.array(r["grip_command"], dtype=float).reshape(n, 2),
            objects=np.array(r["objects"]).reshape(n, len(BODIES), 3),
            objects_quat=np.array(r["objects_quat"]).reshape(n, len(BODIES), 4),
            held=np.array(r["held"], dtype=bool).reshape(n, len(BODIES)),
            stage=np.array(r["stage"], dtype=np.int8), physics=np.array(r["physics"]),
            frame_t=np.array(frame_t), bodies=np.array(BODIES), sources=np.array(SOURCES), stages=np.array(STAGES),
        )
        codec = None
        for name in CAMERAS:
            if frames[name]:
                codec = self._write_video(self.dir / f"{name}.mp4", frames[name])
        (self.dir / "events.json").write_text(json.dumps(events, indent=1))
        steps_by = {name: int((source == i).sum()) for i, name in enumerate(SOURCES)}
        meta = {
            "episode_id": self.id,
            "task": self.task,
            "started_as": self.mode,
            "model": self.model_id,
            "started_at": self.started_at,
            "wall_s": round(time.monotonic() - self.started_wall, 2),
            "sim_s": round(float(r["t"][-1] - r["t"][0]), 2),
            "steps": n,
            "step_s": 0.05,
            "frames": len(frame_t),
            "fps": FPS,
            "video_codec": codec,
            "steps_by_source": steps_by,
            "control_changes": [e for e in events if e["kind"] == "control"],
            "physics_state": "mujoco.mj_getState(mjSTATE_INTEGRATION) before each step",
            "versions": {"dreamlabs": version("dreamlabs"), "mujoco": mujoco.__version__},
            "errors": [f"{type(error).__name__}: {error}"] if error else [],
            **self.extra,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        status = monitor_status or {}
        outcome = {
            "label": "success" if status.get("stage") == "success" else "failure",
            "final_stage": status.get("stage"),
            "failures_seen": [e for e in events if e["kind"] == "monitor"],
            "judge": "monitor.py (online stage rules); replace with the team's shared judge when it exists",
        } if monitor_status is not None else {
            # A run without the failure monitor (model only): its verdict is the rules' (runs/<id>/label.json).
            "label": None, "final_stage": None, "failures_seen": [],
            "judge": "none online (model only); see runs/<id>/label.json",
        }
        (self.dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
        return self.dir
