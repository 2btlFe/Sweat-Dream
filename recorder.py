"""Episode recorder for the Dream YAM Sim.

Every control step (50 ms of simulated time) it records the arms' state, the commanded action, where every object is
and whether a fingertip touches it; every frame the model sees (one step in four) it keeps the three camera images.
On close it writes one folder per episode:

    episodes/<episode_id>/
        meta.json      instruction, model, source, timings, request stats, errors
        steps.npz      t, state (N,14), action (N,14), objects (N,4,3), held (N,4), frame_t (F,)
        top.mp4 left.mp4 right.mp4   the cameras at 5 fps of simulated time (H.264 when available)
        outcome.json   the simulator ground-truth judgement (judge.py)
"""

import json
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import judge
from yam_env import YamEnv

EPISODES_DIR = Path(__file__).resolve().parent / "episodes"
CAMERAS = ("top", "left", "right")
FPS = 5  # one kept frame every 0.2 s of simulated time


def new_episode_id():
    """A sortable, unique episode id: UTC time plus a short random suffix."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


class EpisodeRecorder:
    """Collects one episode's trace and writes it to disk."""

    def __init__(self, env, task, *, model, source, stop_on_success=False, out_dir=EPISODES_DIR, extra=None):
        """
        @param env The YamEnv being recorded (a RecordingEnv calls on_step()).
        @param task Instruction text.
        @param model Model id, or None for a scripted episode.
        @param source Where the actions came from: "dream-1" or "scripted:<variant>".
        @param stop_on_success Whether to set stop_event once the judge sees the task done and settled.
        @param out_dir Parent folder for episode folders.
        @param extra Extra metadata to store in meta.json.
        """
        self.env, self.task, self.model, self.source = env, task, model, source
        self.id = new_episode_id()
        self.dir = Path(out_dir) / self.id
        self.stop_event = threading.Event()
        self.stop_on_success = stop_on_success and judge.parse_task(task) is not None
        self.extra = dict(extra or {})
        self.t, self.state, self.action, self.objects, self.held = [], [], [], [], []
        self.frames = {name: [] for name in CAMERAS}
        self.frame_t = []
        self.rtts, self.errors = [], []
        self.started_wall = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.stopped_early_at = None
        m = env.model
        self._body_ids = [m.body(name).id for name in judge.BODIES]
        self._tip_ids = {m.body(f"{side}_{part}").id for side in ("left", "right")
                         for part in ("tip_left", "tip_right", "gripper")}
        self.targets = {name: env.data.body(name).xpos.copy() for name in judge.TARGETS}

    # -- recording ---------------------------------------------------------------------------------------------

    def _held(self):
        """Which objects touch a fingertip right now."""
        m, d = self.env.model, self.env.data
        held = np.zeros(len(self._body_ids), dtype=bool)
        index = {b: i for i, b in enumerate(self._body_ids)}
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in self._tip_ids and b2 in index:
                held[index[b2]] = True
            elif b2 in self._tip_ids and b1 in index:
                held[index[b1]] = True
        return held

    def on_step(self, left, right):
        """Record the step that just ran (called by RecordingEnv.step after physics)."""
        d = self.env.data
        s = self.env.state()
        self.t.append(float(d.time))
        self.state.append(np.r_[s["left"], s["right"]])
        self.action.append(np.r_[np.asarray(left, float), np.asarray(right, float)])
        self.objects.append(np.array([d.xpos[b] for b in self._body_ids]))
        self.held.append(self._held())
        if self.stop_on_success and len(self.t) % 10 == 0 and not self.stop_event.is_set():
            if judge.is_success_now(self.task, self.t, self.objects, self.held, self.targets):
                self.stopped_early_at = self.t[-1]
                self.stop_event.set()

    def on_frame(self, images):
        """Keep one set of camera images (the frames the model sees)."""
        if not images:
            return
        self.frame_t.append(float(self.env.data.time))
        for name in CAMERAS:
            self.frames[name].append(np.asarray(images[name]))

    # -- writing -----------------------------------------------------------------------------------------------

    def _write_video(self, path, frames):
        """Write frames as H.264 MP4 (browser-playable), falling back to MPEG-4 Part 2."""
        import av

        h, w = frames[0].shape[:2]
        for codec in ("libx264", "h264", "mpeg4"):
            try:
                with av.open(str(path), mode="w") as container:
                    stream = container.add_stream(codec, rate=FPS)
                    stream.width, stream.height, stream.pix_fmt = w - w % 2, h - h % 2, "yuv420p"
                    for rgb in frames:
                        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[: h - h % 2, : w - w % 2]), format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                return codec
            except Exception:
                path.unlink(missing_ok=True)
        return None

    def close(self, error=None):
        """Judge the episode and write its folder.
        @param error The exception that ended the episode, if any.
        @return The outcome dict.
        """
        if error is not None:
            self.errors.append(f"{type(error).__name__}: {error}")
        self.dir.mkdir(parents=True, exist_ok=True)
        t = np.array(self.t)
        objects = np.array(self.objects).reshape(-1, len(judge.BODIES), 3)
        held = np.array(self.held, dtype=bool).reshape(-1, len(judge.BODIES))
        np.savez_compressed(
            self.dir / "steps.npz", t=t, state=np.array(self.state), action=np.array(self.action),
            objects=objects, held=held, frame_t=np.array(self.frame_t),
            targets=np.array([self.targets[k] for k in judge.TARGETS]), bodies=np.array(judge.BODIES),
        )
        codec = None
        for name in CAMERAS:
            if self.frames[name]:
                codec = self._write_video(self.dir / f"{name}.mp4", self.frames[name])
        outcome = judge.judge(self.task, t, objects, held, self.targets)
        if self.errors and outcome["label"] != "success":
            outcome["warnings"].append("episode ended with an error: " + self.errors[-1])
        meta = {
            "episode_id": self.id,
            "task": self.task,
            "model": self.model,
            "source": self.source,
            "started_at": self.started_at,
            "wall_s": round(time.monotonic() - self.started_wall, 2),
            "sim_s": round(float(t[-1] - t[0]), 2) if len(t) else 0.0,
            "steps": len(t),
            "frames": len(self.frame_t),
            "fps": FPS,
            "video_codec": codec,
            "requests": len(self.rtts),
            "rtt_mean_s": round(float(np.mean(self.rtts)), 3) if self.rtts else None,
            "rtt_max_s": round(float(np.max(self.rtts)), 3) if self.rtts else None,
            "stopped_early_at_s": self.stopped_early_at,
            "errors": self.errors,
            **self.extra,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        (self.dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
        return outcome


## Free joints of the movable objects, by body name.
FREE_JOINTS = {"apple": "apple_free", "orange": "orange_free", "banana": "banana_free", "water_bottle": "bottle_free"}


class RecordingEnv(YamEnv):
    """YamEnv that reports every step to an EpisodeRecorder."""

    recorder = None

    def jitter(self, seed, xy_m=0.02, yaw_deg=25.0):
        """Shift each object's start pose by a small seeded random offset, so repeated runs are not identical.
        Offsets of +-2 cm keep the objects >= 10 cm apart and on the table (they start 11+ cm apart).
        @param seed Integer seed; the same seed gives the same scene.
        @param xy_m Largest horizontal shift per axis, metres.
        @param yaw_deg Largest rotation about the vertical axis, degrees.
        @return The offsets applied, {body: [dx, dy, dyaw_deg]}, for meta.json.
        """
        import mujoco

        rng = np.random.default_rng(seed)
        applied = {}
        for body, joint in FREE_JOINTS.items():
            dx, dy = rng.uniform(-xy_m, xy_m, 2)
            yaw = rng.uniform(-yaw_deg, yaw_deg) if body != "water_bottle" else 0.0  # the bottle stands upright
            qpos = self.data.joint(joint).qpos
            qpos[0] += dx
            qpos[1] += dy
            half = np.radians(yaw) / 2
            rotated = np.empty(4)
            mujoco.mju_mulQuat(rotated, np.array([np.cos(half), 0, 0, np.sin(half)]), qpos[3:7].copy())
            qpos[3:7] = rotated
            applied[body] = [round(float(dx), 4), round(float(dy), 4), round(float(yaw), 1)]
        mujoco.mj_forward(self.model, self.data)
        return applied

    def step(self, left, right):
        state = super().step(left, right)
        if self.recorder is not None:
            self.recorder.on_step(left, right)
        return state
