"""Appearance augmentation: re-render recorded episodes with new lighting, colours and camera poses.

The motion is the recorded one: every camera frame is rebuilt from steps.npz (the arms' six joints and gripper
opening, every object's position, and its orientation where recorded), so the simulator's judgement in
outcome.json holds for every variant: same episode, same label, different look. No API calls, no cost.

Per variant, drawn from a seed kept in its params.json:
- lighting: each light's colour, strength and direction, and the headlight's strength;
- surfaces: table, bowl, plate and floor colours, and the water bottle's (a distractor);
- objects to move: apple, banana and orange only mildly tinted, so they stay recognisable for their instruction;
- cameras: the top camera moved up to 1.5 cm and turned up to 2 degrees, the wrist cameras 0.5 cm and 1.5 degrees.
--strength scales every range (0 is the plain replay).

Output: episodes/<id>/aug/<k>/top.mp4 left.mp4 right.mp4 (same frames and fps as the originals) and params.json;
episodes/<id>/aug/index.json lists the variants. Episodes recorded before orientations were kept replay each object
with its starting orientation (from the seed's yaw); `--check` measures how close the plain replay is.

It waits while an episode is being recorded (the GPU is shared with the simulator's cameras).

Usage:
    python rerender.py <episode_id> [...] [--variants 4] [--seed 0] [--strength 1]
    python rerender.py --all [--variants 4]          every episode without variants yet
    python rerender.py <episode_id> --check          plain replay compared with the recorded video
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import dreamfix  # noqa: E402,F401  (puts the sim on sys.path)
import mujoco  # noqa: E402

import episode_io as eio  # noqa: E402  (read/write JSON, read_video, wait_until_idle)
from recorder import CAMERAS, EPISODES_DIR, FREE_JOINTS, RecordingEnv  # noqa: E402

FINGER_OPEN = 0.0475  # each finger's travel when the gripper opening is 1 (the sim's own constant)
TARGET_OBJECTS = ("apple", "banana", "orange")


# ------------------------------------------------------------------- appearance


def _rotate(quat, rng, max_deg):
    """quat turned by a random rotation of up to max_deg about a random axis."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, axis, np.radians(rng.uniform(0, max_deg)))
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, turn, quat)
    return out


class Appearance:
    """Draws and applies one variant's look on a model, and puts the original look back."""

    def __init__(self, model):
        self.model = model
        m = model
        self.original = {
            "light_diffuse": m.light_diffuse.copy(),
            "light_ambient": m.light_ambient.copy(),
            "light_dir": m.light_dir.copy(),
            "head_ambient": m.vis.headlight.ambient.copy(),
            "head_diffuse": m.vis.headlight.diffuse.copy(),
            "mat_rgba": m.mat_rgba.copy(),
            "geom_rgba": m.geom_rgba.copy(),
            "cam_pos": m.cam_pos.copy(),
            "cam_quat": m.cam_quat.copy(),
        }
        self.materials = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i): i for i in range(m.nmat)}
        body = lambda g: m.body(m.geom_bodyid[g]).name  # noqa: E731
        self.floor = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
        self.orange = [g for g in range(m.ngeom) if body(g) == "orange" and m.geom_rgba[g][3] > 0]
        self.bottle = [g for g in range(m.ngeom) if body(g) == "water_bottle" and m.geom_rgba[g][3] > 0]
        self.cameras = {m.camera(i).name: i for i in range(m.ncam)}

    def reset(self):
        m, o = self.model, self.original
        m.light_diffuse[:], m.light_ambient[:], m.light_dir[:] = o["light_diffuse"], o["light_ambient"], o["light_dir"]
        m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = o["head_ambient"], o["head_diffuse"]
        m.mat_rgba[:], m.geom_rgba[:] = o["mat_rgba"], o["geom_rgba"]
        m.cam_pos[:], m.cam_quat[:] = o["cam_pos"], o["cam_quat"]

    def apply(self, seed: int, strength: float = 1.0) -> dict:
        """Draw a look from seed and apply it; return what was drawn."""
        self.reset()
        if strength <= 0:
            return {"seed": seed, "strength": 0.0}
        rng = np.random.default_rng(seed)
        m, s = self.model, strength
        drawn = {"seed": seed, "strength": strength, "lights": []}
        for i in range(m.nlight):
            tint = 1 + s * rng.normal(0, 0.08, 3)
            scale = 1 + s * rng.uniform(-0.4, 0.4)
            m.light_diffuse[i] = np.clip(self.original["light_diffuse"][i] * scale * tint, 0, 1)
            m.light_ambient[i] = np.clip(self.original["light_ambient"][i] * (1 + s * rng.uniform(-0.5, 0.5)), 0, 1)
            turn = _rotate(np.array([1.0, 0, 0, 0]), rng, 20 * s)  # up to 20 degrees, any axis
            direction = np.zeros(3)
            mujoco.mju_rotVecQuat(direction, self.original["light_dir"][i].astype(np.float64), turn)
            m.light_dir[i] = direction / np.linalg.norm(direction)
            drawn["lights"].append({"diffuse": m.light_diffuse[i].round(3).tolist(), "ambient": m.light_ambient[i].round(3).tolist()})
        head = 1 + s * rng.uniform(-0.4, 0.4)
        m.vis.headlight.ambient[:] = np.clip(self.original["head_ambient"] * head, 0, 1)
        m.vis.headlight.diffuse[:] = np.clip(self.original["head_diffuse"] * head, 0, 1)
        drawn["headlight_scale"] = round(float(head), 3)

        def colour(low, high):
            return np.r_[rng.uniform(low, high, 3), 1.0]

        def blend(original, new):  # strength 1 is the new colour, 0 the original
            return np.clip((1 - s) * original + s * new, 0, 1)

        surfaces = {}
        for name, (low, high) in {"oak": (0.55, 1.0), "brown_glaze": (0.3, 1.0), "blue_glaze": (0.3, 1.0)}.items():
            if name in self.materials:
                i = self.materials[name]
                m.mat_rgba[i] = blend(self.original["mat_rgba"][i], colour(low, high))
                surfaces[{"oak": "table", "brown_glaze": "bowl", "blue_glaze": "plate"}[name]] = m.mat_rgba[i][:3].round(3).tolist()
        floor = colour(0.35, 0.95)
        for g in self.floor:
            m.geom_rgba[g] = blend(self.original["geom_rgba"][g], floor)
        surfaces["floor"] = floor[:3].round(3).tolist()
        bottle = colour(0.15, 0.95)
        for g in self.bottle:
            m.geom_rgba[g][:3] = blend(self.original["geom_rgba"][g][:3], bottle[:3])
        surfaces["water_bottle"] = bottle[:3].round(3).tolist()
        drawn["surfaces"] = surfaces
        objects = {}
        for fruit in ("apple", "banana"):  # textured: a mild multiplicative tint keeps them recognisable
            name = f"{fruit}_material"
            if name in self.materials:
                i = self.materials[name]
                tint = np.clip(1 - s * rng.uniform(0, 0.15, 3), 0, 1)
                m.mat_rgba[i][:3] = self.original["mat_rgba"][i][:3] * tint
                objects[fruit] = tint.round(3).tolist()
        tint = np.clip(1 + s * rng.uniform(-0.08, 0.08, 3), 0, None)
        for g in self.orange:
            m.geom_rgba[g][:3] = np.clip(self.original["geom_rgba"][g][:3] * tint, 0, 1)
        objects["orange"] = tint.round(3).tolist()
        drawn["object_tints"] = objects
        cameras = {}
        for name, i in self.cameras.items():
            move, turn = (0.015, 2.0) if name == "top" else (0.005, 1.5)
            offset = rng.uniform(-move, move, 3) * s
            m.cam_pos[i] = self.original["cam_pos"][i] + offset
            m.cam_quat[i] = _rotate(self.original["cam_quat"][i], rng, turn * s)
            cameras[name] = {"moved_m": offset.round(4).tolist()}
        drawn["cameras"] = cameras
        return drawn


# ----------------------------------------------------------------------- replay


class Replayer:
    """Puts a RecordingEnv into the recorded state of any step and renders the three cameras."""

    def __init__(self):
        self.env = RecordingEnv()
        self.appearance = Appearance(self.env.model)

    def load(self, d: Path):
        z = np.load(d / "steps.npz")
        self.t, self.state, self.objects = z["t"], z["state"], z["objects"]
        self.bodies = [str(b) for b in z["bodies"]]
        self.frame_t = z["frame_t"]
        self.quats = z["objects_quat"] if "objects_quat" in z.files else None
        meta = eio.read_json(d / "meta.json", {})
        self.fps = int(meta.get("fps") or 5)
        self.start_quats = {}
        jitter = meta.get("jitter") or {}
        data = self.env.data
        mujoco.mj_resetData(self.env.model, data)
        for body in self.bodies:
            q0 = data.joint(FREE_JOINTS[body]).qpos[3:7].copy()
            yaw = (jitter.get(body) or [0, 0, 0])[2]
            half = np.radians(yaw) / 2
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, np.array([np.cos(half), 0, 0, np.sin(half)]), q0)
            self.start_quats[body] = out
        return len(self.frame_t)

    def pose(self, k: int) -> None:
        """The scene at step k (positions and joints from the trace; velocities zero; no physics step)."""
        data, state = self.env.data, self.state[k]
        for side, base in (("left", 0), ("right", 7)):
            for i in range(6):
                data.joint(f"{side}_joint{i + 1}").qpos[0] = state[base + i]
            for sign in (-1, 1):
                data.joint(f"{side}_finger_{sign}").qpos[0] = float(np.clip(state[base + 6], 0, 1)) * FINGER_OPEN
        for j, body in enumerate(self.bodies):
            qpos = data.joint(FREE_JOINTS[body]).qpos
            qpos[0:3] = self.objects[k][j]
            qpos[3:7] = self.quats[k][j] if self.quats is not None else self.start_quats[body]
        data.qvel[:] = 0
        mujoco.mj_forward(self.env.model, data)

    def frames(self, on_frame=None):
        """The three cameras at every kept frame time, as {camera: HxWx3 uint8}."""
        for n, ft in enumerate(self.frame_t):
            if on_frame is not None:
                on_frame(n)
            self.pose(int(np.argmin(np.abs(self.t - ft))))
            yield self.env.images()

    def close(self):
        self.env.close()


def write_videos(folder: Path, frames, fps: int, on_frame=None) -> int:
    import av

    folder.mkdir(parents=True, exist_ok=True)
    containers, streams, count = {}, {}, 0
    try:
        for images in frames:
            for cam in CAMERAS:
                rgb = images[cam]
                h, w = rgb.shape[0] - rgb.shape[0] % 2, rgb.shape[1] - rgb.shape[1] % 2
                if cam not in containers:
                    containers[cam] = av.open(str(folder / f"{cam}.tmp.mp4"), mode="w")
                    for codec in ("libx264", "h264", "h264_videotoolbox", "mpeg4"):
                        try:
                            streams[cam] = containers[cam].add_stream(codec, rate=fps)
                            break
                        except Exception:
                            continue
                    streams[cam].width, streams[cam].height, streams[cam].pix_fmt = w, h, "yuv420p"
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[:h, :w]), format="rgb24")
                for packet in streams[cam].encode(frame):
                    containers[cam].mux(packet)
            count += 1
    finally:
        for cam, container in containers.items():
            for packet in streams[cam].encode():
                container.mux(packet)
            container.close()
    for cam in CAMERAS:
        (folder / f"{cam}.tmp.mp4").replace(folder / f"{cam}.mp4")
    return count


def episode_seed(episode_id: str, base: int, k: int) -> int:
    return (zlib.crc32(episode_id.encode()) + 1000 * base + k) % (2**31)


def augment(replayer: Replayer, d: Path, variants: int, base_seed: int, strength: float) -> dict:
    started = time.time()
    total = replayer.load(d)
    outcome = eio.read_json(d / "outcome.json", {})
    index = {"episode_id": d.name, "label": outcome.get("label"), "reason": outcome.get("reason"),
             "orientations": "recorded" if replayer.quats is not None else "start orientation only",
             "frames": total, "fps": replayer.fps, "variants": []}
    for k in range(1, variants + 1):
        seed = episode_seed(d.name, base_seed, k)
        drawn = replayer.appearance.apply(seed, strength)
        tick = time.time()
        count = write_videos(d / "aug" / str(k), replayer.frames(lambda n: n % 20 == 0 and eio.wait_until_idle()), replayer.fps)
        params = {"variant": k, "label_from": "outcome.json (same motion)", "label": outcome.get("label"),
                  "frames": count, "seconds": round(time.time() - tick, 1), **drawn}
        eio.write_json(d / "aug" / str(k) / "params.json", params)
        index["variants"].append({"variant": k, "seed": seed, "frames": count})
        print(f"  {d.name} variant {k}: {count} frames in {params['seconds']}s", flush=True)
    replayer.appearance.reset()
    index["seconds"] = round(time.time() - started, 1)
    eio.write_json(d / "aug" / "index.json", index)
    return index


def check(replayer: Replayer, d: Path) -> dict:
    """The plain replay against the recorded video: mean absolute difference per frame (0-255) and PSNR."""
    replayer.load(d)
    replayer.appearance.reset()
    replayed = {cam: [] for cam in CAMERAS}
    for images in replayer.frames():
        for cam in CAMERAS:
            replayed[cam].append(images[cam])
    result = {}
    for cam in CAMERAS:
        original = eio.read_video(d / f"{cam}.mp4")
        diffs, psnrs = [], []
        for a, frame in zip(original, replayed[cam]):
            b = frame[: a.shape[0], : a.shape[1]].astype(np.float32)
            err = np.abs(a.astype(np.float32) - b)
            diffs.append(float(err.mean()))
            mse = float((err**2).mean())
            psnrs.append(99.0 if mse == 0 else 10 * np.log10(255**2 / mse))
        result[cam] = {"mean_abs_diff": round(float(np.median(diffs)), 2), "worst_frame_diff": round(max(diffs), 2),
                       "psnr_db": round(float(np.median(psnrs)), 1)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", nargs="*")
    parser.add_argument("--all", action="store_true", help="every episode without variants yet")
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--check", action="store_true", help="compare the plain replay with the recorded video")
    args = parser.parse_args()
    if not args.episodes and not args.all:
        parser.error("give episode ids, or --all")
    folders = [EPISODES_DIR / e for e in args.episodes] if args.episodes else [
        d for d in sorted(EPISODES_DIR.iterdir())
        if d.is_dir() and (d / "steps.npz").exists() and not (d / "aug" / "index.json").exists()
    ]
    eio.wait_until_idle()
    replayer = Replayer()
    try:
        for d in folders:
            if args.check:
                print(d.name, json.dumps(check(replayer, d)), flush=True)
            else:
                index = augment(replayer, d, args.variants, args.seed, args.strength)
                print(f"{d.name}: {len(index['variants'])} variants of {index['frames']} frames "
                      f"({index['label']}, orientations: {index['orientations']}) in {index['seconds']}s", flush=True)
    finally:
        replayer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
