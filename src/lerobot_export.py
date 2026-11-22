"""Recordings (episodes/<id>/) to the Dream Labs dataset format: LeRobot v2.1, as the fine-tuning upload reads it.

https://platform.dreamlabs.ai/api-documentation/fine-tuning/#4-check-and-upload-your-dataset

    <out>/
        meta/info.json            codebase_version v2.1, fps 5, the features below
        meta/tasks.jsonl          {"task_index", "task"}: each task's text, the prompt it trains on
        meta/episodes.jsonl       {"episode_index", "tasks", "length"}
        meta/episodes_stats.jsonl each episode's min / max / mean / std / count of the numeric features
        data/chunk-000/episode_<NNNNNN>.parquet
                                  observation.state (14,), action (14,), timestamp, frame_index, episode_index,
                                  index, task_index: one row per camera frame
        videos/chunk-000/observation.images.<role>/episode_<NNNNNN>.mp4   H.264, one frame per row
    <out>/../embodiment.json      the embodiment config to register the robot with (client.add_embodiment)

One row per camera frame (5 per second, the frames Dream-1 sees; the service brings 5 fps up to 30):
    observation.state  the arms when the frame was taken; action  the command sent on the next control step.
Vectors follow the Dream YAM layout, right arm first: right 6 joints (rad), right gripper (0 closed .. 1 open),
left 6 joints, left gripper. The recording keeps left first; it is reordered here.
Cameras: top -> overhead, left -> wrist_l, right -> wrist_r.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FPS = 5
CHUNK = 1000
MIN_FRAMES = 9  # 1.6 s at 5 fps: the service's shortest episode (one training sample)
CAMERAS = {"top": "overhead", "left": "wrist_l", "right": "wrist_r"}
PARTS = {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]}
EMBODIMENT = {
    "state": PARTS,
    "action": PARTS,
    "video": {role: f"observation.images.{role}" for role in CAMERAS.values()},
    "head_camera": "overhead",
}
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def right_first(x):
    """(N,14) left 7 + right 7 -> right 7 + left 7."""
    x = np.asarray(x, dtype=np.float32)
    return np.concatenate([x[:, 7:14], x[:, 0:7]], axis=1)


def frame_rows(steps):
    """For each camera frame, the step whose state is the arms at that moment (the step after the frame).
    @return (frame numbers kept, step numbers): frames after the last step have no next action and are left out."""
    t_before = np.asarray(steps["t"]) - float(np.median(np.diff(steps["t"]))) if len(steps["t"]) > 1 else steps["t"]
    rows = np.searchsorted(t_before, np.asarray(steps["frame_t"]) - 1e-6)
    keep = np.flatnonzero(rows < len(t_before))
    return keep, rows[keep]


def read_frames(path):
    import av

    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def write_video(path, frames):
    """H.264 MP4 at FPS, one frame per row. @return (height, width)."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    h, w = h - h % 2, w - w % 2
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for rgb in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb[:h, :w]), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return h, w


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    v = v.reshape(len(v), -1)
    return {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "mean": v.mean(0).tolist(),
            "std": v.std(0).tolist(), "count": [len(v)]}


def export(episodes, out, robot_type="yam"):
    """Write the recordings as one LeRobot v2.1 dataset.
    @param episodes [(episode folder, task text)], in the order they become episodes.
    @param out The dataset folder to create.
    @return {"episodes": [{"episode_index", "source", "frames"}], "skipped": [{"source", "reason"}], "frames": N}."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out)
    tasks, written, skipped, shapes = {}, [], [], {}
    index = 0
    for folder, task in episodes:
        folder = Path(folder)
        try:
            steps = np.load(folder / "steps.npz")
            keep, rows = frame_rows(steps)
            videos = {cam: read_frames(folder / f"{cam}.mp4") for cam in CAMERAS}
        except (OSError, ValueError, KeyError) as error:
            skipped.append({"source": folder.name, "reason": f"cannot read the recording ({error})"})
            continue
        keep_n = min([len(keep)] + [len(v) for v in videos.values()])
        if keep_n < MIN_FRAMES:
            skipped.append({"source": folder.name, "reason": f"{keep_n} frames, shorter than {MIN_FRAMES} (1.6 s)"})
            continue
        keep, rows = keep[:keep_n], rows[:keep_n]
        e = len(written)
        chunk = e // CHUNK
        task_index = tasks.setdefault(task, len(tasks))
        state = right_first(steps["state"][rows])
        action = right_first(steps["action"][rows])
        frame_index = np.arange(keep_n, dtype=np.int64)
        timestamp = (frame_index / FPS).astype(np.float32)
        table = pa.table({
            "observation.state": pa.array(list(state), type=pa.list_(pa.float32())),
            "action": pa.array(list(action), type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index),
            "episode_index": pa.array(np.full(keep_n, e, dtype=np.int64)),
            "index": pa.array(np.arange(index, index + keep_n, dtype=np.int64)),
            "task_index": pa.array(np.full(keep_n, task_index, dtype=np.int64)),
        })
        path = out / DATA_PATH.format(episode_chunk=chunk, episode_index=e)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        for cam, role in CAMERAS.items():
            key = f"observation.images.{role}"
            frames = [videos[cam][i] for i in keep]
            shapes[key] = write_video(out / VIDEO_PATH.format(episode_chunk=chunk, video_key=key, episode_index=e),
                                      frames)
        written.append({"episode_index": e, "source": folder.name, "task": task, "frames": keep_n,
                        "stats": {"observation.state": stats(state), "action": stats(action),
                                  "timestamp": stats(timestamp), "frame_index": stats(frame_index),
                                  "episode_index": stats(np.full(keep_n, e)),
                                  "index": stats(np.arange(index, index + keep_n)),
                                  "task_index": stats(np.full(keep_n, task_index))}})
        index += keep_n
    if not written:
        return {"episodes": [], "skipped": skipped, "frames": 0}

    width = 14
    features = {
        "action": {"dtype": "float32", "shape": [width], "names": list(name_vector())},
        "observation.state": {"dtype": "float32", "shape": [width], "names": list(name_vector())},
    }
    for key, (h, w) in shapes.items():
        features[key] = {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
                         "info": {"video.fps": FPS, "video.height": h, "video.width": w, "video.codec": "h264",
                                  "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}
    for name, dtype in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"),
                        ("index", "int64"), ("task_index", "int64")):
        features[name] = {"dtype": dtype, "shape": [1], "names": None}
    info = {
        "codebase_version": "v2.1", "robot_type": robot_type,
        "total_episodes": len(written), "total_frames": index, "total_tasks": len(tasks),
        "total_videos": len(written) * len(CAMERAS), "total_chunks": (len(written) - 1) // CHUNK + 1,
        "chunks_size": CHUNK, "fps": FPS, "splits": {"train": f"0:{len(written)}"},
        "data_path": DATA_PATH, "video_path": VIDEO_PATH, "features": features,
    }
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    lines = lambda rows: "".join(json.dumps(r) + "\n" for r in rows)  # noqa: E731
    (meta / "tasks.jsonl").write_text(lines({"task_index": i, "task": t} for t, i in tasks.items()))
    (meta / "episodes.jsonl").write_text(lines({"episode_index": w["episode_index"], "tasks": [w["task"]],
                                                "length": w["frames"]} for w in written))
    (meta / "episodes_stats.jsonl").write_text(lines({"episode_index": w["episode_index"], "stats": w["stats"]}
                                                     for w in written))
    return {"episodes": [{k: w[k] for k in ("episode_index", "source", "frames")} for w in written],
            "skipped": skipped, "frames": index}


def name_vector():
    for side in ("right", "left"):
        yield from (f"{side}_joint_{i}" for i in range(1, 7))
        yield f"{side}_gripper"


def validate(path):
    """The Dream Labs checks (dream.datasets.validate_dataset): local, nothing is sent, no key needed.
    @return {"ok", "error", "episodes", "frames", "notes"}, or {"ok": None, "error": why} when it cannot run."""
    try:
        from dream.datasets import validate_dataset

        report = validate_dataset(path, EMBODIMENT)
    except Exception as error:  # the library missing, or a check that cannot run
        return {"ok": None, "error": f"{type(error).__name__}: {error}"}
    return {"ok": bool(report.ok), "error": getattr(report, "error", None),
            "episodes": getattr(report, "episodes", None), "frames": getattr(report, "frames", None),
            "notes": list(getattr(report, "notes", None) or [])}
