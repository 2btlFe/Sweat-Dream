"""Build and inspect a Dream Labs LeRobot v2.1 dataset from Sweat-Dream recordings.

Remote upload/training commands are explicit; build and validate stay local.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import av
import dream
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 5
CAMERAS = {"top": "overhead", "left": "wrist_l", "right": "wrist_r"}
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CONFIG = {
    "state": {"left_arm": [0, 6], "left_gripper": [6, 7],
              "right_arm": [7, 13], "right_gripper": [13, 14]},
    "action": {"left_arm": [0, 6], "left_gripper": [6, 7],
               "right_arm": [7, 13], "right_gripper": [13, 14]},
    "video": {role: f"observation.images.{role}" for role in CAMERAS.values()},
    "head_camera": "overhead",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_video(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def write_video(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def synthetic_episode():
    count = 12
    t = np.arange(count * 4, dtype=float) / 20
    state = np.zeros((len(t), 14), dtype=np.float32)
    action = state.copy()
    action[:, 0] = np.linspace(0, .3, len(t))
    frames = {}
    for camera, role in CAMERAS.items():
        base = {"top": 35, "left": 90, "right": 145}[camera]
        frames[role] = []
        for k in range(count):
            image = np.full((64, 96, 3), base, dtype=np.uint8)
            image[16:32, 8 + k:24 + k] = (240, 95, 40)
            frames[role].append(image)
    return {"task": "put apple into the bowl", "t": t,
            "state": state, "action": action, "frame_t": np.arange(count) / FPS,
            "frames": frames, "source": "synthetic-dummy"}


def recording_episode(folder):
    meta = read_json(folder / "meta.json")
    if meta.get("robot", "yam") != "yam":
        raise ValueError(f"{folder}: only YAM's 14-value state/action layout is supported")
    with np.load(folder / "steps.npz", allow_pickle=False) as data:
        t, state, action, frame_t = (np.asarray(data[key]) for key in ("t", "state", "action", "frame_t"))
    if state.ndim != 2 or action.ndim != 2 or state.shape[1] != 14 or action.shape[1] != 14:
        raise ValueError(f"{folder}: expected 14-value YAM state and action")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{folder}: nonfinite state or action")
    frames = {role: read_video(folder / f"{camera}.mp4") for camera, role in CAMERAS.items()}
    count = min(len(frame_t), *(len(images) for images in frames.values()))
    if count < 9:
        raise ValueError(f"{folder}: need at least 9 synchronized 5fps frames")
    if any(len(images) != count for images in frames.values()) or len(frame_t) != count:
        raise ValueError(f"{folder}: camera and timestamp frame counts differ")
    if len(t) < 2 or np.any(np.diff(t) <= 0) or np.any(np.diff(frame_t) <= 0):
        raise ValueError(f"{folder}: time must increase")
    return {"task": meta["task"], "t": t, "state": state, "action": action,
            "frame_t": frame_t, "frames": frames, "source": str(folder)}


def selected_recordings(roots):
    chosen = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"No recording directory: {root}")
        episodes = root / "episodes" if (root / "episodes").is_dir() else root
        runs = root / "runs"
        labels = {}
        if runs.is_dir():
            for run in runs.iterdir():
                if not (run / "meta.json").is_file() or not (run / "label.json").is_file():
                    continue
                run_meta = read_json(run / "meta.json")
                if read_json(run / "label.json").get("label") == "success" and run_meta.get("episode"):
                    labels[run_meta["episode"]] = run
        if not labels:
            print(f"Warning: {root} has no labeled success runs", file=sys.stderr)
        for episode_id, run in sorted(labels.items()):
            folder = episodes / episode_id
            if (folder / "steps.npz").is_file():
                chosen.append((folder, run))
    return chosen


def build(out, episodes, *, replace=False):
    out = Path(out).expanduser().resolve()
    if not episodes:
        raise ValueError("No eligible episodes")
    if out in (Path("/"), Path.home().resolve(), Path.cwd().resolve()):
        raise ValueError(f"Refusing to use {out} as a dataset output directory")
    if out.exists():
        if not replace:
            raise ValueError(f"{out} already exists; choose another path or pass --replace")
        shutil.rmtree(out)
    task_ids = {}
    episode_rows, provenance = [], []
    total = 0
    video_shapes = None
    for index, entry in enumerate(episodes):
        item = recording_episode(entry) if isinstance(entry, Path) else entry
        if item["task"] not in task_ids:
            task_ids[item["task"]] = len(task_ids)
        frame_count = len(item["frame_t"])
        if frame_count < 9:
            raise ValueError(f"{item['source']}: need at least 9 frames at 5 fps")
        sample = np.searchsorted(item["t"], item["frame_t"], side="left")
        sample = np.clip(sample, 0, len(item["t"]) - 1)
        rows = pa.table({
            "action": pa.array(item["action"][sample].astype(np.float32).tolist()),
            "observation.state": pa.array(item["state"][sample].astype(np.float32).tolist()),
            "timestamp": pa.array(np.arange(frame_count) / FPS),
            "episode_index": pa.array([index] * frame_count, type=pa.int64()),
            "frame_index": pa.array(list(range(frame_count)), type=pa.int64()),
            "task_index": pa.array([task_ids[item["task"]]] * frame_count, type=pa.int64()),
        })
        data_path = out / DATA_PATH.format(episode_chunk=index // 1000, episode_index=index)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(rows, data_path)
        shapes = {}
        for role, images in item["frames"].items():
            if len(images) != frame_count:
                raise ValueError(f"{item['source']}: {role} has a different frame count")
            shapes[role] = list(images[0].shape)
            key = CONFIG["video"][role]
            path = out / VIDEO_PATH.format(episode_chunk=index // 1000, video_key=key, episode_index=index)
            write_video(path, images)
        if video_shapes is None:
            video_shapes = shapes
        elif video_shapes != shapes:
            raise ValueError("All episodes must have matching camera sizes")
        episode_rows.append({"episode_index": index, "tasks": [item["task"]], "length": frame_count})
        provenance.append({"episode_index": index, "source": item["source"]})
        total += frame_count
    features = {"action": {"dtype": "float32", "shape": [14]},
                "observation.state": {"dtype": "float32", "shape": [14]}}
    for role, shape in video_shapes.items():
        features[CONFIG["video"][role]] = {"dtype": "video", "shape": shape,
                                           "names": ["height", "width", "channel"]}
    info = {"codebase_version": "v2.1", "fps": FPS, "chunks_size": 1000,
            "total_episodes": len(episode_rows), "total_frames": total,
            "data_path": DATA_PATH, "video_path": VIDEO_PATH, "features": features}
    meta = out / "meta"
    meta.mkdir(exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    jsonl(meta / "episodes.jsonl", episode_rows)
    jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": task} for task, i in task_ids.items()])
    (out / "embodiment.json").write_text(json.dumps(CONFIG, indent=2))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Built {len(episode_rows)} episodes, {total} frames: {out}")
    validate(out)


def validate(path):
    path = Path(path).expanduser().resolve()
    config = read_json(path / "embodiment.json")
    report = dream.datasets.validate_dataset(path, config)
    if not report.ok:
        raise ValueError(report.error)
    print(f"Dream Labs local validation passed: {report.episodes} episodes, {report.frames} frames")
    return report


def client():
    return dream.DreamClient()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dummy = commands.add_parser("dummy", help="Create a synthetic local format test")
    dummy.add_argument("output")
    dummy.add_argument("--replace", action="store_true")
    collect = commands.add_parser("build", help="Combine labeled successes from teammate repositories")
    collect.add_argument("output")
    collect.add_argument("inputs", nargs="+", help="Teammate repo roots containing runs/ and episodes/")
    collect.add_argument("--replace", action="store_true")
    check = commands.add_parser("validate", help="Run Dream Labs' local data checks")
    check.add_argument("dataset")
    register = commands.add_parser("register", help="Register the YAM embodiment under your account")
    register.add_argument("name", help="Short embodiment name, e.g. sweat-yam")
    upload = commands.add_parser("upload", help="Upload a validated dataset to your account")
    upload.add_argument("dataset")
    upload.add_argument("name")
    upload.add_argument("--embodiment", required=True, help="Existing embodiment name, username/name")
    train = commands.add_parser("train", help="Queue a billable fine-tuning job")
    train.add_argument("dataset", help="Processed dataset name, username/name")
    train.add_argument("suffix")
    train.add_argument("--steps", type=int, default=500)
    train.add_argument("--base-share", type=float, default=0.5)
    status = commands.add_parser("status", help="Inspect dataset or fine-tuning job")
    status.add_argument("id")
    status.add_argument("--dataset", action="store_true", help="ID is ds_... rather than a job ID")
    commands.add_parser("queue", help="Inspect the fine-tuning queue")
    args = parser.parse_args(argv)
    if args.command == "dummy":
        build(args.output, [synthetic_episode()], replace=args.replace)
    elif args.command == "build":
        chosen = selected_recordings(args.inputs)
        build(args.output, [folder for folder, _ in chosen], replace=args.replace)
    elif args.command == "validate":
        validate(args.dataset)
    elif args.command == "register":
        result = client().add_embodiment(args.name, CONFIG)
        print(f"Embodiment: {result.name}")
    elif args.command == "upload":
        validate(args.dataset)
        result = client().upload_dataset(args.dataset, args.name, args.embodiment)
        print(f"Dataset: {result.id} {result.status} ({result.episodes} episodes)")
    elif args.command == "train":
        if args.steps < 500 or args.steps > 2000 or args.steps % 100 or args.base_share not in (0.0, 0.5):
            parser.error("Steps must be 500–2000 in multiples of 100; base-share must be 0 or 0.5")
        result = client().finetune(args.dataset, args.suffix, steps=args.steps, base_share=args.base_share)
        print(json.dumps(result, indent=2))
    elif args.command == "status":
        api = client()
        result = api.get_dataset_status(args.id) if args.dataset else api.get_finetuning_status(args.id)
        print(json.dumps(result if isinstance(result, dict) else vars(result), indent=2, default=str))
    elif args.command == "queue":
        print(json.dumps(dream.fine_tuning.queue(), indent=2))


if __name__ == "__main__":
    main()
