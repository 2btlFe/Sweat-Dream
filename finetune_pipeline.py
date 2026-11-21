"""Test, combine, upload and monitor Dream Labs datasets made by lerobot_export."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import av
import dream
import numpy as np

import lerobot_export

FPS = lerobot_export.FPS
CONFIG = lerobot_export.EMBODIMENT


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(path):
    result = lerobot_export.validate(path)
    if result["ok"] is not True:
        raise ValueError(result["error"])
    print(f"Dream Labs local validation passed: {result['episodes']} episodes, {result['frames']} frames")
    return result


def _write_sample_video(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for frame_number in range(12):
            image = np.full((64, 96, 3), color, dtype=np.uint8)
            image[18:34, 8 + frame_number:24 + frame_number] = (240, 90, 40)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def sample_recording(folder):
    """A deterministic 2.4-second synthetic recording; never a real demonstration."""
    folder.mkdir(parents=True)
    count = 12 * 4
    state = np.zeros((count, 14), dtype=np.float32)
    action = state.copy()
    action[:, 0] = np.linspace(0, .3, count)
    np.savez(folder / "steps.npz", t=np.arange(1, count + 1) / 20,
             state=state, action=action, frame_t=np.arange(12) / FPS)
    for camera, color in (("top", 35), ("left", 90), ("right", 145)):
        _write_sample_video(folder / f"{camera}.mp4", color)
    (folder / "meta.json").write_text(json.dumps({"robot": "yam", "task": "put apple into the bowl"}))


def _check_output(out, replace):
    out = Path(out).expanduser().resolve()
    if out in (Path("/"), Path.home().resolve(), Path.cwd().resolve()):
        raise ValueError(f"Refusing to use {out} as a dataset output directory")
    if out.exists():
        if not replace:
            raise ValueError(f"{out} already exists; choose another path or pass --replace")
        shutil.rmtree(out)
    return out


def _finish(out, result, sources):
    if not result["episodes"]:
        raise ValueError(f"No episodes exported: {result['skipped']}")
    (out.parent / f"{out.name}-embodiment.json").write_text(json.dumps(CONFIG, indent=2))
    (out.parent / f"{out.name}-provenance.json").write_text(json.dumps({"sources": sources, **result}, indent=2))
    print(f"Exported {len(result['episodes'])} episodes; skipped {len(result['skipped'])}: {out}")
    validate(out)


def make_dummy(output, replace=False):
    out = _check_output(output, replace)
    with tempfile.TemporaryDirectory(prefix="sweat-dream-dummy-") as temp:
        folder = Path(temp) / "episode"
        sample_recording(folder)
        result = lerobot_export.export([(folder, "put apple into the bowl")], out)
    _finish(out, result, ["synthetic dummy; do not use as a real demonstration"])


def selected_recordings(inputs):
    selected, seen = [], set()
    for input_path in inputs:
        root = Path(input_path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"No source folder: {root}")
        episodes_dir, runs_dir = root / "episodes", root / "runs"
        if not episodes_dir.is_dir() or not runs_dir.is_dir():
            raise ValueError(f"{root} needs both episodes/ and runs/ (a saved dataset or teammate repo root)")
        manifest = root / "manifest.json"
        if manifest.is_file():
            entries = read_json(manifest).get("runs", [])
            candidates = [(entry["id"], entry["task"]) for entry in entries
                          if entry.get("label") == "success" and entry.get("recorded")]
        else:
            candidates = []
            for run in sorted(runs_dir.iterdir()):
                if not (run / "meta.json").is_file() or not (run / "label.json").is_file():
                    continue
                if read_json(run / "label.json").get("label") != "success":
                    continue
                meta = read_json(run / "meta.json")
                if meta.get("episode"):
                    candidates.append((meta["episode"], meta["task"]))
        for episode_id, task in candidates:
            folder = episodes_dir / episode_id
            if not (folder / "steps.npz").is_file():
                continue
            identity = folder.resolve()
            if identity in seen:
                continue
            seen.add(identity)
            if read_json(folder / "meta.json").get("robot", "yam") != "yam":
                raise ValueError(f"{folder}: only YAM recordings fit this embodiment")
            selected.append((folder, task))
    return selected


def combine(output, inputs, replace=False):
    selected = selected_recordings(inputs)
    if not selected:
        raise ValueError("No labeled successful recordings found")
    out = _check_output(output, replace)
    result = lerobot_export.export(selected, out)
    _finish(out, result, [str(folder) for folder, _ in selected])


def client():
    return dream.DreamClient()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dummy = commands.add_parser("dummy", help="Create a synthetic local format test")
    dummy.add_argument("output")
    dummy.add_argument("--replace", action="store_true")
    build = commands.add_parser("build", help="Combine labeled successes from teammate saved datasets")
    build.add_argument("output")
    build.add_argument("inputs", nargs="+")
    build.add_argument("--replace", action="store_true")
    check = commands.add_parser("validate", help="Run Dream Labs' local data checks")
    check.add_argument("dataset")
    register = commands.add_parser("register", help="Register the YAM embodiment under your account")
    register.add_argument("name")
    upload = commands.add_parser("upload", help="Upload a validated dataset to your account")
    upload.add_argument("dataset")
    upload.add_argument("name")
    upload.add_argument("--embodiment", required=True)
    train = commands.add_parser("train", help="Queue a billable fine-tuning job")
    train.add_argument("dataset", help="Processed dataset name, username/name")
    train.add_argument("suffix")
    train.add_argument("--steps", type=int, default=500)
    train.add_argument("--base-share", type=float, default=0.5)
    status = commands.add_parser("status", help="Inspect dataset or fine-tuning job")
    status.add_argument("id")
    status.add_argument("--dataset", action="store_true")
    commands.add_parser("queue", help="Inspect fine-tuning queue")
    args = parser.parse_args(argv)
    if args.command == "dummy":
        make_dummy(args.output, args.replace)
    elif args.command == "build":
        combine(args.output, args.inputs, args.replace)
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
