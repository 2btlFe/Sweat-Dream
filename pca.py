"""Feature PCA videos for recorded episodes: DINOv2 + NAF, 224 x 224, made after the episode.

For each episode folder with top.mp4 / left.mp4 / right.mp4 (5 fps), writes episodes/<id>/pca/:
    top.mp4 left.mp4 right.mp4   224 x 224 PCA maps, one per camera frame (same count and fps: they play in sync)
    info.json                    method, frames, timings, NAF speed-ups
    embeddings.npy               (frames, 3, 384) float16: DINOv2 mean patch feature per frame and view
    status.json                  waiting / running (done, total) / paused / ready / error

How a map is made: each view is resized to 224 x 224 and passed through DINOv2 ViT-S/14 (16 x 16 patches of
384-d features). One PCA basis per episode is fitted on frames spread over it and all three views, so a colour means
the same thing at every moment and in every view. NAF (Neighborhood Attention Filtering, CVPR 2026) upsamples the
3-component map to 224 x 224 zero-shot, guided by the image (naf_upsampler.py). Each map is averaged with its
neighbours in time (no lag, offline), and a view whose image did not change keeps its previous map.

It never runs while an episode is being recorded (episodes/.recording-<pid>, written by recorder.py): on a laptop
the GPU is shared with the simulator's cameras, and a slower simulator lets the Dream-1 session expire. It pauses,
then continues.

Usage:
    python pca.py <episode_id> [<episode_id> ...]
    python pca.py --all                every episode without finished PCA
    python pca.py --all --watch        keep going: new episodes are processed whenever none is recording
    options: --no-naf (bilinear), --smooth 1, --device mps|cuda|cpu, --force
    NAF_COMPILE=0 runs NAF without torch.compile (it is skipped by itself where it cannot build).

Two pca.py processes never work on the same episode: each holds episodes/<id>/pca/.lock while it does.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EPISODES_DIR = HERE / "episodes"  # same as recorder.EPISODES_DIR (no sim import needed)
CAMERAS = ("top", "left", "right")
SIZE = 224
BASIS_FRAMES = 48  # frames, spread over the episode, the PCA basis is fitted on
UNCHANGED = 0.5  # a view whose image changed less than this (mean |difference|, 0-255) keeps its previous map
METHOD = {
    "naf": "DINOv2 ViT-S/14 at 224x224 -> 16x16 patch features (384-d) -> one PCA basis per episode -> NAF "
    "(Neighborhood Attention Filtering, CVPR 2026) upsamples the projection zero-shot to 224x224, guided by the image",
    "bilinear": "DINOv2 ViT-S/14 at 224x224 -> 16x16 patch features (384-d) -> one PCA basis per episode -> "
    "bilinear upsampling to 224x224",
}


# --------------------------------------------------------------------- state


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def status(d: Path, **fields) -> None:
    write_json(d / "pca" / "status.json", {**fields, "updated": time.time()})


def recording_now() -> list[int]:
    """Pids of episodes being recorded (markers of finished or crashed processes are removed)."""
    import psutil

    live = []
    for marker in EPISODES_DIR.glob(".recording-*"):
        try:
            pid = int(marker.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if psutil.pid_exists(pid):
            live.append(pid)
        else:
            marker.unlink(missing_ok=True)
    return live


def wait_until_idle(d: Path | None = None, done: int = 0, total: int = 0) -> None:
    """Block while an episode is being recorded."""
    noted = False
    while recording_now():
        if d is not None and not noted:
            status(d, state="paused", done=done, total=total, reason="an episode is being recorded")
            noted = True
        time.sleep(2)


def claim(d: Path) -> bool:
    """Take the episode's PCA lock (pca/.lock holds the pid). False if another live pca.py holds it; a lock left by
    a finished or crashed run is taken over."""
    import psutil

    lock = d / "pca" / ".lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(lock.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and pid != os.getpid() and psutil.pid_exists(pid):
                return False
            lock.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return True
    return False


def release(d: Path) -> None:
    (d / "pca" / ".lock").unlink(missing_ok=True)


def pending(force: bool = False) -> list[Path]:
    """Episodes with camera videos and no finished PCA, oldest first."""
    out = []
    for d in sorted(EPISODES_DIR.iterdir()) if EPISODES_DIR.is_dir() else []:
        if d.name.startswith(".") or not (d / "meta.json").exists() or not (d / "top.mp4").exists():
            continue
        state = (read_json(d / "pca" / "status.json") or {}).get("state")
        if force or state not in ("ready", "error"):
            out.append(d)
    return out


# ---------------------------------------------------------------------- video


def read_video(path: Path) -> list[np.ndarray]:
    import av

    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


class VideoWriter:
    """H.264 MP4 (browser-playable), falling back to MPEG-4 Part 2, written frame by frame."""

    def __init__(self, path: Path, fps: int, size: int = SIZE):
        import av

        self.container = av.open(str(path), mode="w")
        for codec in ("libx264", "h264", "h264_videotoolbox", "mpeg4"):
            try:
                self.stream = self.container.add_stream(codec, rate=fps)
                break
            except Exception:
                continue
        self.stream.width = self.stream.height = size
        self.stream.pix_fmt = "yuv420p"

    def write(self, rgb: np.ndarray) -> None:
        import av

        for packet in self.stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


# -------------------------------------------------------------------- process


def process(d: Path, device=None, naf: bool = True, smooth: int = 1) -> dict:
    """Write the PCA videos of one episode; return its final status."""
    import torch

    import feature_pca as fp
    import naf_upsampler as nu

    started = time.time()
    meta = read_json(d / "meta.json", {})
    fps = int(meta.get("fps") or 5)
    status(d, state="loading", done=0, total=0)
    videos = [read_video(d / f"{cam}.mp4") for cam in CAMERAS]
    total = min(len(v) for v in videos)
    if total == 0:
        raise ValueError("no camera frames")
    squares = [[nu.square(v[i]) for v in videos] for i in range(total)]  # [frame][camera] 224x224
    del videos
    wait_until_idle(d, 0, total)
    model = fp.FeaturePCA(device, (SIZE, SIZE))
    upsampler, method, note = None, "bilinear", None
    if naf:
        try:
            upsampler, method = nu.NAFPCA(model), "naf"
        except Exception as error:  # no network for the weights, ...
            note = f"NAF unavailable, bilinear used: {type(error).__name__}: {error}"

    # 1. One basis for the whole episode, from frames spread over it.
    picks = np.unique(np.linspace(0, total - 1, min(total, BASIS_FRAMES)).round().astype(int))
    model.fit(torch.cat([model.features(squares[k]) for k in picks]))
    mean, basis, lo, hi = model.basis

    def bilinear(feats):
        low = torch.einsum("vrcd,dk->vkrc", feats - mean, basis)
        up = torch.nn.functional.interpolate(low, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        rgb = (up - lo.view(1, 3, 1, 1)) / (hi - lo).clamp_min(1e-6).view(1, 3, 1, 1)
        return rgb.clamp(0, 1).permute(0, 2, 3, 1).numpy()

    # 2. Every frame; videos are encoded on a thread while the next maps are computed.
    out_dir = d / "pca"
    out_dir.mkdir(exist_ok=True)
    writers = {cam: VideoWriter(out_dir / f"{cam}.tmp.mp4", fps) for cam in CAMERAS}
    behind: queue.Queue = queue.Queue(maxsize=16)

    def encoder():
        while (maps := behind.get()) is not None:
            for v, cam in enumerate(CAMERAS):
                writers[cam].write((maps[v] * 255).astype(np.uint8))

    thread = threading.Thread(target=encoder, daemon=True)
    thread.start()
    embeddings = np.zeros((total, len(CAMERAS), model.model.num_features), np.float16)
    last_images, last_maps = [None] * len(CAMERAS), [None] * len(CAMERAS)
    window: list = []
    reused, compute_s = 0, 0.0

    def emit(k):  # centred average over the frames on both sides that exist
        lo_k, hi_k = max(0, k - smooth), min(len(window), k + smooth + 1)
        behind.put(np.mean(window[lo_k:hi_k], axis=0))

    for n, images in enumerate(squares):
        if n % 5 == 0:
            wait_until_idle(d, n, total)
        tick = time.perf_counter()
        changed = [
            v for v in range(len(CAMERAS))
            if last_images[v] is None or np.abs(images[v].astype(np.int16) - last_images[v]).mean() >= UNCHANGED
        ]
        reused += len(CAMERAS) - len(changed)
        maps = [None] * len(CAMERAS)
        if changed:
            subset = [images[v] for v in changed]
            feats = model.features(subset)
            new = upsampler.maps(subset, feats) if upsampler is not None else bilinear(feats)
            pooled = feats.mean(dim=(1, 2)).numpy()
            for j, v in enumerate(changed):
                maps[v], last_maps[v], last_images[v] = new[j], new[j], images[v].astype(np.int16)
                embeddings[n, v] = pooled[j]
        for v in range(len(CAMERAS)):
            if maps[v] is None:
                maps[v], embeddings[n, v] = last_maps[v], embeddings[n - 1, v]
        compute_s += time.perf_counter() - tick
        window.append(np.stack(maps).astype(np.float16))
        if len(window) > smooth:
            emit(len(window) - 1 - smooth)
        if len(window) > 2 * smooth + 1:
            window.pop(0)
        if (n + 1) % 10 == 0 or n + 1 == total:
            status(d, state="running", done=n + 1, total=total, method=method, device=str(model.device))
    for k in range(max(0, len(window) - smooth), len(window)):
        emit(k)
    behind.put(None)
    thread.join()
    for cam, writer in writers.items():
        writer.close()
        (out_dir / f"{cam}.tmp.mp4").replace(out_dir / f"{cam}.mp4")
    np.save(out_dir / "embeddings.npy", embeddings)
    info = {
        "method": method,
        "method_text": METHOD[method],
        "frames": total,
        "fps": fps,
        "size": [SIZE, SIZE],
        "grid": [16, 16],
        "smooth_frames_each_side": smooth,
        "basis_frames": int(len(picks)),
        "reused_views": reused,
        "ms_per_frame": round(1000 * compute_s / total),
        "seconds": round(time.time() - started, 1),
        "device": str(model.device),
        "naf_speedups": getattr(getattr(upsampler, "naf", None), "speedups", None),
        **({"note": note} if note else {}),
    }
    write_json(out_dir / "info.json", info)
    final = {"state": "ready", "done": total, "total": total, "method": method, "seconds": info["seconds"]}
    status(d, **final)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", nargs="*", help="episode ids")
    parser.add_argument("--all", action="store_true", help="every episode without finished PCA")
    parser.add_argument("--watch", action="store_true", help="with --all: keep looking for new episodes")
    parser.add_argument("--force", action="store_true", help="redo episodes that already have PCA")
    parser.add_argument("--no-naf", action="store_true", help="bilinear upsampling instead of NAF")
    parser.add_argument("--smooth", type=int, default=1, help="frames on each side to average over (0: none)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if not args.episodes and not args.all:
        parser.error("give episode ids, or --all")
    failed = 0
    while True:
        folders = [EPISODES_DIR / e for e in args.episodes] if args.episodes else pending(args.force)
        for d in folders:
            if not (d / "top.mp4").exists():
                print(f"{d.name}: no camera video, skipped", flush=True)
                continue
            if not claim(d):
                print(f"{d.name}: another pca.py is working on it, skipped", flush=True)
                continue
            try:
                info = process(d, args.device, naf=not args.no_naf, smooth=args.smooth)
                print(f"{d.name}: {info['method']} PCA, {info['frames']} frames in {info['seconds']}s "
                      f"({info['ms_per_frame']} ms/frame, {info['reused_views']} unchanged views reused)", flush=True)
            except Exception as error:
                failed += 1
                status(d, state="error", error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
            finally:
                release(d)
        if not (args.all and args.watch):
            return 1 if failed else 0
        args.force = False
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
