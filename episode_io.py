"""Small helpers shared by the post-processing scripts: JSON files, camera videos, and waiting for recordings.

No simulator import, so any script can use it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EPISODES_DIR = HERE / "episodes"  # same as recorder.EPISODES_DIR


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    """Write JSON atomically (a reader never sees half a file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_video(path: Path) -> list[np.ndarray]:
    """Every frame of a video as HxWx3 uint8 RGB."""
    import av

    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def recording_now() -> list[int]:
    """Pids of episodes being recorded (episodes/.recording-<pid>, written by recorder.py); markers of finished or
    crashed processes are removed."""
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


def wait_until_idle(poll_s: float = 2.0) -> None:
    """Block while an episode is being recorded: GPU work alongside the simulator's cameras slows them enough to let
    the Dream-1 session expire."""
    while recording_now():
        time.sleep(poll_s)
