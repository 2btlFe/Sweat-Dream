"""Fixes that let the Dream YAM Sim run reliably on this Windows laptop.

Measured 2026-11-21 (Intel Iris Xe, Windows 11, dreamlabs 1.0.42 / dreamlabs-client 1.1.7):
- Rendering the three cameras takes ~420 ms per 50 ms control step, so one 48-action chunk took ~21 s. The gateway
  drops a connection idle for ~5-10 s (SSL EOF) and expires the session at ~10 s idle, and the client does not reopen
  the connection on its own, so the second request always failed.
- Fix 1: render the cameras only on the steps whose frames the model sees (1 in 4); other steps reuse the last images.
- Fix 2: when get_action() fails with DreamProtocolError (dropped connection), ask again, up to twice.

Import this module before running anything from dream_control; it puts the sim's folder on sys.path.
"""

import sys
import time
from pathlib import Path

def _sim_dir() -> Path:
    """The installed Dream YAM Sim (Windows venvs keep packages in Lib/site-packages, macOS and Linux in
    lib/pythonX.Y/site-packages)."""
    import importlib.util

    spec = importlib.util.find_spec("dream_examples")
    if spec is not None and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))) / "yam_mujoco"
    return Path(sys.prefix) / "Lib/site-packages/dream_examples/yam_mujoco"


SIM_DIR = _sim_dir()
sys.path.insert(0, str(SIM_DIR))

import dream  # noqa: E402
import dream_control  # noqa: E402

STATS = {"retries": 0}

_capture = dream_control.capture
_last_images = {}


def capture_kept_only(env, windows, keep=True):
    """Render only frames the model sees; other steps hand back the last rendered images (the viewer panel)."""
    if keep or not _last_images:
        _last_images.clear()
        _last_images.update(_capture(env, windows, keep))
    return dict(_last_images)


_get_action = dream.DreamClient.get_action


def get_action_retry(self, *args, **kwargs):
    """A connection the gateway dropped while idle is reopened by asking again, up to twice."""
    for attempt in range(3):
        try:
            return _get_action(self, *args, **kwargs)
        except dream.DreamProtocolError:
            if attempt == 2:
                raise
            STATS["retries"] += 1
            print("  Connection dropped; asking again.", flush=True)
            time.sleep(0.2)


def apply():
    """Install both fixes (idempotent)."""
    dream_control.capture = capture_kept_only
    dream.DreamClient.get_action = get_action_retry


apply()
