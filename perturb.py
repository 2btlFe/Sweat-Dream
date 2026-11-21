"""Failure-recovery data: replay a clean episode to a chosen moment, disturb it, and let Dream-1 try to recover.

A clean episode (one the simulator judged a success) is replayed action by action from its seed's start; MuJoCo is
deterministic, so the replay reaches exactly the recorded state (0 mm error on every object, checked). At the
trigger the disturbance is applied, then Dream-1 takes over from that state, recorded and judged like any episode.
The replayed prefix costs nothing; only the recovery calls Dream-1.

Each run is a new episode (source "perturb:<kind>") with perturbation.json beside outcome.json:
    kind, params, trigger, applied_at_s, window_s, clean_episode, object, target,
    recovered (the judge's success after the disturbance), recovery_time_s (success time - applied_at)

Disturbances (SPEC is kind[:key=value,...]@trigger, e.g. drop@grasp+0.5, push:vy=0.4@lift, freeze:seconds=2@3.0):
    drop        open the gripper holding the object for `seconds` (0.6): it falls
    push        velocity kick to an object: vx, vy, vz m/s (0, 0.4, 0); object (the task's)
    shift       move an object instantly by dx, dy m (0.05, 0); object (the task's); before a grasp
    nudge_arm   turn one joint of an arm by `rad` (0.3) at once: side (the holding or nearest arm), joint 1-6 (2)
    move_target move the bowl or plate by dx, dy m (0.08, 0): the target changes mid-task
    freeze      the arms hold still for `seconds` (1.5): a stall or dropped connection
    noise       Gaussian noise of `std` rad (0.05) on every joint command for `seconds` (2); seed
    occlude     a camera sends black frames for `seconds` (2): camera top | left | right (left)
Triggers: a time in simulated seconds (3.0); grasp (the clean run first holds the object); lift (held and 3 cm up);
release (let go after the lift); each optionally +seconds. A trigger the clean run never reaches is skipped.

Usage:
    python perturb.py <clean_episode_id> --perturb drop@grasp+0.5 [--perturb ...] [--seconds 60] [--no-dream]
    python perturb.py --clean --perturb drop@grasp+0.5 --perturb push:vy=0.4@lift --repeats 2   # every clean success
    python perturb.py --clean --perturb push@lift --repeats 5 --delay-jitter 1.0 --scale-jitter 0.3      # variations
    python perturb.py --list
--no-dream keeps the clean run's last command after the disturbance instead of calling Dream-1 (no cost): a
no-recovery baseline and a check of the disturbance itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EPISODES_DIR = HERE / "episodes"
BATCHES_DIR = HERE / "batches"

DEFAULTS = {
    "drop": {"seconds": 0.6},
    "push": {"vx": 0.0, "vy": 0.4, "vz": 0.0},
    "shift": {"dx": 0.05, "dy": 0.0},
    "nudge_arm": {"joint": 2, "rad": 0.3},
    "move_target": {"dx": 0.08, "dy": 0.0},
    "freeze": {"seconds": 1.5},
    "noise": {"std": 0.05, "seconds": 2.0, "seed": 0},
    "occlude": {"camera": "left", "seconds": 2.0},
}
INSTANT = {"push", "shift", "nudge_arm", "move_target"}
FINGER_OPEN = 0.0475
LIFT_M = 0.03


def parse_spec(spec: str) -> dict:
    """'kind[:k=v,...]@trigger[+delay]' -> {kind, params, trigger, delay}."""
    m = re.fullmatch(r"([a-z_]+)(?::([^@]*))?@([a-z]+|[0-9.]+)(?:\+([0-9.]+))?", spec.strip())
    if not m or m.group(1) not in DEFAULTS:
        raise ValueError(f"bad disturbance {spec!r}: kind[:key=value,...]@trigger[+delay], kinds {sorted(DEFAULTS)}")
    kind = m.group(1)
    params = dict(DEFAULTS[kind])
    for pair in filter(None, (m.group(2) or "").split(",")):
        key, _, value = pair.partition("=")
        if key not in ("object", "side", "camera", "target"):
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"{spec!r}: {key} must be a number") from None
        params[key] = value
    trigger = m.group(3)
    if not re.fullmatch(r"[0-9.]+", trigger) and trigger not in ("grasp", "lift", "release"):
        raise ValueError(f"{spec!r}: trigger must be seconds, grasp, lift or release")
    return {"spec": spec, "kind": kind, "params": params, "trigger": trigger, "delay": float(m.group(4) or 0)}


def trigger_step(trigger: str, delay: float, t, objects, held, j) -> int | None:
    """The step index of a trigger in the clean trace (None if never reached)."""
    k = None
    if re.fullmatch(r"[0-9.]+", trigger):
        k = int(np.searchsorted(t, t[0] + float(trigger)))
    elif trigger == "grasp":
        hits = np.flatnonzero(held[:, j])
        k = int(hits[0]) if len(hits) else None
    else:
        lifted = np.flatnonzero(held[:, j] & (objects[:, j, 2] > objects[0, j, 2] + LIFT_M))
        if len(lifted):
            k = int(lifted[0])
            if trigger == "release":
                after = np.flatnonzero(~held[k:, j])
                k = k + int(after[0]) if len(after) else None
    if k is None:
        return None
    k = int(np.searchsorted(t, t[k] + delay))
    return k if k < len(t) else None


# ------------------------------------------------------------------- one run


def vary(spec: dict, clean_id: str, repeat: int, delay_jitter: float, scale_jitter: float) -> dict:
    """A variation of a disturbance, drawn from (clean episode, spec, repeat): a later trigger by up to delay_jitter s
    and every magnitude scaled by up to +-scale_jitter. Repeat 0 with no jitter is the spec as written."""
    import zlib

    if not delay_jitter and not scale_jitter:
        return {**spec, "variation": None}
    rng = np.random.default_rng(zlib.crc32(f"{clean_id}|{spec['spec']}|{repeat}".encode()))
    params = dict(spec["params"])
    scaled = {}
    for key, value in spec["params"].items():
        if isinstance(value, float) and key not in ("seed", "joint"):
            params[key] = round(value * (1 + rng.uniform(-scale_jitter, scale_jitter)), 4)
            scaled[key] = params[key]
    extra = round(float(rng.uniform(0, delay_jitter)), 3)
    return {**spec, "params": params, "delay": spec["delay"] + extra,
            "variation": {"extra_delay_s": extra, "scaled": scaled, "delay_jitter": delay_jitter, "scale_jitter": scale_jitter}}


def run_one(clean_id: str, spec: dict, seconds: float, use_dream: bool, hold_seconds: float, repeat: int, batch_id=None):
    """One disturbed run; use_dream False holds the arms after the disturbance (no API calls)."""
    import dreamfix
    import dream
    import dream_control
    import mujoco

    import judge
    from recorder import FREE_JOINTS, EpisodeRecorder, RecordingEnv

    clean = EPISODES_DIR / clean_id
    meta = json.loads((clean / "meta.json").read_text())
    outcome = json.loads((clean / "outcome.json").read_text())
    z = np.load(clean / "steps.npz")
    t, actions, objects, held = z["t"], z["action"], z["objects"], z["held"].astype(bool)
    task = meta["task"]
    parsed = judge.parse_task(task)
    if parsed is None:
        raise ValueError(f"the judge does not understand {task!r}")
    obj, target = parsed
    obj = spec["params"].get("object", obj)
    j = judge.BODIES.index(obj)
    k = trigger_step(spec["trigger"], spec["delay"], t, objects, held, j)
    if k is None:
        print(f"SKIPPED: the clean run never reaches {spec['trigger']} (+{spec['delay']}s)", flush=True)
        return None
    window = spec["params"].get("seconds", 0.0) if spec["kind"] not in INSTANT else 0.0

    class Disturbance:
        """Changes the commands (drop, freeze, noise) and camera frames (occlude) while its window is open."""

        def __init__(self):
            self.start, self.end, self.side, self.frozen = None, None, None, None
            self.rng = np.random.default_rng(int(spec["params"].get("seed", 0)) + repeat)

        def active(self, now):
            return self.start is not None and self.start <= now < self.end

        def commands(self, env, left, right):
            if not self.active(env.data.time) or spec["kind"] not in ("drop", "freeze", "noise"):
                return left, right
            left, right = np.array(left, float), np.array(right, float)
            if spec["kind"] == "drop":
                (left if self.side == "left" else right)[6] = 1.0
            elif spec["kind"] == "freeze":
                if self.frozen is None:  # the last command before the stall stays in force, grip included
                    self.frozen = tuple(np.array(c, float) for c in env.last_command)
                left, right = self.frozen[0].copy(), self.frozen[1].copy()
            else:
                std = float(spec["params"]["std"])
                left[:6] += self.rng.normal(0, std, 6)
                right[:6] += self.rng.normal(0, std, 6)
            return left, right

    disturbance = Disturbance()

    class PerturbEnv(RecordingEnv):
        last_command = None  # (left, right) as last commanded, before any disturbance

        def step(self, left, right):
            if not disturbance.active(self.data.time) or self.last_command is None:
                self.last_command = (np.array(left, float), np.array(right, float))
            left, right = disturbance.commands(self, left, right)
            return super().step(left, right)

    env = PerturbEnv()
    seed = meta.get("seed")
    offsets = env.jitter(seed) if seed is not None else None
    model_id = dream_control.BETA_MODEL if use_dream else None
    extra = {"seed": seed, "jitter": offsets, "batch_id": batch_id, "max_seconds": seconds,
             "perturbation": {"spec": spec["spec"], "kind": spec["kind"], "clean_episode": clean_id, "repeat": repeat}}
    rec = EpisodeRecorder(env, task, model=model_id, source=f"perturb:{spec['kind']}", stop_on_success=True, extra=extra)
    env.recorder = rec

    # 1. Replay the clean run up to the trigger, keeping a frame every 4 steps as the recorder does.
    for i in range(k):
        a = actions[i]
        env.step(a[:7], a[7:])
        if (i + 1) % 4 == 0:
            rec.on_frame(env.images())
    drift = float(np.abs(np.array([env.data.body(b).xpos for b in judge.BODIES]) - objects[k - 1]).max()) if k else 0.0

    # 2. The disturbance.
    data, model = env.data, env.model
    now = float(data.time)
    applied = {}
    holder = None
    tcp = {side: data.site(f"{side}_tcp").xpos.copy() for side in ("left", "right")}
    nearest = min(tcp, key=lambda side: np.linalg.norm(tcp[side] - data.body(obj).xpos))
    holder = spec["params"].get("side", nearest)
    if spec["kind"] == "push":
        joint = data.joint(FREE_JOINTS[obj])
        kick = np.array([spec["params"]["vx"], spec["params"]["vy"], spec["params"]["vz"]], float)
        joint.qvel[0:3] += kick
        applied = {"object": obj, "velocity_mps": kick.tolist()}
    elif spec["kind"] == "shift":
        joint = data.joint(FREE_JOINTS[obj])
        joint.qpos[0] += spec["params"]["dx"]
        joint.qpos[1] += spec["params"]["dy"]
        joint.qvel[:] = 0
        applied = {"object": obj, "moved_m": [spec["params"]["dx"], spec["params"]["dy"]], "was_held": bool(held[k - 1, j])}
    elif spec["kind"] == "nudge_arm":
        name = f"{holder}_joint{int(spec['params']['joint'])}"
        data.joint(name).qpos[0] += spec["params"]["rad"]
        applied = {"joint": name, "rad": spec["params"]["rad"]}
    elif spec["kind"] == "move_target":
        tgt = spec["params"].get("target", target)
        body = model.body(tgt)
        body.pos[0] += spec["params"]["dx"]
        body.pos[1] += spec["params"]["dy"]
        applied = {"target": tgt, "moved_m": [spec["params"]["dx"], spec["params"]["dy"]]}
    else:
        disturbance.start, disturbance.end, disturbance.side = now, now + float(window), holder
        applied = {"window_s": [round(now, 3), round(now + float(window), 3)], "side": holder,
                   **({"camera": spec["params"]["camera"]} if spec["kind"] == "occlude" else {})}
    mujoco.mj_forward(model, data)
    rec.targets = {name: data.body(name).xpos.copy() for name in judge.TARGETS}  # a moved target is the new goal

    # 3. Dream-1 takes over (or the arms hold, with --no-dream).
    error = None
    if use_dream:
        fixed_capture = dream_control.capture

        def capture_and_record(env_, windows, keep=True):
            if spec["kind"] == "occlude" and disturbance.active(env_.data.time):
                cam = spec["params"]["camera"]
                original = env_.images

                def blacked():
                    images = original()
                    images[cam] = np.zeros_like(images[cam])
                    return images

                env_.images = blacked
                try:
                    images = fixed_capture(env_, windows, keep)
                finally:
                    del env_.images
            else:
                images = fixed_capture(env_, windows, keep)
            if keep:
                rec.on_frame(images)
            return images

        timed_get_action = dream.DreamClient.get_action

        def get_action_timed(self, *args, **kwargs):
            started = time.monotonic()
            reply = timed_get_action(self, *args, **kwargs)
            rec.rtts.append(time.monotonic() - started)
            return reply

        dream_control.capture, dream.DreamClient.get_action = capture_and_record, get_action_timed
        argv = ["--seconds", str(seconds), "--task", task, "--model", model_id, "--headless", "--quiet"]
        try:
            dream_control.main(argv, shared_env=env, stop_event=rec.stop_event)
        except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001
            error = e
        finally:
            dream_control.capture, dream.DreamClient.get_action = fixed_capture, timed_get_action
    else:
        steps = int(hold_seconds / 0.05)
        hold = tuple(np.array(c, float) for c in env.last_command)  # the clean run's last command, grip included
        for i in range(steps):
            env.step(hold[0].copy(), hold[1].copy())
            if (i + 1) % 4 == 0:
                images = env.images()
                if spec["kind"] == "occlude" and disturbance.active(env.data.time):
                    images[spec["params"]["camera"]] = np.zeros_like(images[spec["params"]["camera"]])
                rec.on_frame(images)
            if rec.stop_event.is_set():
                break
    out = rec.close(error)
    env.close()
    recovered = out["label"] == "success"
    record = {
        "spec": spec["spec"], "kind": spec["kind"], "params": spec["params"], "trigger": spec["trigger"],
        "delay_s": spec["delay"], "trigger_step": k, "applied_at_s": round(now, 3), "window_s": window,
        "applied": applied, "clean_episode": clean_id, "clean_label": outcome.get("label"), "object": obj,
        "target": spec["params"].get("target", target), "policy": "dream-1" if use_dream else "hold (no recovery)",
        "replay_drift_m": round(drift, 6), "recovered": recovered,
        "recovery_time_s": round(out["success_time_s"] - now, 2) if recovered and out.get("success_time_s") is not None else None,
        "outcome_label": out["label"], "outcome_reason": out["reason"], "repeat": repeat, "variation": spec.get("variation"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (rec.dir / "perturbation.json").write_text(json.dumps(record, indent=2))
    print(f"\nEpisode {rec.dir.name}: {spec['spec']} at {now:.2f}s on {clean_id} -> "
          f"{'RECOVERED' if recovered else 'NOT RECOVERED'} ({out['reason']})"
          + (f" in {record['recovery_time_s']}s" if record["recovery_time_s"] is not None else "")
          + f" · replay drift {drift * 1000:.3f} mm", flush=True)
    print(f"EPISODE_DIR={rec.dir}", flush=True)
    return rec.dir


# ------------------------------------------------------------------- many runs


def clean_episodes() -> list[str]:
    """Episodes the simulator judged a success, that are not themselves disturbed runs."""
    out = []
    for d in sorted(EPISODES_DIR.iterdir()) if EPISODES_DIR.is_dir() else []:
        try:
            outcome = json.loads((d / "outcome.json").read_text())
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            continue
        if outcome.get("label") == "success" and not str(meta.get("source", "")).startswith("perturb") and (d / "steps.npz").exists():
            out.append(d.name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", nargs="*", help="clean episode ids")
    parser.add_argument("--clean", action="store_true", help="every clean success")
    parser.add_argument("--perturb", action="append", default=[], help="SPEC, repeatable")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=60, help="Dream-1's time to recover (wall clock)")
    parser.add_argument("--no-dream", action="store_true", help="hold still after the disturbance (no API calls)")
    parser.add_argument("--hold-seconds", type=float, default=6.0)
    parser.add_argument("--delay-jitter", type=float, default=0.0, help="each repeat triggers up to this many s later")
    parser.add_argument("--scale-jitter", type=float, default=0.0, help="each repeat scales magnitudes by up to +-this (0.3 = 30%%)")
    parser.add_argument("--list", action="store_true", help="list disturbances and their defaults")
    parser.add_argument("--job", help=argparse.SUPPRESS)  # one run, in a child process: clean_id|spec|repeat
    parser.add_argument("--batch-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.list:
        for kind, params in DEFAULTS.items():
            print(f"{kind:<12} {params}")
        return 0
    if args.job:
        clean_id, spec, repeat = args.job.split("|")
        varied = vary(parse_spec(spec), clean_id, int(repeat), args.delay_jitter, args.scale_jitter)
        run_one(clean_id, varied, args.seconds, not args.no_dream, args.hold_seconds, int(repeat), args.batch_id)
        return 0
    specs = [parse_spec(s) for s in args.perturb]
    if not specs:
        parser.error("give at least one --perturb SPEC (see --list)")
    sources = args.episodes or (clean_episodes() if args.clean else [])
    if not sources:
        parser.error("give clean episode ids, or --clean")
    batch_id = "p" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    jobs = [(e, s["spec"], r) for r in range(args.repeats) for e in sources for s in specs]
    BATCHES_DIR.mkdir(exist_ok=True)
    progress = {"batch_id": batch_id, "kind": "perturb", "args": vars(args), "jobs": len(jobs), "done": {}, "skipped": [], "failed": {}}
    print(f"Perturbation batch {batch_id}: {len(jobs)} runs ({len(sources)} clean episodes x {len(specs)} disturbances "
          f"x {args.repeats}), {'Dream-1 recovers' if not args.no_dream else 'no recovery (hold)'}", flush=True)
    for n, (clean_id, spec, repeat) in enumerate(jobs, 1):
        print(f"[{n}/{len(jobs)}] {spec} on {clean_id} (repeat {repeat}) ...", flush=True)
        cmd = [sys.executable, str(HERE / "perturb.py"), "--job", f"{clean_id}|{spec}|{repeat}", "--seconds", str(args.seconds),
               "--hold-seconds", str(args.hold_seconds), "--batch-id", batch_id, "--delay-jitter", str(args.delay_jitter),
               "--scale-jitter", str(args.scale_jitter)] + (["--no-dream"] if args.no_dream else [])
        result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1", "PYTHONWARNINGS": "ignore"},
                                timeout=args.seconds + 300)
        out = result.stdout
        found = re.search(r"EPISODE_DIR=(.+)", out)
        line = next((l for l in out.splitlines() if l.startswith("Episode ")), "")
        key = f"{clean_id}|{spec}|{repeat}"
        if "SKIPPED" in out:
            progress["skipped"].append(key)
            print("  skipped: trigger not reached in the clean run", flush=True)
        elif found:
            progress["done"][key] = Path(found.group(1)).name
            print("  " + line.replace("Episode ", ""), flush=True)
        else:
            progress["failed"][key] = (result.stderr or out)[-800:]
            print(f"  failed: {(result.stderr or out).strip().splitlines()[-1:]} ", flush=True)
        (BATCHES_DIR / f"{batch_id}.json").write_text(json.dumps(progress, indent=2))
    print(f"\nDone: {len(progress['done'])} runs, {len(progress['skipped'])} skipped, {len(progress['failed'])} failed "
          f"-> batches/{batch_id}.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
