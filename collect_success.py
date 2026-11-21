"""Collect independent Dream-1 attempts and export judged successes per teammate."""

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import judge
import simulator_settings as sim_settings

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
BATCHES = HERE / "batches"
OBJECTS = ("apple", "orange", "banana", "bottle")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def rules_snapshot():
    path = HERE / "collection_rules.json"
    return judge.normalize_rules(json.loads(path.read_text(encoding="utf-8")))


def validate_training_trace(trajectory, folder):
    times = trajectory.get("t", [])
    actions = trajectory.get("actions", {})
    frames = json.loads((folder / "frames.json").read_text(encoding="utf-8"))
    if not times or any(len(actions.get(arm, [])) != len(times) for arm in ("left", "right")):
        raise ValueError("action trace is missing or does not align with time")
    if not frames or any(not (folder / "frames" / item["file"]).is_file() for item in frames):
        raise ValueError("timestamped camera frames are missing")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", type=int, required=True, help="Unique team member number, 1 to 6")
    parser.add_argument("--object", choices=OBJECTS, required=True)
    parser.add_argument("--attempts", type=int, default=30, help="Total attempts; rerun resumes unfinished attempts")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--budget-minutes", type=float, default=120)
    parser.add_argument("--robot", choices=("yam", "openyam"), default="yam")
    parser.add_argument("--room", choices=tuple(sim_settings.ROOMS), default="yam_tabletop")
    parser.add_argument("--lighting", choices=tuple(sim_settings.LIGHTING), default="standard")
    args = parser.parse_args(argv)
    if not 1 <= args.member <= 6 or not 1 <= args.attempts <= 999999 or args.seconds <= 0 or args.budget_minutes <= 0:
        parser.error("Member must be 1–6; attempts, seconds, and budget must be positive")
    credentials = Path.home() / ".dreamlabs" / "credentials"
    if not os.environ.get("DREAM_API_KEY") and not credentials.is_file():
        parser.error("Set DREAM_API_KEY or add it to ~/.dreamlabs/credentials before collecting")

    task = f"pick {args.object} into the bowl"
    rules = rules_snapshot()
    settings = {"robot": args.robot, "room": args.room, "lighting": args.lighting, "layout": "varied"}
    collection = BATCHES / f"member-{args.member}-{args.object}"
    manifest_path = collection / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["task"] != task or manifest["rules"] != rules or manifest["settings"] != settings or manifest["seconds"] != args.seconds:
            parser.error(f"{manifest_path} has different settings; use the original settings or archive that collection first")
    else:
        manifest = {"schema_version": 1, "member": args.member, "task": task, "rules": rules,
                    "settings": settings, "seconds": args.seconds, "attempts": []}
        save_json(manifest_path, manifest)

    completed = {entry["attempt"] for entry in manifest["attempts"]}
    started = time.monotonic()
    for attempt in range(1, args.attempts + 1):
        if attempt in completed:
            continue
        if time.monotonic() - started >= args.budget_minutes * 60:
            print("Time budget reached; rerun the same command to resume.", flush=True)
            break
        seed = args.member * 1_000_000 + attempt
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        folder = RUNS / run_id
        command = [sys.executable, str(HERE / "sim.py"), task, "--seconds", str(args.seconds),
                   "--robot", args.robot, "--room", args.room, "--lighting", args.lighting,
                   "--layout", "varied", "--seed", str(seed)]
        print(f"[{attempt}/{args.attempts}] {task} · seed {seed}", flush=True)
        try:
            result = subprocess.run(command, cwd=HERE, env={**os.environ, "RUN_DIR": str(folder)},
                                    capture_output=True, text=True, timeout=args.seconds + 180)
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            trajectory = json.loads((folder / "trajectory.json").read_text(encoding="utf-8"))
            verdict = judge.evaluate(meta, trajectory, rules)
            if verdict["label"] == "success":
                validate_training_trace(trajectory, folder)
        except (subprocess.TimeoutExpired, OSError, ValueError, KeyError, TypeError) as error:
            print(f"Invalid or incomplete episode: {error}. Collection stopped; inspect {folder}.", flush=True)
            break

        label = {"label": verdict["label"], "reason": verdict["reason"],
                 "success_time_s": verdict.get("success_time_s"), "rules": rules,
                 "saved_at": datetime.now().isoformat(timespec="seconds")}
        save_json(folder / "label.json", label)
        if result.returncode or verdict["label"] == "invalid":
            print(f"Invalid episode ({verdict['reason']}); collection stopped. Inspect {folder}.", flush=True)
            break
        exported = None
        if verdict["label"] == "success":
            destination = collection / "success" / run_id
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(folder, destination)
            exported = str(destination.relative_to(HERE))
        manifest["attempts"].append({"attempt": attempt, "seed": seed, "run_id": run_id,
                                     "label": verdict["label"], "reason": verdict["reason"],
                                     "run_path": str(folder.relative_to(HERE)), "export_path": exported})
        save_json(manifest_path, manifest)
        successes = sum(item["label"] == "success" for item in manifest["attempts"])
        print(f"  {verdict['label']}: {verdict['reason']} · {successes}/{len(manifest['attempts'])} successes", flush=True)

    successes = sum(item["label"] == "success" for item in manifest["attempts"])
    print(f"Saved {successes} successes from {len(manifest['attempts'])} valid attempts in {collection}")
    print("Share the collection folder, including manifest.json and success/. Do not commit the large run data to Git.")


if __name__ == "__main__":
    main()
