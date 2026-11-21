"""Collect and label independent Dream-1 episodes with the current UI rules.

Example: python batch.py --repeats 10 --seconds 60 --budget-minutes 60
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import judge
import simulator_settings as sim_settings

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TASKS = ("put apple into the bowl", "put orange into the bowl", "put banana on the plate")


def rules_snapshot():
    path = RUNS / "rules.json"
    if not path.exists():
        return dict(judge.DEFAULT_RULES)
    return judge.normalize_rules(json.loads(path.read_text(encoding="utf-8")))


def settings_snapshot():
    path = RUNS / "simulator_settings.json"
    return sim_settings.normalize(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else dict(sim_settings.DEFAULT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed for varied layout; overrides the UI seed")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--budget-minutes", type=float, default=60)
    args = parser.parse_args()
    if args.repeats < 1 or args.seconds <= 0 or args.budget_minutes <= 0:
        parser.error("Use positive repeats, seconds, and budget-minutes")
    if any(judge.parse_task(task) is None for task in args.tasks):
        parser.error("Tasks must name apple, orange, banana, or bottle and a bowl or plate")
    rules = rules_snapshot()
    settings = settings_snapshot()
    if not sim_settings.local_supported(settings):
        parser.error("Selected robot or room is Sandbox-only; choose YAM tabletop in the UI")
    base_seed = args.seed if args.seed is not None else (settings["seed"] if settings["seed"] is not None else 0)
    counts = Counter()
    total = args.repeats * len(args.tasks)
    started = time.monotonic()
    RUNS.mkdir(exist_ok=True)
    for n, (seed, task) in enumerate(((base_seed + i, task) for i in range(args.repeats)
                                      for task in args.tasks), 1):
        if time.monotonic() - started >= args.budget_minutes * 60:
            print("Wall-time budget reached; remaining episodes were not started.")
            break
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        folder = RUNS / run_id
        print(f"[{n}/{total}] {task} ({'seed ' + str(seed) if settings['layout'] == 'varied' else 'fixed layout'})", flush=True)
        command = [sys.executable, str(HERE / "sim.py"), task, "--seconds", str(args.seconds),
                   "--layout", settings["layout"], "--lighting", settings["lighting"]]
        if settings["layout"] == "varied":
            command += ["--seed", str(seed)]
        try:
            proc = subprocess.run(
                command, cwd=HERE, env={**os.environ, "RUN_DIR": str(folder)},
                capture_output=True, text=True, timeout=args.seconds + 180)
        except subprocess.TimeoutExpired:
            counts["invalid"] += 1
            print("Run timed out; stopping the batch.")
            break
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            trajectory = json.loads((folder / "trajectory.json").read_text(encoding="utf-8"))
            verdict = judge.evaluate(meta, trajectory, rules)
        except (OSError, ValueError, TypeError, KeyError) as error:
            counts["invalid"] += 1
            print(f"No complete result ({error}); stopping the batch.")
            break
        label = {"label": verdict["label"], "reason": verdict["reason"],
                 "success_time_s": verdict.get("success_time_s"), "rules": rules,
                 "saved_at": datetime.now().isoformat(timespec="seconds")}
        (folder / "label.json").write_text(json.dumps(label, indent=2), encoding="utf-8")
        counts[verdict["label"]] += 1
        print(f"  {verdict['label']}: {verdict['reason']} · {folder}", flush=True)
        if proc.returncode != 0 or verdict["label"] == "invalid":
            print("Invalid episode; stopping the batch so the configuration can be checked.")
            break
    valid = counts["success"] + counts["failure"]
    print(f"Success: {counts['success']}/{valid} valid episodes"
          + (f" ({counts['success'] / valid:.1%})" if valid else "")
          + f"; invalid: {counts['invalid']}")


if __name__ == "__main__":
    main()
