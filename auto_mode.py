"""Auto mode: collect, auto-label, and multiply the data with variations, in one command.

    1. collect      Dream-1 episodes (batch.py) and/or scripted known-outcome episodes, if asked
    2. auto-label   every episode from the simulator's judgement (autolabel.py): label, confidence, review queue
    3. disturb      every high-confidence clean success, once per --perturb SPEC x --perturb-repeats, with
                    --delay-jitter / --scale-jitter variations; Dream-1 tries to recover (perturb.py)
    4. auto-label   the failure-recovery runs: recovered or not, and how fast
    5. looks        --looks appearance variants of every high-confidence episode (rerender.py); each variant
                    carries its episode's label
    6. PCA          feature PCA videos (pca.py), with --pca
    7. report       batches/auto-<time>.json and a table: labels, review queue, recovery rate per disturbance

Low-confidence episodes are neither disturbed nor multiplied: they wait in the review queue of the labeling UI.
Every step runs in its own process and is skipped when it has nothing to do.

Usage:
    python auto_mode.py --perturb drop@grasp+0.5 --perturb push:vy=0.4@lift --perturb-repeats 2 --looks 2
    python auto_mode.py --dream-repeats 3 --scripted --perturb freeze:seconds=2@lift --looks 4 --pca
    python auto_mode.py --perturb drop@grasp+0.5 --no-dream-recovery            # disturbances only, no API cost
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPISODES_DIR = HERE / "episodes"
BATCHES_DIR = HERE / "batches"


def run(step: str, args: list[str], log: list) -> int:
    started = time.time()
    print(f"\n== {step}: {' '.join(args)}", flush=True)
    code = subprocess.call([sys.executable, *args], cwd=str(HERE))
    log.append({"step": step, "command": args, "exit": code, "seconds": round(time.time() - started, 1)})
    return code


def confident(*, clean_success: bool = False, exclude_perturbed: bool = True) -> list[str]:
    """Episodes whose auto label is high-confidence (optionally: clean successes only)."""
    out = []
    for d in sorted(EPISODES_DIR.iterdir()) if EPISODES_DIR.is_dir() else []:
        try:
            auto = json.loads((d / "auto_label.json").read_text())
        except (OSError, ValueError):
            continue
        if auto.get("confidence") != "high":
            continue
        if clean_success and "clean_success" not in auto.get("tags", []):
            continue
        if exclude_perturbed and "recovery_run" in auto.get("tags", []):
            continue
        out.append(d.name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dream-repeats", type=int, default=0, help="collect Dream-1 episodes first: seeds per task")
    parser.add_argument("--tasks", nargs="+", help="tasks for the Dream-1 collection (batch.py's default otherwise)")
    parser.add_argument("--budget-minutes", type=float, default=30)
    parser.add_argument("--scripted", action="store_true", help="collect the scripted known-outcome episodes first")
    parser.add_argument("--scripted-repeats", type=int, default=1)
    parser.add_argument("--perturb", action="append", default=[], help="disturbance SPEC for every clean success (perturb.py)")
    parser.add_argument("--perturb-repeats", type=int, default=1)
    parser.add_argument("--delay-jitter", type=float, default=0.0)
    parser.add_argument("--scale-jitter", type=float, default=0.0)
    parser.add_argument("--max-clean", type=int, default=0, help="disturb at most this many clean successes (0: all)")
    parser.add_argument("--recovery-seconds", type=float, default=60)
    parser.add_argument("--no-dream-recovery", action="store_true", help="after a disturbance hold still (no API calls)")
    parser.add_argument("--looks", type=int, default=0, help="appearance variants per high-confidence episode")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--pca", action="store_true", help="also make the feature PCA videos")
    args = parser.parse_args()
    run_id = "auto-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log: list = []
    before = {d.name for d in EPISODES_DIR.iterdir()} if EPISODES_DIR.is_dir() else set()

    if args.dream_repeats or args.scripted:
        cmd = ["batch.py", "--repeats", str(max(args.dream_repeats, args.scripted_repeats if args.scripted else 0)),
               "--budget-minutes", str(args.budget_minutes)]
        cmd += ["--tasks", *args.tasks] if args.tasks else []
        cmd += ["--scripted"] if args.scripted else []
        cmd += ["--no-dream"] if not args.dream_repeats else []
        run("collect", cmd, log)
    run("auto-label", ["autolabel.py"], log)

    if args.perturb:
        sources = confident(clean_success=True)
        if args.max_clean:
            sources = sources[-args.max_clean:]
        if sources:
            cmd = ["perturb.py", *sources, "--repeats", str(args.perturb_repeats), "--seconds", str(args.recovery_seconds),
                   "--delay-jitter", str(args.delay_jitter), "--scale-jitter", str(args.scale_jitter)]
            for spec in args.perturb:
                cmd += ["--perturb", spec]
            cmd += ["--no-dream"] if args.no_dream_recovery else []
            run("disturb", cmd, log)
            run("auto-label", ["autolabel.py"], log)
        else:
            print("\n== disturb: no high-confidence clean success to disturb yet", flush=True)

    if args.looks:
        todo = [e for e in confident(exclude_perturbed=False) if not (EPISODES_DIR / e / "aug" / "index.json").exists()]
        if todo:
            run("looks", ["rerender.py", *todo, "--variants", str(args.looks), "--strength", str(args.strength)], log)
            run("auto-label", ["autolabel.py"], log)
    if args.pca:
        run("pca", ["pca.py", "--all"], log)

    summary = json.loads((EPISODES_DIR / "auto_label_summary.json").read_text())
    after = {d.name for d in EPISODES_DIR.iterdir()}
    report = {"run": run_id, "args": vars(args), "steps": log, "new_episodes": sorted(after - before - {"auto_label_summary.json", "index.csv"}),
              "summary": summary, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    BATCHES_DIR.mkdir(exist_ok=True)
    (BATCHES_DIR / f"{run_id}.json").write_text(json.dumps(report, indent=2))
    print(f"\n== report: batches/{run_id}.json · {len(report['new_episodes'])} new episodes", flush=True)
    sys.path.insert(0, str(HERE))
    import autolabel

    autolabel.print_summary(summary)
    return 0 if all(s["exit"] == 0 for s in log) else 1


if __name__ == "__main__":
    sys.exit(main())
