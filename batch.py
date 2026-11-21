"""Batch runner: many recorded, judged episodes unattended, with a cost budget and resume.

Each episode runs in its own Python process, so a MuJoCo or network crash ends that episode, not the batch.

Usage:
    python batch.py --repeats 3                                  # Dream-1 on the default tasks, 3 seeds each
    python batch.py --tasks "put apple into the bowl" --repeats 5 --budget-minutes 10
    python batch.py --scripted --repeats 2 --no-dream            # scripted episodes only (no API, no cost)
    python batch.py --resume <batch_id>                          # finish an interrupted batch
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import summary
EPISODES_DIR = Path(__file__).resolve().parent / "episodes"  # same as recorder.EPISODES_DIR (no sim import needed)

HERE = Path(__file__).resolve().parent
BATCHES_DIR = HERE / "batches"
DEFAULT_TASKS = ["put apple into the bowl", "put orange into the bowl", "put banana on the plate"]
SCRIPTED_VARIANTS = ["success", "orange_success", "outside", "drop", "wrong_object", "hold", "no_attempt"]


def plan(args):
    """The batch's jobs, in run order: each task (or variant) once per seed, seeds interleaved."""
    jobs = []
    for r in range(args.repeats):
        seed = args.seed + r
        if not args.no_dream:
            jobs += [{"key": f"dream|{t}|{seed}", "kind": "dream", "task": t, "seed": seed} for t in args.tasks]
        if args.scripted:
            jobs += [{"key": f"scripted|{v}|{seed}", "kind": "scripted", "variant": v, "seed": seed}
                     for v in SCRIPTED_VARIANTS]
    return jobs


def command(job, args, batch_id):
    """The command line that runs one job."""
    py = sys.executable
    if job["kind"] == "dream":
        cmd = [py, str(HERE / "run_episode.py"), job["task"], "--seconds", str(args.seconds),
               "--seed", str(job["seed"]), "--batch-id", batch_id]
        if args.view:
            cmd.append("--view")
        return cmd
    return [py, str(HERE / "scripted.py"), job["variant"], "--seed", str(job["seed"]), "--batch-id", batch_id]


def run_job(job, args, batch_id):
    """Run one job in a child process. @return (episode dir or None, error text or None, wall seconds)."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONWARNINGS": "ignore"}
    timeout = args.seconds + 180 if job["kind"] == "dream" else 600
    started = time.monotonic()
    try:
        proc = subprocess.run(command(job, args, batch_id), cwd=HERE, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s", time.monotonic() - started
    wall = time.monotonic() - started
    folder = next((line.split("=", 1)[1].strip() for line in proc.stdout.splitlines()
                   if line.startswith("EPISODE_DIR=")), None)
    if folder is None:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return None, f"exit {proc.returncode}: " + " / ".join(tail), wall
    return folder, None, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--repeats", type=int, default=3, help="Seeds per task")
    parser.add_argument("--seed", type=int, default=0, help="First seed")
    parser.add_argument("--seconds", type=float, default=90, help="Longest Dream-1 episode (stops early on success)")
    parser.add_argument("--budget-minutes", type=float, default=30,
                        help="Stop starting Dream-1 episodes once their total wall time passes this")
    parser.add_argument("--scripted", action="store_true", help="Also run the scripted known-outcome episodes")
    parser.add_argument("--no-dream", action="store_true", help="Skip Dream-1 episodes")
    parser.add_argument("--view", action="store_true", help="Show the viewer window for Dream-1 episodes")
    parser.add_argument("--resume", metavar="BATCH_ID")
    args = parser.parse_args()

    BATCHES_DIR.mkdir(exist_ok=True)
    if args.resume:
        state_path = BATCHES_DIR / f"{args.resume}.json"
        state = json.loads(state_path.read_text())
        args = argparse.Namespace(**{**vars(args), **state["args"], "resume": args.resume})
    else:
        batch_id = "b" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        state_path = BATCHES_DIR / f"{batch_id}.json"
        state = {"batch_id": batch_id, "args": {k: v for k, v in vars(args).items() if k != "resume"},
                 "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "done": {}, "failed": {}}
    batch_id = state["batch_id"]
    jobs = plan(args)
    todo = [j for j in jobs if j["key"] not in state["done"]]
    dream_wall = sum(v.get("wall_s", 0) for v in state["done"].values() if v.get("kind") == "dream")
    print(f"Batch {batch_id}: {len(jobs)} jobs, {len(todo)} to run, Dream-1 budget {args.budget_minutes} min", flush=True)

    for n, job in enumerate(todo, 1):
        if job["kind"] == "dream" and dream_wall / 60 >= args.budget_minutes:
            print(f"Budget reached ({dream_wall / 60:.1f} min); skipping the remaining Dream-1 episodes.", flush=True)
            if all(j["kind"] == "dream" for j in todo[n - 1:]):
                break
            continue
        label = job.get("task") or job["variant"]
        print(f"[{n}/{len(todo)}] {job['kind']:8s} {label} (seed {job['seed']}) ...", end=" ", flush=True)
        folder, error, wall = run_job(job, args, batch_id)
        if job["kind"] == "dream":
            dream_wall += wall
        if folder:
            out = json.loads((Path(folder) / "outcome.json").read_text())
            state["done"][job["key"]] = {"kind": job["kind"], "episode": Path(folder).name, "wall_s": round(wall, 1),
                                         "label": out["label"], "reason": out["reason"]}
            state["failed"].pop(job["key"], None)
            print(f"{out['label']} ({out['reason']}) {wall:.0f}s", flush=True)
        else:
            state["failed"][job["key"]] = {"kind": job["kind"], "error": error, "wall_s": round(wall, 1)}
            print(f"NOT RECORDED: {error}", flush=True)
        state_path.write_text(json.dumps(state, indent=2))

    state["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state_path.write_text(json.dumps(state, indent=2))
    print("\n" + summary.report(summary.load(batch_id)) if state["done"] else "\nNo episodes recorded.")
    if state["failed"]:
        print(f"{len(state['failed'])} jobs not recorded (see {state_path}); rerun with --resume {batch_id}")
    print(f"Index: {summary.write_index(summary.load())}")


if __name__ == "__main__":
    main()
