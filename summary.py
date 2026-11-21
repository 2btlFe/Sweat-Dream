"""Dataset index and summary over recorded episodes.

Usage: python summary.py [--batch <batch_id>]

Writes episodes/index.csv (one row per episode) and prints success rates and failure reasons per source and task.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

EPISODES_DIR = Path(__file__).resolve().parent / "episodes"  # same as recorder.EPISODES_DIR (no sim import needed)

COST_PER_HOUR = 3.0  # Dream-1 list price, USD per hour of use (the flash model is $2)
FIELDS = ["episode_id", "batch_id", "source", "task", "seed", "label", "reason", "success_time_s", "failure_time_s",
          "object_moved_m", "final_distance_to_target_m", "others_moved", "wall_s", "sim_s", "requests", "rtt_mean_s",
          "rtt_max_s", "connection_retries", "stopped_early_at_s", "errors"]


def load(batch=None):
    """Every episode as one flat dict, oldest first; only one batch's when `batch` is given."""
    rows = []
    for d in sorted(p for p in EPISODES_DIR.iterdir() if (p / "outcome.json").exists() and (p / "meta.json").exists()):
        meta = json.loads((d / "meta.json").read_text())
        if batch and meta.get("batch_id") != batch:
            continue
        out = json.loads((d / "outcome.json").read_text())
        m = out.get("metrics", {})
        rows.append({
            **{k: meta.get(k) for k in ("episode_id", "batch_id", "source", "task", "seed", "wall_s", "sim_s",
                                         "requests", "rtt_mean_s", "rtt_max_s", "connection_retries",
                                         "stopped_early_at_s")},
            "label": out["label"], "reason": out["reason"],
            "success_time_s": out["success_time_s"], "failure_time_s": out["failure_time_s"],
            "object_moved_m": m.get("object_moved_m"), "final_distance_to_target_m": m.get("final_distance_to_target_m"),
            "others_moved": ";".join(f"{k}={v}" for k, v in (m.get("others_moved") or {}).items()),
            "errors": " | ".join(meta.get("errors") or []),
        })
    return rows


def write_index(rows, path=EPISODES_DIR / "index.csv"):
    """Write the flat episode table as CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})
    return path


def report(rows):
    """Summary text: per source group and task, success rate, failure reasons, timing and reliability."""
    groups = defaultdict(list)
    for r in rows:
        groups[("dream-1" if r["source"] == "dream-1" else "scripted", r["task"])].append(r)
    lines = []
    for (source, task), rs in sorted(groups.items()):
        ok = sum(r["label"] == "success" for r in rs)
        reasons = Counter(r["reason"] for r in rs if r["label"] != "success")
        errors = sum(bool(r["errors"]) for r in rs)
        lines.append(f"[{source}] {task}: {ok}/{len(rs)} success ({ok / len(rs):.0%})")
        if reasons:
            lines.append("    failures: " + ", ".join(f"{k} x{v}" for k, v in reasons.most_common()))
        if source == "dream-1":
            wall = sum(r["wall_s"] or 0 for r in rs)
            reqs = sum(r["requests"] or 0 for r in rs)
            retries = sum(r["connection_retries"] or 0 for r in rs)
            rtts = [r["rtt_mean_s"] for r in rs if r["rtt_mean_s"]]
            lines.append(f"    {reqs} requests, mean RTT {sum(rtts) / len(rtts) if rtts else 0:.2f}s, "
                         f"{retries} connection retries, {errors} episodes ended with an error, "
                         f"{wall / 60:.1f} min (~${wall / 3600 * COST_PER_HOUR:.2f})")
        elif errors:
            lines.append(f"    {errors} episodes ended with an error")
    total = len(rows)
    ok = sum(r["label"] == "success" for r in rows)
    lines.append(f"Total: {total} episodes, {ok} success, {total - ok} failure/unknown")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch")
    args = parser.parse_args()
    rows = load(args.batch)
    path = write_index(load())
    print(report(rows) if rows else "No episodes.")
    print(f"Index: {path}")


if __name__ == "__main__":
    main()
