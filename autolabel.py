"""Auto labeling: every episode gets a label from the simulator's judgement, a confidence, and a review flag.

For each episode folder, writes auto_label.json next to outcome.json (which it never changes) and never touches a
human label (label.json). The label is the judge's; confidence is "low", and the episode goes to the review queue,
when something makes that judgement less certain:

    unknown          the judge does not understand the instruction
    judge_warning    the judge itself warned (its warnings are copied)
    run_error        the episode ended with an error
    near_edge        the object ends within 1 cm of the target region's edge (in or out)
    ended_moving     the object was still moving when the episode ended
    others_moved     another object was moved too
    replay_drift     a disturbed run's replayed prefix did not match its clean run
    human_disagrees  a human label exists and differs (the human label wins; auto is kept for comparison)

Failure-recovery runs (perturb.py) also get their recovery: kind, when, recovered, recovery time. Tags make the
dataset easy to slice: clean_success, failure, recovery_run, recovered / not_recovered, scripted, looks:<n>.

Usage:
    python autolabel.py                 every episode (again, so new rules apply)
    python autolabel.py <episode_id> ...
    python autolabel.py --summary       label and review counts, recovery rates per disturbance
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPISODES_DIR = HERE / "episodes"
VERSION = 1
EDGE_M = 0.01
RADIUS = {"bowl": 0.065, "plate": 0.065}  # judge.TARGETS (no sim import needed)
MOVING_M_S = 0.05


def read(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def episode_dirs() -> list[Path]:
    if not EPISODES_DIR.is_dir():
        return []
    return [d for d in sorted(EPISODES_DIR.iterdir()) if d.is_dir() and (d / "outcome.json").exists() and (d / "meta.json").exists()]


def auto_label(d: Path) -> dict:
    meta, outcome = read(d / "meta.json", {}), read(d / "outcome.json", {})
    perturbation, human, aug = read(d / "perturbation.json"), read(d / "label.json"), read(d / "aug" / "index.json")
    m = outcome.get("metrics") or {}
    label = outcome.get("label", "unknown")
    review = []
    if label not in ("success", "failure"):
        review.append("unknown")
    if outcome.get("warnings"):
        review.append("judge_warning")
    if meta.get("errors"):
        review.append("run_error")
    distance, target = m.get("final_distance_to_target_m"), m.get("target")
    if distance is not None and target in RADIUS and abs(distance - RADIUS[target]) < EDGE_M:
        review.append("near_edge")
    if (m.get("final_speed_m_s") or 0) > MOVING_M_S:
        review.append("ended_moving")
    if m.get("others_moved"):
        review.append("others_moved")
    if perturbation and (perturbation.get("replay_drift_m") or 0) > 0.001:
        review.append("replay_drift")
    if human and human.get("label") in ("success", "failure") and human["label"] != label:
        review.append("human_disagrees")
    source = str(meta.get("source") or "")
    tags = []
    if perturbation:
        tags += ["recovery_run", "recovered" if perturbation.get("recovered") else "not_recovered",
                 f"disturbance:{perturbation.get('kind')}", "policy:" + ("dream-1" if perturbation.get("policy") == "dream-1" else "hold")]
    elif label == "success":
        tags.append("clean_success")
    elif label == "failure":
        tags.append("failure")
    if source.startswith("scripted"):
        tags.append("scripted")
    if aug and aug.get("variants"):
        tags.append(f"looks:{len(aug['variants'])}")
    record = {
        "label": label,
        "reason": outcome.get("reason"),
        "success_time_s": outcome.get("success_time_s"),
        "failure_time_s": outcome.get("failure_time_s"),
        "source": "simulator judge (judge.py)",
        "confidence": "low" if review else "high",
        "review": bool(review),
        "review_reasons": review,
        "warnings": outcome.get("warnings") or [],
        "tags": tags,
        "variants": {"looks": len(aug["variants"]) if aug and aug.get("variants") else 0,
                     "label_applies_to_looks": bool(aug)},
        "version": VERSION,
        "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if perturbation:
        record["recovery"] = {k: perturbation.get(k) for k in (
            "kind", "spec", "applied_at_s", "window_s", "recovered", "recovery_time_s", "clean_episode", "policy")}
    if human:
        record["human_label"] = human.get("label")
    (d / "auto_label.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def summary(records: dict[str, dict]) -> dict:
    labels = Counter(r["label"] for r in records.values())
    reasons = Counter(x for r in records.values() for x in r["review_reasons"])
    recovery = defaultdict(lambda: {"runs": 0, "recovered": 0, "times": []})
    for r in records.values():
        rec = r.get("recovery")
        if rec and rec.get("policy") == "dream-1":
            row = recovery[rec["kind"]]
            row["runs"] += 1
            row["recovered"] += bool(rec.get("recovered"))
            if rec.get("recovery_time_s") is not None:
                row["times"].append(rec["recovery_time_s"])
    return {
        "episodes": len(records),
        "labels": dict(labels),
        "high_confidence": sum(r["confidence"] == "high" for r in records.values()),
        "review_queue": sum(r["review"] for r in records.values()),
        "review_reasons": dict(reasons),
        "clean_successes": sum("clean_success" in r["tags"] for r in records.values()),
        "look_variants": sum(r["variants"]["looks"] for r in records.values()),
        "recovery_by_disturbance": {k: {"runs": v["runs"], "recovered": v["recovered"],
                                        "rate": round(v["recovered"] / v["runs"], 3) if v["runs"] else None,
                                        "mean_recovery_s": round(sum(v["times"]) / len(v["times"]), 1) if v["times"] else None}
                                    for k, v in sorted(recovery.items())},
    }


def print_summary(s: dict) -> None:
    print(f"{s['episodes']} episodes: {s['labels']} · high confidence {s['high_confidence']} · review queue {s['review_queue']}")
    if s["review_reasons"]:
        print("  review because: " + ", ".join(f"{k} x{v}" for k, v in s["review_reasons"].items()))
    print(f"  clean successes {s['clean_successes']} · look variants {s['look_variants']} (each labelled like its episode)")
    for kind, r in s["recovery_by_disturbance"].items():
        mean = f", mean {r['mean_recovery_s']}s to recover" if r["mean_recovery_s"] is not None else ""
        print(f"  recovery after {kind:<12} {r['recovered']}/{r['runs']} ({r['rate']:.0%}){mean}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", nargs="*")
    parser.add_argument("--summary", action="store_true", help="only print the totals")
    args = parser.parse_args()
    dirs = [EPISODES_DIR / e for e in args.episodes] if args.episodes else episode_dirs()
    records = {}
    for d in dirs:
        records[d.name] = read(d / "auto_label.json") if args.summary else auto_label(d)
    records = {k: v for k, v in records.items() if v}
    s = summary(records)
    if not args.summary:
        (EPISODES_DIR / "auto_label_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print_summary(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
