"""Server-side evaluation of saved Dream YAM trajectories.

The task selects the object and target. User rules select horizontal target
radius, whether release is required, and the final settling duration.
"""

import math
import re

OBJECTS = {"apple": "apple", "orange": "orange", "banana": "banana",
           "bottle": "water_bottle", "water bottle": "water_bottle"}
TASK = re.compile(r"\b(?:put|place|move|drop|set)\s+(?:the\s+|a\s+|an\s+)?"
                  r"(water bottle|bottle|apple|orange|banana)\s+"
                  r"(?:into|in|onto|on|inside|in to|on to)\s+(?:the\s+|a\s+)?(bowl|plate)\b", re.I)
DEFAULT_RULES = {"distance_cm": 6.5, "release": True, "still_seconds": 1.0}
HEIGHT_BANDS = {"bowl": (0.755, 0.83), "plate": (0.76, 0.82)}
REST_SPEED_M_S = 0.05
MOVED_M = 0.03
OFF_TABLE_Z = 0.70


def parse_task(task):
    match = TASK.search(task or "")
    return (OBJECTS[match.group(1).lower()], match.group(2).lower()) if match else None


def normalize_rules(value):
    """Return validated, reproducible rule values without silent clamping."""
    if not isinstance(value, dict) or type(value.get("release")) is not bool:
        raise ValueError("Rules need distance_cm, release, and still_seconds.")
    try:
        distance = float(value["distance_cm"])
        seconds = float(value["still_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Rules need distance_cm, release, and still_seconds.") from exc
    if not math.isfinite(distance) or not 0.5 <= distance <= 30:
        raise ValueError("Distance must be between 0.5 and 30 cm.")
    if not math.isfinite(seconds) or not 0 <= seconds <= 10:
        raise ValueError("Still time must be between 0 and 10 seconds.")
    return {"distance_cm": distance, "release": value["release"], "still_seconds": seconds}


def invalid(reason, details=None):
    return {"label": "invalid", "reason": reason, "details": details, "checks": [],
            "success": None, "dist": [], "touch": [], "t": []}


def evaluate(meta, trajectory, rules):
    """Judge one completed run using the selected rules and its MuJoCo trace."""
    rules = normalize_rules(rules)
    if meta.get("error"):
        return invalid("execution_error", meta.get("error"))
    if meta.get("stopped_by_user"):
        return invalid("user_stopped")
    parsed = parse_task(meta.get("task"))
    if parsed is None:
        return invalid("unsupported_task")
    if meta.get("model_responses", 0) < 1 or meta.get("model_actions", 0) < 1:
        return invalid("no_model_actions")
    obj, target = parsed
    if obj not in trajectory.get("positions", {}) or target not in trajectory.get("positions", {}):
        return invalid("not_on_table")
    try:
        t = trajectory["t"]
        positions = trajectory["positions"]
        p = positions[obj]
        target_positions = positions[target]
        touch = trajectory["touching"][obj]
        n = len(t)
        if n < 2 or any(len(v) != n for v in (p, target_positions, touch)):
            return invalid("incomplete_trajectory")
        if any(len(v) != 3 for v in (*p, *target_positions)):
            return invalid("incomplete_trajectory")
        if not all(math.isfinite(float(v)) for series in (t, *p, *target_positions) for v in series):
            return invalid("nonfinite_trajectory")
        if any(t[i] <= t[i - 1] for i in range(1, n)):
            return invalid("nonmonotonic_time")
    except (KeyError, TypeError, ValueError, IndexError):
        return invalid("incomplete_trajectory")

    z_min, z_max = HEIGHT_BANDS[target]
    dist = [math.hypot(p[i][0] - target_positions[i][0], p[i][1] - target_positions[i][1]) * 100
            for i in range(n)]
    in_target = [dist[i] < rules["distance_cm"] and z_min < p[i][2] < z_max for i in range(n)]
    released = [not bool(touch[i]) for i in range(n)]
    placed = [in_target[i] and (released[i] or not rules["release"]) for i in range(n)]
    speed = [math.dist(p[i], p[i - 1]) / (t[i] - t[i - 1]) for i in range(1, n)]
    required = rules["still_seconds"]
    start = n - 1
    while start > 0 and placed[start - 1]:
        start -= 1
    final_speed = [speed[i - 1] for i in range(1, n) if t[i] > t[-1] - required]
    stable = required == 0 or (bool(final_speed) and max(final_speed) < REST_SPEED_M_S)
    long_enough = placed[-1] and t[-1] - t[start] >= required - 1e-9
    success = bool(long_enough and stable)
    moved = max(math.dist(pos, p[0]) for pos in p)
    if success:
        reason = "placed_and_stable"
    elif min(pos[2] for pos in p) < OFF_TABLE_Z:
        reason = "object_fell_off_table"
    elif in_target[-1] and rules["release"] and not released[-1]:
        reason = "still_holding"
    elif in_target[-1]:
        reason = "not_stable"
    elif moved <= MOVED_M and any(touch):
        reason = "touched_but_not_moved"
    elif moved <= MOVED_M:
        reason = "object_not_moved"
    elif any(in_target):
        reason = "left_target"
    else:
        reason = "outside_target"
    checks = [
        {"ok": in_target[-1], "text": f"Target: {obj} {'inside' if in_target[-1] else 'outside'} {target}"
         f" (distance {dist[-1]:.1f} cm; limit {rules['distance_cm']:.1f} cm)"},
        {"ok": released[-1] or not rules["release"],
         "text": "Release: " + ("not required" if not rules["release"] else
                               "released" if released[-1] else "still touching gripper")},
        {"ok": bool(long_enough and stable), "text": f"Stable: final {required:g} s; max speed"
         f" {max(final_speed) if final_speed else 0:.3f} m/s (limit {REST_SPEED_M_S:.3f})"},
    ]
    return {"label": "success" if success else "failure", "reason": reason, "checks": checks,
            "success": {"start": t[start], "end": t[-1]} if success else None,
            "success_time_s": t[-1] if success else None, "dist": dist, "touch": touch, "t": t,
            "metrics": {"object_moved_m": round(moved, 4), "distance_to_target_cm": round(dist[-1], 2),
                        "max_final_speed_m_s": round(max(final_speed), 4) if final_speed else None,
                        "model_responses": meta["model_responses"], "model_actions": meta["model_actions"]}}
