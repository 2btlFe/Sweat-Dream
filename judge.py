"""Simulator ground-truth judge: did the episode do what the instruction asked?

The MuJoCo scene knows where every object is, so the outcome of a pick-and-place instruction can be decided from
physics state alone, with a reason and the moment it happened. This is the label every other part of the pipeline
(human labels, the VLM verifier) is measured against.
"""

import re

import numpy as np

## Movable objects in the Dream YAM Sim scene, by MuJoCo body name, and the words a task may use for each.
OBJECTS = {"apple": "apple", "orange": "orange", "banana": "banana", "bottle": "water_bottle", "water bottle": "water_bottle"}
BODIES = ["apple", "orange", "banana", "water_bottle"]

## Target regions, by MuJoCo body name: horizontal radius from the body's origin and the height band an object
## resting in or on it occupies (metres, world frame; the table top is at z 0.76).
TARGETS = {
    "bowl": {"radius": 0.065, "z": (0.755, 0.83)},
    "plate": {"radius": 0.065, "z": (0.76, 0.82)},
}

TASK_RE = re.compile(
    r"\b(?:put|place|move|drop|set)\s+(?:the\s+|a\s+|an\s+)?(?P<obj>water bottle|bottle|apple|orange|banana)\s+"
    r"(?:into|in|onto|on|inside|in to|on to)\s+(?:the\s+|a\s+)?(?P<tgt>bowl|plate)\b",
    re.IGNORECASE,
)

MOVED_M = 0.03  # an object displaced more than this was moved (not just nudged by contact noise)
OFF_TABLE_Z = 0.70  # below this an object has fallen off the table
REST_SPEED = 0.05  # m/s: slower than this over the last second counts as resting
SETTLE_S = 1.0  # the object must stay placed and released this long for success
DROP_HEIGHT_M = 0.04  # released this far above the target's height band, away from the target: dropped in transit


def parse_task(task):
    """The object and target a pick-and-place instruction names.
    @param task Instruction text, such as "put apple into the bowl".
    @return (object body, target body), or None when the instruction is not one this judge understands.
    """
    m = TASK_RE.search(task)
    if not m:
        return None
    return OBJECTS[m.group("obj").lower()], m.group("tgt").lower()


def in_target(pos, target_pos, target):
    """Whether a position lies inside a target's region.
    @param pos Object position (3,) or positions (N, 3).
    @param target_pos The target body's position (3,).
    @param target Target body name.
    @return Bool, or a bool array for (N, 3).
    """
    spec = TARGETS[target]
    pos = np.asarray(pos)
    xy = np.linalg.norm(pos[..., :2] - target_pos[:2], axis=-1)
    return (xy < spec["radius"]) & (pos[..., 2] > spec["z"][0]) & (pos[..., 2] < spec["z"][1])


def judge(task, t, objects, held, targets):
    """Decide the outcome of one episode from its recorded trace.
    @param task Instruction text.
    @param t Simulated time of each recorded step, seconds (N,).
    @param objects Object positions per step (N, len(BODIES), 3), in BODIES order.
    @param held Whether each object touched a gripper fingertip at each step (N, len(BODIES)).
    @param targets Target body positions by name, {"bowl": (3,), "plate": (3,)}.
    @return A dict: label (success / failure / unknown), reason, success_time_s, failure_time_s, metrics, warnings.
    """
    t = np.asarray(t, dtype=float)
    objects = np.asarray(objects, dtype=float)
    held = np.asarray(held, dtype=bool)
    parsed = parse_task(task)
    result = {"label": "unknown", "reason": None, "success_time_s": None, "failure_time_s": None,
              "metrics": {}, "warnings": []}
    if parsed is None:
        result["reason"] = "task_not_understood"
        return result
    if len(t) < 2:
        result["reason"] = "too_short"
        return result
    obj, target = parsed
    k = BODIES.index(obj)
    path = objects[:, k]
    displacement = np.linalg.norm(objects - objects[:1], axis=-1)  # (N, objects)
    max_moved = displacement.max(axis=0)
    inside = in_target(path, np.asarray(targets[target]), target)
    placed = inside & ~held[:, k]

    dt = np.median(np.diff(t))
    last = t >= t[-1] - SETTLE_S
    speed = np.linalg.norm(np.diff(path[last], axis=0), axis=-1).max() / dt if last.sum() > 1 else 0.0
    others = [j for j in range(len(BODIES)) if j != k and max_moved[j] > MOVED_M]

    # Success: placed (inside, released) from some moment through the end, for at least SETTLE_S, and resting.
    success_time = None
    if placed[-1]:
        start = len(placed) - 1
        while start > 0 and placed[start - 1]:
            start -= 1
        if t[-1] - t[start] >= SETTLE_S and speed < REST_SPEED:
            success_time = float(t[start])

    result["metrics"] = {
        "object": obj, "target": target,
        "object_moved_m": round(float(max_moved[k]), 4),
        "final_distance_to_target_m": round(float(np.linalg.norm(path[-1, :2] - np.asarray(targets[target])[:2])), 4),
        "final_speed_m_s": round(float(speed), 4),
        "ever_touched": bool(held[:, k].any()),
        "ever_inside_target": bool(inside.any()),
        "others_moved": {BODIES[j]: round(float(max_moved[j]), 4) for j in others},
        "duration_s": round(float(t[-1] - t[0]), 2),
    }
    for j in others:
        result["warnings"].append(f"{BODIES[j]} moved {max_moved[j] * 100:.1f} cm")

    if success_time is not None:
        result.update(label="success", reason="placed_in_target", success_time_s=round(success_time, 2))
        return result

    def first(mask):
        idx = np.flatnonzero(mask)
        return round(float(t[idx[0]]), 2) if len(idx) else None

    result["label"] = "failure"
    off_table = path[:, 2] < OFF_TABLE_Z
    if off_table.any():
        result.update(reason="object_fell_off_table", failure_time_s=first(off_table))
    elif held[-1, k] and max_moved[k] > MOVED_M:
        over = np.linalg.norm(path[-1, :2] - np.asarray(targets[target])[:2]) < TARGETS[target]["radius"]
        result.update(reason="still_holding_over_target" if over else "still_holding",
                      failure_time_s=round(float(t[-1]), 2))
    elif inside[-1]:
        result.update(reason="not_settled_in_target", failure_time_s=round(float(t[-1]), 2))
    elif max_moved[k] <= MOVED_M and others:
        j = max(others, key=lambda j: max_moved[j])
        result.update(reason="wrong_object_moved", failure_time_s=first(displacement[:, j] > MOVED_M))
    elif max_moved[k] <= MOVED_M and held[:, k].any():
        result.update(reason="touched_but_not_moved", failure_time_s=first(held[:, k]))
    elif max_moved[k] <= MOVED_M:
        result.update(reason="object_not_moved", failure_time_s=None)
    elif inside.any():
        # It was in the target and left it again: the moment it left.
        left_at = np.flatnonzero(inside[:-1] & ~inside[1:])
        result.update(reason="left_target_after_placing",
                      failure_time_s=round(float(t[left_at[-1] + 1]), 2) if len(left_at) else None)
    elif (released := np.flatnonzero(held[:-1, k] & ~held[1:, k])).size and \
            path[released[-1], 2] > TARGETS[target]["z"][1] + DROP_HEIGHT_M and \
            np.linalg.norm(path[released[-1], :2] - np.asarray(targets[target])[:2]) > TARGETS[target]["radius"]:
        result.update(reason="dropped_in_transit", failure_time_s=round(float(t[released[-1] + 1]), 2))
    else:
        result.update(reason="placed_outside_target",
                      failure_time_s=round(float(t[released[-1] + 1]), 2) if len(released) else first(displacement[:, k] > MOVED_M))
    return result


def is_success_now(task, t, objects, held, targets):
    """Cheap online check for stopping an episode early once it has succeeded (same rule as judge())."""
    return judge(task, t, objects, held, targets)["label"] == "success"
