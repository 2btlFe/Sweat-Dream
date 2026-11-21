"""Server-side evaluation of saved Dream YAM trajectories, by task type.

A run's task has a type, and each type has its own rules (set on the page, kept in runs/rules.json):

  pick_place    put an object into the bowl / on the plate (several pairs at once)
                rules: distance to the target's centre, release required, still time at the end
  move_to_area  move an object (or any object) to the center / left / right / front / back of the workspace
                rules: area size, release required, still time at the end
  lift          lift an object (or any object)
                rules: height above its start, time held up there
  push          push an object (or any object) along the table
                rules: distance moved, lifting allowed or not

The type and its objects come from the task's words (classify), or from the spec chosen for a typed prompt
(meta["judge"]). Add a type: its rules to RULES, a judge_<type> function, and a branch in classify().

The numbers of the YAM table are the defaults (HEIGHT_BANDS, OFF_TABLE_Z, WORKSPACE_CENTER). A run on another
table (OpenYAM's Sandbox table) says its own in meta.json: "relative_height_bands" (above the bowl / plate),
"off_table_z", and "workspace" ({"center", "left", "front"}: its middle and the camera's left and front directions).
"""

import math
import re

OBJECTS = {"apple": "apple", "orange": "orange", "banana": "banana",
           "bottle": "water_bottle", "water bottle": "water_bottle"}
OBJECT_BODIES = ("apple", "orange", "banana", "water_bottle")
TASK = re.compile(r"\b(?:put|place|move|drop|set)\s+(?:the\s+|a\s+|an\s+)?"
                  r"(water bottle|bottle|apple|orange|banana)\s+"
                  r"(?:into|in|onto|on|inside|in to|on to)\s+(?:the\s+|a\s+)?(bowl|plate)\b", re.I)
HEIGHT_BANDS = {"bowl": (0.755, 0.83), "plate": (0.76, 0.82)}
REST_SPEED_M_S = 0.05
MOVED_M = 0.03
OFF_TABLE_Z = 0.70
LIFTED_M = 0.03  # an object this far above its start is off the table (not resting, not pushed)
MAX_PAIRS = 4

## Workspace areas (world frame, metres; +x is the left of the top camera's image, +y towards the arms). An area
## starts zone_cm from WORKSPACE_CENTER: the center is a disc of that radius, the sides lie beyond that distance.
WORKSPACE_CENTER = (0.0, -0.02)
AREAS = ("center", "left", "right", "front", "back")
AREA_WORDS = {"center": "center", "centre": "center", "middle": "center", "left": "left", "right": "right",
              "front": "front", "near": "front", "back": "back", "far": "back"}

## Each task type's rules: name -> (default, minimum, maximum) for numbers, or a default bool.
RULES = {
    "pick_place": {"distance_cm": (6.5, 0.5, 30), "release": True, "still_seconds": (1.0, 0, 10)},
    "move_to_area": {"zone_cm": (10.0, 2, 30), "release": True, "still_seconds": (1.0, 0, 10)},
    "lift": {"height_cm": (5.0, 1, 30), "hold_seconds": (1.0, 0, 10)},
    "push": {"distance_cm": (5.0, 1, 40), "no_lift": True},
}
TYPES = tuple(RULES)
DEFAULT_RULES = {kind: {name: (spec[0] if isinstance(spec, tuple) else spec) for name, spec in rules.items()}
                 for kind, rules in RULES.items()}


def parse_task(task):
    match = TASK.search(task or "")
    return (OBJECTS[match.group(1).lower()], match.group(2).lower()) if match else None


def classify(task):
    """The judge spec a task's words give, or None when they give none (the run is then not judged).
      "put apple in the bowl and banana on the plate" -> pick_place, pairs apple -> bowl, banana -> plate
      "Move an object to the center of the workspace"   -> move_to_area, any object -> center
      "lift the orange"                                 -> lift, orange
      "push the bottle"                                 -> push, water_bottle
    """
    text = (task or "").lower()
    named = [(m.start(), "water_bottle" if "bottle" in m.group() else m.group())
             for m in re.finditer(r"water bottle|bottle|apple|orange|banana|bowl|plate", text)]
    objects = list(dict.fromkeys(name for _, name in named if name in OBJECT_BODIES))
    targets = [(at, name) for at, name in named if name in HEIGHT_BANDS]
    if objects and targets:  # each object goes to the next bowl or plate named after it
        pairs = []
        for name in objects[:MAX_PAIRS]:
            at = next(a for a, n in named if n == name)
            pairs.append({"object": name, "target": next((g for a, g in targets if a > at), targets[-1][1])})
        return {"type": "pick_place", "pairs": pairs}
    obj = objects[0] if objects else "any"
    area = next((AREA_WORDS[w] for w in re.findall(r"\b(?:center|centre|middle|left|right|front|near|back|far)\b", text)), None)
    if area and re.search(r"\b(?:move|put|place|bring|push|slide|drag|take|carry|shift)\b", text):
        return {"type": "move_to_area", "object": obj, "area": area}
    if re.search(r"\b(?:lift|raise|hold up|pick up)\b|\bpick\b.*\bup\b", text):
        return {"type": "lift", "object": obj}
    if re.search(r"\b(?:push|slide|nudge|shove|poke)\b", text):
        return {"type": "push", "object": obj}
    return None


def normalize_spec(spec):
    """A judge spec, checked. Accepts the older forms ({"pairs": [...]}, {"object", "target"}) as pick_place.
    @return {"type", ...} or None when it is not a valid spec."""
    if not isinstance(spec, dict):
        return None
    kind = spec.get("type") or "pick_place"
    if kind == "pick_place":
        given = spec.get("pairs") if isinstance(spec.get("pairs"), list) else ([spec] if spec.get("object") else [])
        pairs = [(p.get("object"), p.get("target")) for p in given if isinstance(p, dict)]
        objects = [o for o, _ in pairs]
        if (0 < len(pairs) <= MAX_PAIRS and len(set(objects)) == len(objects)
                and all(o in OBJECT_BODIES and g in HEIGHT_BANDS for o, g in pairs)):
            return {"type": "pick_place", "pairs": [{"object": o, "target": g} for o, g in pairs]}
        return None
    obj = spec.get("object") or "any"
    if kind in TYPES and obj in (*OBJECT_BODIES, "any"):
        if kind == "move_to_area":
            return {"type": kind, "object": obj, "area": spec["area"]} if spec.get("area") in AREAS else None
        return {"type": kind, "object": obj}
    return None


def judged_spec(meta):
    """The spec a run is judged on: the one chosen for it (meta["judge"]), else the one its task's words give."""
    if meta.get("judge"):
        return normalize_spec(meta["judge"])
    return classify(meta.get("task"))


def judged_pairs(meta):
    """@return The pick_place pairs of a run [(object, target)], or None (kept for older callers)."""
    spec = judged_spec(meta)
    return [(p["object"], p["target"]) for p in spec["pairs"]] if spec and spec["type"] == "pick_place" else None


def normalize_rules(value):
    """Every task type's rules, checked (no silent clamping). Types or values left out keep their defaults; the
    older flat form {"distance_cm", "release", "still_seconds"} is the pick_place rules."""
    if not isinstance(value, dict):
        raise ValueError("Rules must be an object.")
    if "distance_cm" in value and "release" in value:  # the older flat form
        value = {"pick_place": value}
    result = {}
    for kind, specs in RULES.items():
        given = value.get(kind) or {}
        if not isinstance(given, dict):
            raise ValueError(f"Rules for {kind} must be an object.")
        result[kind] = {}
        for name, spec in specs.items():
            v = given.get(name, DEFAULT_RULES[kind][name])
            if isinstance(spec, tuple):
                try:
                    v = float(v)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{kind}.{name} must be a number.") from exc
                if not math.isfinite(v) or not spec[1] <= v <= spec[2]:
                    raise ValueError(f"{kind}.{name} must be between {spec[1]:g} and {spec[2]:g}.")
            elif type(v) is not bool:
                raise ValueError(f"{kind}.{name} must be true or false.")
            result[kind][name] = v
    return result


def invalid(reason, details=None):
    return {"label": "invalid", "reason": reason, "details": details, "checks": [],
            "success": None, "dist": [], "touch": [], "t": []}


def evaluate(meta, trajectory, rules):
    """Judge one completed run with its task type's rules and its MuJoCo trace."""
    rules = normalize_rules(rules)
    if meta.get("error"):
        return invalid("execution_error", meta.get("error"))
    if meta.get("stopped_by_user"):
        return invalid("user_stopped")
    spec = judged_spec(meta)
    if spec is None:
        return invalid("unsupported_task")
    if meta.get("model_responses", 0) < 1 or meta.get("model_actions", 0) < 1:
        return invalid("no_model_actions")
    positions = trajectory.get("positions", {}) if isinstance(trajectory, dict) else {}
    kind = spec["type"]
    if kind == "pick_place":
        jobs = [(p["object"], p["target"]) for p in spec["pairs"]]
    elif spec["object"] == "any":
        jobs = [(o, None) for o in OBJECT_BODIES if o in positions]
    else:
        jobs = [(spec["object"], None)]
    if not jobs or any(o not in positions or (g and g not in positions) for o, g in jobs):
        return invalid("not_on_table")
    try:
        t = trajectory["t"]
        n = len(t)
        series = [(positions[o], positions[g] if g else None, trajectory["touching"][o]) for o, g in jobs]
        if n < 2 or any(len(v) != n for s3 in series for v in s3 if v is not None):
            return invalid("incomplete_trajectory")
        if any(len(v) != 3 for p, q, _ in series for v in (*p, *(q or []))):
            return invalid("incomplete_trajectory")
        if not all(math.isfinite(float(v)) for p, q, _ in series for xs in (t, *p, *(q or [])) for v in xs):
            return invalid("nonfinite_trajectory")
        if any(t[i] <= t[i - 1] for i in range(1, n)):
            return invalid("nonmonotonic_time")
    except (KeyError, TypeError, ValueError, IndexError):
        return invalid("incomplete_trajectory")

    try:
        frame = table_frame(meta)
    except (KeyError, TypeError, ValueError):
        return invalid("invalid_height_band")
    model = {"model_responses": meta["model_responses"], "model_actions": meta["model_actions"]}
    type_rules = rules[kind]
    if kind == "pick_place":
        results = [judge_pair(t, p, q, touch, obj, target, type_rules, frame)
                   for (obj, target), (p, q, touch) in zip(jobs, series)]
        chart = {"y": "distance to target (cm)", "limit": type_rules["distance_cm"]}
    else:
        judge_one = {"move_to_area": judge_area, "lift": judge_lift, "push": judge_push}[kind]
        results = [judge_one(t, p, touch, obj, spec, type_rules, frame) for (obj, _), (p, _, touch) in zip(jobs, series)]
        chart = {"move_to_area": {"y": f"distance to the {spec.get('area')} area (cm)", "limit": 0},
                 "lift": {"y": "height above start (cm)", "limit": type_rules.get("height_cm")},
                 "push": {"y": "moved from start (cm)", "limit": type_rules.get("distance_cm")}}[kind]
    per_pair = [{"object": r["object"], "target": r["target"], "label": "success" if r["success"] else "failure",
                 "reason": r["reason"]} for r in results]
    plot = [{"object": r["object"], "target": r["target"], "dist": r["dist"], "touch": r["touch"]} for r in results]
    common = {"type": kind, "spec": spec, "chart": chart, "t": t, "pairs": per_pair, "series": plot}

    if kind != "pick_place" and len(results) > 1:
        # Any object will do: the run succeeds when one does; otherwise the reason is the one that came closest.
        best = next((r for r in results if r["success"]), None) or max(results, key=lambda r: r["progress"])
        checks = [{**c, "text": f"{best['object']} · {c['text']}"} for c in best["checks"]]
        return {**common, "label": "success" if best["success"] else "failure",
                "reason": f"{best['reason']} ({best['object']})", "checks": checks,
                "success": {"start": best["start"], "end": best["end"]} if best["success"] else None,
                "success_time_s": best["end"] if best["success"] else None, "dist": best["dist"], "touch": best["touch"],
                "metrics": {"object": best["object"], **best["metrics"], **model}}
    if len(results) == 1:
        r = results[0]
        return {**common, "label": "success" if r["success"] else "failure", "reason": r["reason"], "checks": r["checks"],
                "success": {"start": r["start"], "end": r.get("end", t[-1])} if r["success"] else None,
                "success_time_s": r.get("end", t[-1]) if r["success"] else None, "dist": r["dist"], "touch": r["touch"],
                "metrics": {**r["metrics"], **model}}
    # Several pick_place pairs: success only when every pair holds its rules at the end.
    placed = sum(r["success"] for r in results)
    success = placed == len(results)
    first_failed = next((r for r in results if not r["success"]), None)
    reason = "placed_and_stable" if success else \
        f"{placed}/{len(results)} placed; {first_failed['object']}: {first_failed['reason']}"
    checks = [{**c, "text": f"{r['object']} → {r['target']} · {c['text']}"} for r in results for c in r["checks"]]
    return {**common, "label": "success" if success else "failure", "reason": reason, "checks": checks,
            "success": {"start": max(r["start"] for r in results), "end": t[-1]} if success else None,
            "success_time_s": t[-1] if success else None, "dist": results[0]["dist"], "touch": results[0]["touch"],
            "metrics": {"placed": placed, "pairs": len(results), **model,
                        "per_pair": [{"object": r["object"], **r["metrics"]} for r in results]}}


def table_frame(meta):
    """The table a run was on: height bands of the bowl and plate (absolute, or relative to them), the floor level,
    and the workspace (its middle, and the unit vectors the top camera sees as left and as front)."""
    bands = meta.get("relative_height_bands")
    if bands is not None:
        for low, high in bands.values():
            if not (math.isfinite(float(low)) and math.isfinite(float(high)) and low < high):
                raise ValueError("bad height band")
    ws = meta.get("workspace") or {}
    return {"bands": bands, "off_table_z": float(meta.get("off_table_z", OFF_TABLE_Z)),
            "center": tuple(ws.get("center", WORKSPACE_CENTER)), "left": tuple(ws.get("left", (1.0, 0.0))),
            "front": tuple(ws.get("front", (0.0, 1.0)))}


DEFAULT_FRAME = {"bands": None, "off_table_z": OFF_TABLE_Z, "center": WORKSPACE_CENTER, "left": (1.0, 0.0),
                 "front": (0.0, 1.0)}


def final_window(t, ok, required):
    """The final stretch where ok holds: (index where it began, whether it lasted `required` seconds)."""
    n = len(t)
    start = n - 1
    while start > 0 and ok[start - 1]:
        start -= 1
    return start, bool(ok[-1]) and t[-1] - t[start] >= required - 1e-9


def _offsets(x, y, frame):
    """A point's offset from the workspace's middle, along the camera's left and front directions (m)."""
    ex, ey = x - frame["center"][0], y - frame["center"][1]
    return ex * frame["left"][0] + ey * frame["left"][1], ex * frame["front"][0] + ey * frame["front"][1]


def in_area(x, y, area, zone, frame=DEFAULT_FRAME):
    dx, dy = _offsets(x, y, frame)
    return {"center": math.hypot(dx, dy) <= zone, "left": dx >= zone, "right": dx <= -zone,
            "front": dy >= zone, "back": dy <= -zone}[area]


def area_gap_cm(x, y, area, zone, frame=DEFAULT_FRAME):
    """How far (cm) a point is from the area; 0 inside."""
    dx, dy = _offsets(x, y, frame)
    gap = {"center": math.hypot(dx, dy) - zone, "left": zone - dx, "right": dx + zone,
           "front": zone - dy, "back": dy + zone}[area]
    return max(0.0, gap) * 100


def judge_area(t, p, touch, obj, spec, rules, frame=DEFAULT_FRAME):
    """move_to_area: the object ends in the area, resting on the table (released if required) and still."""
    n, area, zone = len(t), spec["area"], rules["zone_cm"] / 100
    z0 = p[0][2]
    gap = [area_gap_cm(p[i][0], p[i][1], area, zone, frame) for i in range(n)]
    resting = [frame["off_table_z"] < p[i][2] < z0 + LIFTED_M for i in range(n)]
    inside = [gap[i] == 0 and resting[i] for i in range(n)]
    released = [not bool(touch[i]) for i in range(n)]
    placed = [inside[i] and (released[i] or not rules["release"]) for i in range(n)]
    speed = [math.dist(p[i], p[i - 1]) / (t[i] - t[i - 1]) for i in range(1, n)]
    required = rules["still_seconds"]
    start, long_enough = final_window(t, placed, required)
    final_speed = [speed[i - 1] for i in range(1, n) if t[i] > t[-1] - required]
    stable = required == 0 or (bool(final_speed) and max(final_speed) < REST_SPEED_M_S)
    moved = max(math.dist(pos, p[0]) for pos in p)
    success = bool(long_enough and stable and moved > MOVED_M)
    if success:
        reason = "moved_to_area"
    elif min(pos[2] for pos in p) < frame["off_table_z"]:
        reason = "object_fell_off_table"
    elif moved <= MOVED_M:
        reason = "already_there" if inside[0] else ("touched_but_not_moved" if any(touch) else "object_not_moved")
    elif inside[-1] and rules["release"] and not released[-1]:
        reason = "still_holding"
    elif inside[-1]:
        reason = "not_stable"
    elif any(inside):
        reason = "left_area"
    else:
        reason = "outside_area"
    checks = [
        {"ok": inside[-1] and moved > MOVED_M, "text": f"Area: {obj} {'in' if inside[-1] else 'outside'} the {area} area"
         + (f" ({gap[-1]:.1f} cm away)" if gap[-1] else "") + ("" if moved > MOVED_M else "; it did not move")},
        {"ok": released[-1] or not rules["release"],
         "text": "Release: " + ("not required" if not rules["release"] else "released" if released[-1] else "still touching gripper")},
        {"ok": bool(long_enough and stable), "text": f"Stable: final {required:g} s; max speed"
         f" {max(final_speed) if final_speed else 0:.3f} m/s (limit {REST_SPEED_M_S:.3f})"},
    ]
    return {"success": success, "reason": reason, "checks": checks, "start": t[start], "end": t[-1], "dist": gap,
            "touch": touch, "object": obj, "target": area, "progress": -gap[-1] + (100 if moved > MOVED_M else 0),
            "metrics": {"object_moved_m": round(moved, 4), "distance_to_area_cm": round(gap[-1], 2)}}


def judge_lift(t, p, touch, obj, spec, rules, frame=DEFAULT_FRAME):
    """lift: the object rises height_cm above its start and stays that high for hold_seconds (in one stretch)."""
    n, need = len(t), rules["height_cm"]
    height = [(p[i][2] - p[0][2]) * 100 for i in range(n)]
    high = [h >= need for h in height]
    best = (0.0, None, None)  # longest stretch up there: (seconds, start index, end index)
    i = 0
    while i < n:
        if not high[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and high[j + 1]:
            j += 1
        if t[j] - t[i] > best[0] or best[1] is None:
            best = (t[j] - t[i], i, j)
        i = j + 1
    held, first, last = best
    success = first is not None and held >= rules["hold_seconds"] - 1e-9
    top = max(height)
    moved = max(math.dist(pos, p[0]) for pos in p)
    if success:
        reason = "lifted_and_held"
    elif first is not None:
        reason = "not_held_long_enough"
    elif top > LIFTED_M * 100 / 3:
        reason = "not_high_enough"
    elif moved > MOVED_M:
        reason = "moved_but_not_lifted"
    else:
        reason = "touched_but_not_lifted" if any(touch) else "object_not_moved"
    checks = [
        {"ok": first is not None, "text": f"Height: {obj} up to {top:.1f} cm above its start (need {need:.1f} cm)"},
        {"ok": success, "text": f"Hold: {held:.1f} s up there (need {rules['hold_seconds']:g} s)"},
    ]
    return {"success": success, "reason": reason, "checks": checks,
            "start": t[first] if first is not None else t[-1], "end": t[last] if last is not None else t[-1],
            "dist": height, "touch": touch, "object": obj, "target": "lift", "progress": top + 100 * held,
            "metrics": {"max_height_cm": round(top, 2), "held_s": round(held, 2), "object_moved_m": round(moved, 4)}}


def judge_push(t, p, touch, obj, spec, rules, frame=DEFAULT_FRAME):
    """push: the object ends distance_cm or more from its start, on the table (and never lifted, if required)."""
    n, need = len(t), rules["distance_cm"]
    moved = [math.hypot(p[i][0] - p[0][0], p[i][1] - p[0][1]) * 100 for i in range(n)]
    lift = max((pos[2] - p[0][2]) * 100 for pos in p)
    on_table = p[-1][2] > frame["off_table_z"]
    lifted = lift >= LIFTED_M * 100
    far = [m >= need for m in moved]
    start, _ = final_window(t, far, 0)
    success = far[-1] and on_table and not (rules["no_lift"] and lifted)
    if success:
        reason = "pushed"
    elif not on_table:
        reason = "object_fell_off_table"
    elif rules["no_lift"] and lifted and far[-1]:
        reason = "lifted_not_pushed"
    elif max(moved) * 1e-2 > MOVED_M:
        reason = "not_far_enough"
    else:
        reason = "touched_but_not_moved" if any(touch) else "object_not_moved"
    checks = [
        {"ok": far[-1], "text": f"Moved: {obj} ends {moved[-1]:.1f} cm from its start (need {need:.1f} cm)"},
        {"ok": on_table, "text": "On the table at the end" if on_table else "Fell off the table"},
        {"ok": not (rules["no_lift"] and lifted),
         "text": f"Lifting: {'not allowed' if rules['no_lift'] else 'allowed'}; highest {lift:.1f} cm above its start"},
    ]
    return {"success": success, "reason": reason, "checks": checks, "start": t[start], "end": t[-1], "dist": moved,
            "touch": touch, "object": obj, "target": "push", "progress": moved[-1],
            "metrics": {"moved_cm": round(moved[-1], 2), "max_lift_cm": round(lift, 2)}}


def judge_pair(t, p, target_positions, touch, obj, target, rules, frame=DEFAULT_FRAME):
    """Judge one object against one target with the rules.
    @return {success, reason, checks, start (when the final placement began), dist, touch, metrics, object, target}.
    """
    n = len(t)
    if frame["bands"] is not None:  # heights above the bowl / plate itself (another table)
        low, high = frame["bands"][target]
        bands = [(target_positions[i][2] + low, target_positions[i][2] + high) for i in range(n)]
    else:
        bands = [HEIGHT_BANDS[target]] * n
    dist = [math.hypot(p[i][0] - target_positions[i][0], p[i][1] - target_positions[i][1]) * 100
            for i in range(n)]
    in_target = [dist[i] < rules["distance_cm"] and bands[i][0] < p[i][2] < bands[i][1] for i in range(n)]
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
    elif min(pos[2] for pos in p) < frame["off_table_z"]:
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
    return {"success": success, "reason": reason, "checks": checks, "start": t[start], "dist": dist, "touch": touch,
            "object": obj, "target": target,
            "metrics": {"object_moved_m": round(moved, 4), "distance_to_target_cm": round(dist[-1], 2),
                        "max_final_speed_m_s": round(max(final_speed), 4) if final_speed else None}}
