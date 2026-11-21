"""Online failure monitor: decides, while a run is going, when Dream-1 needs a human and when the human is done.

It reads the simulator's ground truth (object and hand positions) every control step. Thresholds and task parsing
follow the team's judge.py (commit 98d9797); replace this with the shared stage judge once it exists.

What it watches: the object and target of the task's words ("put the apple into the bowl"), or, when the run has a
goal scene (judge.py "goal", set by dragging objects on the page), each object of the goal in turn, each with its
relation: in the bowl / on the plate, beside an item (that far from where the item is now: it may have moved), or a
fixed spot. Once an object is done the next one is watched; the run is done when all are.

Stages of the object being watched, in order:
    reach     the object has not been picked up yet
    carry     the object is lifted and next to a hand
    placed    the object is at its goal (in the target, or at its spot)
    success   placed, released and resting for SETTLE_S

When Dream-1 is in control, a failure event asks for a human:
    dropped         the object was carried, then fell back to the table outside the target
    stalled         no progress for STALL_S (no new stage, the hand not getting closer to the object)
    off_table       the object fell off the table: cannot be recovered, the run should end

Recovery goal shown to the human, and the moment to give the arms back (key T):
    failed before or while carrying (dropped, stalled in reach)  ->  object lifted in a hand again (stage carry)
    stalled while carrying or placed                             ->  task success
"""

from __future__ import annotations

import re

import numpy as np

OBJECTS = {"apple": "apple", "orange": "orange", "banana": "banana", "bottle": "water_bottle",
           "water bottle": "water_bottle"}
TARGETS = {"bowl": {"radius": 0.065, "z": (0.755, 0.83)}, "plate": {"radius": 0.065, "z": (0.76, 0.82)}}
TASK_RE = re.compile(
    r"\b(?:put|place|move|drop|set)\s+(?:the\s+|a\s+|an\s+)?(?P<obj>water bottle|bottle|apple|orange|banana)\s+"
    r"(?:into|in|onto|on|inside|in to|on to)\s+(?:the\s+|a\s+)?(?P<tgt>bowl|plate)\b",
    re.IGNORECASE,
)
STEP_S = 0.05
LIFTED_M = 0.03  # object this far above its resting height counts as lifted
NEAR_HAND_M = 0.07  # object centre this close to a hand (TCP), and touched by its fingertips: in that hand
FINGERTIPS = [f"{side}_{part}" for side in ("left", "right") for part in ("tip_left", "tip_right", "gripper")]
OFF_TABLE_Z = 0.70
REST_SPEED = 0.05  # m/s
SETTLE_S = 1.0
STALL_S = 12.0  # no progress this long: stalled
PROGRESS_M = 0.02  # the hand got this much closer to the object than ever before: progress
STAGES = ["reach", "carry", "placed", "success"]
GOALS = {"carry": "pick the {obj} up again (lift it in a gripper)", "success": "put the {obj} {where}"}
TABLE_Z = 0.76  # table top (world metres)
SPOT_TOLERANCE_CM = 4.0  # a spot beside an item / on the table is reached this close (judge.py goal tolerance_cm)
SPOT_HEIGHT_M = 0.03  # ... and at its resting height (judge.py GOAL_HEIGHT_M)


def parse_task(task):
    m = TASK_RE.search(task)
    return (OBJECTS[m.group("obj").lower()], m.group("tgt").lower()) if m else None


def where_words(entry):
    """How a goal entry reads: "into the bowl", "12 cm left of the apple", "at its spot"."""
    if "in" in entry:
        return "into the bowl" if entry["in"] == "bowl" else "onto the plate"
    if "near" in entry:
        left, front = entry["offset_cm"]
        parts = [f"{abs(left):.0f} cm {'left' if left > 0 else 'right'}"] if abs(left) >= 1 else []
        parts += [f"{abs(front):.0f} cm {'forward' if front > 0 else 'back'}"] if abs(front) >= 1 else []
        return f"{' and '.join(parts) or 'right at'} from the {entry['near'].replace('water_', '')}"
    return "at its spot on the table"


class Monitor:
    def __init__(self, task, env, goal=None, tolerance_cm=None):
        """@param goal A goal-scene judge spec ({"type": "goal", "objects": {...}}), or None for the task's words.
        @param tolerance_cm How close a spot beside an item / on the table must be reached."""
        self.task = task
        self.goal_scene = bool(goal and goal.get("type") == "goal")
        if self.goal_scene:
            self.plan = list(goal["objects"].items())
        else:
            parsed = parse_task(task)
            self.plan = [(parsed[0], {"in": parsed[1]})] if parsed else []
        self.tolerance = (tolerance_cm or SPOT_TOLERANCE_CM) / 100
        self.env = env
        self.done = []  # objects at their goal, in the order they got there
        self.t = 0.0
        self.event = None  # the failure that asked for help, until the human gives the arms back
        self.goal = None
        self.goal_reached = False
        self._watch(env)

    def _watch(self, env):
        """Watch the next object not at its goal yet (the last one again when all are)."""
        left = [obj for obj, _ in self.plan if obj not in self.done]
        self.object = left[0] if left else (self.plan[-1][0] if self.plan else None)
        self.entry = dict(self.plan).get(self.object)
        self.target = (self.entry.get("in") or self.entry.get("near")) if self.entry else None
        self.rest_z = float(env.data.body(self.object).xpos[2]) if self.object else None
        self.stage = "success" if self.object in self.done else "reach"
        self.best_stage = self.stage
        self.last_progress_t = self.t
        self.closest = np.inf
        self.placed_since = None
        self.previous = None

    def understood(self):
        return self.object is not None

    def is_done(self, obj):
        return obj in self.done

    def spot(self, env, obj=None):
        """Where an object goes: its container's origin, or its spot (beside its item as the item is now) on the
        table. @return [x, y, z] (z: the container's origin or the table top)."""
        entry = dict(self.plan)[obj or self.object]
        if "in" in entry:
            return env.data.body(entry["in"]).xpos.copy()
        if "near" in entry:
            item = env.data.body(entry["near"]).xpos
            return np.array([item[0] + entry["offset_cm"][0] / 100, item[1] + entry["offset_cm"][1] / 100, TABLE_Z])
        return np.array([entry["at"][0], entry["at"][1], TABLE_Z])

    def at_goal(self, env, position, obj=None):
        entry = dict(self.plan)[obj or self.object]
        if "in" in entry:
            target, centre = TARGETS[entry["in"]], env.data.body(entry["in"]).xpos
            # Anywhere in the bowl / on the plate (its own size, as judge.py IN_RADIUS_CM), whatever the distance rule.
            radius = 0.07 if self.goal_scene else target["radius"]
            return (np.linalg.norm(position[:2] - centre[:2]) < radius
                    and target["z"][0] < position[2] < target["z"][1])
        spot = self.spot(env, obj)
        height = entry["z"] if "near" in entry else entry["at"][2]
        return np.linalg.norm(position[:2] - spot[:2]) < self.tolerance and abs(position[2] - height) < SPOT_HEIGHT_M

    def update(self, env, watching=True):
        """One control step. @param watching Whether failures are looked for (Dream-1 has the arms).
        @return A new failure event name when one happens, else None."""
        if not self.understood():
            return None
        self.t += STEP_S
        position = env.data.body(self.object).xpos.copy()
        hands = [env.data.site(f"{side}_tcp").xpos for side in ("left", "right")]
        hand_distance = min(np.linalg.norm(position - h) for h in hands)
        speed = 0.0 if self.previous is None else np.linalg.norm(position - self.previous) / STEP_S
        self.previous = position
        inside = self.at_goal(env, position)
        lifted = position[2] > self.rest_z + LIFTED_M
        # In a hand: near it and touched by its fingertips (an open hand that only hovers near it does not count,
        # e.g. one that could not retreat far after setting the object down).
        in_hand = hand_distance < NEAR_HAND_M and self._touched(env)

        before = self.stage
        if inside and not in_hand and speed < REST_SPEED:
            self.placed_since = self.t if self.placed_since is None else self.placed_since
            self.stage = "success" if self.t - self.placed_since >= SETTLE_S else "placed"
            if self.stage == "success" and self.object not in self.done:
                self.done.append(self.object)
                if len(self.done) < len(self.plan) and self.goal is None:
                    self._watch(env)  # on to the next object of the goal scene
                    return None
        else:
            self.placed_since = None
            self.stage = "placed" if inside else "carry" if (lifted and in_hand) else "reach"

        # Progress: a stage not reached before, or the hand closer to the object than ever.
        if STAGES.index(self.stage) > STAGES.index(self.best_stage):
            self.best_stage = self.stage
            self.last_progress_t = self.t
        if hand_distance < self.closest - PROGRESS_M:
            self.closest = hand_distance
            self.last_progress_t = self.t

        if self.goal is not None:
            self.goal_reached = STAGES.index(self.stage) >= STAGES.index(self.goal)

        if self.event is not None or not watching:
            if not watching and self.event is None:
                self.last_progress_t = self.t  # stall time counts only while Dream-1 has the arms
            return None
        if position[2] < OFF_TABLE_Z:
            return self._fail("off_table", None)
        if before == "carry" and self.stage == "reach" and not lifted:
            return self._fail("dropped", "carry")
        if self.stage != "success" and self.t - self.last_progress_t > STALL_S:
            return self._fail("stalled", "carry" if self.stage == "reach" else "success")
        return None

    def _touched(self, env):
        m, d = env.model, env.data
        body = m.body(self.object).id
        tips = {m.body(name).id for name in FINGERTIPS}
        for contact in d.contact[: d.ncon]:
            a, b = m.geom_bodyid[contact.geom1], m.geom_bodyid[contact.geom2]
            if (a in tips and b == body) or (b in tips and a == body):
                return True
        return False

    def recovered(self):
        """The human gave the arms back: watch for the next failure from here."""
        self.event, self.goal, self.goal_reached = None, None, False
        if self.object in self.done and len(self.done) < len(self.plan):
            self._watch(self.env)  # the recovery finished this object: on to the next one
            return
        self.last_progress_t = self.t
        self.closest = np.inf
        self.best_stage = self.stage

    def status(self):
        name = self.object.replace("water_", "") if self.object else None
        where = where_words(self.entry) if self.entry else ""
        return {
            "understood": self.understood(),
            "stage": self.stage,
            "object": self.object,
            "done": list(self.done),
            "objects": [obj for obj, _ in self.plan],
            "event": self.event,
            "goal": GOALS[self.goal].format(obj=name, where=where) if self.goal else None,
            "goal_reached": self.goal_reached,
            "seconds_without_progress": round(self.t - self.last_progress_t, 1),
        }

    def _fail(self, event, goal):
        self.event, self.goal, self.goal_reached = event, goal, False
        return event
