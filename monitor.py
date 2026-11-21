"""Online failure monitor: decides, while a run is going, when Dream-1 needs a human and when the human is done.

It reads the simulator's ground truth (object and hand positions) every control step. Thresholds and task parsing
follow the team's judge.py (commit 98d9797); replace this with the shared stage judge once it exists.

Stages of a pick-and-place task, in order:
    reach     the object has not been picked up yet
    carry     the object is lifted and next to a hand
    placed    the object is inside the target region
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
NEAR_HAND_M = 0.07  # object centre this close to a hand (TCP) counts as in that hand
OFF_TABLE_Z = 0.70
REST_SPEED = 0.05  # m/s
SETTLE_S = 1.0
STALL_S = 12.0  # no progress this long: stalled
PROGRESS_M = 0.02  # the hand got this much closer to the object than ever before: progress
STAGES = ["reach", "carry", "placed", "success"]
GOALS = {"carry": "pick the {obj} up again (lift it in a gripper)", "success": "finish the task: {task}"}


def parse_task(task):
    m = TASK_RE.search(task)
    return (OBJECTS[m.group("obj").lower()], m.group("tgt").lower()) if m else None


class Monitor:
    def __init__(self, task, env):
        self.task = task
        parsed = parse_task(task)
        self.object, self.target = parsed if parsed else (None, None)
        self.rest_z = float(env.data.body(self.object).xpos[2]) if self.object else None
        self.t = 0.0
        self.stage = "reach"
        self.best_stage = "reach"
        self.last_progress_t = 0.0
        self.closest = np.inf
        self.placed_since = None
        self.previous = None
        self.event = None  # the failure that asked for help, until the human gives the arms back
        self.goal = None
        self.goal_reached = False

    def understood(self):
        return self.object is not None

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
        target = TARGETS[self.target]
        target_pos = env.data.body(self.target).xpos
        inside = (np.linalg.norm(position[:2] - target_pos[:2]) < target["radius"]
                  and target["z"][0] < position[2] < target["z"][1])
        lifted = position[2] > self.rest_z + LIFTED_M
        in_hand = hand_distance < NEAR_HAND_M

        before = self.stage
        if inside and not in_hand and speed < REST_SPEED:
            self.placed_since = self.t if self.placed_since is None else self.placed_since
            self.stage = "success" if self.t - self.placed_since >= SETTLE_S else "placed"
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

    def recovered(self):
        """The human gave the arms back: watch for the next failure from here."""
        self.event, self.goal, self.goal_reached = None, None, False
        self.last_progress_t = self.t
        self.closest = np.inf
        self.best_stage = self.stage

    def status(self):
        name = self.object.replace("water_", "") if self.object else None
        return {
            "understood": self.understood(),
            "stage": self.stage,
            "event": self.event,
            "goal": GOALS[self.goal].format(obj=name, task=self.task) if self.goal else None,
            "goal_reached": self.goal_reached,
            "seconds_without_progress": round(self.t - self.last_progress_t, 1),
        }

    def _fail(self, event, goal):
        self.event, self.goal, self.goal_reached = event, goal, False
        return event
