"""Scripted recovery: when Dream-1 fails, a script tries to recover first; a human is asked only if it cannot.

Used by sim.py. The failure monitor (monitor.py) detects the failure and sets the recovery goal; this script then
drives the arms through teleop.py (the same hand targets, IK and smooth motion the human uses, in mode "script").
It reads the simulator's ground truth (object pose) every step, so changes are followed:
    - the object moves while the hand approaches: the hand re-aims at the new position;
    - it moves while the hand goes down, or an elongated object (banana) turns: the attempt starts again
      (the hand must go back up and turn first);
    - every new attempt computes everything from the scene as it is then.

Recovery goal "carry" (pick the object up again):
    prepare   open the gripper; lift the hand to above the object's height if it is lower
    point     point the hand down (for an elongated object such as the banana: across its length)
    approach  10 cm above the grasp point
    descend   down to the grasp point
    close     close the gripper
    lift      12 cm up
    lifted?   the monitor sees the object lifted in a hand (stage carry)
Recovery goal "success" (finish the task): the steps above unless the object is already held, then
    transport above its goal: the bowl or plate, or its spot beside an item / on the table (goal scene, monitor.spot)
    set_down  down until the object's bottom reaches the target
    open      open the gripper
    retreat   12 cm up (the monitor counts a placed object as done only once the hand has left it)
    placed?   the monitor sees the task done (stage success)

Relay (goal "success"), when no arm reaches both the object and its goal (one reaches the object, only the other the
goal): the first arm sets the object down at a free spot on the table both arms reach, and the other arm picks it
up there and finishes:
    pick (first arm)  relay  set_down  open  retreat  switch  pick (other arm)  transport ... placed?

An attempt fails when a step takes longer than STEP_TIMEOUT_S, the arm cannot reach a target, or a check fails.
After a failed attempt the script tries again from the current scene, up to RETRIES times; then it gives up and
the arms are held for a human (mode paused), as without the script. The human can take over at any time (T, or
a click on the picture).

A water bottle lying on its side is grasped like the banana: from above, across its length, at its middle, and
set down lying. Standing it back up is left to a human.
"""

from __future__ import annotations

import numpy as np

from teleop import (ABOVE_OBJECT_M, ABOVE_PLACE_M, BOUNDS, DOWN, GRASP_DZ, GRIP_SECONDS, LIFT_M, LONG_AXIS,
                    RELEASE_M, SIDES, STEP_S, TABLE_Z, yaw)

RETRIES = 2  # failed attempts tried again before the human is asked (so at most 1 + RETRIES attempts)
STEP_TIMEOUT_S = 20.0  # a step that takes longer fails the attempt (the script moves at Dream-1's slow pace)
ARRIVED_M = 0.01  # the hand target is this close to where it should go
SETTLE_S = 0.3  # after arriving, wait this long so the arm (position servos) catches up with the target
REAIM_M = 0.01  # the object moved this much since the hand was aimed at it: aim again
CHECK_S = 2.5  # how long a check waits for the monitor to see the expected stage
APPROACH_HEIGHTS_M = (ABOVE_OBJECT_M, 0.07, 0.05)  # the hand starts down from the first one it can reach
LIFT_HEIGHTS_M = (LIFT_M, 0.08, 0.05)  # lift / retreat: the first height the arm can reach
# Transport: the hand goes above the target at the first of these heights the arm can reach (a target close to the
# robots' base, such as a bowl right in front of them, cannot be reached 12 cm up by either arm, but can lower).
PLACE_HEIGHTS_M = (ABOVE_PLACE_M, 0.10, 0.08, 0.06)
PLACE_SLACK_M = 0.03  # transport / set_down: a hand that stops this close (sideways) above the target is close enough
DROP_M = 0.05  # set_down: ... and the object's bottom at most this far above where it would rest
REACH_TURNS = np.linspace(0.0, np.pi, 4, endpoint=False)  # hand turns tried when checking a point can be reached
TURNED_DEG = 20.0  # an elongated object (banana) turned this much after the hand was turned for it: attempt again
UPRIGHT_COS = 0.8  # a water bottle whose axis is tilted further than ~37 degrees from vertical is lying down
LYING_BOTTLE_DZ = 0.0  # a lying bottle is grasped at its middle (the body origin is the middle of the cylinder)
PICK = ["prepare", "point", "approach", "descend", "close", "lift", "lifted?"]
PLACE = ["transport", "set_down", "open", "retreat", "placed?"]
RELAY = ["relay", "set_down", "open", "retreat", "switch"]  # set it down where the other arm can pick it up
ITEMS = ["apple", "orange", "banana", "water_bottle", "bowl", "plate"]
RELAY_CLEAR_M = 0.09  # a relay spot is this far (on the table) from every other item
RELAY_GRID = [(x, y) for x in np.linspace(-0.16, 0.16, 9) for y in np.linspace(-0.16, 0.20, 10)]


def lying(env, obj):
    """A water bottle on its side."""
    return obj == "water_bottle" and env.data.body(obj).xmat.reshape(3, 3)[2, 2] < UPRIGHT_COS


def long_axis(env, obj):
    """The body axis (0 x, 1 y, 2 z) along an elongated object's length, or None for a round or standing one."""
    if lying(env, obj):
        return 2  # the bottle's cylinder is along its z axis
    return LONG_AXIS.get(obj)


def grasp_dz(env, obj):
    """Grasp height above the object's body origin."""
    return LYING_BOTTLE_DZ if lying(env, obj) else GRASP_DZ.get(obj, 0.0)


def clip(position):
    """A hand position inside the box teleop allows (teleop.BOUNDS), as teleop itself would clip it."""
    return np.clip(position, BOUNDS[:, 0], BOUNDS[:, 1])


class Failed(Exception):
    """The current attempt failed (the message says why)."""


class Recovery:
    def __init__(self, teleop, monitor):
        self.teleop = teleop
        self.monitor = monitor
        self.active = False
        self.event = None  # the failure being recovered from
        self.goal = None  # "carry" or "success" (the monitor's recovery goal)
        self.attempt = 0
        self.steps = []
        self.index = 0
        self.step_t = 0.0  # seconds in the current step
        self.wait_t = 0.0  # seconds waited after arriving / for a check
        self.entered = False
        self.side = None
        self.aim = None  # object position the current hand target was computed from
        self.aim_heading = None  # an elongated object's direction the hand was turned for
        self.approach_m = ABOVE_OBJECT_M  # height above the grasp point the hand starts down from
        self.other_arm = False  # use the arm farther from the object (after the nearer one could not reach)
        self.release = RELEASE_M  # set-down height above the target's origin
        self.place_point = None
        self.relay = None  # the relay spot on the table, when one arm hands the object to the other (RELAY)
        self.forced = None  # the arm the next pick must use (a relay's first or second arm)
        self.placer = None  # a relay's second arm
        self.result = None  # "recovered", "gave_up: <why>", "human took over"
        self.message = ""
        self.results = []  # (event, goal, result, attempts, seconds), one per recovery, for statistics
        self.t = 0.0

    # ---- called by sim.py ---------------------------------------------------------------------------------------

    def start(self, env, event, goal):
        """The monitor reported a failure with a recovery goal: take the arms and start the first attempt."""
        self.event, self.goal = event, goal
        self.obj = self.monitor.object  # the object this recovery is for (the monitor may move on to the next one)
        self.attempt, self.t, self.result, self.side = 0, 0.0, None, None
        self.other_arm = False
        self.active = True
        self.teleop.start_script(env, event)
        print(f"  Recovery: script starts ({event}, goal {goal})", flush=True)
        self._plan(env)

    def tick(self, env):
        """One control step (after env.step and the monitor's update)."""
        if not self.active:
            return
        if self.teleop.mode != "script":  # the human took over (T or a click)
            self._finish("human took over")
            return
        self.t += STEP_S
        self.step_t += STEP_S
        try:
            if self.step_t > STEP_TIMEOUT_S:
                raise Failed(f"{self.steps[self.index]} took longer than {STEP_TIMEOUT_S:g} s")
            if self._step(env):
                self.index += 1
                self.step_t, self.wait_t, self.entered = 0.0, 0.0, False
                if self.index == len(self.steps):
                    self._finish("recovered")
                    self.teleop.end_script(recovered=True)
        except Failed as failure:
            self.attempt += 1
            if "cannot reach" in str(failure) or "cannot point" in str(failure):
                self.other_arm = not self.other_arm
            print(f"  Recovery: attempt {self.attempt} failed: {failure}", flush=True)
            if self.attempt > RETRIES:
                self._give_up(env, str(failure))
            else:
                self._plan(env)

    def status(self):
        return {
            "active": self.active,
            "event": self.event,
            "goal": self.goal,
            "step": self.steps[self.index] if self.active and self.index < len(self.steps) else None,
            "attempt": self.attempt + 1 if self.active else self.attempt,
            "max_attempts": 1 + RETRIES,
            "arm": self.side,
            "result": self.result,
            "message": self.message,
        }

    # ---- attempts -----------------------------------------------------------------------------------------------

    def _plan(self, env):
        """Steps for a new attempt, from the scene as it is now."""
        held = self._held_by(env)
        self.relay, self.forced, self.placer = None, None, None
        relay = self._relay_plan(env, held) if self.goal == "success" else None
        if self.goal == "success" and held:
            self.side = held
            self.steps = RELAY + PICK + PLACE if relay else list(PLACE)
        else:
            self.side = None
            self.steps = (PICK + RELAY + PICK + PLACE if relay
                          else PICK + (PLACE if self.goal == "success" else []))
        if relay:
            self.relay, self.forced, self.placer = relay
            print(f"  Recovery: relay: the {self.forced} arm sets the {self.obj} down at "
                  f"({self.relay[0]:.2f}, {self.relay[1]:.2f}) for the {self.placer} arm", flush=True)
        self.index, self.step_t, self.wait_t, self.entered = 0, 0.0, 0.0, False
        self.aim, self.aim_heading = None, None
        self.message = f"attempt {self.attempt + 1} of {1 + RETRIES}"

    def _give_up(self, env, why):
        self._finish(f"gave up: {why}")
        self.teleop.end_script(recovered=False, reason=f"script gave up ({self.event})")
        print("  Recovery: the script gave up; the arms are held for a human.", flush=True)

    def _finish(self, result):
        self.active = False
        self.result = result
        self.message = result
        self.results.append((self.event, self.goal, result, self.attempt + (result == "recovered"), round(self.t, 1)))
        print(f"  Recovery: {result} ({self.t:.1f} s)", flush=True)

    def _step(self, env):
        """Run the current step. @return True when it is done. @raise Failed when the attempt cannot go on."""
        name = self.steps[self.index]
        first = not self.entered
        self.entered = True
        t = self.teleop
        obj = self.obj
        with t.lock:
            if name == "prepare":
                if first:
                    position = env.data.body(obj).xpos
                    nearer = "left" if position[0] > 0 else "right"
                    # The arm that can reach both the object and where it goes (its goal may be on the other side:
                    # the plate, a spot beside an item); the nearer one first. After an attempt the arm could not
                    # reach, the next one uses the other arm.
                    order = [nearer, "right" if nearer == "left" else "left"]
                    if self.monitor.understood():  # also when only picking it up: it goes there next
                        spot = self.monitor.spot(env, obj)
                        able = [s for s in order if self._picks(env, s, position)
                                and self._place_height(env, s, spot) is not None]
                        order = able + [s for s in order if s not in able]
                    self.side = self.forced or (order[0] if not self.other_arm else order[1])
                    t.active = self.side
                    t.grip[self.side] = 1.0
                    hand = t.targets[self.side][0]
                    clear_z = position[2] + grasp_dz(env, obj) + ABOVE_OBJECT_M
                    if hand[2] < clear_z:
                        t.destinations[self.side] = clip(np.r_[hand[:2], clear_z])
                # The lift may be refused (a level hand can barely move); pointing down next fixes that.
                return self._gripper_done() and self.side not in t.destinations and self._settled()
            if name == "point":
                if first and not self._orient(env, self.side, obj):
                    raise Failed(f"the {self.side} hand cannot point down at the {obj} from here")
                return self._joints_done() and self._settled()
            if name in ("approach", "descend"):
                if self._turned(env, obj):  # the hand was turned for the old direction: start the attempt again
                    raise Failed(f"the {obj} turned more than {TURNED_DEG:g} degrees")
                grasp = self._grasp_point(env, obj)
                destination = clip(grasp + [0, 0, self.approach_m] if name == "approach" else grasp)
                if first or np.linalg.norm(grasp - self.aim) > REAIM_M:  # the object moved: aim again
                    self.aim = grasp
                    if not first and name == "descend" and long_axis(env, obj) is not None:
                        # An elongated object that slid must be grasped across its new direction: start again. A round
                        # one (apple, orange, a standing bottle) is followed down where it went.
                        raise Failed(f"the {obj} moved while the hand was going down")
                    t.destinations[self.side] = destination
                return self._arrived(destination, name)
            if name == "close":
                t.grip[self.side] = 0.0
                return self._gripper_done() and self._waited(0.2)
            if name in ("lift", "retreat"):
                if first:
                    # As high as the arm can reach, up to LIFT_M (a hand holding a tall bottle cannot go 12 cm up).
                    position, rotation = t.targets[self.side]
                    for height in LIFT_HEIGHTS_M:
                        self.aim = clip(position + [0, 0, height])
                        if t._ik(env, self.side, (self.aim, rotation), [t.goals[self.side], t.joints[self.side]])[1]:
                            break
                    t.destinations[self.side] = self.aim
                return self._arrived(self.aim, name)
            if name == "switch":  # the relay is done: the other arm picks the object up from the relay spot
                self.side, self.forced = self.placer, self.placer
                self.aim, self.aim_heading = None, None
                return True
            if name in ("transport", "relay"):
                if first:
                    # Its goal: the bowl / plate, or its spot beside an item (as the item is now) or on the table;
                    # for a relay, the relay spot on the table.
                    target = self.relay if name == "relay" else self.monitor.spot(env, obj)
                    # As a click on the target does (teleop.clicked): set down until the object's bottom reaches it.
                    height = t.grasp_height.get(self.side)
                    self.release = height - 0.02 if height is not None else RELEASE_M
                    self.place_point = target
                    # Above it at the first height the arm reaches: high enough for the object to clear the target's
                    # rim if it can, else lower (a target right in front of the robots' base).
                    heights = [max(h, self.release + 0.05) for h in PLACE_HEIGHTS_M] + list(PLACE_HEIGHTS_M)
                    above = next((h for h in heights if self._reaches(env, self.side, target + [0, 0, h])),
                                 max(ABOVE_PLACE_M, self.release + 0.05))
                    self.aim = clip(target + [0, 0, above])
                    t.active = self.side
                    t.destinations[self.side] = self.aim
                return self._arrived(self.aim, name)
            if name == "set_down":
                if first:
                    # Never below the hand's lowest allowed height (teleop.BOUNDS): the object then rests on the target.
                    self.aim = clip(self.place_point + [0, 0, self.release])
                    t.destinations[self.side] = self.aim
                return self._arrived(self.aim, name)
            if name == "open":
                t.grip[self.side] = 1.0
                return self._gripper_done() and self._waited(0.3)
        # Checks (the monitor's stage; no lock needed).
        if name == "lifted?":
            if self.monitor.stage in ("carry", "placed", "success"):
                return self._waited(SETTLE_S)
            if self._waited(CHECK_S):
                raise Failed(f"the {obj} was not lifted")
            return False
        if name == "placed?":
            if self.monitor.is_done(obj) or self.monitor.stage == "success":
                return True
            if self._waited(CHECK_S):
                raise Failed(f"the {obj} is not resting at its goal")
            return False
        raise ValueError(name)

    # ---- helpers (teleop lock held where they touch teleop) -----------------------------------------------------

    def _reaches(self, env, side, point):
        """Whether the arm can hold its hand pointing down at a point (IK only, from a few starts)."""
        t = self.teleop
        limits = env.model.jnt_range[[env.model.joint(f"{side}_joint{i}").id for i in range(1, 7)]]
        rng = np.random.default_rng(1)
        seeds = [t.joints[side]] + [rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(5)]
        point = clip(point)
        return any(t._ik(env, side, (point, yaw(turn) @ DOWN), seeds)[1] for turn in REACH_TURNS)

    def _picks(self, env, side, position):
        """Whether the arm can hold its hand pointing down above an object at one of the approach heights."""
        return any(self._reaches(env, side, position + [0, 0, h]) for h in APPROACH_HEIGHTS_M)

    def _relay_plan(self, env, held):
        """A relay when no arm reaches both the object and its goal: (spot, first arm, second arm), or None (one arm
        can do it alone, or no relay spot works)."""
        position, goal = env.data.body(self.obj).xpos.copy(), self.monitor.spot(env, self.obj)
        pickers = [held] if held else [s for s in SIDES if self._picks(env, s, position)]
        if not pickers or any(self._place_height(env, s, goal) is not None for s in pickers):
            return None
        for first in pickers:
            second = "left" if first == "right" else "right"
            if self._place_height(env, second, goal) is None:
                continue
            spot = self._relay_spot(env, first, second, position, goal)
            if spot is not None:
                return spot, first, second
        return None

    def _relay_spot(self, env, first, second, position, goal):
        """A free spot on the table the first arm can set the object down at and the second can pick it up from;
        the one nearest the middle of the object and its goal first."""
        others = [env.data.body(n).xpos[:2] for n in ITEMS if n != self.obj]
        middle = (position[:2] + goal[:2]) / 2
        for x, y in sorted(RELAY_GRID, key=lambda p: np.linalg.norm(np.array(p) - middle)):
            xy = np.array([x, y])
            if min(np.linalg.norm(xy - o) for o in others) < RELAY_CLEAR_M:
                continue
            table, resting = np.r_[xy, TABLE_Z], np.r_[xy, position[2]]
            if self._place_height(env, first, table) is not None and self._picks(env, second, resting):
                return table
        return None

    def _place_height(self, env, side, spot):
        """The first height of PLACE_HEIGHTS_M above a target at which the arm can hold its hand pointing down, or
        None when it reaches none."""
        return next((h for h in PLACE_HEIGHTS_M if self._reaches(env, side, spot + [0, 0, h])), None)

    def _grasp_point(self, env, obj):
        grasp = env.data.body(obj).xpos + [0, 0, grasp_dz(env, obj)]
        self.teleop.grasp_height[self.side] = grasp[2] - TABLE_Z  # used to set the object down at the right height
        return grasp

    def _orient(self, env, side, obj):
        """Point the hand down, turned to grasp the object (across its length if it is elongated). Several hand
        turns are tried (the grasp is symmetric), the closest reachable one wins. @return False if none is."""
        t = self.teleop
        if long_axis(env, obj) is not None:
            self.aim_heading = self._heading(env, obj)
            across = self.aim_heading + np.pi / 2  # DOWN closes the fingers along world x
            turns = [across, across + np.pi]
        else:
            turns = [0.0, np.pi / 2, -np.pi / 2, np.pi]
        # Turn the hand where it will start the approach (above the object): a hand far above the object (a tall
        # bottle) may not be able to point down where it is. If that fails, where the hand is now.
        grasp = env.data.body(obj).xpos + [0, 0, grasp_dz(env, obj)]
        # Lower approach heights are tried too: a downward hand cannot reach far above a tall object (bottle).
        starts = [(clip(grasp + [0, 0, h]), h) for h in APPROACH_HEIGHTS_M] + [(t.targets[side][0].copy(), None)]
        limits = env.model.jnt_range[[env.model.joint(f"{side}_joint{i}").id for i in range(1, 7)]]
        rng = np.random.default_rng(0)
        seeds = [t.joints[side]] + [rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(8)]
        for position, height in starts:
            found = []
            for turn in turns:
                rotation = yaw(turn) @ DOWN
                for seed in seeds:
                    q, ok = t._ik(env, side, (position, rotation), [seed])
                    if ok:
                        margin = np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))
                        found.append((np.linalg.norm(q - t.joints[side]) - 2.0 * margin, q, rotation))
            if found:
                # The joints move there smoothly (teleop limits joint speed); prepare lifted the hand clear first.
                _, q, rotation = min(found, key=lambda item: item[0])
                t.goals[side] = q
                t.targets[side] = (position, rotation)
                self.approach_m = height if height is not None else ABOVE_OBJECT_M
                return True
        return False

    @staticmethod
    def _heading(env, obj):
        """Direction of an elongated object's length on the table, radians."""
        axis = env.data.body(obj).xmat.reshape(3, 3)[:, long_axis(env, obj)]
        return float(np.arctan2(axis[1], axis[0]))

    def _turned(self, env, obj):
        """An elongated object turned on the table since the hand was turned to grasp it (its ends are alike, so
        a half turn counts as no turn)."""
        if long_axis(env, obj) is None or self.aim_heading is None:
            return False
        change = (self._heading(env, obj) - self.aim_heading) % np.pi
        return np.degrees(min(change, np.pi - change)) > TURNED_DEG

    def _arrived(self, destination, name):
        t = self.teleop
        if self.side in t.destinations:
            return False
        hand = t.targets[self.side][0]
        if name in ("transport", "relay", "set_down") and self.place_point is not None:
            # At the edge of the arm's reach (a target right in front of the robots' base) the hand may stop short:
            # close enough above the target, the object is set down (or let go a little higher) from there.
            sideways = np.linalg.norm(hand[:2] - self.place_point[:2])
            high = hand[2] - (self.place_point[2] + self.release)
            if sideways < PLACE_SLACK_M and (name != "set_down" or high < DROP_M):
                return self._joints_done() and self._settled()
        if np.linalg.norm(hand - destination) > ARRIVED_M:
            # teleop dropped the destination: IK cannot reach it (reach or joint limit), or it lies outside BOUNDS.
            raise Failed(f"{name}: the {self.side} arm cannot reach there")
        return self._joints_done() and self._settled()

    def _joints_done(self):
        t = self.teleop
        return np.max(np.abs(t.goals[self.side] - t.joints[self.side])) < 1e-3

    def _gripper_done(self):
        t = self.teleop
        return abs(t.grip_now[self.side] - t.grip[self.side]) < 1e-3

    def _settled(self):
        return self._waited(SETTLE_S)

    def _waited(self, seconds):
        self.wait_t += STEP_S
        return self.wait_t >= seconds

    def _held_by(self, env):
        """The arm whose closed gripper holds the object, or None."""
        if self.monitor.stage not in ("carry", "placed"):
            return None
        obj = env.data.body(self.obj).xpos
        t = self.teleop
        with t.lock:
            closed = [s for s in SIDES if t.grip[s] < 0.5]
        if not closed:
            return None
        return min(closed, key=lambda s: np.linalg.norm(env.data.site(f"{s}_tcp").xpos - obj))
