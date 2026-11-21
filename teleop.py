"""Human control of the two simulated YAM arms: click where the hand should go, or steer it with keys.

Used by sim.py. Every 50 ms control step turns the hand (TCP) targets into the seven values env.step() expects
(six joint angles + gripper opening) by IK.

Mouse (the camera pictures on the website):
    click an object      the nearer arm points its hand down and moves 10 cm above the object
                         (in a wrist camera picture, that camera's arm; drag works in the top camera only)
    click the bowl/table the arm holding something (else the nearer one) moves 12 cm above that point
    Lower                down to grasp height (or to release height above the bowl/plate)
    Grip                 close / open the gripper
    Lift                 12 cm up
Keys (fine adjustment, directions as seen in the top camera):
    W / S  away from / toward the camera    A / D  left / right in the picture    R / F  up / down
    Q / E  turn the hand    Z / X  tilt the hand    G  grasp from above <-> tilted (from the side)
    Space  gripper    Tab  switch arm    V  point the hand down    T  take over / give back (in a human-only run: hand the arms to Dream-1 from here)
    H      the active arm back to its start pose (lifted clear of the table first; the gripper stays as it is)

Modes:
    dream       Dream-1's actions go to the simulator unchanged.
    paused      the failure monitor asked for help: the arms hold still, Dream-1 is not called, waiting for the human.
    human       the human's actions go to the simulator; Dream-1 is not called but keeps receiving camera frames.
    script      the failure monitor reported a failure and the recovery script (recovery.py) moves the arms through
                the same hand targets a human would; Dream-1 is not called. It ends in returning (recovered) or
                paused (it gave up after its retries). T or a click on the picture takes over from it.
    returning   the human (or the recovery script) gave the arms back: they hold still until Dream-1 starts its next chunk, so it never
                executes the rest of a chunk planned before the human moved the arms.
"""

from __future__ import annotations

import threading

import mujoco
import numpy as np

SIDES = ("left", "right")
STEP_S = 0.05  # one control step (20 Hz)
SPEED_M_S = 0.12  # hand speed while a direction key is held
GOTO_SPEED_M_S = 0.2  # hand speed toward a clicked or dragged-to point
TURN_RAD_S = 1.2  # hand turn speed while Q or E is held
ACCEL_M_S2 = 0.8  # the hand speeds up and slows down at this rate (no jerky starts and stops)
MAX_JOINT_STEP = 0.08  # rad per control step for the joint that moves most; the others are scaled with it
JOINT_ACCEL_RAD_S2 = 6.0  # joints speed up and slow down at this rate (large moves such as turning the hand down)
GRIP_SECONDS = 0.6  # a full open <-> close takes this long
# The recovery script (mode script) moves at Dream-1's pace, not a human's, so its recovery segments look like the
# policy's own motion. Measured on two Dream-1 runs (episodes 20261121-175207-1983 and -185725-5602): hand speed while
# moving median 0.03 m/s, 90th percentile 0.07 m/s; fastest joint 90th percentile 0.24 rad/s.
SCRIPT_SPEED_M_S = 0.07  # hand speed toward a target (human: GOTO_SPEED_M_S)
SCRIPT_ACCEL_M_S2 = 0.25  # (human: ACCEL_M_S2)
SCRIPT_MAX_JOINT_RAD_S = 0.3  # fastest joint (human: MAX_JOINT_STEP / STEP_S = 1.6 rad/s)
SCRIPT_JOINT_ACCEL_RAD_S2 = 1.5  # (human: JOINT_ACCEL_RAD_S2)
SCRIPT_GRIP_SECONDS = 1.5  # full open <-> close (human: GRIP_SECONDS)
IK_ITERATIONS = 30
IK_TOLERANCE_M = 0.005  # a hand target IK cannot reach this closely is refused (reach or joint limits)
# The hand may not go below the table or outside this box (world metres).
BOUNDS = np.array([[-0.55, 0.55], [-0.35, 0.30], [0.775, 1.25]])
MOVES = {"w": (1, -1), "s": (1, 1), "a": (0, 1), "d": (0, -1), "r": (2, 1), "f": (2, -1)}
# Pointing straight down: the TCP's z axis along world -z (as the sim's scripted grasps use).
DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# Grasp height relative to each object's body origin (from the team's scripted.py); objects the hand can pick up.
GRASP_DZ = {"apple": 0.003, "orange": -0.009, "banana": 0.003, "water_bottle": 0.03}
ABOVE_OBJECT_M = 0.10
ABOVE_PLACE_M = 0.12
RELEASE_M = 0.07  # release height above a clicked place
LIFT_M = 0.12
# After a hand-back a closed gripper stays commanded closed until Dream-1 itself either grips (commands it at least
# GRIP_SQUEEZE_MARGIN tighter than measured) or lets go (GRIP_RELEASE_MARGIN more open than measured).
GRIP_SQUEEZE_MARGIN = 0.05
GRIP_RELEASE_MARGIN = 0.15
HOME_CLEAR_Z = 0.90  # before going home, a hand lower than this is first lifted to it, so it does not sweep the table


TILTS_DEG = (45, 30, 60)  # tilted grasps tried, in this order (0 = from above; 90 = level, out of reach at table height)
PRE_GRASP_M = 0.10  # a tilted grasp starts this far back along its approach line
LONG_AXIS = {"banana": 1}
TABLE_Z = 0.76  # table top height (world metres)  # elongated objects: the body axis (0 x, 1 y, 2 z) along their length


def tilted(direction, tilt_deg):
    """Hand rotation for a grasp tilted `tilt_deg` from vertical, leaning toward the horizontal `direction`.
    The fingers (the TCP's y axis) stay level, so they close on the object from both sides."""
    up = np.array([0.0, 0.0, 1.0])
    d = np.array([direction[0], direction[1], 0.0])
    d /= np.linalg.norm(d)
    t = np.radians(tilt_deg)
    z = -np.cos(t) * up + np.sin(t) * d  # approach
    y = np.cross(up, d)
    y /= np.linalg.norm(y)
    return np.c_[np.cross(y, z), y, z]


def about_local_y(angle):
    """Rotation about the hand's own finger axis (applied on the right: R @ about_local_y(a))."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Teleop:
    """Hand targets for both arms, the input, and who controls the arms (thread-safe)."""

    def __init__(self, mode="dream"):
        self.lock = threading.Lock()
        self.mode = mode
        self.dream_available = True  # False in keyboard-only runs: T does nothing
        self.active = "left"
        self.held: set[str] = set()
        self.targets = {}  # side -> (position (3,), rotation (3, 3))
        self.joints = {}  # side -> last commanded joint angles (6,)
        self.goals = {}  # side -> joint angles that reach the hand target (6,); joints move toward it each step
        self.destinations = {}  # side -> hand position the hand is travelling to (click, Lower, Lift)
        self.lower_to = {}  # side -> where Lower goes: grasp point of the clicked object, or release point
        self.grip = {side: 1.0 for side in SIDES}  # what the human asked for: 1 open, 0 closed
        self.grip_now = {side: 1.0 for side in SIDES}  # the commanded opening, moving toward self.grip
        self.velocity = {side: np.zeros(3) for side in SIDES}  # hand target velocity, m/s
        self.joint_velocity = {side: np.zeros(6) for side in SIDES}  # rad/s
        self.last_action = None  # (left, right) last sent to the simulator, by anyone
        self.reach_limited = False
        self.message = ""
        self.reason = None  # why the monitor paused Dream-1
        self.click = None  # (u, v) waiting for the main thread to look it up in the camera
        self.drag = None  # latest (u, v) of a mouse drag, same
        self.home = {}  # side -> start joint angles (6,), set by set_home()
        self.homing = set()  # sides lifting clear of the table on their way home
        self.grasp_style = "top"  # "top": from straight above; "tilted": at an angle, from the side (key G)
        self.on_control = None  # called with (mode, reason) whenever control changes hands
        self.grip_hold = {}  # side -> Dream-1 steps a closed gripper has been kept closed since a hand-back
        self.on_grip_hold = None  # called with (side, "start"/"end", why) when that hold starts or ends
        self.grasp_height = {}  # side -> how high above its resting surface the hand grasped the clicked object
        self.events = []  # (step, mode) changes, for the record
        self.steps = 0

    # ---- input from the website -------------------------------------------------------------------------------

    def set_held(self, keys):
        with self.lock:
            self.held = {str(k).lower() for k in keys} & (set(MOVES) | {"q", "e", "z", "x"})
            if self.held:
                self.destinations.pop(self.active, None)  # keys take over from a click move
                self.homing.discard(self.active)

    def press(self, key, env):
        """One press (not held): space/grip, tab, v, t, lower, lift."""
        key = str(key).lower()
        with self.lock:
            if key in (" ", "space", "grip"):
                self.grip[self.active] = 0.0 if self.grip[self.active] > 0.5 else 1.0
            elif key == "tab":
                self.active = "right" if self.active == "left" else "left"
            elif key == "g":
                self.grasp_style = "tilted" if self.grasp_style == "top" else "top"
                self.message = ("Grasp: tilted (from the side, at an angle). Click an object." if self.grasp_style ==
                                "tilted" else "Grasp: from above. Click an object.")
            elif key == "v" and self.mode == "human":
                self.reach_limited = not self._point_down(env, self.active)
            elif key == "lower" and self.mode == "human":
                position = self.targets[self.active][0]
                self.destinations[self.active] = self.lower_to.get(self.active, position - [0, 0, ABOVE_OBJECT_M])
            elif key in ("h", "home") and self.mode == "human":
                self._start_home(env, self.active)
            elif key == "lift" and self.mode == "human":
                self.destinations[self.active] = self.targets[self.active][0] + [0, 0, LIFT_M]
            elif key == "t":
                if self.mode == "script" or (self.mode in ("dream", "paused")
                                             and (self.dream_available or self.mode == "paused")):
                    self._take_over(env)
                    self._set_mode("human")
                elif self.mode == "human":
                    # Give the arms (back) to Dream-1. In a human-only run this starts Dream-1 from the scene as the
                    # human left it (sim.py sees the request and starts the Dream-1 loop).
                    self.dream_available = True
                    self._set_mode("returning")
                    self._start_grip_hold()  # as after the recovery script: do not let the object slip out

    def queue_click(self, u, v, camera="top"):
        """A click on a camera picture (top, left or right wrist), u and v from 0 to 1 (left/top).
        Looked up by the main thread."""
        with self.lock:
            self.click = (float(u), float(v), camera if camera in ("top", *SIDES) else "top")

    def take_click(self):
        with self.lock:
            click, self.click = self.click, None
            return click

    def queue_drag(self, u, v):
        """The mouse dragged to (u, v) on the top camera picture: the hand follows it at its current height."""
        with self.lock:
            self.drag = (float(u), float(v))

    def take_drag(self):
        with self.lock:
            drag, self.drag = self.drag, None
            return drag

    def dragged(self, env, camera_position, point):
        """The main thread found the ray of the cursor (camera -> point): follow it at the active hand's height."""
        with self.lock:
            if self.mode != "human":
                if self.mode in ("dream", "returning") and self.dream_available:
                    self.message = "Dream-1 has the arms: press T (or wait for it to ask for help) before dragging."
                    return
                self._take_over(env)
                self._set_mode("human")
            side = self.active
            if (self.grasp_style == "top" and np.linalg.norm(rotation_error(DOWN, self.targets[side][1])) > 0.1
                    and not self._point_down(env, side)):
                self.message = f"The {side} hand cannot point down from here."
                return
            # Straight above the spot the cursor points at (not where the camera ray crosses the hand's height:
            # the top camera looks at the table at an angle, so that would land several cm toward the camera).
            height = self.targets[self.active][0][2]
            above = np.r_[np.asarray(point)[:2], height]
            self.destinations[self.active] = np.clip(above, BOUNDS[:, 0], BOUNDS[:, 1])
            self.message = f"{self.active} hand follows the mouse. Wheel: up / down."

    def wheel(self, notches):
        """Mouse wheel over the picture: the active hand up (positive) or down, 1 cm per notch."""
        with self.lock:
            if self.mode != "human":
                return
            start = self.destinations.get(self.active, self.targets[self.active][0])
            self.destinations[self.active] = np.clip(start + [0, 0, 0.01 * float(notches)], BOUNDS[:, 0],
                                                     BOUNDS[:, 1])

    def clicked(self, env, body, point, camera="top"):
        """The main thread found what the click hit: move a hand above it. A click in a wrist camera moves that
        camera's arm; a click in the top camera moves the nearer arm (or the one holding something, for a place)."""
        with self.lock:
            if body.startswith(SIDES):
                self.message = "That is the robot itself: click an object, the bowl, the plate or the table."
                return
            if self.mode == "dream" and self.dream_available:
                self.message = "Dream-1 has the arms: press T (or wait for it to ask for help) before clicking."
                return
            if self.mode in ("paused", "dream", "script"):  # a click during a scripted recovery takes over from it
                self._take_over(env)
                self._set_mode("human")
            if self.mode != "human":
                return
            point = np.asarray(point, float)
            if body in GRASP_DZ and self.grasp_style == "tilted":
                side = camera if camera in SIDES else "left" if point[0] > 0 else "right"
                self.active = side
                self._tilted_grasp(env, side, body)
                return
            if body in GRASP_DZ:
                side = camera if camera in SIDES else "left" if point[0] > 0 else "right"
                grasp = env.data.body(body).xpos + [0, 0, GRASP_DZ[body]]
                destination, self.lower_to[side] = grasp + [0, 0, ABOVE_OBJECT_M], grasp
                self.grasp_height[side] = grasp[2] - TABLE_Z
                what = "the " + body.replace("water_", "")
            else:
                holding = [s for s in SIDES if self.grip[s] < 0.5]
                side = camera if camera in SIDES else self.active if self.active in holding else (
                    holding[0] if holding else "left" if point[0] > 0 else "right")
                # Set down until the held object's bottom touches the spot: aim 2 cm below the height the object was
                # grasped at (the position servos sag about 2 cm under a load, and the surface stops the rest), so a
                # tall bottle is neither let go in the air nor pushed hard into the table.
                release = self.grasp_height.get(side, RELEASE_M) - 0.02 if self.grip[side] < 0.5 else RELEASE_M
                destination = point + [0, 0, max(ABOVE_PLACE_M, release + 0.05)]
                self.lower_to[side] = point + [0, 0, release]
                what = f"the {body}" if body in ("bowl", "plate") else "that spot"
            self.active = side
            if (self.grasp_style == "top" or body in GRASP_DZ) and np.linalg.norm(
                    rotation_error(DOWN, self.targets[side][1])) > 0.1 and not self._point_down(env, side):
                self.message = f"The {side} hand cannot point down from here."
                return
            self.destinations[side] = np.clip(destination, BOUNDS[:, 0], BOUNDS[:, 1])
            self.message = (f"{side} hand → above {what}. Then: Lower, Grip (opens to let go), Lift." if body not in
                            GRASP_DZ else f"{side} hand → above {what}. Then: Lower, Grip, Lift.")

    def pause(self, env, reason):
        """The failure monitor asks for a human: hold the arms where Dream-1 left them."""
        with self.lock:
            if self.mode != "dream":
                return False
            self._take_over(env)
            self.reason = reason
            self._set_mode("paused")
            return True

    def start_script(self, env, reason):
        """The failure monitor reported a failure and the recovery script (recovery.py) takes the arms."""
        with self.lock:
            if self.mode != "dream":
                return False
            self._take_over(env)
            self.reason = reason
            self._set_mode("script")
            return True

    def end_script(self, recovered, reason=None):
        """The recovery script is done: give the arms back to Dream-1, or hold them for a human (it gave up)."""
        with self.lock:
            if self.mode != "script":
                return
            self.destinations.clear()
            self.velocity = {side: np.zeros(3) for side in SIDES}
            if recovered:
                self._set_mode("returning")
                self._start_grip_hold()
            else:
                self.reason = reason or self.reason
                self._set_mode("paused")

    def chunk_started(self):
        """Dream-1 is about to be asked for a chunk. @return False when it must not be asked (a human has the arms)."""
        with self.lock:
            if self.mode == "returning":
                self._set_mode("dream")
                self.reason = None
            return self.mode == "dream"

    # ---- the control step --------------------------------------------------------------------------------------

    def set_home(self, env):
        """Remember the arms' start pose (call once, right after the simulator starts)."""
        with self.lock:
            state = env.state()
            self.home = {side: state[side][:6].copy() for side in SIDES}

    def start_human(self, env):
        """Keyboard-only runs: take the arms at once."""
        with self.lock:
            self.dream_available = False
            self._take_over(env)
            self._set_mode("human")
            # The arms stay in the start pose (hands level). Clicking or dragging points the hand down first; with the
            # keys, press V (a level hand can barely move sideways, the wrist being near its limits).

    def filter(self, env, left, right):
        """Called with the action about to be sent to env.step(); returns the action to send instead."""
        with self.lock:
            self.steps += 1
            if self.mode == "dream":
                left, right = np.array(left, float), np.array(right, float)
                if self.grip_hold:
                    self._hold_grip(env, {"left": left, "right": right})
                self.last_action = (left, right)
                return left, right
            if not self.targets:
                self._take_over(env)
            if self.mode in ("human", "script"):
                self._move_hand(env)
            action = tuple(self._solve(env, side) for side in SIDES)
            self.last_action = action
            return action

    def record_view(self):
        """(hand targets (2,3), gripper commands (2,)) for the recorder; NaN before the human first had the arms."""
        with self.lock:
            hands = [self.targets[side][0] if side in self.targets else np.full(3, np.nan) for side in SIDES]
            grips = [self.grip_now.get(side, np.nan) for side in SIDES]
        return np.array(hands, dtype=float), np.array(grips, dtype=float)

    def status(self):
        with self.lock:
            hands = {
                side: [round(float(v), 3) for v in self.targets[side][0]] if side in self.targets else None
                for side in SIDES
            }
            return {
                "mode": self.mode,
                "dream_available": self.dream_available,
                "reason": self.reason,
                "active": self.active,
                "grip": {side: "open" if g > 0.5 else "closed" for side, g in self.grip.items()},
                "hand": hands,
                "moving": self.active in self.destinations,
                "reach_limited": self.reach_limited,
                "grasp_style": self.grasp_style,
                "message": self.message,
            }

    # ---- internals (lock held) ---------------------------------------------------------------------------------

    def _limit(self, name):
        """A speed limit: Dream-1's pace while the recovery script moves the arms, a human's otherwise."""
        if self.mode == "script":
            return {"goto": SCRIPT_SPEED_M_S, "accel": SCRIPT_ACCEL_M_S2, "joint": SCRIPT_MAX_JOINT_RAD_S,
                    "joint_accel": SCRIPT_JOINT_ACCEL_RAD_S2, "grip_s": SCRIPT_GRIP_SECONDS}[name]
        return {"goto": GOTO_SPEED_M_S, "accel": ACCEL_M_S2, "joint": MAX_JOINT_STEP / STEP_S,
                "joint_accel": JOINT_ACCEL_RAD_S2, "grip_s": GRIP_SECONDS}[name]

    def _start_grip_hold(self):
        """The arms go back to Dream-1: keep each closed gripper closed until Dream-1 grips or lets go (_hold_grip)."""
        for side in SIDES:
            if self.grip[side] < 0.5:
                self.grip_hold[side] = 0  # Dream-1 steps held so far
                self._grip_hold_event(side, "start", "handed back holding")

    def _hold_grip(self, env, actions):
        """Right after a hand-back, Dream-1 tends to command the gripper to the opening it measures (an object between
        the fingers keeps it ~0.7 open). A position-servo gripper then squeezes nothing and the object slips out.
        So a closed gripper stays commanded closed until Dream-1 grips on its own (commands it GRIP_SQUEEZE_MARGIN
        tighter than measured) or clearly lets go (GRIP_RELEASE_MARGIN more open, e.g. over the bowl): from then on its
        command goes through unchanged. A change of control (a new failure) ends the hold too.
        @param actions By side, the action about to be sent (changed in place)."""
        state = env.state()
        for side in list(self.grip_hold):
            command, measured = actions[side][6], state[side][6]
            if command > measured + GRIP_RELEASE_MARGIN or command < measured - GRIP_SQUEEZE_MARGIN:
                steps = self.grip_hold.pop(side)
                why = "Dream-1 lets go" if command > measured else "Dream-1 grips on its own"
                self._grip_hold_event(side, "end", f"{why} (after {steps} steps)")
                continue
            actions[side][6] = 0.0
            self.grip_hold[side] += 1

    def _grip_hold_event(self, side, what, why):
        if self.on_grip_hold is not None:
            self.on_grip_hold(side, what, why)  # the recorder logs it: these steps' gripper is not Dream-1's command
        print(f"  Grip hold ({side}): {what} · {why}", flush=True)

    def _set_mode(self, mode):
        if mode not in ("dream", "returning") and self.grip_hold:
            for side in list(self.grip_hold):
                self._grip_hold_event(side, "end", f"control went to {mode}")
            self.grip_hold.clear()
        self.mode = mode
        self.events.append((self.steps, mode))
        if self.on_control is not None:
            self.on_control(mode, self.reason)  # e.g. the recorder logs every change of control
        print(f"  Control: {mode}" + (f" ({self.reason})" if mode == "paused" else ""), flush=True)

    def _take_over(self, env):
        """Start from where the arms are: the last commanded joints if known, else the measured ones."""
        state = env.state()
        for i, side in enumerate(SIDES):
            if self.last_action is not None:
                q, grip = np.asarray(self.last_action[i][:6], float), float(self.last_action[i][6])
            else:
                q, grip = state[side][:6].copy(), float(state[side][6])
            self.joints[side] = self._clip_joints(env, side, q)
            self.goals[side] = self.joints[side].copy()
            self.grip[side] = 1.0 if grip > 0.5 else 0.0
            self.grip_now[side] = float(np.clip(grip, 0.0, 1.0))
            self.velocity[side] = np.zeros(3)
            self.joint_velocity[side] = np.zeros(6)
            self.targets[side] = self._forward(env, side, self.joints[side])
        self.held.clear()
        self.destinations.clear()
        self.lower_to.clear()

    def _move_hand(self, env):
        """Move the active hand's target one step: toward the keys' direction or the destination, with the speed
        ramped up and down at ACCEL_M_S2 so the hand starts, turns and stops smoothly."""
        side = self.active
        position, rotation = self.targets[side]
        direction = np.zeros(3)
        for key in self.held & set(MOVES):
            axis, sign = MOVES[key]
            direction[axis] += sign
        destination = self.destinations.get(side)
        if direction.any():
            wanted = direction / np.linalg.norm(direction) * SPEED_M_S
        elif destination is not None:
            gap = destination - position
            distance = np.linalg.norm(gap)
            if distance < 0.001 and np.linalg.norm(self.velocity[side]) < 0.02:
                self.destinations.pop(side)
                self.velocity[side] = np.zeros(3)
                if side in self.homing:
                    self._go_home(env, side)  # lifted clear: now the joints go back to the start pose
                return
            # As fast as allowed, but slow enough to stop at the destination: v = sqrt(2 a d).
            speed = min(self._limit('goto'), np.sqrt(2 * self._limit('accel') * distance))
            wanted = gap / max(distance, 1e-9) * speed
        else:
            wanted = np.zeros(3)
        change = wanted - self.velocity[side]
        limit = self._limit('accel') * STEP_S
        if np.linalg.norm(change) > limit:
            change = change / np.linalg.norm(change) * limit
        self.velocity[side] = self.velocity[side] + change
        turn = (("q" in self.held) - ("e" in self.held)) * TURN_RAD_S * STEP_S
        tilt = (("z" in self.held) - ("x" in self.held)) * TURN_RAD_S * STEP_S
        if tilt:
            rotation = rotation @ about_local_y(tilt)
        if not self.velocity[side].any() and not turn and not tilt:
            self.reach_limited = False
            return
        step = self.velocity[side] * STEP_S
        if destination is not None and not direction.any() and np.linalg.norm(step) > np.linalg.norm(
                destination - position):
            step = destination - position  # never overshoot the destination
        new_position = np.clip(position + step, BOUNDS[:, 0], BOUNDS[:, 1])
        if not self._try_target(env, side, new_position, yaw(turn) @ rotation if turn else rotation):
            self.velocity[side] = np.zeros(3)
            if self.destinations.pop(side, None) is not None:
                self.message = f"The {side} hand cannot go further that way (reach or joint limit)."
            elif np.linalg.norm(rotation_error(DOWN, rotation)) > 0.1:
                # A level hand (the start pose) can barely move sideways: the wrist is near its limits.
                self.message = "The hand is level, so it can barely move: press V to point it down."

    def _tilted_grasp(self, env, side, body):
        """Find a tilted grasp of the object this arm can reach (tilt and lean direction), go to its pre-grasp pose;
        Lower then slides in along the approach line."""
        grasp = env.data.body(body).xpos + [0, 0, GRASP_DZ[body]]
        base = env.data.body(f"{side}_base").xpos
        away = np.arctan2(*(grasp - base)[1::-1])  # lean direction pointing away from the arm's base, tried first
        directions = [away + np.radians(a) for a in (0, 45, -45, 90, -90, 135, -135, 180)]
        if body in LONG_AXIS:
            # A long object: lean along its length, so the level fingers close across it (its narrow side).
            axis = env.data.body(body).xmat.reshape(3, 3)[:, LONG_AXIS[body]]
            along = np.arctan2(axis[1], axis[0])
            ends = sorted([along, along + np.pi], key=lambda a: -np.cos(a - away))
            directions = ends + [a + np.radians(d) for a in ends for d in (20, -20)]
        limits = env.model.jnt_range[[env.model.joint(f"{side}_joint{i}").id for i in range(1, 7)]]
        rng = np.random.default_rng(0)
        seeds = [self.joints[side]] + [rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(6)]
        for tilt in TILTS_DEG:
            for a in directions:
                rotation = tilted((np.cos(a), np.sin(a)), tilt)
                for seed in seeds:
                    q, ok = self._ik(env, side, (grasp, rotation), [seed])
                    if not ok:
                        continue
                    pre = grasp - rotation[:, 2] * PRE_GRASP_M
                    q_pre, ok = self._ik(env, side, (pre, rotation), [q])
                    if ok:
                        self.goals[side] = q_pre  # joints go there smoothly (speed and acceleration limits)
                        self.targets[side] = (pre, rotation)
                        self.destinations.pop(side, None)
                        self.velocity[side] = np.zeros(3)
                        self.lower_to[side] = grasp
                        self.grasp_height[side] = grasp[2] - TABLE_Z
                        self.message = (f"{side} hand → tilted {tilt}° grasp of the {body.replace('water_', '')}. "
                                        "Then: Lower (slides in), Grip, Lift.")
                        return True
        self.message = f"The {side} arm cannot reach the {body.replace('water_', '')} at an angle: try the other arm or G."
        return False

    def _start_home(self, env, side):
        if side not in self.home:
            self.message = "The start pose is not known."
            return
        position = self.targets[side][0]
        if position[2] < HOME_CLEAR_Z:
            self.destinations[side] = np.r_[position[:2], HOME_CLEAR_Z]
            self.homing.add(side)
            self.message = f"{side} arm: lifting clear, then back to the start pose."
        else:
            self._go_home(env, side)

    def _go_home(self, env, side):
        """Joints straight to the start pose (smoothly, at the joint speed and acceleration limits)."""
        self.homing.discard(side)
        self.destinations.pop(side, None)
        self.velocity[side] = np.zeros(3)
        self.goals[side] = self.home[side].copy()
        self.targets[side] = self._forward(env, side, self.home[side])
        self.message = f"{side} arm → start pose."

    def _try_target(self, env, side, position, rotation):
        q, ok = self._ik(env, side, (position, rotation), [self.goals[side], self.joints[side]])
        self.reach_limited = not ok
        if ok:
            self.targets[side] = (position, rotation)
            self.goals[side] = q
        return ok

    def _solve(self, env, side):
        """One control step toward the side's goal joints, at most MAX_JOINT_STEP per joint."""
        start = self.joints[side]
        gap = self.goals[side] - start
        largest = np.max(np.abs(gap))
        if largest < 1e-5:
            self.joint_velocity[side] = np.zeros(6)
            q = self.goals[side].copy()
        else:
            # Speed of the joint that has furthest to go: capped, and low enough to stop at the goal (v = sqrt(2 a d));
            # the other joints are scaled with it, so all arrive together and the hand keeps its path.
            speed = min(self._limit('joint'), np.sqrt(2 * self._limit('joint_accel') * largest))
            wanted = gap / largest * speed
            limit = self._limit('joint_accel') * STEP_S
            self.joint_velocity[side] = self.joint_velocity[side] + np.clip(wanted - self.joint_velocity[side],
                                                                            -limit, limit)
            change = self.joint_velocity[side] * STEP_S
            if np.max(np.abs(change)) >= largest:
                change, self.joint_velocity[side] = gap, gap / STEP_S  # arrive exactly, without overshooting
            q = start + change
        q = self._clip_joints(env, side, q)
        self.joints[side] = q
        grip_step = STEP_S / self._limit('grip_s')
        self.grip_now[side] += float(np.clip(self.grip[side] - self.grip_now[side], -grip_step, grip_step))
        return np.r_[q, self.grip_now[side]]

    def _ik(self, env, side, target, seeds):
        """Damped least-squares IK (kinematics only, no physics), tried from each seed in turn.
        @return (joint angles, reached): reached is False when no seed gets within IK_TOLERANCE_M and 0.1 rad.
        """
        model = env.model
        data = mujoco.MjData(model)
        data.qpos[:] = env.data.qpos
        ids = [model.joint(f"{side}_joint{i}").id for i in range(1, 7)]
        qadr, vadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
        site = model.site(f"{side}_tcp").id
        target_p, target_r = target
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        best = (np.inf, np.asarray(seeds[0], float))
        for seed in seeds:
            q = self._clip_joints(env, side, np.asarray(seed, float))
            for _ in range(IK_ITERATIONS):
                data.qpos[qadr] = q
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                error_p = target_p - data.site_xpos[site]
                error_r = rotation_error(target_r, data.site_xmat[site].reshape(3, 3))
                size = np.linalg.norm(error_p) + 0.05 * np.linalg.norm(error_r)
                if size < best[0]:
                    best = (size, q.copy())
                if np.linalg.norm(error_p) < 0.002 and np.linalg.norm(error_r) < 0.02:
                    return q, True
                mujoco.mj_jacSite(model, data, jp, jr, site)
                jacobian = np.vstack([jp[:, vadr], 0.3 * jr[:, vadr]])
                error = np.r_[error_p, 0.3 * error_r]
                dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(6), error)
                q = self._clip_joints(env, side, q + np.clip(dq, -0.2, 0.2))
            data.qpos[qadr] = q
            mujoco.mj_kinematics(model, data)
            if (np.linalg.norm(target_p - data.site_xpos[site]) < IK_TOLERANCE_M
                    and np.linalg.norm(rotation_error(target_r, data.site_xmat[site].reshape(3, 3))) < 0.1):
                return q, True
        return best[1], False

    def _point_down(self, env, side):
        """The same hand position, pointing down. Searched from many starts, since the wrist may need to turn far."""
        position, _ = self.targets[side]
        rng = np.random.default_rng(0)
        limits = env.model.jnt_range[[env.model.joint(f"{side}_joint{i}").id for i in range(1, 7)]]
        seeds = [self.joints[side]] + [rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(12)]
        found = []
        for seed in seeds:
            q, ok = self._ik(env, side, (position, DOWN), [seed])
            if ok:
                margin = np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))  # distance from the nearest limit
                found.append((np.linalg.norm(q - self.joints[side]) - 2.0 * margin, q))
        if not found:
            return False
        self.goals[side] = min(found, key=lambda item: item[0])[1]
        self.targets[side] = (position, DOWN.copy())
        return True

    def _forward(self, env, side, q):
        model = env.model
        data = mujoco.MjData(model)
        data.qpos[:] = env.data.qpos
        ids = [model.joint(f"{side}_joint{i}").id for i in range(1, 7)]
        data.qpos[model.jnt_qposadr[ids]] = q
        mujoco.mj_kinematics(model, data)
        site = model.site(f"{side}_tcp").id
        return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()

    @staticmethod
    def _clip_joints(env, side, q):
        limits = env.model.jnt_range[[env.model.joint(f"{side}_joint{i}").id for i in range(1, 7)]]
        return np.clip(q, limits[:, 0], limits[:, 1])


def rotation_error(target, current):
    """Axis-angle vector that turns `current` into `target` (world frame)."""
    r = target @ current.T
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.linalg.norm(axis) < 1e-9:  # 180 degrees: any perpendicular axis
        return np.array([np.pi, 0.0, 0.0])
    return axis / np.linalg.norm(axis) * angle


def select(env, u, v, camera="top"):
    """What a click on a camera picture hits. u, v from 0 to 1 (left, top) of the 640x360 picture env.images() gives.
    Main thread only (it uses the renderer). @return (body name, world point), or (None, None) for nothing.
    """
    if env.renderer is None:
        env.images()
    renderer = env.renderer
    renderer.update_scene(env.data, camera=camera)
    point = np.zeros(3)
    geom, flex, skin = (np.array([-1], np.int32) for _ in range(3))
    height, width = renderer.height, renderer.width
    # env.images() crops rows 60..420 of the 480-row render; MuJoCo counts rows from the bottom.
    body = mujoco.mjv_select(env.model, env.data, mujoco.MjvOption(), width / height, u,
                             1 - (60 + v * 360) / height, renderer.scene, point, geom, flex, skin)
    if body < 0:
        return None, None
    name = env.model.body(body).name
    return ("table" if name == "world" else name), point


def camera_position(env):
    """World position of the camera the last select() rendered from (the middle of its two eyes)."""
    cameras = env.renderer.scene.camera
    return (np.asarray(cameras[0].pos) + np.asarray(cameras[1].pos)) / 2
