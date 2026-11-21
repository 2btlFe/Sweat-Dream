"""Scripted pick-and-place episodes with a known outcome, recorded like Dream-1 episodes (no API calls).

They test the judge and seed the dataset with labelled successes and failures of each kind. The grasp and IK follow
the Dream YAM Sim's own mock_pickups.py (contact-only grasps, no attachments or teleporting).

Usage: python scripted.py [variant ...]      variants: success outside drop wrong_object hold no_attempt
"""

import sys
import warnings

warnings.filterwarnings("ignore")
import dreamfix  # noqa: F401,E402  (puts the sim on sys.path)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from recorder import EpisodeRecorder, RecordingEnv  # noqa: E402

HOME = np.r_[np.zeros(6), 1.0]
LIFT = 0.15
GRASP_DZ = {"orange": -0.009, "apple": 0.003, "banana": 0.003}
GRASP_YAW = {"banana": np.pi / 2}

## variant: (task, object actually handled, target actually used, what goes wrong)
VARIANTS = {
    "success": ("put apple into the bowl", "apple", "bowl", None),
    "outside": ("put apple into the bowl", "apple", "bowl", "release_offset"),
    "drop": ("put apple into the bowl", "apple", "bowl", "drop_midway"),
    "wrong_object": ("put apple into the bowl", "orange", "bowl", None),
    "hold": ("put apple into the bowl", "apple", "bowl", "never_release"),
    "no_attempt": ("put apple into the bowl", None, "bowl", None),
    "orange_success": ("put orange into the bowl", "orange", "bowl", None),
}


def solve(env, side, target, rotation, seed):
    """Joint-limited IK for the arm's TCP site (from the sim's mock_pickups.py)."""
    m = env.model
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, env.data)
    ids = [m.joint(f"{side}_joint{i}").id for i in range(1, 7)]
    qa, va, limits = m.jnt_qposadr[ids], m.jnt_dofadr[ids], m.jnt_range[ids]
    site = m.site(f"{side}_tcp").id
    rng = np.random.default_rng(42)
    best = (1e9, None)
    for initial in [seed, np.zeros(6)] + [rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(14)]:
        q = np.array(initial).copy()
        for _ in range(180):
            d.qpos[qa] = q
            mujoco.mj_kinematics(m, d)  # kinematics only: a trial pose inside the table must not run collisions
            mujoco.mj_comPos(m, d)
            rot = rotation @ d.site_xmat[site].reshape(3, 3).T
            angle = np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))
            axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
            error_r = axis * angle / max(2 * np.sin(angle), 1e-8)
            error = target - d.site_xpos[site]
            score = np.linalg.norm(error) + 0.1 * angle
            if score < best[0]:
                best = (score, q.copy())
            if np.linalg.norm(error) < 0.004 and angle < 0.06:
                return q
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, site)
            J = np.vstack([jp[:, va], jr[:, va] * 0.15])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[error, error_r * 0.15])
            q = np.clip(q + np.clip(dq, -0.12, 0.12), limits[:, 0], limits[:, 1])
    if best[0] < 0.03:
        return best[1]
    raise ValueError(f"IK failed {side} {np.round(target, 3)}: score {best[0]:.4f}")


def run(variant, seed=None, batch_id=None):
    """Run and record one scripted episode.
    @param seed When set, the objects' start poses are jittered with this seed.
    @param batch_id Batch this episode belongs to, stored in meta.json.
    @return (episode folder, outcome dict).
    """
    task, obj, target, fault = VARIANTS[variant]
    env = RecordingEnv()
    offsets = env.jitter(seed) if seed is not None else None
    rec = EpisodeRecorder(env, task, model=None, source=f"scripted:{variant}",
                          extra={"expected": variant, "seed": seed, "jitter": offsets, "batch_id": batch_id})
    env.recorder = rec
    steps = {"n": 0, "last": (HOME, HOME)}
    error = None

    def step(left, right):
        steps["last"] = (left, right)
        env.step(left, right)
        if steps["n"] % 4 == 0:
            rec.on_frame(env.images())
        steps["n"] += 1

    try:
        for _ in range(20):
            step(HOME, HOME)
        if obj is not None:
            p = env.data.body(obj).xpos.copy()
            yaw = GRASP_YAW.get(obj, 0.0)
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])
            goal0 = env.data.body(target).xpos
            # The nearer arm first; the other when the nearer one cannot reach the object and the target.
            side = None
            for candidate in (["left", "right"] if p[0] > 0 else ["right", "left"]):
                try:
                    for point in (p + [0, 0, LIFT - 0.01], p + [0, 0, GRASP_DZ.get(obj, 0.003)],
                                  goal0 + [0, 0, LIFT + 0.02], goal0 + [0, 0, 0.09]):
                        solve(env, candidate, point, rotation, np.zeros(6))
                    side = candidate
                    break
                except ValueError:
                    continue
            if side is None:
                raise ValueError(f"neither arm reaches {obj} at {np.round(p, 3)} and the {target}")
            limits = env.model.jnt_range[[env.model.joint(f"{side}_joint{i}").id for i in range(1, 7)]]
            q = np.clip(env.state()[side][:6], limits[:, 0], limits[:, 1])

            def command(qv, opening):
                arm = np.r_[qv, opening]
                step(arm, HOME) if side == "left" else step(HOME, arm)

            def move(point, opening, n, release_at=None):
                nonlocal q
                end = solve(env, side, np.asarray(point), rotation, q)
                start = q.copy()
                for i, a in enumerate(np.linspace(0, 1, n)):
                    o = 1.0 if release_at is not None and i >= release_at * n else opening
                    command(start + (end - start) * a, o)
                q = end

            move(p + [0, 0, LIFT - 0.01], 1, 50)
            move(p + [0, 0, GRASP_DZ.get(obj, 0.003)], 1, 50)
            for o in np.linspace(1, 0, 25):
                command(q, o)
            move(p + [0, 0, LIFT], 0, 50)
            goal = env.data.body(target).xpos.copy()
            if fault == "release_offset":
                goal = goal + [0.0, 0.16, 0.0]
            above = goal + [0, 0, LIFT + 0.02]
            move(above, 0, 70, release_at=0.5 if fault == "drop_midway" else None)
            move(goal + [0, 0, 0.09], 0, 40)
            if fault != "never_release":
                for o in np.linspace(0, 1, 15):
                    command(q, o)
                move(goal + [0, 0, 0.13], 1, 40)
            for _ in range(40):
                command(q, 0.0 if fault == "never_release" else 1.0)
        else:
            for _ in range(200):
                step(HOME, HOME)
        for _ in range(30):  # settle, holding the last command (a blocked gripper must stay closed)
            step(*steps["last"])
    except Exception as e:  # noqa: BLE001
        error = e
    outcome = rec.close(error)
    env.close()
    return rec.dir, outcome


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*", help=f"any of: {' '.join(VARIANTS)} (default: all)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-id")
    args = parser.parse_args()
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s): {unknown}")
    for name in args.variants or list(VARIANTS):
        folder, outcome = run(name, seed=args.seed, batch_id=args.batch_id)
        print(f"EPISODE_DIR={folder}", flush=True)
        m = outcome["metrics"]
        print(f"{name:15s} -> {outcome['label']:8s} {outcome['reason']:28s} "
              f"fail_t={outcome['failure_time_s']} ok_t={outcome['success_time_s']} "
              f"moved={m.get('object_moved_m')} dist={m.get('final_distance_to_target_m')} warn={outcome['warnings']}  [{folder.name}]",
              flush=True)
