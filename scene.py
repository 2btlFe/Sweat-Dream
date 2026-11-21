"""Random start scenes from a seed: where the tabletop objects start, reproducible from (setting, seed).

Self-contained module; sim.py only calls add_arguments() and build_env() (the website side is seed_web.py + static/seed.js):

    python sim.py "put apple into the bowl"                       a new random train seed (printed)
    python sim.py "put apple into the bowl" --seed 42             the start scene of seed 42
    python sim.py "put apple into the bowl" --seed 1000007 --split eval
    python sim.py "put banana on the plate" --setting fruits_only --seed 42
    python sim.py "put apple into the bowl" --default-scene       the package's fixed scene

    python scene.py 42 --image scene42.jpg     a seed's start scene as the website shows it (top camera + both
                                               wrist cameras), without running Dream (no API key needed);
                                               app.py's /api/preview uses this
    python scene.py --check                    check that seeds still give the same scenes as recorded (REFERENCE)
    python scene.py --build-reach-map          rebuild scene_reach_map.npz (only when the robot or table changes)

REPRODUCIBLE: the start scene is a pure function of (SCENE_VERSION, setting, seed). The task text does not
change it, so different tasks on one seed start from the same scene. All randomness comes from rng_for(), which
derives an independent stream per (setting, seed, purpose); other modules (OOD injection, task sampling, ...)
should take their randomness from rng_for(seed, "their-purpose", setting=...) too, so a run stays reproducible
from its seed alone.

SETTINGS: a Setting says which objects move and how. "default" moves all six objects. Add a Setting to
SETTINGS for another task or scene; it gets its own scenes for the same seeds, and existing settings are not
affected. Objects a setting does not move stay at the package's default pose and are kept clear of.

What a seed randomizes (default setting):
  - the spot of each object: apple, orange, banana, water_bottle, bowl and plate alike;
  - each object's turn about the vertical axis, 0 to 360 degrees;
  - the order the objects are placed in (so no object is always placed last, into the leftover gaps).
Every object lies where BOTH arms can reach (scene_reach_map.npz) and the top camera sees, fully on the table,
at least min_gap from every other object, and not under an arm at home or hidden behind one in the top camera
view. Spots near the edge of that area are drawn less often (edge_falloff), so most scenes are comfortably
reachable and a few are near the limit.

Seeds are split so evaluation scenes are never trained on:
  train  0 .. 999_999
  eval   1_000_000 and up; the fixed evaluation set EVAL_SEEDS has 50 seeds, a third each from the easy,
         medium and hard scenes (by difficulty()) of the first 1000 eval seeds.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPoint, box
from shapely.ops import unary_union
from shapely.prepared import prep

spec = importlib.util.find_spec("dream_examples")
if spec is None:
    sys.exit("The simulator is missing: run  pip install dreamlabs")
sys.path.insert(0, str(Path(next(iter(spec.submodule_search_locations))) / "yam_mujoco"))

import mujoco  # noqa: E402
from yam_env import build_xml as _package_xml  # noqa: E402  taken now, before ood.py can patch it

REACH_MAP = Path(__file__).resolve().parent / "scene_reach_map.npz"

# Bump when the sampling rules change (the same seed then gives a different scene), and refresh REFERENCE.
SCENE_VERSION = 2

TRAIN_SEEDS = range(0, 1_000_000)
EVAL_SEED_START = 1_000_000
EVAL_POOL = range(EVAL_SEED_START, EVAL_SEED_START + 1000)  # the stratified eval set is drawn from these
EVAL_SET_SIZE = 50
TRIES_PER_OBJECT = 300
SCENE_ATTEMPTS = 100
CAMERA_MARGIN = 10  # px kept between an object and the top camera's image edge
LIFT = 0.002  # m dropped onto the table, so a turned object never starts inside it


@dataclasses.dataclass(frozen=True)
class Setting:
    """How a start scene is randomized. Every field takes part in the scenes it makes."""

    objects: tuple  # bodies placed at random; the others stay at the package's default pose
    reach: str = "both"  # "both": every object within reach of both arms; "either": of at least one arm
    min_gap: float = 0.03  # m between the edges of any two objects
    yaw_deg: tuple = (0.0, 360.0)  # range of each object's turn about the vertical axis
    edge_falloff: float = 0.08  # m: spots closer than this to the edge of the free area are drawn less often
    edge_min_weight: float = 0.25  # how often a spot right at the edge is drawn, relative to the middle
    description: str = ""

    def key(self):
        """@return A number that changes when any field changes (feeds the random streams)."""
        fields = dataclasses.asdict(self)
        fields.pop("description")
        return int(hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:15], 16)


SETTINGS = {
    "default": Setting(
        objects=("bowl", "plate", "water_bottle", "banana", "apple", "orange"),
        description="all six objects at random, within reach of both arms",
    ),
    "fruits_only": Setting(
        objects=("banana", "apple", "orange"),
        description="only the fruits move; bowl, plate and bottle stay where the package puts them",
    ),
}

# Fingerprints of a few seeds' scenes, per setting, and of the eval set. If they stop matching, the same seed no
# longer gives the same scene (dreamlabs, the camera, the reach map or this code changed): check_reproducible()
# says so. After a deliberate change, bump SCENE_VERSION and refresh with  python scene.py --check --update
REFERENCE = {
    "default": {0: "8eb50efed92d694a", 42: "2426e2750bd7bf00", 999999: "c30a489f7bb042b8",
                1000000: "1a58b9c2f68bdab4", 1000049: "2f5e272c5251f480"},
    "fruits_only": {0: "e1af86a63042200c", 42: "b8db6b7a9a9572b1"},
}
EVAL_FINGERPRINT = "f885497727d1d571"
CONTROLS_FINGERPRINT = "86b9558a52e56a0f"  # the ood.py scene controls drawn from seeds (draw_controls)
# The fixed eval set of the default setting: 17 easy, 17 medium, 16 hard (select_eval_seeds() remakes it).
EVAL_SEEDS = (1000000, 1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000007, 1000008, 1000009, 1000010,
              1000011, 1000012, 1000013, 1000014, 1000015, 1000016, 1000017, 1000018, 1000019, 1000020, 1000021,
              1000022, 1000023, 1000024, 1000025, 1000026, 1000027, 1000028, 1000029, 1000030, 1000031, 1000032,
              1000033, 1000034, 1000035, 1000036, 1000037, 1000038, 1000039, 1000040, 1000041, 1000042, 1000043,
              1000044, 1000045, 1000049, 1000051, 1000054, 1000056)


# ---------------------------------------------------------------------------------------------- seeds and streams

def setting_of(name):
    """@return The Setting called `name`."""
    if name not in SETTINGS:
        raise ValueError(f"Unknown setting {name!r}; known: {', '.join(SETTINGS)}")
    return SETTINGS[name]


def rng_for(seed, *purpose, setting="default"):
    """The random stream of one purpose of one seed: same (setting, seed, purpose), same numbers, anywhere.
    @param purpose Names of what the numbers are for, e.g. "placement" or "ood", "lighting".
    """
    words = [int(hashlib.sha256(str(p).encode()).hexdigest()[:15], 16) for p in purpose]
    return np.random.default_rng(np.random.SeedSequence([SCENE_VERSION, setting_of(setting).key(), int(seed), *words]))


def split_of(seed):
    """@return "train" or "eval" for a seed."""
    if seed < 0:
        raise ValueError("A seed is a whole number 0 or above.")
    return "train" if seed in TRAIN_SEEDS else "eval"


def check_split(seed, split):
    """Refuse a seed of the other split (an eval scene must never be trained on)."""
    if split_of(seed) != split:
        raise ValueError(f"Seed {seed} is for {split_of(seed)}, not {split}: train seeds are "
                         f"{TRAIN_SEEDS.start}..{TRAIN_SEEDS.stop - 1}, eval seeds {EVAL_SEED_START} and up.")


def new_seed(split="train", setting="default", model=None):
    """@return A fresh random seed of the split (eval: one of the setting's fixed eval seeds)."""
    rng = np.random.default_rng()
    if split == "train":
        return int(rng.integers(TRAIN_SEEDS.start, TRAIN_SEEDS.stop))
    seeds = eval_seeds(setting, model)
    return int(seeds[rng.integers(len(seeds))])


# ---------------------------------------------------------------------------------------------- scene geometry

def yaw_quat(yaw):
    """@return MuJoCo quaternion (w, x, y, z) of a turn of `yaw` radians about the vertical axis."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def geom_points(model, data, g):
    """A geom's mesh vertices (or points on its outline / box corners), in world coordinates."""
    kind = model.geom_type[g]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        local = model.mesh_vert[start:start + count]
    elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE):
        size = model.geom_size[g]
        radius = max(size[:2]) if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID else size[0]
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        local = np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros_like(a)], 1)
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:  # along its own z axis: both ends and both caps
            half = size[1]
            local = np.vstack([local + [0, 0, half], local - [0, 0, half],
                               local + [0, 0, half + radius], local - [0, 0, half + radius]])
    else:
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        local = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return data.geom_xpos[g] + local @ data.geom_xmat[g].reshape(3, 3).T


def body_points(model, data, bodies):
    """@return World points of every geom of the given body ids."""
    return np.vstack([geom_points(model, data, g) for g in range(model.ngeom) if model.geom_bodyid[g] in bodies])


class Layout:
    """What does not depend on the seed or the setting: object outlines and default poses, the arms' shadows,
    and the table cells each arm reaches and the top camera sees. Built from the model before any scene."""

    def __init__(self, model):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.table_top = float(model.geom("table").pos[2] + model.geom("table").size[2])
        self.bases = np.array([data.body(f"{side}_base").xpos[:2] for side in ("left", "right")])
        names = {n for s in SETTINGS.values() for n in s.objects}
        names |= {model.body(model.jnt_bodyid[j]).name for j in range(model.njnt)
                  if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
        self.defaults, self.outlines = {}, {}
        for name in sorted(names):
            body = model.body(name)
            free = body.jntnum[0] > 0
            if free:
                adr = model.jnt_qposadr[body.jntadr[0]]
                pos, quat = model.qpos0[adr:adr + 3].copy(), model.qpos0[adr + 3:adr + 7].copy()
            else:
                pos, quat = model.body_pos[body.id].copy(), model.body_quat[body.id].copy()
            self.defaults[name] = {"pos": pos, "quat": quat, "free": free}
            footprint = body_points(model, data, {body.id})[:, :2] - data.xpos[body.id][:2]
            self.outlines[name] = MultiPoint([tuple(p) for p in footprint]).convex_hull.simplify(0.001)
        self.arm_shadows = [self._arm_shadow(model, data, side) for side in ("left", "right")]
        if not REACH_MAP.exists():
            sys.exit(f"{REACH_MAP.name} is missing: run  python scene.py --build-reach-map")
        saved = np.load(REACH_MAP)
        self.reach = {"left": saved["left"], "right": saved["right"]}
        self.origin, self.cell = saved["origin"], float(saved["cell"])
        ny, nx = saved["left"].shape
        ys, xs = np.mgrid[0:ny, 0:nx]
        centers = np.stack([self.origin[0] + (xs + 0.5) * self.cell, self.origin[1] + (ys + 0.5) * self.cell], -1)
        self.visible = self._visible(model, data, centers.reshape(-1, 2)).reshape(ny, nx)
        self.areas = {}

    def _arm_shadow(self, model, data, side):
        """Table area an arm at home takes up or hides from the top camera: its points pushed along the
        camera's rays down to the table. An object there would be under the arm or hidden behind it."""
        bodies = {i for i in range(model.nbody) if model.body(i).name.startswith(f"{side}_")}
        points = body_points(model, data, bodies)
        eye = data.cam("top").xpos
        above = points[points[:, 2] > self.table_top + 1e-3]
        t = (self.table_top - eye[2]) / (above[:, 2] - eye[2])
        shadow = eye[:2] + t[:, None] * (above[:, :2] - eye[:2])
        return MultiPoint([tuple(p) for p in np.vstack([points[:, :2], shadow])]).convex_hull

    def _visible(self, model, data, xy):
        """Whether table points show in the top camera image (the 640x360 crop the model gets)."""
        cam = data.cam("top")
        local = (np.c_[xy, np.full(len(xy), self.table_top)] - cam.xpos) @ cam.xmat.reshape(3, 3)
        depth = -local[:, 2]
        f = 240 / np.tan(np.radians(model.cam_fovy[model.camera("top").id]) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = f * local[:, 0] / depth + 320, -f * local[:, 1] / depth + 240
        m = CAMERA_MARGIN
        return (depth > 0) & (u > m) & (u < 640 - m) & (v > 60 + m) & (v < 420 - m)

    def area(self, setting):
        """Where a setting may place objects: the polygon, its cells, and how often each cell is drawn."""
        key = (setting.reach, setting.edge_falloff, setting.edge_min_weight)
        if key not in self.areas:
            from scipy import ndimage

            if setting.reach == "both":
                reach = self.reach["left"] & self.reach["right"]
            elif setting.reach == "either":
                reach = self.reach["left"] | self.reach["right"]
            else:
                raise ValueError("reach is 'both' or 'either'")
            allowed = reach & self.visible
            o, c = self.origin, self.cell
            polygon = unary_union([box(o[0] + x * c, o[1] + y * c, o[0] + (x + 1) * c, o[1] + (y + 1) * c)
                                   for y, x in zip(*np.nonzero(allowed))])
            cells = np.argwhere(allowed)
            # Spots near the edge of the area are drawn less often: weight rises from edge_min_weight at the
            # edge to 1 at edge_falloff inside.
            inside = ndimage.distance_transform_edt(allowed) * c
            weight = np.clip(inside[allowed] / max(setting.edge_falloff, 1e-9), setting.edge_min_weight, 1.0)
            self.areas[key] = (prep(polygon.buffer(1e-6)), cells, weight / weight.sum())
        return self.areas[key]

    def placed(self, name, x, y, yaw):
        """@return The footprint of the object at (x, y), turned by yaw radians."""
        return affinity.translate(affinity.rotate(self.outlines[name], yaw, origin=(0, 0), use_radians=True), x, y)


_LAYOUT = None


def layout_for(model=None):
    """The Layout every scene is drawn from, built once from the package's UNCHANGED scene (default camera,
    lights and table), whatever model is passed: camera or lighting changes (ood.py) never move a seed's objects.
    """
    global _LAYOUT
    if _LAYOUT is None:
        _LAYOUT = Layout(mujoco.MjModel.from_xml_string(_package_xml()))
    return _LAYOUT


# ---------------------------------------------------------------------------------------------- sample and apply

def sample_scene(seed, model, setting="default"):
    """Draw the start scene of a seed under a setting.
    @return {"version", "setting", "seed", "split", "objects": {name: {"x", "y", "z", "yaw_deg"}}}, in metres.
    """
    seed, rules = int(seed), setting_of(setting)
    layout = layout_for(model)
    missing = [n for n in rules.objects if n not in layout.outlines]
    if missing:
        raise ValueError(f"Setting {setting!r} names objects the scene does not have: {missing}")
    allowed, cells, weights = layout.area(rules)
    # Objects the setting does not move stay at their default pose and are kept clear of, like the arms.
    fixed = [affinity.translate(layout.outlines[n], *layout.defaults[n]["pos"][:2])
             for n in layout.outlines if n not in rules.objects and n in SETTINGS["default"].objects]
    rng = rng_for(seed, "placement", setting=setting)
    lo, hi = np.radians(rules.yaw_deg[0]), np.radians(rules.yaw_deg[1])
    for _attempt in range(SCENE_ATTEMPTS):
        shapes, objects = list(layout.arm_shadows) + fixed, {}
        for name in rng.permutation(rules.objects):
            for _try in range(TRIES_PER_OBJECT):
                i, j = cells[rng.choice(len(cells), p=weights)]
                x = layout.origin[0] + (j + rng.random()) * layout.cell
                y = layout.origin[1] + (i + rng.random()) * layout.cell
                yaw = rng.uniform(lo, hi)
                shape = layout.placed(name, x, y, yaw)
                if allowed.contains(shape) and all(shape.distance(s) >= rules.min_gap for s in shapes):
                    break
            else:
                break  # this object found no spot: start the whole scene again (same stream, so still reproducible)
            shapes.append(shape)
            z = layout.defaults[name]["pos"][2] + (LIFT if layout.defaults[name]["free"] else 0)
            objects[str(name)] = {"x": round(float(x), 5), "y": round(float(y), 5), "z": round(float(z), 5),
                                  "yaw_deg": round(float(np.degrees(yaw)), 3)}
        else:
            return {"version": SCENE_VERSION, "setting": setting, "seed": seed, "split": split_of(seed),
                    "objects": dict(sorted(objects.items()))}
    raise RuntimeError(f"Seed {seed}, setting {setting}: no layout found in {SCENE_ATTEMPTS} attempts.")


def difficulty(scene, model):
    """How far the objects are from the arms: the mean distance (m) from each object to the nearer arm base."""
    bases = layout_for(model).bases
    return float(np.mean([np.min(np.hypot(*(bases - [o["x"], o["y"]]).T)) for o in scene["objects"].values()]))


def select_eval_seeds(model, setting="default"):
    """The fixed eval set: of the EVAL_POOL seeds' scenes sorted by difficulty, the first seeds of the easy,
    medium and hard thirds, EVAL_SET_SIZE in all. Same model and setting, same set."""
    scores = np.array([difficulty(sample_scene(s, model, setting), model) for s in EVAL_POOL])
    edges = np.quantile(scores, [1 / 3, 2 / 3])
    tiers = np.digitize(scores, edges)
    sizes = [EVAL_SET_SIZE // 3 + (k < EVAL_SET_SIZE % 3) for k in range(3)]
    chosen = [s for k in range(3) for s in [p for p, t in zip(EVAL_POOL, tiers) if t == k][:sizes[k]]]
    return tuple(sorted(chosen))


_EVAL_CACHE = {}


def eval_seeds(setting="default", model=None):
    """@return The setting's fixed eval seeds (EVAL_SEEDS for the default setting)."""
    if setting == "default" and EVAL_SEEDS:
        return EVAL_SEEDS
    if setting not in _EVAL_CACHE:
        if model is None:
            from yam_env import YamEnv

            model = YamEnv().model
        _EVAL_CACHE[setting] = select_eval_seeds(model, setting)
    return _EVAL_CACHE[setting]


def apply_scene(env, scene):
    """Put the objects where the scene says, and make that the state a reset returns to.
    @param env A YamEnv (anything with .model and .data), before the run starts.
    """
    model, data = env.model, env.data
    layout = layout_for(model)
    for name, place in scene["objects"].items():
        default = layout.defaults[name]
        pos = np.array([place["x"], place["y"], place["z"]])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, yaw_quat(np.radians(place["yaw_deg"])), default["quat"])
        body = model.body(name)
        if default["free"]:
            adr = model.jnt_qposadr[body.jntadr[0]]
            model.qpos0[adr:adr + 3], model.qpos0[adr + 3:adr + 7] = pos, quat
            data.qpos[adr:adr + 3], data.qpos[adr + 3:adr + 7] = pos, quat
            data.qvel[model.jnt_dofadr[body.jntadr[0]]:][:6] = 0
        else:
            model.body_pos[body.id], model.body_quat[body.id] = pos, quat
    mujoco.mj_forward(model, data)


def settle(env, seconds=1.0):
    """Let the objects come to rest with the arms holding still, then check them.
    @return {name: {"moved_cm", "on_table"}} and whether every object is at rest on the table.
    """
    data, layout = env.data, layout_for(env.model)
    names = [n for n in layout.outlines if n in SETTINGS["default"].objects or layout.defaults[n]["free"]]
    start = {name: data.body(name).xpos.copy() for name in names}
    for _ in range(int(seconds / 0.05)):
        state = env.state()
        env.step(state["left"], state["right"])
    report, ok = {}, True
    for name in names:
        pos = data.body(name).xpos
        moved = float(np.linalg.norm(pos[:2] - start[name][:2]))
        on_table = bool(pos[2] > layout.table_top - 0.005)
        speed = float(np.linalg.norm(data.body(name).cvel[3:])) if layout.defaults[name]["free"] else 0.0
        report[name] = {"moved_cm": round(moved * 100, 2), "on_table": on_table}
        ok &= on_table and moved < 0.02 and speed < 0.02
    return report, ok


# ---------------------------------------------------------------------------------------------- scene controls

NOT_RANDOM = {"camera_target"}  # ood.py controls that pick what to edit in the page, not a scene value
ARM_ATTEMPTS = 30  # random arm angles that collide are drawn again (deterministically) up to this many times


def random_controls():
    """@return The ood.py scene controls that can be drawn from the seed."""
    import ood

    return sorted(set(ood.FIELDS) - NOT_RANDOM)


def draw_controls(seed, names, setting="default", attempt=0):
    """Values for the ood.py scene controls named (camera, arm angles, blur, light, table, ...), drawn from the seed
    within each control's range and step. Same (setting, seed, control), same value; each control has its own
    stream, so choosing more or fewer random controls never changes the others' values. `attempt` > 0 redraws only
    the arm angles (after a drawn pose collided).
    """
    import ood

    values = {}
    for name in sorted(names):
        field = ood.FIELDS.get(name)
        if field is None or name in NOT_RANDOM:
            raise ValueError(f"Not a random scene control: {name}")
        redraw = (attempt,) if attempt and name in ood.ARM_FIELDS else ()
        rng = rng_for(seed, "ood", name, *redraw, setting=setting)
        if "choices" in field:
            options = sorted(field["choices"])
            values[name] = options[int(rng.integers(len(options)))]
        else:
            steps = int(round((field["max"] - field["min"]) / field["step"]))
            values[name] = round(field["min"] + field["step"] * int(rng.integers(steps + 1)), 6)
    return values


def resolve_controls(seed, manual, random_names, setting="default", attempt=0):
    """The scene-control values of a run: the manual ones, with the random ones drawn from the seed.
    Random camera fine-tuning goes to the workspace (top) camera; other cameras keep their manual values.
    """
    import ood

    if random_names and seed is None:
        raise ValueError("Random scene controls need a seed (not --default-scene).")
    controls = {**manual}
    drawn = draw_controls(seed, random_names, setting, attempt) if random_names else {}
    camera = {k: v for k, v in drawn.items() if k in ood.CAMERA_FIELDS}
    controls.update({k: v for k, v in drawn.items() if k not in camera})
    if camera:
        target = controls.pop("camera_target", "top")
        manual_camera = {k: controls.pop(k) for k in list(controls) if k in ood.CAMERA_FIELDS}
        views = {name: dict(view) for name, view in controls.pop("camera_views", {}).items()}
        if manual_camera:
            views[target] = {**views.get(target, {}), **manual_camera}
        views["top"] = {**views.get("top", {}), **camera}
        controls["camera_views"] = views
    return controls


# ---------------------------------------------------------------------------------------------- reproducibility

def fingerprint(value):
    """@return A short hash of a scene (or any JSON value): same contents, same hash."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def reference_now(model=None):
    """@return What REFERENCE, EVAL_FINGERPRINT, EVAL_SEEDS and CONTROLS_FINGERPRINT would be now."""
    reference = {name: {seed: fingerprint(sample_scene(seed, model, name)) for seed in seeds}
                 for name, seeds in REFERENCE.items()}
    seeds = select_eval_seeds(model)
    return reference, fingerprint(list(seeds)), seeds, controls_fingerprint()


def controls_fingerprint():
    """@return A hash of every ood.py scene control drawn at random for a few seeds."""
    return fingerprint([draw_controls(seed, random_controls()) for seed in (0, 42, 1000000)])


def check_reproducible(model=None):
    """@return What no longer matches the recorded fingerprints, as text lines (empty: everything reproduces)."""
    changed = [f"setting {name} seed {seed}" for name, seeds in REFERENCE.items() for seed, expected in seeds.items()
               if fingerprint(sample_scene(seed, model, name)) != expected]
    if fingerprint(list(EVAL_SEEDS)) != EVAL_FINGERPRINT:
        changed.append("EVAL_SEEDS was edited")
    if controls_fingerprint() != CONTROLS_FINGERPRINT:
        changed.append("random scene controls (ood.py ranges or drawing)")
    return changed


def warn_if_not_reproducible(model=None):
    """Print a warning when a seed would no longer give the same scene as recorded."""
    changed = check_reproducible(model)
    if changed:
        print(f"  WARNING: {', '.join(changed)}: not what SCENE_VERSION {SCENE_VERSION} recorded. The "
              "environment changed (dreamlabs, reach map, ood.py or scene.py); seeds from before may not reproduce.",
              flush=True)


# ---------------------------------------------------------------------------------------------- hook for sim.py

def add_arguments(parser):
    """Add --seed, --split, --setting, --ood-random and --default-scene to a run's argument parser."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, help="the start scene's seed (a new random one when left out)")
    group.add_argument("--default-scene", action="store_true", help="the package's fixed scene, not random")
    parser.add_argument("--split", choices=("train", "eval"),
                        help="refuse a seed of the other split; with no --seed, draw a new seed of this split")
    parser.add_argument("--setting", default="default", choices=sorted(SETTINGS),
                        help="which objects move and how (scene.py SETTINGS)")
    if not any("--ood" in action.option_strings for action in parser._actions):
        parser.add_argument("--ood", type=json.loads, default={}, help="scene settings (ood.py) as a JSON object")
    parser.add_argument("--ood-random", default="", type=lambda text: [n for n in text.split(",") if n],
                        help="comma-separated ood.py scene controls to draw from the seed (others: --ood values)")


def pick_seed(args):
    """@return The run's seed (None with --default-scene): --seed, or a new one of --split. Stops on a bad seed."""
    if args.default_scene:
        return None
    try:
        seed = args.seed if args.seed is not None else new_seed(args.split or "train", args.setting)
        check_split(seed, args.split or split_of(seed))
    except ValueError as error:
        print(f"Stopped: {error}", flush=True)
        sys.exit(1)
    return seed


def build_env(args):
    """Make the run's simulator: the seed's object layout, the scene controls (ood.py: --ood values, with the
    --ood-random ones drawn from the seed) and the arms' initial angles; print what a rerun needs.
    @return The environment (for dream_control.main(shared_env=...)) and the scene-control changes in effect.
    """
    import ood
    import yam_env

    layout_for()  # from the unchanged package scene, before ood.py patches it
    seed = pick_seed(args)

    def stop(message):
        print(f"Stopped: {message}", flush=True)
        sys.exit(1)

    random_arm = set(args.ood_random) & ood.ARM_FIELDS
    for attempt in range(ARM_ATTEMPTS if random_arm else 1):
        try:
            changes = ood.validate(resolve_controls(seed, args.ood, args.ood_random, args.setting, attempt))
        except ValueError as error:
            stop(error)
        if seed is None:
            print("Scene: default (not random)", flush=True)
            env = ood.build_environment(yam_env, changes) if changes else yam_env.YamEnv()
            break
        # The arms are posed after the objects are placed, so the pose is checked against the seed's objects.
        angles = {name: value for name, value in changes.items() if name in ood.ARM_FIELDS}
        rest = {name: value for name, value in changes.items() if name not in ood.ARM_FIELDS}
        env = ood.build_environment(yam_env, rest) if rest else yam_env.YamEnv()
        try:
            scene = sample_scene(seed, None, args.setting)
        except (ValueError, RuntimeError) as error:
            stop(error)
        apply_scene(env, scene)
        try:
            if angles:
                ood._set_arm_pose(env, angles)
                # A pose deep inside something can overflow MuJoCo's contact memory, and then the overlap check
                # above may not see every contact: count that as an overlap too.
                if any(env.data.warning[w].number for w in (mujoco.mjtWarning.mjWARN_CONTACTFULL,
                                                            mujoco.mjtWarning.mjWARN_CNSTRFULL)):
                    raise ValueError("Initial arm pose overlaps another part of the scene. Adjust the joint angles.")
            break
        except ValueError as error:
            env.close()
            if not random_arm or attempt == ARM_ATTEMPTS - 1:
                stop(error)
    if seed is None:
        return env, changes
    _report, settled = settle(env)
    again = f"--seed {seed}" + (f" --setting {args.setting}" if args.setting != "default" else "")
    if args.ood_random:
        again += f" --ood-random {','.join(args.ood_random)}"
    print(f"Scene: seed {seed} ({scene['split']}, setting {args.setting}); run again with {again} for the same start",
          flush=True)
    if changes:
        print(f"Scene controls: {json.dumps(changes, sort_keys=True)}"
              + (f" (from the seed: {', '.join(sorted(args.ood_random))})" if args.ood_random else ""), flush=True)
    if not settled:
        print("  Note: an object did not come to rest before the start.", flush=True)
    warn_if_not_reproducible()
    return env, changes


# ---------------------------------------------------------------------------------------------- reach map

def build_reach_map(samples=2_000_000, seed=0):
    """Map the 1 cm table cells each arm reaches, and save it to scene_reach_map.npz. Rebuilding changes every
    scene: bump SCENE_VERSION and refresh REFERENCE afterwards.

    Random joint angles inside the URDF limits go through forward kinematics. A configuration counts when the
    gripper tip (tcp) is at grasp height above the table, the gripper points down at least somewhat (not
    sideways or up from below), and no part of that arm touches the table. Small holes between hits are closed.
    """
    from scipy import ndimage
    from yam_env import YamEnv

    cell, grasp_z, min_down = 0.01, (0.015, 0.07), 0.3
    env = YamEnv()
    m, d = env.model, env.data
    table = m.geom("table")
    top = table.pos[2] + table.size[2]
    x0, y0 = table.pos[0] - table.size[0], table.pos[1] - table.size[1]
    nx, ny = int(round(2 * table.size[0] / cell)), int(round(2 * table.size[1] / cell))
    rng = np.random.default_rng(seed)
    masks = {}
    for side in ("left", "right"):
        joints = [m.joint(f"{side}_joint{i}") for i in range(1, 7)]
        qadr = [j.qposadr[0] for j in joints]
        lo, hi = np.array([j.range[0] for j in joints]), np.array([j.range[1] for j in joints])
        arm = {i for i in range(m.nbody) if m.body(i).name.startswith(f"{side}_")}
        tcp, link6 = m.site(f"{side}_tcp").id, m.body(f"{side}_link6").id
        hits = np.zeros((ny, nx), dtype=bool)
        for q in rng.uniform(lo, hi, size=(samples, 6)):
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            p = d.site_xpos[tcp]
            i, j = int((p[1] - y0) / cell), int((p[0] - x0) / cell)
            if not (top + grasp_z[0] <= p[2] <= top + grasp_z[1] and 0 <= i < ny and 0 <= j < nx):
                continue
            approach = p - d.xpos[link6]
            if -approach[2] / np.linalg.norm(approach) < min_down:
                continue
            mujoco.mj_collision(m, d)
            if any(table.id in (c.geom1, c.geom2) and (m.geom_bodyid[c.geom1] in arm or m.geom_bodyid[c.geom2] in arm)
                   for c in d.contact[: d.ncon]):
                continue
            hits[i, j] = True
        masks[side] = ndimage.binary_fill_holes(ndimage.binary_closing(hits, np.ones((3, 3)), iterations=2))
        print(f"{side} arm: {masks[side].sum()} cells", flush=True)
    reach = masks["left"] | masks["right"]
    np.savez_compressed(REACH_MAP, reach=reach, left=masks["left"], right=masks["right"],
                        origin=np.array([x0, y0]), cell=cell, samples=samples, seed=seed)
    print(f"reachable by either arm: {reach.sum()} cm^2, by both: {(masks['left'] & masks['right']).sum()} cm^2 "
          f"-> {REACH_MAP.name}")


# ---------------------------------------------------------------------------------------------- command line

def save_views(images, path):
    """The top camera on the left, the two wrist cameras stacked on the right, as one JPEG (the website's layout)."""
    from PIL import Image

    canvas = Image.new("RGB", (720, 270))
    canvas.paste(Image.fromarray(images["top"]).resize((480, 270)), (0, 0))
    canvas.paste(Image.fromarray(images["left"]).resize((240, 135)), (480, 0))
    canvas.paste(Image.fromarray(images["right"]).resize((240, 135)), (480, 135))
    canvas.save(str(path) + ".tmp", format="JPEG", quality=85)
    Path(str(path) + ".tmp").replace(path)


def main():
    parser = argparse.ArgumentParser(description="Show the random start scene of a seed")
    parser.add_argument("seed", type=int, nargs="?")
    parser.add_argument("--setting", default="default", choices=sorted(SETTINGS))
    parser.add_argument("--image", help="save the camera views of the start scene to this JPEG file")
    parser.add_argument("--check", action="store_true", help="check that recorded seeds still reproduce")
    parser.add_argument("--update", action="store_true", help="with --check: print new REFERENCE values")
    parser.add_argument("--build-reach-map", action="store_true", help="rebuild scene_reach_map.npz")
    parser.add_argument("--samples", type=int, default=2_000_000, help="with --build-reach-map: configurations per arm")
    args = parser.parse_args()
    if args.build_reach_map:
        return build_reach_map(args.samples)

    from yam_env import YamEnv

    env = YamEnv()
    if args.check:
        if args.update:
            reference, eval_print, seeds, controls_print = reference_now()
            print(f"REFERENCE = {reference!r}\nEVAL_FINGERPRINT = {eval_print!r}\nEVAL_SEEDS = {seeds!r}\n"
                  f"CONTROLS_FINGERPRINT = {controls_print!r}")
            return
        changed = check_reproducible()
        count = sum(len(s) for s in REFERENCE.values())
        print(f"reproducible: all {count} reference scenes and the eval set match" if not changed
              else "NOT reproducible: " + ", ".join(changed))
        sys.exit(1 if changed else 0)
    if args.seed is None:
        parser.error("give a seed, --check or --build-reach-map")
    warn_if_not_reproducible()
    scene = sample_scene(args.seed, env.model, args.setting)
    apply_scene(env, scene)
    report, ok = settle(env)
    print(json.dumps(scene, indent=1))
    print(f"difficulty: {difficulty(scene, env.model):.3f} m (mean distance to the nearer arm base)")
    print("settled:" if ok else "NOT settled:", json.dumps(report))
    if args.image:
        save_views(env.images(), args.image)
        print(f"camera views saved to {args.image}")
    env.close()


if __name__ == "__main__":
    main()
