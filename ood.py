"""OOD controls shared by the web API and the MuJoCo scene builder."""

import math
import xml.etree.ElementTree as ET
from unittest.mock import patch


def _number(label, default, minimum, maximum, step):
    return dict(label=label, default=default, min=minimum, max=maximum, step=step)


def _choice(label, initial, **choices):
    return dict(label=label, default=initial, choices=choices)


# Whole-degree bounds within the installed YAM joint limits, in joint order.
JOINT_BOUNDS = ((-149, 179), (0, 209), (0, 179), (-90, 90), (-90, 90), (-120, 120))
ARM_GROUPS = {
    f"{side.title()} arm · initial angles": {
        f"{side}_joint{index}": _number(f"Joint {index} (°)", 0, minimum, maximum, 1)
        for index, (minimum, maximum) in enumerate(JOINT_BOUNDS, 1)
    }
    for side in ("left", "right")
}
ARM_FIELDS = {name for fields in ARM_GROUPS.values() for name in fields}

GROUPS = {
    "Start camera view": {
        "camera_preset": _choice(
            "Workspace viewpoint", "default", default="Default", overhead="Overhead",
            front="Front angle", left="Left diagonal", right="Right diagonal",
        ),
        "camera_target": _choice(
            "Camera to fine-tune", "top", top="Workspace camera", left="Left wrist", right="Right wrist",
        ),
        "camera_x_cm": _number("X offset (cm, mount axes)", 0, -10, 10, 0.5),
        "camera_y_cm": _number("Y offset (cm, mount axes)", 0, -10, 10, 0.5),
        "camera_z_cm": _number("Z offset (cm, mount axes)", 0, -10, 10, 0.5),
        "camera_pitch_deg": _number("Tilt (°)", 0, -30, 30, 1),
        "camera_yaw_deg": _number("Pan (°)", 0, -30, 30, 1),
        "camera_roll_deg": _number("Roll (°)", 0, -25, 25, 1),
        "camera_fov_offset_deg": _number("Field of view change (°)", 0, -20, 20, 1),
    },
    **ARM_GROUPS,
    "Image effects": {
        "blur_px": _number("Gaussian blur radius (px)", 0, 0, 12, 0.5),
        "blur_camera": _choice("Blur target", "all", all="All cameras", top="Top camera", left="Left wrist", right="Right wrist"),
    },
    "Lighting & surface": {
        "light": _number("Light intensity (×)", 1, 0.25, 2, 0.05),
        "table": _choice("Work surface", "oak", oak="Default texture", gray="Matte gray"),
    },
}

FIELDS = {name: field for group in GROUPS.values() for name, field in group.items()}


CAMERA_FIELDS = {
    name: field for name, field in GROUPS["Start camera view"].items() if "min" in field
}
CAMERA_TARGETS = FIELDS["camera_target"]["choices"]


def _nondefaults(values, fields):
    if not isinstance(values, dict):
        raise ValueError("Scene settings must be a JSON object.")
    changes = {}
    for name, value in values.items():
        field = fields.get(name)
        if field is None:
            raise ValueError(f"Unknown scene control: {name}")
        if "choices" in field:
            valid = isinstance(value, str) and value in field["choices"]
        else:
            valid = type(value) in (int, float) and field["min"] <= value <= field["max"]
        if not valid:
            raise ValueError(f"Invalid value for {field['label']}.")
        if value != field["default"]:
            changes[name] = value
    return changes


def validate(values):
    if not isinstance(values, dict):
        raise ValueError("Scene settings must be a JSON object.")
    changes = _nondefaults({name: value for name, value in values.items() if name != "camera_views"}, FIELDS)
    if "camera_views" in values:
        views = values["camera_views"]
        if not isinstance(views, dict) or views.keys() - CAMERA_TARGETS.keys():
            raise ValueError("Invalid camera views.")
        if changes.keys() & (CAMERA_FIELDS.keys() | {"camera_target"}):
            raise ValueError("Use either per-camera settings or a single camera target.")
        views = {name: _nondefaults(view, CAMERA_FIELDS) for name, view in views.items()}
        if any(views.values()):
            changes["camera_views"] = {name: view for name, view in views.items() if view}
    if "blur_px" not in changes:
        changes.pop("blur_camera", None)
    return changes


def _set_arm_pose(env, angles):
    import mujoco

    for name, degrees in angles.items():
        joint = env.model.joint(name)
        angle = math.radians(degrees)
        if not joint.range[0] <= angle <= joint.range[1]:
            raise ValueError(f"{name} is outside the simulator joint limits.")
        env.data.joint(name).qpos[0] = angle
        env.data.actuator(name).ctrl[0] = angle
    mujoco.mj_forward(env.model, env.data)
    if any(contact.dist < -0.001 for contact in env.data.contact):
        raise ValueError("Initial arm pose overlaps another part of the scene. Adjust the joint angles.")


def _with_blur(capture, radius, target):
    from PIL import Image, ImageFilter
    import numpy as np

    filter_ = ImageFilter.GaussianBlur(radius)

    def images():
        return {
            name: np.asarray(Image.fromarray(frame).filter(filter_)) if target in ("all", name) else frame
            for name, frame in capture().items()
        }

    return images


# World-space positions in metres, aimed at the center of the workspace.
VIEWPOINTS = {
    "overhead": (0.0, -0.08, 1.65),
    "front": (0.0, 0.58, 1.43),
    "left": (-0.55, 0.37, 1.52),
    "right": (0.55, 0.37, 1.52),
}


def _set_viewpoint(camera, preset):
    import mujoco
    import numpy as np

    position = np.array(VIEWPOINTS[preset])
    backward = position - [0.0, -0.08, 0.77]
    backward /= np.linalg.norm(backward)
    up = [0, 1, 0] if preset == "overhead" else [0, 0, 1]
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    basis = np.column_stack((right, np.cross(backward, right), backward))
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, basis.ravel())
    camera.set("pos", " ".join(map(str, position)))
    camera.set("quat", " ".join(map(str, quaternion)))


def _rotate_camera(camera, changes):
    import mujoco
    import numpy as np

    quaternion = np.fromstring(camera.attrib["quat"], sep=" ")
    for axis, field in enumerate(("camera_pitch_deg", "camera_yaw_deg", "camera_roll_deg")):
        angle = math.radians(changes.get(field, 0)) / 2
        rotation = np.zeros(4)
        rotation[0] = math.cos(angle)
        rotation[axis + 1] = math.sin(angle)
        result = np.empty(4)
        mujoco.mju_mulQuat(result, quaternion, rotation)
        quaternion = result
    camera.set("quat", " ".join(map(str, quaternion)))


def _adjust_camera(camera, changes):
    offsets = [changes.get(f"camera_{axis}_cm", 0) for axis in "xyz"]
    if any(offsets):
        position = map(float, camera.attrib["pos"].split())
        camera.set("pos", " ".join(f"{axis + cm / 100:.12g}" for axis, cm in zip(position, offsets)))
    if {"camera_pitch_deg", "camera_yaw_deg", "camera_roll_deg"} & changes.keys():
        _rotate_camera(camera, changes)
    if "camera_fov_offset_deg" in changes:
        offset = changes["camera_fov_offset_deg"]
        if "focalpixel" in camera.attrib:
            fx, fy = map(float, camera.attrib["focalpixel"].split())
            half_height = float(camera.attrib["resolution"].split()[1]) / 2
            half_fov = math.atan(half_height / fy)
            scale = math.tan(half_fov) / math.tan(half_fov + math.radians(offset) / 2)
            camera.set("focalpixel", f"{fx * scale} {fy * scale}")
        else:
            camera.set("fovy", str(float(camera.get("fovy", "45")) + offset))


def scene_xml(xml, changes):
    """Adjust the cameras and appearance before MuJoCo compiles the scene."""
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    if "camera_preset" in changes:
        _set_viewpoint(world.find("camera[@name='top']"), changes["camera_preset"])
    views = changes.get("camera_views", {changes.get("camera_target", "top"): changes})
    for name, view in views.items():
        _adjust_camera(world.find(f".//camera[@name='{name}']"), view)
    if "light" in changes:
        for light in world.findall("light"):
            for color in ("ambient", "diffuse"):
                if color in light.attrib:
                    light.set(color, " ".join(str(float(v) * changes["light"]) for v in light.get(color).split()))
    if changes.get("table") == "gray":
        table = world.find("geom[@name='table']")
        table.attrib.pop("material", None)
        table.set("rgba", ".36 .37 .39 1")
    return ET.tostring(root, encoding="unicode")


def build_environment(sdk, changes):
    """Construct a standard SDK environment with an adjusted initial scene."""
    model_settings = CAMERA_FIELDS.keys() | {"camera_preset", "camera_views", "light", "table"}
    if model_settings & changes.keys():
        xml = scene_xml(sdk.build_xml(), changes)
        with patch.object(sdk, "build_xml", return_value=xml):
            env = sdk.YamEnv()
    else:
        env = sdk.YamEnv()

    angles = {name: degrees for name, degrees in changes.items() if name in ARM_FIELDS}
    if angles:
        _set_arm_pose(env, angles)
    if "blur_px" in changes:
        env.images = _with_blur(env.images, changes["blur_px"], changes.get("blur_camera", "all"))
    return env
