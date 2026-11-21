"""OOD controls shared by the web API and the MuJoCo scene builder."""

import math
import xml.etree.ElementTree as ET
from unittest.mock import patch


def _number(label, default, minimum, maximum, step):
    return dict(label=label, default=default, min=minimum, max=maximum, step=step)


def _choice(label, default, **choices):
    return dict(label=label, default=default, choices=choices)


GROUPS = {
    "Camera view": {
        "camera_pitch_deg": _number("Top camera tilt (°)", 0, -20, 20, 1),
        "camera_yaw_deg": _number("Top camera pan (°)", 0, -20, 20, 1),
        "camera_roll_deg": _number("Top camera roll (°)", 0, -15, 15, 1),
        "camera_fov_offset_deg": _number("Top camera FOV change (°)", 0, -15, 15, 1),
        "camera_x_cm": _number("Top camera X offset (cm)", 0, -5, 5, 0.5),
        "camera_y_cm": _number("Top camera Y offset (cm)", 0, -5, 5, 0.5),
    },
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


def validate(values):
    if not isinstance(values, dict):
        raise ValueError("Scene settings must be an object.")
    changes = {}
    for name, value in values.items():
        field = FIELDS.get(name)
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
    if "blur_px" not in changes:
        changes.pop("blur_camera", None)
    return changes


def _shift(element, x_cm, y_cm):
    x, y, z = map(float, element.attrib["pos"].split())
    element.set("pos", f"{x + x_cm / 100:g} {y + y_cm / 100:g} {z:g}")


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


def scene_xml(xml, changes):
    """Adjust the SDK's scene before MuJoCo compiles it."""
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    camera = world.find("camera[@name='top']")

    if {"camera_x_cm", "camera_y_cm"} & changes.keys():
        _shift(camera, changes.get("camera_x_cm", 0), changes.get("camera_y_cm", 0))
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
    xml = scene_xml(sdk.build_xml(), changes)
    with patch.object(sdk, "build_xml", return_value=xml):
        env = sdk.YamEnv()
    if "blur_px" in changes:
        env.images = _with_blur(env.images, changes["blur_px"], changes.get("blur_camera", "all"))
    return env
