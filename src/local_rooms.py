"""YAM room styling from the public Sandbox room catalog.

The Sandbox's baked room meshes render in Three.js. The bundled YAM MuJoCo
example has a different frame and scene graph, so only desk texture, desk color,
background color, and lighting are mapped here. Wall panels are local geometry.
No Dream Labs asset is included in this repository.
"""

import io
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


# Source: https://platform.dreamlabs.ai/sim-app/assets/rooms-eS8SKP-n.js
# These values are the public room descriptors, not an exported MuJoCo room.
ROOMS = {
    "kitchen": {"desk": (.603, .4789, .3562), "background": (.4887, .4557, .3839),
                "light": (1, .96, .9), "texture": "desk-CsEjigx0.webp"},
    "sunny_apartment": {"desk": (.6518, .5156, .365), "background": (.445, .4138, .3553),
                        "light": (1, .96, .9), "texture": "desk-O-Sf6QWg.webp"},
    "cottage": {"desk": (.6518, .5156, .365), "background": (.2535, .3553, .4666),
                "light": (1, .98, .95), "texture": "desk-O-Sf6QWg.webp"},
    "walnut_study": {"desk": (.4894, .3542, .2587), "background": (.1836, .1025, .0596),
                     "light": (1, .92, .82), "texture": "desk-eQj0s_IU.webp"},
    "evening_den": {"desk": (.4837, .3346, .2358), "background": (.0254, .011, .005),
                    "light": (1, .84, .66), "texture": "desk-C_25yVaI.webp"},
}
ASSET_ORIGIN = "https://platform.dreamlabs.ai/sim-app/assets/"


def desk_texture(room):
    """Fetch the publicly served desk image to a temporary cache, without vendoring it."""
    name = ROOMS[room]["texture"]
    path = Path(tempfile.gettempdir()) / "sweat-dream-room-cache" / name.replace(".webp", ".png")
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ASSET_ORIGIN + name, timeout=10) as response:
        if response.status != 200:
            raise OSError(f"Room texture HTTP {response.status}")
        payload = response.read(8_000_001)
    if len(payload) > 8_000_000:
        raise ValueError("Room texture exceeds 8 MB")
    with Image.open(io.BytesIO(payload)) as source:
        source.convert("RGB").save(path, format="PNG")
    return path


def _rgb(values):
    return " ".join(str(v) for v in (*values, 1))


def themed_xml(base_xml, room, texture_path=None):
    if room == "yam_tabletop":
        return base_xml
    profile = ROOMS[room]
    root = ET.fromstring(base_xml)
    world = root.find("worldbody")
    table = world.find(".//geom[@name='table']")
    if table is None:
        raise ValueError("Local scene has no table geometry")
    table.attrib.pop("material", None)
    table.set("rgba", _rgb(profile["desk"]))
    if texture_path is not None:
        asset = root.find("asset")
        ET.SubElement(asset, "texture", name="sandbox_desk", type="2d", file=str(texture_path))
        ET.SubElement(asset, "material", name="sandbox_desk_material", texture="sandbox_desk",
                      texrepeat="1 1", rgba=_rgb(profile["desk"]))
        table.set("material", "sandbox_desk_material")
        table.set("rgba", "1 1 1 1")
    for geom in world.findall(".//geom"):
        if geom.get("type") == "plane":
            geom.set("rgba", _rgb(profile["background"]))
    for light in world.findall(".//light"):
        light.set("diffuse", " ".join(str(v) for v in profile["light"]))
    # Nonphysical background walls; no task object or collision parameters change.
    color = _rgb(profile["background"])
    for name, pos, size in (
        ("back", "0 .95 1.35", "1.55 .015 1.35"),
        ("left", "-1.55 0 1.35", ".015 1.1 1.35"),
        ("right", "1.55 0 1.35", ".015 1.1 1.35"),
    ):
        ET.SubElement(world, "geom", name=f"room_{name}", type="box", pos=pos,
                      size=size, rgba=color, contype="0", conaffinity="0")
    return ET.tostring(root, encoding="unicode")
