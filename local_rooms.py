"""Original lightweight MuJoCo room backdrops for the bundled YAM tabletop.

These are local approximations of Sandbox room themes, not Sandbox scene assets.
The table, robot, objects, and policy cameras remain in the calibrated YAM scene.
"""

import xml.etree.ElementTree as ET


PALETTES = {
    "kitchen": (".88 .88 .83 1", ".94 .93 .86 1", ".65 .68 .69 1", ".78 .81 .79 1", ".86 .87 .84 1"),
    "sunny_apartment": (".93 .88 .76 1", ".97 .91 .79 1", ".74 .81 .87 1", ".76 .65 .49 1", ".87 .75 .55 1"),
    "cottage": (".80 .72 .60 1", ".91 .83 .69 1", ".64 .75 .62 1", ".58 .39 .26 1", ".70 .48 .28 1"),
    "walnut_study": (".43 .37 .33 1", ".65 .57 .48 1", ".35 .43 .48 1", ".34 .23 .17 1", ".43 .30 .22 1"),
    "evening_den": (".28 .32 .43 1", ".42 .45 .57 1", ".35 .39 .51 1", ".26 .25 .31 1", ".48 .47 .56 1"),
}


def themed_xml(base_xml, room):
    if room == "yam_tabletop":
        return base_xml
    if room not in PALETTES:
        raise ValueError(f"Unknown local room: {room}")
    root = ET.fromstring(base_xml)
    world = root.find("worldbody")
    wall, ceiling, accent, trim, tabletop = PALETTES[room]
    table = world.find("geom[@name='table']")
    table.attrib.pop("material", None)
    table.set("rgba", tabletop)
    for geom in world.findall("geom"):
        if geom.get("type") == "plane":
            geom.set("rgba", wall)

    def panel(name, pos, size, color):
        ET.SubElement(world, "geom", name=f"room_{name}", type="box", pos=pos,
                      size=size, rgba=color, contype="0", conaffinity="0")

    panel("back_wall", "0 .95 1.35", "1.55 .015 1.35", wall)
    panel("left_wall", "-1.55 0 1.35", ".015 1.1 1.35", wall)
    panel("right_wall", "1.55 0 1.35", ".015 1.1 1.35", wall)
    panel("ceiling", "0 0 2.68", "1.55 1.1 .015", ceiling)
    panel("baseboard", "0 .925 .13", "1.55 .025 .075", trim)
    # A simple nonphysical interior cue stays outside the tabletop workspace.
    if room in ("sunny_apartment", "cottage", "kitchen"):
        panel("window", ".9 .925 1.6", ".38 .018 .43", accent)
        panel("window_h", ".9 .90 1.6", ".39 .022 .025", trim)
        panel("window_v", ".9 .90 1.6", ".025 .022 .44", trim)
    else:
        panel("art", ".9 .925 1.55", ".32 .02 .38", accent)
        panel("art_frame", ".9 .90 1.15", ".37 .025 .025", trim)
    return ET.tostring(root, encoding="unicode")
