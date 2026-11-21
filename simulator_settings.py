"""Choices shown in the UI and the subset supported by the bundled local sim."""

ROBOTS = {
    "yam": "YAM",
    "widowx_ai": "Trossen WidowX AI",
    "metal": "MakerMods Metal",
    "so101": "SO-101",
    "openyam": "OpenYAM",
    "piper": "AgileX PiPER",
    "ur5e": "Universal Robots UR5e",
    "rebot_b601": "Seeed reBot B601",
    "maker_arm": "MakerMods Maker Arm",
}
ROOMS = {
    "yam_tabletop": "YAM tabletop",
    "kitchen": "Kitchen",
    "sunny_apartment": "Sunny apartment",
    "cottage": "Cottage",
    "walnut_study": "Walnut study",
    "evening_den": "Evening den",
}
LIGHTING = {"standard": "Standard", "warm": "Warm", "cool": "Cool"}
LAYOUTS = {"fixed": "Fixed starting layout", "varied": "Seeded object variation"}
# The robots the page offers: the team's target robot only. The others stay known (older runs, saved settings), and
# adding a key here shows it again ("openyam" runs here too).
TARGET_ROBOTS = ("yam",)
DEFAULT = {"robot": "yam", "room": "yam_tabletop", "lighting": "standard",
           "layout": "varied", "seed": None}


def normalize(value):
    """Validate a UI selection; remote choices can be saved but cannot run locally."""
    if not isinstance(value, dict):
        raise ValueError("Simulator settings must be an object.")
    selected = {**DEFAULT, **value}
    for key, choices in (("robot", ROBOTS), ("room", ROOMS),
                         ("lighting", LIGHTING), ("layout", LAYOUTS)):
        if not isinstance(selected[key], str) or selected[key] not in choices:
            raise ValueError(f"Unknown {key} setting: {selected[key]!r}.")
    seed = selected["seed"]
    if seed in ("", None):
        seed = None
    elif isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be an integer from 0 to 4294967295 or empty.")
    selected["seed"] = seed
    return selected


def local_supported(settings):
    return settings["robot"] in ("yam", "openyam") and settings["room"] in ROOMS


def catalog():
    return {"robots": ROBOTS, "rooms": ROOMS, "lighting": LIGHTING,
            "layouts": LAYOUTS, "local_robots": ["yam", "openyam"], "target_robots": list(TARGET_ROBOTS),
            "local_rooms": list(ROOMS),
            "sandbox_url": "https://platform.dreamlabs.ai/sandbox"}
