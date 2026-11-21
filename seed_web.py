"""The website's side of the seed (scene.py): standard library only, like app.py.

app.py passes seeding_args(body, ood.FIELDS) to sim.py for /api/run and /api/preview, serves static/seed.js,
and returns drawn_header(stdout) with the preview so the page can show what the seed drew.
"""

import json
import re

NOT_RANDOM = {"camera_target"}  # scene.py NOT_RANDOM: page selectors, not scene values


def seeding_args(body, fields):
    """sim.py arguments for the run's seed and the scene controls drawn from it.
    @param body The request: "seed" (empty: sim.py picks a new one) and "ood_random" (control names).
    @param fields ood.FIELDS, the scene controls.
    """
    seed = "" if body.get("seed") is None else str(body["seed"]).strip()
    if seed and not seed.isdigit():
        raise ValueError("The seed must be a whole number 0 or above, or empty.")
    names = body.get("ood_random") or []
    if not isinstance(names, list) or any(name not in fields or name in NOT_RANDOM for name in names):
        raise ValueError("Unknown random scene control.")
    return (["--seed", seed] if seed else []) + (["--ood-random", ",".join(names)] if names else [])


def drawn_header(stdout):
    """@return The X-Scene-Controls header: the scene-control values sim.py printed ("{}" when none)."""
    for line in stdout.splitlines():
        match = re.match(r"Scene controls: (\{.*\})", line)
        if match:
            json.loads(match.group(1))  # only ever pass on valid JSON
            return {"X-Scene-Controls": match.group(1)}
    return {"X-Scene-Controls": "{}"}
