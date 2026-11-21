"""The website's side of the seed (scene.py): standard library only, like app.py.

app.py passes seeding_args(body, ood.FIELDS) to sim.py for /api/run and /api/preview, serves static/seed.js,
and returns drawn_header(stdout) with the preview so the page can show what the seed drew.
"""

import hashlib
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


def episode_seed(batch_seed, index):
    """The start-scene seed of episode `index` (0, 1, ...) in a batch with seed `batch_seed`: one seed is the whole
    batch. Every episode gets its own layout, the same batch seed gives the same episodes again, and batches with
    other seeds share none of them (two of the 10^6 train seeds can coincide by chance, rarely). A train seed."""
    digest = hashlib.sha256(f"sweat-dream batch {int(batch_seed)} episode {int(index)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 1_000_000


def drawn_header(stdout):
    """@return The X-Scene-Controls header (the scene-control values sim.py printed) and X-Scene-Layout (where each
    item starts, so the page's custom layout can start from the seed's); "{}" when sim.py printed none."""
    headers = {"X-Scene-Controls": "{}", "X-Scene-Layout": "{}"}
    for line in stdout.splitlines():
        match = re.match(r"Scene (controls|layout): (\{.*\})", line)
        if match:
            json.loads(match.group(2))  # only ever pass on valid JSON
            headers["X-Scene-" + match.group(1).title()] = match.group(2)
    return headers
