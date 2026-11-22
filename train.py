"""Train Custom Model!: fine-tune Dream-1 on a dataset saved from the collect page, through the Dream Labs API.

The steps, with the signed-in API key (app.py's PAGE_KEY):
    1. a saved dataset (datasets/<folder>/dream_format/, LeRobot v2.1) is checked here (nothing sent) and uploaded
       under a name, with the robot's embodiment (registered once as "yam-sim" from embodiment.json): free;
    2. Dream Labs processes it (ready_to_process -> processing -> processed): free;
    3. a fine-tuning job on a processed dataset: charged, PRICE_PER_STEP per step, held from the account's credits
       when it is queued; a queued job cannot be cancelled;
    4. the job runs (queued -> running -> succeeded) and publishes a model, dream-1-ft:<account>-<suffix>.
Everything Dream Labs knows (embodiments, datasets, jobs) is read back from it, so the page finds them again after a
restart. An upload's outcome is also kept in datasets/<folder>/dream_upload.json.
https://platform.dreamlabs.ai/api-documentation/fine-tuning/
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path

PRICE_PER_STEP = 0.20  # USD, at the time of writing; the Billing page shows the current price
STEPS = (500, 10000)  # a multiple of 100 in this range
NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
SUFFIX_RE = re.compile(r"^[a-z0-9-]{1,40}$")
EMBODIMENT_NAME = "yam-sim"
HIDDEN = ".hidden"  # a saved dataset's folder holds this file while it is hidden on the Train page (it stays on disk)
UPLOADS = {}  # folder name -> {"state": "checking" | "uploading" | "done" | "failed", "name", "error", "dataset_id"}
LOCK = threading.Lock()


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def suggested_name(folder):
    """A Dream Labs dataset name from a folder name: 20261122-005915-Banana_Rule_based_Only_50 ->
    banana-rule-based-only-50."""
    words = re.sub(r"^\d{8}-\d{6}-", "", folder)
    return (re.sub(r"[^a-z0-9]+", "-", words.lower()).strip("-") or "dataset")[:64]


def local_datasets(root):
    """The saved datasets that have a Dream Labs copy (dream_format/), newest first."""
    rows = []
    for folder in sorted(Path(root).glob("*"), reverse=True) if Path(root).is_dir() else []:
        manifest = _read(folder / "manifest.json")
        if not manifest or not (folder / "dream_format" / "meta" / "info.json").exists():
            continue
        dream = manifest.get("dream_format") or {}
        tasks = sorted({r.get("task") for r in manifest.get("runs", []) if r.get("task")})
        with LOCK:
            upload = dict(UPLOADS.get(folder.name) or {}) or _read(folder / "dream_upload.json")
        rows.append({"folder": folder.name, "hidden": (folder / HIDDEN).exists(),
                     "created_at": manifest.get("created_at"), "runs": manifest.get("count"),
                     "labels": manifest.get("labels"), "episodes": dream.get("episodes"), "frames": dream.get("frames"),
                     "checked": (dream.get("check") or {}).get("ok"), "tasks": tasks,
                     "suggested_name": suggested_name(folder.name), "upload": upload})
    return rows


def set_hidden(root, folder, hidden):
    """Hide a saved dataset on the Train page, or show it again; nothing is deleted. @throws ValueError"""
    path = Path(root) / folder
    if "/" in folder or folder.startswith(".") or not (path / "manifest.json").exists():
        raise ValueError("No such saved dataset.")
    marker = path / HIDDEN
    if hidden:
        marker.write_text("hidden on the Train page\n")
    else:
        marker.unlink(missing_ok=True)


def account(key):
    """What Dream Labs has for this key: embodiments, datasets, fine-tuning jobs, the queue (all free reads)."""
    from dream import datasets, fine_tuning

    return {
        "embodiments": [{"name": e.name, "created_at": _iso(e.created_at)} for e in datasets.list_embodiments(api_key=key)],
        "datasets": [{"id": d.id, "name": d.name, "embodiment": d.embodiment, "status": d.status, "episodes": d.episodes,
                      "size_bytes": d.size_bytes, "error": d.error, "created_at": _iso(d.created_at)}
                     for d in datasets.list_datasets(api_key=key)],
        "jobs": fine_tuning.list(api_key=key),
        "queue": fine_tuning.queue(api_key=key),
        "price_per_step": PRICE_PER_STEP,
        "steps": {"min": STEPS[0], "max": STEPS[1], "every": 100, "default": 1000},
    }


def ensure_embodiment(key, config):
    """The account's YAM-sim embodiment with this config, registered on first use (free; it cannot be changed or
    deleted afterwards, so a different config gets a new name). @return Its full name, <username>/yam-sim."""
    from dream import datasets

    existing = datasets.list_embodiments(api_key=key)
    for e in existing:
        if e.name.split("/")[-1].startswith(EMBODIMENT_NAME) and e.embodiment_config == config:
            return e.name
    taken = {e.name.split("/")[-1] for e in existing}
    name = EMBODIMENT_NAME if EMBODIMENT_NAME not in taken else next(
        f"{EMBODIMENT_NAME}-{i}" for i in range(2, 1000) if f"{EMBODIMENT_NAME}-{i}" not in taken)
    return datasets.add_embodiment(name, config, api_key=key).name


def start_upload(key, root, folder, name):
    """Check and upload one saved dataset in the background. @throws ValueError for a bad folder or name, or one
    already being uploaded."""
    path = Path(root) / folder
    if "/" in folder or folder.startswith(".") or not (path / "dream_format").is_dir():
        raise ValueError("That dataset has no Dream Labs copy (dream_format/).")
    if not NAME_RE.match(name or ""):
        raise ValueError("Dataset names are lowercase letters, numbers and hyphens, up to 64 characters.")
    with LOCK:
        if (UPLOADS.get(folder) or {}).get("state") in ("checking", "uploading"):
            raise ValueError("That dataset is being uploaded already.")
        UPLOADS[folder] = {"state": "checking", "name": name, "error": None, "dataset_id": None,
                           "started_at": datetime.now().isoformat(timespec="seconds")}
    threading.Thread(target=_upload, args=(key, path, name), daemon=True).start()


def _upload(key, path, name):
    from dream import datasets

    def note(**changes):
        with LOCK:
            UPLOADS[path.name] = {**UPLOADS[path.name], **changes}
            record = dict(UPLOADS[path.name])
        if record["state"] in ("done", "failed"):
            (path / "dream_upload.json").write_text(json.dumps(record, indent=2))

    try:
        config = _read(path / "embodiment.json", {}).get("config")
        if not config:
            raise ValueError("embodiment.json is missing: save the dataset again with the Dream Labs format.")
        report = datasets.validate_dataset(path / "dream_format", config)
        if not report.ok:
            raise ValueError(f"The Dream Labs checks refused it: {report.error}")
        embodiment = ensure_embodiment(key, config)
        note(state="uploading", embodiment=embodiment, episodes=report.episodes)
        dataset = datasets.upload_dataset(path / "dream_format", name, embodiment, api_key=key)
        note(state="done", dataset_id=dataset.id, remote_name=dataset.name, status=dataset.status,
             finished_at=datetime.now().isoformat(timespec="seconds"))
    except Exception as error:  # noqa: BLE001  shown on the page as it is
        note(state="failed", error=f"{type(error).__name__}: {error}",
             finished_at=datetime.now().isoformat(timespec="seconds"))


def price(steps):
    return round(int(steps) * PRICE_PER_STEP, 2)


def start_job(key, body):
    """Queue a fine-tuning job: charged, and it cannot be cancelled. The page must send the price it showed and the
    person confirmed (`price`); a mismatch is refused. @return The job as Dream Labs answered it.
    @throws ValueError for a bad request."""
    from dream import fine_tuning

    dataset, suffix = str(body.get("dataset") or ""), str(body.get("suffix") or "")
    if not dataset:
        raise ValueError("Pick a processed dataset.")
    if not SUFFIX_RE.match(suffix):
        raise ValueError("The model name is lowercase letters, numbers and hyphens, up to 40 characters.")
    try:
        steps = int(body.get("steps"))
    except (TypeError, ValueError) as error:
        raise ValueError("Steps must be a number.") from error
    if not (STEPS[0] <= steps <= STEPS[1] and steps % 100 == 0):
        raise ValueError(f"Steps are a multiple of 100 from {STEPS[0]} to {STEPS[1]}.")
    if body.get("price") is None or abs(float(body["price"]) - price(steps)) > 0.005:
        raise ValueError("The price changed: look at it again before starting.")
    schedule = body.get("lr_schedule") or "constant"
    if schedule not in ("constant", "cosine"):
        raise ValueError("The learning-rate schedule is constant or cosine.")
    share = float(body.get("base_share", 0.5))
    if share not in (0.5, 0.0):
        raise ValueError("Training data: mix in Dream Labs data (0.5) or my data only (0.0).")
    return fine_tuning.create(dataset, suffix, steps=steps, lr_schedule=schedule, base_share=share, api_key=key)
