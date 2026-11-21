"""Re-run the judge on every recorded episode (after a judge change) and print a summary table."""
import json, sys
from pathlib import Path
import numpy as np
import judge
root = Path(__file__).resolve().parent / "episodes"
for d in sorted(p for p in root.iterdir() if (p / "steps.npz").exists()):
    meta = json.loads((d / "meta.json").read_text())
    z = np.load(d / "steps.npz")
    targets = dict(zip(judge.TARGETS, z["targets"]))
    out = judge.judge(meta["task"], z["t"], z["objects"], z["held"], targets)
    if meta["errors"] and out["label"] != "success":
        out["warnings"].append("episode ended with an error: " + meta["errors"][-1])
    (d / "outcome.json").write_text(json.dumps(out, indent=2))
    print(f"{d.name}  {meta['source']:24s} {out['label']:8s} {out['reason']:26s} fail_t={out['failure_time_s']} ok_t={out['success_time_s']}")
