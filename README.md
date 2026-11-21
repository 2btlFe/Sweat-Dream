# Sweat-Dream

Run Dream-1 in the simulator, set your own success rules, save labels.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install dreamlabs
```

Get an API key at https://platform.dreamlabs.ai.

## Run

```bash
python app.py
```

Open http://localhost:8000.

1. Paste your API key → **Use key** (kept in memory only; or put `DREAM_API_KEY=...` in `~/.dreamlabs/credentials`)
2. Type a task → **Run**
3. Set the 3 rules
4. **Save labels**

## Rules

| Rule | Meaning |
| --- | --- |
| Distance | How close the object must get to the bowl/plate center (cm) |
| Release | Must the gripper let go? |
| Still time | How long the object must then stay still (s) |

Reloading the page clears the list. Runs stay saved in `runs/<id>/`, labels in `runs/<id>/label.json`.
