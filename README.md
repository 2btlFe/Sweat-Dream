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

1. Paste your API key → **Use key** (kept in memory only; **Clear** removes it)
2. Set up the scene: pick objects and a bowl/plate, drag them on the map (robot: YAM, the only one in this simulator)
3. Type a task → **Run**
4. Set the 3 rules
5. **Save labels**

## Rules

| Rule | Meaning |
| --- | --- |
| Distance | How close the object must get to the bowl/plate center (cm) |
| Release | Must the gripper let go? |
| Still time | How long the object must then stay still (s) |

Reloading the page clears the list. Runs stay saved in `runs/<id>/`, labels in `runs/<id>/label.json`.
