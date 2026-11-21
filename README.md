# Sweat-Dream

Run Dream-1 in the simulator, set your own success rules, save labels.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install dreamlabs
```

Put your API key (from https://platform.dreamlabs.ai) in `~/.dreamlabs/credentials`:

```
DREAM_API_KEY=your-key-here
```

## Run

```bash
python app.py
```

Open http://localhost:8000.

1. Type a task → **Run**
2. Set the 3 rules
3. **Save labels**

## Rules

| Rule | Meaning |
| --- | --- |
| Distance | How close the object must get to the bowl/plate center (cm) |
| Release | Must the gripper let go? |
| Still time | How long the object must then stay still (s) |

Runs are saved in `runs/<id>/`, labels in `runs/<id>/label.json`.
