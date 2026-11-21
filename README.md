# Sweat-Dream

A small website for running **Dream-1** in the Dream Labs robot simulator.
Type a task such as `put apple into the bowl`, press **Run**, and watch the two robot arms live in your browser.

## 1. Install

You need Python 3.12.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install dreamlabs
```

**Windows (PowerShell)**

```powershell
$env:PYTHONUTF8 = "1"
python -m venv .venv
.venv\Scripts\activate
pip install dreamlabs
```

## 2. Add your API key

Get a key at https://platform.dreamlabs.ai and put it in the file `~/.dreamlabs/credentials`:

```
DREAM_API_KEY=your-key-here
```

## 3. Start the website

```bash
python app.py
```

Then open http://localhost:8000.

## 4. Use it

1. Type a task, or click one of the examples.
2. Press **Run**. The cameras appear after a few seconds.
3. Press **Stop** at any time, or let it run until the time limit.

## 5. Decide what counts as success

Under **Success rules** there are three settings. A run is a success when all three hold:

1. **Distance to target**: how close (in cm) the object must get to the centre of the bowl or plate.
2. **Letting go**: whether the robot must release the object, or may still be holding it.
3. **Time standing still**: how many seconds the object must then stay put.

Every run under **Runs** is judged again as soon as you change a setting. Click a run to see which rule failed,
with a chart of the distance over time. **Save labels for all runs** writes each result to `runs/<id>/label.json`.

Every run is saved in `runs/<id>/` (`trajectory.json` has the object positions and gripper contact at each step).

## Files

| File | What it does |
| --- | --- |
| `app.py` | The website |
| `static/index.html` | The page |
| `sim.py` | Runs one task in the simulator and saves its trajectory. Also works alone: `python sim.py "put apple into the bowl"` |
