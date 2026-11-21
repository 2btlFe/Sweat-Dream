# Sweat-Dream

A small website for running **Dream-1** in the Dream Labs robot simulator.
Type a task such as `Move an object to the center of the workspace`, press **Run**, and watch the two robot arms live in your browser.

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

## Files

| File | What it does |
| --- | --- |
| `app.py` | The website |
| `static/index.html` | The page |
| `sim.py` | Runs one task in the simulator. Also works alone: `python sim.py "Move an object to the center of the workspace"` |
