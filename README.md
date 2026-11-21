# Sweat-Dream

Run Dream-1 in the simulator, set success rules, collect episodes, and save reproducible labels.

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

1. Paste your API key → **Use key**. It is kept in memory only; **Clear** removes it.
2. **Seed**: the start scene comes from it (objects placed around their usual spots, within reach of both arms; see
   [SCENE_SEED.md](SCENE_SEED.md)). **↻** draws a new one; the same seed always gives the same start.
3. **Camera, robot & environment settings**: workspace viewpoint and camera offsets, the arms' initial angles, blur,
   light and table. Tick **random** on a setting to draw it from the seed instead. The camera view shows the start
   scene as you change these (no Dream API call).
4. Type a task such as `put apple into the bowl` → **Run**. The note under the task says what the run is judged on;
   open-ended tasks (such as "Move an object to the center of the workspace") run but are not judged.
5. Set the distance, release, and still-time rules. Existing runs are re-evaluated immediately.
6. Select a run to inspect its final frame, checks, and distance chart. **Save labels** writes the current
   server-side verdicts to `label.json`.

**Custom layout** (optional, single runs): tick *Use my own layout*, pick the objects and targets on the table and
drag them on the map (the robot is YAM, the only one in this simulator). It replaces the seed's layout; settings
marked random are then not drawn (they need the seed).

### Batch (many episodes in the background)

Switch to **Batch**. Set the number of episodes and the seconds per episode, and the seeds: **Seed, Seed + 1, …**
(train seeds, from the Seed field) or **Eval set (50)** (the fixed evaluation seeds, easy / medium / hard). Each
episode starts from its seed's scene, with the scene settings above; the ones marked random are drawn from each
episode's seed. The cameras are not shown; the page shows progress, success / failure / invalid counts and the time
left instead.

Tasks: pick the example prompts, or type your own prompt and **Add** it. Dream-1 gets your prompt as written. The
judge only parses fixed sentences, so each own prompt is judged on the **judge as** pairs below it: object → target,
filled in from the prompt's words (each object goes to the next bowl or plate named after it) and editable.

**Several objects**: add pairs with **+ object** (up to 4, each object once; several objects may go to the same
target), e.g. "put the apple in the bowl and the banana on the plate" → apple · bowl, banana · plate. A run succeeds
only when every pair holds the rules at the end; otherwise the reason says how many were placed and which object
failed first (e.g. `1/2 placed; banana: outside_target`). The run's chart has one line per object. Give these
episodes more time (90–120 s for two objects).

Every run keeps its seed, split, scene settings and the values drawn from the seed in `meta.json`, so its start can
be made again (`python sim.py "..." --seed N --ood-random ...`). Each batch episode is labeled with the rules as they were when the batch started (`runs/<id>/label.json`, `source: batch`). A batch
stops on a run error such as a rejected API key, after three invalid episodes in a row, or on **Stop batch**. Pick a
batch in the run list to see only its runs and its success rate; select a run for its chart.

The run list also shows episodes collected from the command line, including after a page reload.

To collect 30 episodes across the three sample tasks with ten seeded starting layouts:

```bash
python batch.py --repeats 10 --seconds 60 --budget-minutes 60
```

The batch uses the default scene, the key in `~/.dreamlabs/credentials` (`DREAM_API_KEY=...`), and a snapshot of the rules saved in the UI. It writes each episode to `runs/<id>/` and stops on an invalid run, such as an API authentication error. Check the API key and endpoint before starting a large batch. `--tasks`, `--seed`, `--repeats`, and `--seconds` are adjustable.

## Rules

| Rule | Meaning |
| --- | --- |
| Distance | Final horizontal distance from the named object to the bowl/plate center. The object's center must also be in the target's fixed height band. |
| Release | Whether the named object must stop touching a gripper fingertip. |
| Still time | How long the final placement and release must persist. Maximum object speed in that final window must stay below 0.05 m/s. |

The current UI exposes distance from 1 to 15 cm and still time from 0 to 5 seconds. The backend accepts 0.5 to 30 cm and 0 to 10 seconds. Target height bands and the 0.05 m/s speed limit are currently fixed. An execution error, user stop, missing model response, or incomplete trajectory is labeled **invalid**, never **failure**. Saved labels include the rules snapshot so later rule changes do not silently rewrite them.

## Files

| File | What it does |
| --- | --- |
| `app.py` | The website: runs, previews, batches, rules, labels |
| `static/index.html`, `static/ood.js`, `static/seed.js` | The page; scene settings form; seed field and random boxes |
| `sim.py` | Runs one task (or renders a start scene with `--preview`) and saves its trajectory |
| `scene.py`, `scene_reach_map.npz` | Start scenes from a seed (see `SCENE_SEED.md`) |
| `ood.py`, `seed_web.py` | Scene settings (camera, arms, blur, light, table); the web side of the seed |
| `judge.py`, `test_judge.py` | Success / failure from a trajectory and the rules; its tests |
| `batch.py` | Batches from the command line |
