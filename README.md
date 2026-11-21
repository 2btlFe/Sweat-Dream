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
2. **Robot** and **Room**: YAM (with the seed scenes and scene settings below) or OpenYAM (on the Dream Labs Sandbox
   table; needs [Node.js](https://nodejs.org) to fetch it, and a seed only shifts its objects by up to 2 cm). The other
   robots in the list run only in the [Dream Labs Sandbox](https://platform.dreamlabs.ai/sandbox). A room styles the
   table and background (kitchen, cottage, ...). The choice is kept in `runs/simulator_settings.json`.
3. **Seed**: the start scene comes from it (objects placed around their usual spots, within reach of both arms; see
   [SCENE_SEED.md](SCENE_SEED.md)). **↻** draws a new one; the same seed always gives the same start.
4. **Camera, robot & environment settings**: workspace viewpoint and camera offsets, the arms' initial angles, blur,
   light and table. Tick **random** on a setting to draw it from the seed instead. The camera view shows the start
   scene as you change these (no Dream API call).
5. Type a task such as `put apple into the bowl` → **Run**. The note under the task says what the run is judged on;
   open-ended tasks (such as "Move an object to the center of the workspace") run but are not judged.
6. Set the distance, release, and still-time rules. Existing runs are re-evaluated immediately.
7. Select a run to inspect its final frame, checks, and distance chart. **Save labels** writes the current
   server-side verdicts to `label.json`.
8. **Your label**: decide a run yourself with **Success / Failure / Invalid** (keys **1 / 2 / 3**; **0** goes back to
   the rules), with an optional note. **J / K** move to the next / previous run. A label set by hand wins over the
   rules: changing the rules or **Save labels** never replaces it, and `label.json` keeps the rules' verdict beside
   it (`source: "human"`, `rules_verdict`). The run list marks these runs with ✋, and its filter can show only them
   or only the ones where you and the rules disagree.

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

## Rules, by task type

Each run is judged by its task's type, with that type's rules (the **Success rules** panel has one tab per type).
The type and its objects come from the task's words, and the note under the task says how it was read; a batch's own
prompts take them from **judge as** (a type, then its objects). A task that names none of these runs but is not judged.

| Type | Example tasks | Success when, at the end of the run | Rules |
| --- | --- | --- | --- |
| Pick & place | `put apple into the bowl`, `put apple in the bowl and banana on the plate` | every named object is in its bowl/plate, released if required, and still | distance to the target's center, release, still time |
| Move to area | `Move an object to the center of the workspace`, `push the apple to the left` | the object (or, for "an object", any object) has moved, rests in the area, released if required, and still | area size (center: radius; left/right/front/back: beyond this from the middle), release, still time |
| Lift | `lift the apple`, `pick the bottle up` | the object rose this high above its start and stayed up for the hold time in one stretch | height, hold time |
| Push | `push the orange` | the object ends this far from its start, on the table (never lifted 3 cm, if lifting is not allowed) | distance, lifting allowed |

"Still" means under 0.05 m/s. Left / right / front / back are as the workspace camera sees the table. An execution
error, user stop, missing model response, or incomplete trajectory is labeled **invalid**, never **failure**. Saved
labels include the rules snapshot, so later rule changes do not silently rewrite them. `runs/rules.json` keeps every
type's rules (the older single set is read as the pick & place rules). To add a task type: its rules in `judge.RULES`,
a `judge_<type>` function and a branch in `judge.classify()`, and its controls in the page's `TYPE_UI`.

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
| `simulator_settings.py`, `test_simulator_settings.py` | Robot and room choices, and which run here; its tests |
| `sandbox_env.py`, `sandbox_scene.mjs`, `local_rooms.py` | OpenYAM on the Sandbox table (fetched at run time); room styling |
