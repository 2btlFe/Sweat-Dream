# Sweet Dream · Usage

The full guide to every page and option. For the overview, see the [README](../README.md).

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

Open http://localhost:8000. The site has several pages:

- **Start** (`/`): **Sweet Dream** and a sign-in box. Paste your Dream API key → **Sign in**: Dream Labs is asked
  whether it accepts the key (it lists the models: free, nothing runs). The key is kept in memory only.
- **Sections** (`/sections`): **Collect Custom Dataset!** (the page below), **Train Custom Model!** (below) and
  **Test Custom Model!** (`/test`). **Sign out** clears the key.
- **Train Custom Model!** (`/train`, `train.py`), with the signed-in key:
  1. **Upload a dataset**: every dataset saved with the Dream Labs format (`datasets/<folder>/dream_format/`) is
     listed. **Import dataset ZIP** accepts a Sweet Dream saved dataset ZIP, or a LeRobot ZIP with `embodiment.json`;
     it validates locally and adds a row. **Upload** checks the row again, registers the robot once as `yam-sim`
     (from `embodiment.json`,
     reused after that) and uploads it under the name given (free). Dream Labs then processes it (a few minutes);
     the page follows it. The outcome is kept in `datasets/<folder>/dream_upload.json`.
  2. **Train**: a processed dataset, a new model name, the steps (500 to 10,000), and `base_share` (0.5 for half
     Dream-1 general training data and half your episodes, or 0.0 for only your data). The learning-rate schedule
     uses the default constant setting. The estimated price shows as you choose ($0.20 a step at the time of writing). Billing shows
     your current balance; the page links there. **Start training…** asks first: the price is held from your credits
     and a queued job cannot be cancelled. The server refuses a start whose price is not the
     one shown.
  3. **Your training jobs**: read from Dream Labs every 30 s (status, phase, step, loss, and the model id
     `dream-1-ft:<account>-<name>` when it is done), so they show again after a restart.
  **Hide** takes a saved dataset off the list (a `.hidden` file in its folder; nothing is deleted); **Show hidden**
  lists them again, faded, with **Unhide**.
  Credits and the current price are on the Dream Labs Billing page (an API key cannot read them).
- **Collect Custom Dataset!** (`/collect`): runs, batches, labels and datasets, as follows.
- **Test Custom Model!** (`/test`, `static/test.html`): for now a copy of the Collect page, to try a fine-tune: pick it
  in the **Model** bar (below), then run it once or in a batch. It shares the runs, rules and model choice with the
  Collect page.

1. The API key from the sign-in is used (no key box here): the top right shows its last characters, and **Sign out**
   clears it and goes back to the start page. Without a key (signed out, or the server restarted) the page goes there too.
   **Model** (the bar under the title): the model Dream-1 runs as, from the models this key may run: the public ones
   (`dream-1-beta`, `dream-1-beta-flash`) and **Your fine-tunes** (`dream-1-ft:<account>-<suffix>`; one not finished
   yet is greyed out, and **↻** lists them again). Single runs and new batches use it; a batch keeps the model it
   started with, and each run keeps its model in `meta.json` (`model`, also in a dataset's manifest). A fine-tune runs
   only with the key of the account that trained it, and a new key or Sign out goes back to the beta. A fine-tune that
   is not running yet is started by its first call (`DA-2000 pool_warming`): the run waits, up to 10 minutes, and that
   wait is not counted in its seconds; the camera view shows a spinning ring with the time waited until it answers.
   A fine-tune can overshoot the gripper a little (such as 1.07, fully open): values up to 0.25 outside 0..1 are
   clipped to 0..1 (`meta.json`: `gripper_clipped`), and only ones further out stop the run as a wrong reply.
   Prompt a fine-tune the way it was trained. From the command line:
   `python sim.py "..." --model dream-1-ft:<account>-<suffix>` (or `batch.py --model ...`).
2. **Robot** and **Room**: the list offers the team's target robot, YAM (`simulator_settings.TARGET_ROBOTS`; add
   `"openyam"` there to offer it again). YAM (with the seed scenes and scene settings below) or OpenYAM (on the Dream Labs Sandbox
   table; needs [Node.js](https://nodejs.org) to fetch it, and a seed only shifts its objects by up to 2 cm). The other
   robots in the list run only in the [Dream Labs Sandbox](https://platform.dreamlabs.ai/sandbox). A room styles the
   table and background (kitchen, cottage, ...). The choice is kept in `runs/simulator_settings.json`.
3. **Seed**: with **Next seed after each run or batch** (ticked by default) every single run, and every batch, starts
   from a new seed, so the start layouts keep changing; each run keeps its seed in
   `meta.json`, and **Same start (seed N)** in a run's detail puts that seed and its random settings back. The start scene comes from it (objects placed around their usual spots, within reach of both arms; see
   [SCENE_SEED.md](SCENE_SEED.md)). **↻** draws a new one; the same seed always gives the same start.
4. **Camera, robot & environment settings**: workspace viewpoint and camera offsets, the arms' initial angles, blur,
   light and table. Tick **random** on a setting to draw it from the seed instead (**All** / **None** above the
   settings, or **all random** in a group's title, tick many at once). The camera view shows the start scene as you
   change these (no Dream API call). If the start cannot be built (the arms' start pose collides with something, a
   random pose or camera hides an object, a seed has no layout), the error banner says why, **Run** waits, and the
   banner offers **Draw again (new seed)** and **Default settings**.
   **Default** (top of Run setup) goes back to the default start: YAM tabletop, every setting at its default, nothing
   random, no custom layout; the seed stays.
5. Type a task such as `put apple into the bowl` → **Run**. The note under the task says what the run is judged on;
   open-ended tasks (such as "Move an object to the center of the workspace") run but are not judged.
6. Set the distance, release, and still-time rules. Existing runs are re-evaluated immediately.
7. Select a run to inspect it: a recorded run plays its camera video (**Top / Left wrist / Right wrist**,
   or **Last frame**), with its checks and distance chart. **Save labels** writes the current
   server-side verdicts to `label.json`.
8. **Your label**: decide a run yourself with **Success / Failure / Invalid** (keys **1 / 2 / 3**; **0** goes back to
   the rules), with an optional note. **J / K** move to the next / previous run. A label set by hand wins over the
   rules: changing the rules or **Save labels** never replaces it, and `label.json` keeps the rules' verdict beside
   it (`source: "human"`, `rules_verdict`). The run list marks these runs with ✋, and its filter can show only them
   or only the ones where you and the rules disagree.
9. **Datasets**: each row's **+ / −** puts the run in the dataset or takes it out; **+ Add successes** picks every
   success the list shows (e.g. one batch). **Save dataset…** opens a popup: a name, and the folder on this computer to
   save in (type a full path, or browse: Project datasets, Home, Desktop, Documents, Downloads, then into folders;
   the last folder is remembered). It writes `<folder>/<time>-<name>/` (`datasets/` by default): `runs/<id>/` and the
   recording `episodes/<id>/` of every picked run (hard links on the same disk: no extra space, and they stay if a run
   is deleted later; copies on another disk),
   and `manifest.json` with each run's final label (yours first), the rules' verdict, the judge spec, who acted, and
   its settings. Each run's settings (mode, robot and room, seed or layout, scene settings and the ones drawn from
   the seed, data collection settings, time limit, batch) show under its task in the list and in full in its detail.
   With **Dream Labs format** ticked in the popup (the default), the recordings are also written as one LeRobot v2.1 dataset in
   `dream_format/`, the format Dream Labs fine-tuning uploads, and checked with the Dream Labs checks
   (`dream.datasets.validate_dataset`, on this computer, nothing sent). One row per camera frame (5 per second):
   `observation.state` and `action` are 14 values, right arm first (6 joints in rad, gripper 0 closed .. 1 open, then
   the left arm); cameras `overhead`, `wrist_l`, `wrist_r`. `embodiment.json` holds the config to register the robot
   with; then `client.upload_dataset("<folder>/dream_format", "<name>", "<embodiment>")` uploads it.
   Runs without a recording (Plain mode) and recordings shorter than 1.6 s are left out (`lerobot_export.py`).
10. **Delete**: **Delete this run** (in a run's detail), **Delete shown** (every run the list shows, e.g. one batch) or
   **Delete** in Recordings removes a finished run for good: `runs/<id>` and its recording `episodes/<id>`. The page
   asks first; the run going now is never deleted. **Delete picked** (next to Save dataset) removes the runs picked
   with **+**, so several unwanted episodes go at once.

**Custom layout** (optional, single runs): tick *Use my own layout*, pick the objects and targets on the table and
drag them on the map (the robot is YAM, the only one in this simulator). The map starts from the seed's scene (as
the preview shows it), with items kept inside the layout area. It replaces the seed's layout; settings
marked random are then not drawn (they need the seed).

### Run modes, human control and recovery (single runs)

| Mode | What happens |
| --- | --- |
| **Data collection** (default) | Dream-1; when the failure monitor (`monitor.py`) sees it fail, the recovery script (`recovery.py`) takes over and gives the arms back once it has recovered. No human input. If the script gives up, or the object falls off the table, the run ends there: recover it by hand later in **Recordings**. |
| **Rule-based only** | The recovery script (`recovery.py`) does the whole task from the first step, object by object for a goal scene: no Dream-1, so no API key or credits. It runs in real time (20 steps a second, as Dream-1's chunks are played), and the settings that apply are the script's speed and ending once the task is done. If the script gives up, the run ends (recover it later in **Recordings**). YAM with all six items only. |
| **Test** | Dream-1 and you take turns with `T` at any time; when Dream-1 fails the arms are held for you. |
| **Human only** | You have the arms from the start (no API key needed); `T` hands them to Dream-1 from the scene as you left it. |
| **Dream-1 only** | No monitor, script or human control. OpenYAM, and custom layouts without all six items, always run this way. |

Failures the monitor sees (pick & place tasks): **dropped** (carried, then fell back outside the target), **stalled**
(12 s without progress), **off_table** (cannot be recovered).

**Data collection settings** (shown in data collection mode, single runs and batches), for faster successes:

| Setting | Default | What it does |
| --- | --- | --- |
| Hand over after | 6 s | seconds without progress before the script takes over (counted from Dream-1's first move) |
| Quick signals | on | also hand over at once on a **missed grasp** (gripper closed next to the object, not lifted 2 s later) or an object **pushed away** (slid 6 cm, further from the target) |
| The script | finishes the task | picks the object up, puts it in the target, lets go; or *picks it up, back to Dream-1* (the old behaviour) |
| Script speed | Dream-1's pace | 1.5×, 2× or 3× makes recoveries faster, but less like Dream-1's motion |
| End when done | on | the run ends as soon as the task is done (placed, released, at rest), so nothing undoes it |

Measured (3 real runs at once, same seeds, 60 s): 3 of 3 successes, ended 35-45 s in (Dream-1 handed over at 15-25 s,
once on a missed grasp); with the old behaviour (12 s, pick up and hand back) 1 of 3.

When Dream-1 is at capacity (`DA-2000 NO_CAPACITY`, shared by everyone), a batch tries that episode again (same seed)
after 5, 10, 15, ... s, up to 5 times, and drops the attempt. The recovery goal is to pick the object up again, or to
finish the task if it was already carried; the page turns green when it is reached, and `T` gives the arms back.

**Mouse** on the camera pictures: click an object (the nearer arm moves its hand 10 cm above it; a click in a wrist
camera moves that arm), then **Lower → Grip → Lift**; drag in the top camera to move the hand along the table, wheel
for up / down. **Keys** (click the picture first): `W/S` away / toward the camera, `A/D` left / right, `R/F` up /
down, `Q/E` turn, `Z/X` tilt, `G` grasp from above ↔ tilted, `Space` gripper, `Tab` switch arm, `V` point down,
`H` arm home, `T` take over / give back. **Reset** starts the same run again from its start.

**Record** (ticked by default) also saves these runs to `episodes/<run id>/` (`recorder.py`): every step's arm state,
the action and **who sent it** (`dream`, `human`, `script`, `hold`), the human's inputs, the cameras, and the **full
physics state before every step**; `runs/<id>/meta.json` gets `mode`, `episode` and `steps_by_source`.

### Recordings: turn a failure into a recovery

The **Recordings** panel lists the recorded runs (failures only, by default). **Choose** one, pick a step with the
slider (it starts 1 s before the failure the monitor saw), and **Resume here**: the simulator rebuilds the run's scene
(seed, scene settings, room or custom layout) and restores exactly that recorded state, with you in control. The
result is a new run whose recording starts with the original's steps (`resumed_from`), so training can use the human
recovery and leave out the steps that led into the failure. From the command line:
`python sim.py --resume <run id> --resume-step <k>`. A human-only or resumed run that you end with **Stop** is judged
like any other run (Stop is its normal end).

### Batch (many episodes in the background)

**Original** (the baseline, in **Each episode**: Data collection / Dream-1 only / **Original**): the seed and the
environment are fixed. Every episode starts from the simulator's default scene (the same object layout each time, no
seed) with default scene settings, and Dream-1 runs with the failure monitor and the recovery script, as in Data
collection (its settings apply). The task, its judge, the episode count and the seconds are the ones set above.

**Run as original**: pick a finished batch in the run list's filter and press **Run as original**: as many episodes,
with that batch's task, judge and time limit, as an Original batch (default scene and settings, recovery script).
Its runs are named "original of <batch>" (`meta.json`: `original_of`), and with either batch picked the summary
compares the two success rates.

Switch to **Batch**. Pick what each episode does (**Data collection**: Dream-1, then the recovery script on a
failure; **Rule-based only**: the script from the start, no API key needed; or **Dream-1 only**), how many run **at once** (1 to 4), the number of episodes and the seconds per
episode, and the seeds: **From the seed**
(one train seed is the whole batch: each episode's own seed is derived from it, `seed_web.episode_seed`, so the
same seed gives the same episodes again and other seeds share none) or **Eval set (50)** (the fixed evaluation seeds, easy / medium / hard).
**Scene per episode**: **Same settings** (the default) uses the scene settings and the room picked on the page for every
episode; **Vary per episode** draws, from each episode's own seed and close to the defaults, the workspace camera's
position (±3 cm), angle (±8°, roll ±5°) and field of view (±8°), the light (0.8× to 1.2×), and the room (YAM tabletop
or Evening den, `seed_web.episode_room`); the same batch seed gives the same scenes again. Original keeps its scene fixed. Each
episode starts from its seed's scene, with the scene settings above; the ones marked random are drawn from each
episode's seed. The page shows progress, success / failure / invalid counts, the time left, and the camera view of
each episode going now (one tile per episode).

Tasks: pick the example prompts, or type your own prompt and **Add** it. Dream-1 gets your prompt as written. The
judge only parses fixed sentences, so each own prompt is judged on the **judge as** pairs below it: object → target,
filled in from the prompt's words (each object goes to the next bowl or plate named after it) and editable.

**Several objects**: add pairs with **+ object** (up to 4, each object once; several objects may go to the same
target), e.g. "put the apple in the bowl and the banana on the plate" → apple · bowl, banana · plate. A run succeeds
only when every pair holds the rules at the end; otherwise the reason says how many were placed and which object
failed first (e.g. `1/2 placed; banana: outside_target`). The run's chart has one line per object. Give these
episodes more time (90–120 s for two objects).

**At once**: each episode is its own simulator process, so several run side by side; episode *i* still gets the
*i*-th seed and task, whatever order they finish in. On an M3 (8 cores), 3 Dream-1 episodes at once ran as fast as one
(simulated time, response times) with no session errors; more at once can slow every simulator, and a Dream-1
session expires after about 10 s without a request, so raise it with care. Each episode's output is kept in a log
file while the server runs; **Stop batch** stops every episode that is going.

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

Errors (preview, run, batch, API key, ...) appear in one banner at the top of the page, in plain words with a fix where
one is known; **Details** shows the original message or the run's log.

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

## Goal scene: set success by dragging objects

Instead of judging by the task's words, you can show what "done" looks like. Under **Run**, press **Set goal scene…**:
a popup shows the robot's top camera on the start scene (same seed and settings as the run). Drag each object to
where it should end up; when you let go, the simulator drops it there and lets it settle (into the bowl, onto the
plate), and the picture shows the result. **Use this goal**, then **Run**: the run is judged on that goal.

Every object you moved must end within the distance of its goal spot (**Success rules › Goal scene**), at its height
(in the bowl, not beside it), released if required, and still. Objects you did not move are not judged. If the seed
or the scene settings change after the goal is set, Run asks before dropping the goal.

## Files

| File | What it does |
| --- | --- |
| `app.py` | The website: runs, previews, batches, rules, labels |
| `static/index.html`, `static/ood.js`, `static/seed.js` | The page; scene settings form; seed field and random boxes |
| `static/test.html` | The Test Custom Model page (for now a copy of `index.html` with its own title) |
| `sim.py` | Runs one task (or renders a start scene with `--preview`) and saves its trajectory |
| `scene.py`, `scene_reach_map.npz` | Start scenes from a seed (see `SCENE_SEED.md`) |
| `ood.py`, `seed_web.py` | Scene settings (camera, arms, blur, light, table); the web side of the seed |
| `judge.py`, `test_judge.py` | Success / failure from a trajectory and the rules; its tests |
| `batch.py` | Batches from the command line |
| `simulator_settings.py`, `test_simulator_settings.py` | Robot and room choices, and which run here; its tests |
| `sandbox_env.py`, `sandbox_scene.mjs`, `local_rooms.py` | OpenYAM on the Sandbox table (fetched at run time); room styling |
| `teleop.py` | Human control: mouse and keys to hand targets, IK to joint angles, Dream-1 / human handover |
| `monitor.py` | Failure monitor: task stage, when Dream-1 needs help, the recovery goal |
| `recovery.py` | Recovery script: reaches the recovery goal itself first (up to 3 attempts), then hands over to a human |
| `recorder.py` | Records runs to `episodes/`: states, actions, who controlled each step, inputs, cameras, physics state |
