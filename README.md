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
2. **Scene** (right): pick the objects and a bowl/plate and drag them on the map. The robot is YAM, the only one in this simulator.
3. Type a task such as `put apple into the bowl` → **Run**.
4. Set the distance, release, and still-time rules. Existing runs are re-evaluated immediately.
5. Select a run to inspect its final frame, checks, and distance chart. **Save labels** writes the current server-side verdicts to `label.json`.

### Batch (many episodes in the background)

Switch to **Batch** above the task. Set the number of episodes, the seconds per episode, the tasks (taken in turn) and
the start positions, then **Start batch**. The cameras are not shown; the page shows progress, success / failure /
invalid counts and the time left instead.

- **Scene ± 2 cm**: the scene set on the right, every object shifted a little (seeded).
- **Random layout**: the items on the right placed anywhere on the reachable table, at least 2 cm apart (seeded).

Episode *i* uses seed *seed + i* (the seed is random unless you set one), so a layout can be made again. Each episode
is labeled with the rules as they were when the batch started (`runs/<id>/label.json`, `source: batch`). A batch
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
