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
4. Press **Reset**, or reload the page (Cmd/Ctrl-R), to start the same run again from the beginning. Opening the page
   in a new tab does not reset a run.

## 5. Recover with the mouse (or keyboard)

**When does a human take over?** While Dream-1 runs, `monitor.py` watches the simulator. When Dream-1 fails it holds
the arms, stops calling Dream-1 and shows **Dream-1 needs help** with a recovery goal:

| Failure | When | Recovery goal |
| --- | --- | --- |
| dropped | the object was carried, then fell back to the table outside the target | pick it up again |
| stalled | 12 s without progress (no new stage, the hand not getting closer) | pick it up (or finish the task if it was already carried) |
| off_table | the object fell off the table | none: cannot be recovered |

When the goal is reached the page turns green: press `T` (or **Give back**) and Dream-1 continues from there.

**Recovery script first.** Before a human is asked, `recovery.py` tries to reach the recovery goal itself, with the
simulator's ground truth: open the gripper, point the hand down (across the banana's length), go above the object,
down, grip, lift, and for "finish the task" also carry it to the bowl or plate, set it down, let go and pull the hand
away. It follows the object if it moves. It moves at Dream-1's pace (hand up to 0.07 m/s, joints up to 0.3 rad/s, as measured on Dream-1 runs). A failed attempt (a step over 20 s, an arm that cannot reach, the object not
lifted or not placed) is tried again from the scene as it is, up to 2 more times (the other arm after a reach
failure); then the arms are held for a human as above. `T` or a click takes over from the script at any time.
Run with `--recover human` (sim.py) to ask a human at once. A lying water bottle is grasped like the banana and set down lying; standing it up is left to a human.
You can also take over any time with `T` / **Take over**.

**Mouse** (on the camera pictures; drag works in the top camera only, since the wrist views move with the hand):

1. **Click an object**: the nearer arm points its hand down and moves 10 cm above it. Clicking in a **wrist camera**
   picture moves that camera's own arm (left wrist view → left arm).
2. **Lower** → **Grip** → **Lift**.
   **Tilted grasp**: press `G` (or the **Grasp** button) before clicking an object. The arm then grasps it at about
   45° from the side: it finds a tilt and direction it can reach (along the length of a banana), waits back along the
   approach line, and **Lower** slides in along that line. A level (90°) side grasp is out of reach at table height
   for these arms, so the tilt stays between 30° and 60°.
3. **Drag** to move the hand along the table (it goes straight above the spot under the cursor), **wheel** for up/down.
4. **Lower** (or wheel) and **Grip** again to let go.

**Keys** (fine adjustment; click the picture first so keys do not go into the task box):
`W/S` away / toward the camera, `A/D` left / right, `R/F` up / down, `Q/E` turn, `Z/X` tilt, `G` grasp style,
`Space` gripper, `Tab` switch arm,
`V` point down, `H` (or **Home**) arm back to its start pose, `T` take over / give back.

- **Human only**: tick it before **Run**. Dream-1 is not called, so no API key or credits are needed.
  Press `T` (or **Hand to Dream-1**) to hand the arms to Dream-1 from the scene as you left it, for the time that
  is left; from then on it is a Dream-1 run (credits are used) and `T` takes over / gives back as usual.
- The arms start level and stay still until you move them. Clicking or dragging points the hand down first; with
  the keys, press `V` first. `H` returns the arm to that start pose.
- While a human has the arms Dream-1 is not called (no credits); it still receives the camera frames. On give-back the
  arms hold still until Dream-1 starts a fresh chunk, so it never finishes a chunk planned before the human moved them.

## 6. Recording

Every run is saved to `episodes/<id>/` (untick **Record** to skip it), whoever controls the arms:

| File | What |
| --- | --- |
| `steps.npz` | per 50 ms step: arm state, the action sent, **who sent it** (`dream`, `human`, `script`, `hold`), the human's hand targets and gripper commands, object poses, fingertip contacts, task stage, and the **full physics state before the step** |
| `events.json` | every key, click, drag and button, every change of control, every failure the monitor saw, with its step |
| `top.mp4`, `left.mp4`, `right.mp4` | the cameras, 5 frames/s (the frames the model sees) |
| `meta.json`, `outcome.json` | task, model, steps per source, versions; final stage and success |

The physics state makes any moment restorable exactly: `mujoco.mj_setState(model, data, physics[k], mjSTATE_INTEGRATION)`
then `env.step(action[k])`, `env.step(action[k+1])`, … reproduces the recording (0 mm error, checked). The layout of
`steps.npz` and `meta.json` follows the team's earlier recorder (commit 98d9797) and adds to it.

## Data collection, then human recovery

1. **Collect** (a normal Dream-1 run): Dream-1 works; when the monitor sees it fail, the recovery script
   (`recovery.py`) takes over and gives the arms back once it has recovered. No human input is taken. If the script
   gives up, or the object falls off the table, the run **ends there** and is saved as a failure.
2. **Recover afterwards** (Recordings panel, below): pick a failed run; Resume starts 1 s before the moment
   *Dream-1* failed (before the script's attempt), with a human in control.

The steps say who acted (`dream`, `script`, `human`, `hold`), so each kind of recovery can be used on its own.

## 7. Turn a recorded failure into a recovery

The **Recordings** panel lists the saved runs (failures only, by default). **Choose** one, pick a step with the slider
(it starts 1 s before the failure the monitor saw; the picture shows the top camera at that step), and press
**Resume here**. The simulator restarts at exactly that recorded state (the full physics state is restored), with you
in control: recover it with the mouse or keys, or press `T` to give it to Dream-1.

The result is a new episode: the original's steps up to that point, then yours, as one trajectory. Its `meta.json`
says `"resumed_from": {"episode", "step"}` and `steps.npz` labels every step's source, so training can use the human
recovery and leave out the original's steps that led into the failure. You can resume the same moment many times.

From the command line: `python sim.py --resume <episode id> --resume-step <k>`.

## Files

| File | What it does |
| --- | --- |
| `app.py` | The website |
| `static/index.html` | The page |
| `sim.py` | Runs one task in the simulator. Also works alone: `python sim.py "put apple into the bowl"` |
| `teleop.py` | Human control: mouse and keys to hand targets, IK to joint angles, Dream-1 / human handover |
| `recorder.py` | Saves every run: states, actions, who controlled each step, inputs, cameras, physics state |
| `monitor.py` | Failure monitor: task stage, asks for a human when Dream-1 fails, recovery goal |
| `recovery.py` | Recovery script: on a failure it tries to reach the recovery goal first (up to 3 attempts), then hands over to a human |
