# Dream Labs Auto-Labeling Pipeline

Stage 1: an episode recorder and a simulator ground-truth judge for Dream-1 runs in the Dream YAM Sim.

## Setup (Windows)

```powershell
$env:PYTHONUTF8 = "1"; python -m venv C:\dream\.venv
C:\dream\.venv\Scripts\python.exe -m pip install dreamlabs
```

Put the API key in `~/.dreamlabs/credentials` as `DREAM_API_KEY=...`.

## Setup (macOS)

```bash
uv venv --seed --python 3.12 ~/dream-env && source ~/dream-env/bin/activate
pip install dreamlabs
```

Run the same scripts with `~/dream-env/bin/python`.

## Usage

```powershell
$env:PYTHONUTF8 = "1"
C:\dream\.venv\Scripts\python.exe run_episode.py "put apple into the bowl" --seconds 90          # headless
C:\dream\.venv\Scripts\python.exe run_episode.py "put apple into the bowl" --view               # viewer window
C:\dream\.venv\Scripts\python.exe scripted.py                                                   # known-outcome episodes, no API
C:\dream\.venv\Scripts\python.exe rejudge.py                                                    # re-judge every episode
C:\dream\.venv\Scripts\python.exe batch.py --repeats 3 --budget-minutes 10                      # Dream-1 x default tasks x 3 seeds
C:\dream\.venv\Scripts\python.exe batch.py --scripted --no-dream --repeats 2                    # scripted only, no cost
C:\dream\.venv\Scripts\python.exe batch.py --resume <batch_id>                                  # finish an interrupted batch
C:\dream\.venv\Scripts\python.exe summary.py [--batch <batch_id>]                               # rates, reasons, index.csv
C:\dream\.venv\Scripts\python.exe rerender.py --all --variants 4                                 # appearance augmentation
C:\dream\.venv\Scripts\python.exe rerender.py <episode_id> --check                              # replay vs recorded video
C:\dream\.venv\Scripts\python.exe autolabel.py                                                  # auto labels + review queue
C:\dream\.venv\Scripts\python.exe perturb.py --clean --perturb drop@grasp+0.5 --repeats 2          # failure-recovery runs
C:\dream\.venv\Scripts\python.exe auto_mode.py --perturb drop@grasp+0.5 --perturb push@lift --looks 2   # all of it
```

Labeling UI: `C:\dream\.venv\Scripts\python.exe label_server.py --labeler <name>` then open http://localhost:8765.
Keys: Space play, arrows step a frame, 1/2/3 Success/Failure/Uncertain, M mark the failure time at the current frame,
Enter save and go to the next episode, J/K next/previous. Blind mode hides the simulator and verifier judgements until
you save, so human labels are independent. Labels go to `episodes/<id>/label.json`; `outcome.json` is never changed.

`--seed` jitters every object's start pose (+-2 cm, +-25 deg yaw) so repeated runs differ; the seed and offsets are
stored in `meta.json`. Each batch episode runs in its own process (a crash ends one episode, not the batch); batch
progress is saved in `batches/<batch_id>.json`, and Dream-1 episodes stop being started once `--budget-minutes` of
their wall time is used ($3/hour list price).

Each episode is written to `episodes/<id>/`: `meta.json`, `steps.npz` (t, state, action, object positions,
fingertip contact; object orientations from now on), `top.mp4` / `left.mp4` / `right.mp4` (H.264, 5 fps), `outcome.json`.

## Auto labeling, failure-recovery data and auto mode

`autolabel.py` gives every episode `auto_label.json`: the simulator judge's label, a confidence and a review flag,
without touching `outcome.json` or a human `label.json`. It sends an episode to the review queue when the judgement is
less certain: unknown instruction, judge warning, run error, object ending within 1 cm of the target's edge, object
still moving at the end, another object moved, replay drift, or a human label that disagrees. Tags slice the data:
`clean_success`, `failure`, `recovery_run`, `recovered` / `not_recovered`, `disturbance:<kind>`, `scripted`, `looks:<n>`.

`perturb.py` makes failure-recovery data from clean runs. A clean success is replayed action by action from its seed's
start (MuJoCo is deterministic: 0 mm replay error, checked on Dream-1 and scripted episodes) up to a trigger, a
disturbance is applied, and Dream-1 takes over from that state; the run is recorded and judged like any episode, with
`perturbation.json` (what, when, recovered, recovery time). The replayed prefix costs nothing.

| Disturbance | Effect (defaults) |
| --- | --- |
| `drop` | the gripper holding the object opens for `seconds` (0.6) |
| `push` | velocity kick to the object: `vx, vy, vz` m/s (0, 0.4, 0) |
| `shift` | the object moves instantly by `dx, dy` m (0.05, 0) |
| `nudge_arm` | one joint (`joint` 2) of the holding or nearest arm turns by `rad` (0.3) |
| `move_target` | the bowl or plate moves by `dx, dy` m (0.08, 0): the goal changes mid-task |
| `freeze` | the last command stays in force for `seconds` (1.5): a stall |
| `noise` | Gaussian noise `std` rad (0.05) on every joint command for `seconds` (2) |
| `occlude` | a `camera` (left) sends black frames to the model for `seconds` (2) |

A SPEC is `kind[:key=value,...]@trigger[+delay]`; triggers are a time in seconds, `grasp`, `lift` or `release` of the
clean run. Variations: `--repeats` with `--delay-jitter` (a later trigger, up to s) and `--scale-jitter` (magnitudes
scaled by up to +-x), drawn from a seed per repeat and kept in `perturbation.json`. `--no-dream` keeps the last command
after the disturbance instead of calling Dream-1: a no-recovery baseline at no cost (the UI marks it "baseline").

`auto_mode.py` runs it end to end: optional collection (`--dream-repeats`, `--scripted`), auto labels, disturbances on
every high-confidence clean success (`--perturb ...`, `--perturb-repeats`, jitters, `--no-dream-recovery`), auto labels
again, `--looks` re-rendered variants of every high-confidence episode, and a report in
`batches/auto-<time>.json` with label counts, the review queue and the recovery rate per disturbance. Low-confidence
episodes are neither disturbed nor multiplied; they wait for review. The labeling UI shows the auto label, the
disturbance (with a mark on the timeline), and filters for the review queue, recovery runs and recovered runs.

## Appearance augmentation (re-rendering)

`rerender.py` replays a recorded episode from `steps.npz` (the arms' joints and gripper opening, every object's
position, and orientation: `objects_quat`, recorded from now on) and renders the three cameras again under a new
look: light colour, strength and direction, table / bowl / plate / floor colours, the water bottle's colour, a mild
tint on the fruit (they stay recognisable for their instruction), and small camera moves (top 1.5 cm / 2 degrees,
wrists 0.5 cm / 1.5 degrees). The motion is the recorded one, so `outcome.json` (the simulator's judgement) is the
label of every variant: more labelled data without API calls. Output: `episodes/<id>/aug/<k>/{top,left,right}.mp4`
(same frames and fps) and `params.json` (seed and every drawn value); `aug/index.json` lists them. The labeling UI
switches between the recorded look and each variant. `--strength` scales every range; `--check` compares the plain
replay with the recorded video. Episodes recorded before `objects_quat` replay objects in their starting orientation.
It waits while an episode is being recorded (`episodes/.recording-<pid>`, written by `recorder.py`): its GPU work
slows the simulator's cameras, and a slower simulator lets the Dream-1 session expire (see Findings).

## Files

| File | Role |
| --- | --- |
| `dreamfix.py` | Windows fixes: render only the frames the model sees; retry a dropped connection |
| `recorder.py` | `RecordingEnv` + `EpisodeRecorder`: per-step trace, camera video, judge on close, early stop on success |
| `judge.py` | Ground-truth outcome from physics: label, reason, success/failure time, metrics, warnings |
| `run_episode.py` | One Dream-1 episode, recorded and judged |
| `scripted.py` | Scripted success/failure episodes with a known outcome |
| `rejudge.py` | Re-run the judge on all recorded episodes |
| `label_server.py` + `static/index.html` | Labeling UI (standard library only): synced 3-camera playback, timeline with sim success/failure marks, Success/Failure/Uncertain + reason + failure time, blind mode, agreement stats |
| `batch.py` | Many episodes unattended: tasks x seeds, scripted variants, cost budget, resume |
| `summary.py` | `episodes/index.csv` and success rates / failure reasons per source and task |
| `live_view.py` | Plain viewer run without recording |
| `rerender.py` | Appearance augmentation: re-render recorded episodes under new lighting, colours and camera poses |
| `episode_io.py` | Shared helpers: JSON files, reading videos, waiting while an episode is recording |
| `autolabel.py` | Auto labels from the simulator judge, confidence, review queue, tags (`auto_label.json`) |
| `perturb.py` | Failure-recovery runs: replay a clean episode, disturb it, let Dream-1 recover (`perturbation.json`) |
| `auto_mode.py` | Collect, auto-label, disturb, re-render and report in one command |

## Judge outcomes

`success / placed_in_target` when the object is inside the target region, released, and resting for 1 s.
Failure reasons: `object_fell_off_table`, `still_holding`, `still_holding_over_target`, `not_settled_in_target`,
`wrong_object_moved`, `touched_but_not_moved`, `object_not_moved`, `left_target_after_placing`,
`dropped_in_transit`, `placed_outside_target`. Instructions the parser does not understand get `unknown`.

## Findings (Windows 11, Intel Iris Xe, dreamlabs 1.0.42 / dreamlabs-client 1.1.7)

- `dreamcli` does not start on Windows (`import fcntl`); run the sim scripts with Python directly.
- `pip install dreamlabs` needs `PYTHONUTF8=1` on a Korean-locale Windows (cp949 decode error in `portal`).
- Rendering three cameras takes ~420 ms per 50 ms step. With the stock loop a 48-action chunk took ~21 s, the
  gateway dropped the idle connection (SSL EOF) and the client did not reconnect: every run died on request 2.
- Idle gaps of 5 s worked 12/12; 10 s expired the session every time (`DreamSessionExpired`), not the documented 20 s.
- macOS (M3): running GPU post-processing alongside a headless batch slowed the sim ~10x and the session restarted on
  nearly every request (no grasps in 7 episodes); hence `rerender.py` waits while an episode is recording.
