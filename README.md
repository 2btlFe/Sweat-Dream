# Dream Labs Auto-Labeling Pipeline

Stage 1: an episode recorder and a simulator ground-truth judge for Dream-1 runs in the Dream YAM Sim.

## Setup (Windows)

```powershell
$env:PYTHONUTF8 = "1"; python -m venv C:\dream\.venv
C:\dream\.venv\Scripts\python.exe -m pip install dreamlabs
```

Put the API key in `~/.dreamlabs/credentials` as `DREAM_API_KEY=...`.

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
fingertip contact), `top.mp4` / `left.mp4` / `right.mp4` (H.264, 5 fps), `outcome.json`.

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
