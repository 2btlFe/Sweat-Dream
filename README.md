# Dream Labs Auto-Labeling Pipeline

Stage 1: an episode recorder and a simulator ground-truth judge for Dream-1 runs in the Dream YAM Sim.

## Setup (Windows)

```powershell
$env:PYTHONUTF8 = "1"; python -m venv C:\dream\.venv
C:\dream\.venv\Scripts\python.exe -m pip install dreamlabs
C:\dream\.venv\Scripts\python.exe -m pip install torch timm "huggingface_hub<1.0" "click<8.2.0"   # only for pca.py
```

Without CUDA, `pca.py` runs on the CPU (on an M3's CPU cores ~0.9 s per frame when all three cameras changed;
expect slower on a laptop CPU). With CUDA but no Triton, NAF runs without `torch.compile` by itself.

Put the API key in `~/.dreamlabs/credentials` as `DREAM_API_KEY=...`.

## Setup (macOS)

```bash
uv venv --seed --python 3.12 ~/dream-env && source ~/dream-env/bin/activate
pip install dreamlabs torch timm "huggingface_hub<1.0" "click<8.2.0"   # torch + timm for pca.py
```

`huggingface_hub<1.0` and `click<8.2.0` keep `pip check` clean: newer huggingface_hub wants a click that dreamlabs refuses.
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
C:\dream\.venv\Scripts\python.exe pca.py --all                                                  # feature PCA videos (after episodes)
C:\dream\.venv\Scripts\python.exe pca.py --all --watch                                          # ...and keep doing new ones
C:\dream\.venv\Scripts\python.exe rerender.py --all --variants 4                                 # appearance augmentation
C:\dream\.venv\Scripts\python.exe rerender.py <episode_id> --check                              # replay vs recorded video
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
It waits while an episode is being recorded, like `pca.py`.

## Feature PCA (DINOv2 + NAF)

`pca.py` adds `episodes/<id>/pca/`: `top.mp4` / `left.mp4` / `right.mp4` with a 224x224 feature PCA map for every
camera frame (same frame count and fps, so the labeling UI plays them in sync with the cameras: Camera / Feature PCA /
Camera + PCA), `info.json`, `embeddings.npy` (per frame and camera, the DINOv2 mean patch feature) and `status.json`.

1. Each frame is resized to 224x224 and passed through DINOv2 ViT-S/14: 16x16 patches of 384-d features. Dream-1's
   own features are not available through its API, so these show how a general vision model groups the scene.
2. One PCA basis per episode, fitted on 48 frames spread over it and all three cameras: a colour means the same at
   every moment and in every view. The first three components are shown as RGB.
3. NAF ([Zero-Shot Feature Upsampling via Neighborhood Attention Filtering](https://github.com/valeoai/NAF),
   CVPR 2026, Apache-2.0, vendored in `third_party/naf`) upsamples the map to 224x224, guided by the image. PCA is
   linear and NAF averages each attention head's values, so the 3 projected values per head are upsampled instead of
   384 features (same result). Without NATTEN (CUDA) a plain PyTorch neighbourhood attention runs it on any GPU or
   CPU; cached RoPE tables, split padding, float16 and `torch.compile` make it ~1.9x faster than released
   (M3: ~270 ms for three cameras; ~200-390 ms per frame with DINOv2, depending on how many cameras changed).
4. Each map is averaged with its neighbours in time, and a camera whose image did not change keeps its last map.

It pauses while an episode is being recorded (`episodes/.recording-<pid>`, written by `recorder.py`): its GPU work
slows the simulator's cameras, and a slower simulator lets the Dream-1 session expire (see Findings). `--no-naf`
upsamples bilinearly; the NAF weights download on first use (torch hub cache), the DINOv2 weights from Hugging Face.

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
| `pca.py` | Feature PCA videos for recorded episodes (DINOv2 + NAF, 224x224), paused while an episode records |
| `rerender.py` | Appearance augmentation: re-render recorded episodes under new lighting, colours and camera poses |
| `feature_pca.py` | DINOv2 patch features and the per-episode PCA basis |
| `naf_upsampler.py` | NAF upsampling of the PCA projection, with the speed-ups; `third_party/naf` is the vendored NAF |

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
  nearly every request (no grasps in 7 episodes); hence `pca.py` waits while an episode is recording.
