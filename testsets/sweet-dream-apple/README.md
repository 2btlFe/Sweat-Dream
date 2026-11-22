---
license: cc-by-nc-4.0
configs:
  - config_name: batchseed07_50
    data_files: batchseed07_50/starts.parquet
  - config_name: batchseed08_50
    data_files: batchseed08_50/starts.parquet
  - config_name: batchseed09_50
    data_files: batchseed09_50/starts.parquet
  - config_name: batchseed10_50
    data_files: batchseed10_50/starts.parquet
pretty_name: Sweet Dream · apple-into-bowl test sets
task_categories:
  - robotics
tags:
  - robotics
  - simulation
  - mujoco
  - dream-1
  - yam
  - evaluation
size_categories:
  - n<1K
---

# Sweet Dream · apple-into-bowl test sets

Four test sets of **start conditions** for the task **"Put the apple into the bowl"** in the Dream YAM simulator
(MuJoCo, bimanual YAM arms, Dream Labs). Each set holds 50 starts; 200 in all. No episode was run to make them:
a test set is the list of starts a model is evaluated on, so any model can be measured on exactly the same scenes.

| Folder | Batch seed | Starts | First episode seeds |
| --- | --- | --- | --- |
| `batchseed07_50/` | 7 | 50 | 151282, 768654, … |
| `batchseed08_50/` | 8 | 50 | 496220, 141530, … |
| `batchseed09_50/` | 9 | 50 | 867170, 752288, … |
| `batchseed10_50/` | 10 | 50 | 372258, 570977, … |

The 200 episode seeds are all different, and so are the 200 layouts. The starts are the same as in
[Dangni/sweet-dream-banana](https://huggingface.co/datasets/Dangni/sweet-dream-banana); only the task and its judge
differ.

## Contents of each folder

- `testset.json`: the set's settings and its 50 starts.
- `previews/epNN_seedS.jpg`: the start scene of each episode as the robot sees it (workspace camera on the left,
  the two wrist cameras on the right), rendered without calling the model.
- `contact_sheet.jpg`: the 50 workspace-camera views on one page, in episode order.

### `testset.json`

| Key | Meaning |
| --- | --- |
| `batch_seed` | The batch seed; episode *i*'s seed is derived from it (see below). |
| `count` | Number of starts (50). |
| `task` | The instruction given to the model: `Put the apple into the bowl`. |
| `judge` | How success is judged: pick & place, `apple` → `bowl` (in the bowl's area, released, still for 1 s). |
| `robot`, `room` | `yam`, `yam_tabletop`. |
| `scene_settings`, `ood_random` | Camera, light, table and initial arm angles: all at their defaults, none drawn at random. |
| `seconds` | Time limit per episode: 60 s. |
| `episodes[]` | `index`, `seed` (the episode's start-scene seed), `layout` (x, y in metres of `apple`, `orange`, `banana`, `water_bottle`, `bowl`, `plate`, world frame of the simulator) and `preview` (relative path). |

## Start states (tables)

Each folder also has **`starts.parquet`**: one row per start, with everything needed to restore it without the
seed or the scene code. The dataset viewer shows it as a table (configs below), and
`datasets.load_dataset("Dangni/sweet-dream-apple", "batchseed07_50")` loads it.

| Column | Meaning |
| --- | --- |
| `index`, `seed`, `batch_seed` | Episode index, its start-scene seed, the batch seed. |
| `task`, `judge_object`, `judge_target` | The instruction and what success is judged on. |
| `observation_state` | The 14 values a policy reads: `left_joint1..6, left_gripper, right_joint1..6, right_gripper` (radians; gripper opening 0..1). |
| `qpos` (44), `qvel` (40) | The full MuJoCo state at the start. |
| `<item>_pos`, `<item>_quat_wxyz` | Position (x, y, z, m) and orientation (w, x, y, z) of `apple`, `orange`, `banana`, `water_bottle`, `bowl`, `plate`. |
| `image_top`, `image_left`, `image_right` | The three cameras at the start (640 x 360 JPEG): workspace, left wrist, right wrist. |

The same images are under `cameras/`, and `testset.json` now carries each episode's item poses
(`items`) and `observation_state`, with the `qpos` layout under `start_state`.

### Restoring a start in MuJoCo

```python
import mujoco
from yam_env import YamEnv          # dreamlabs: dream_examples/yam_mujoco

env = YamEnv(); m, d = env.model, env.data
for name in ("bowl", "plate"):      # fixed bodies: their pose lives in the model, not in qpos
    m.body_pos[m.body(name).id] = row[f"{name}_pos"]
    m.body_quat[m.body(name).id] = row[f"{name}_quat_wxyz"]
d.qpos[:] = row["qpos"]; d.qvel[:] = row["qvel"]
mujoco.mj_forward(m, d)
```

All 200 starts restore this way with zero error in every item pose and joint.

## How the starts are made

One batch seed is a whole batch: episode *i* of the batch with seed *b* starts from seed

```
int.from_bytes(sha256(f"sweat-dream batch {b} episode {i}").digest()[:8], "big") % 1_000_000
```

and that seed places the objects around their usual spots, within reach of both arms (scene version 3 of the
Sweet Dream scene sampler). The same seed always gives the same layout; re-rendering a stored seed reproduces its
`layout` exactly.

## Using a test set

Run each start with the model under test alone (no recovery script, no human), the task above and the 60 s limit,
and count the episodes the judge marks a success. In the Sweet Dream app this is a batch with **From the seed**,
seed *b*, 50 episodes, default scene settings, **Dream-1 only**, task "Put the apple into the bowl", and the model
picked. Comparing models on the same set compares them on identical scenes. These seeds were never used for
training.

## License

[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/):
you may share and adapt this dataset for non-commercial purposes, with attribution.
