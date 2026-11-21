# Sweat-Dream

Run Dream-1 in the simulator, set success rules, collect episodes, and save reproducible labels.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install dreamlabs
```

OpenYAM scene loading also needs Node.js on `PATH`. The first OpenYAM run downloads the publicly served Sandbox MuJoCo package into a temporary cache; later runs reuse it. The web build's catalog URL is not a documented API and may change.

Put your API key (from https://platform.dreamlabs.ai) in `~/.dreamlabs/credentials`:

```
DREAM_API_KEY=your-key-here
```

## Run

```bash
python app.py
```

Open http://localhost:8000.

1. Type a task such as `put apple into the bowl` → **Run**.
2. Open **Simulation setup** in the right panel. Choose YAM and a room, then choose a fixed or seeded object layout and Standard, Warm, or Cool lighting.
3. Set the distance, release, and still-time rules. Existing runs are re-evaluated immediately.
4. Select a run to inspect its final frame, checks, and distance chart. **Save labels** writes the current server-side verdicts to `label.json`.

## Simulator selection

Dream Labs Sandbox currently lists YAM, Trossen WidowX AI, MakerMods Metal, SO-101, OpenYAM, AgileX PiPER, Universal Robots UR5e, Seeed reBot B601, and MakerMods Maker Arm. Its rooms are Kitchen, Sunny apartment, Cottage, Walnut study, and Evening den. Objects can be selected from Food, Toys, Kitchen, Home and tools, Containers, and Pictures, up to eight on the table. The public Playground's **SIM 1–4** buttons switch between shared live rooms; they are not local scene presets.

The bundled `dream_examples/yam_mujoco` package provides YAM. OpenYAM uses the public Sandbox `fruit_table` MuJoCo robot, scene, camera, and object packages, with a local Dream-1 control adapter. **Run** supports YAM and OpenYAM in the tabletop and five simplified local room styles. The room styles use publicly served Sandbox desk textures and color/light descriptors at runtime; the room walls are local geometry because the Sandbox's baked room meshes render in Three.js, outside MuJoCo. No Dream Labs asset is committed to this repository. Other robots are marked **unavailable** until their distinct action mapping and gripper contact rules are validated. Lighting changes rendering. Each run records its robot, room, layout, lighting, and seed in `meta.json`.

The run list also shows episodes collected from the command line, including after a page reload.

Select a run and press **Play** to replay it. New runs save camera frames at 5 fps and support scrubbing. Earlier runs save only object coordinates and the last camera image, so they replay as a top-down object path without robot motion. The full camera replay is not recoverable for earlier runs.

To collect 30 episodes across the three sample tasks with ten seeded starting layouts:

```bash
python batch.py --repeats 10 --seconds 60 --budget-minutes 60
```

The batch uses a snapshot of the rules saved in the UI. It writes each episode to `runs/<id>/` and stops on an invalid run, such as an API authentication error. Check the API key and endpoint before starting a large batch. `--tasks`, `--seed`, `--repeats`, and `--seconds` are adjustable.

## Rules

| Rule | Meaning |
| --- | --- |
| Distance | Final horizontal distance from the named object to the bowl/plate center. The object's center must also be in the target's fixed height band. |
| Release | Whether the named object must stop touching a gripper fingertip. |
| Still time | How long the final placement and release must persist. Maximum object speed in that final window must stay below 0.05 m/s. |

The current UI exposes distance from 1 to 15 cm and still time from 0 to 5 seconds. The backend accepts 0.5 to 30 cm and 0 to 10 seconds. Target height bands and the 0.05 m/s speed limit are currently fixed. An execution error, user stop, missing model response, or incomplete trajectory is labeled **invalid**, never **failure**. Saved labels include the rules snapshot so later rule changes do not silently rewrite them.
