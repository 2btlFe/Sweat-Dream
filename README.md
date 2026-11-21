# Sweat-Dream

Run Dream-1 in the simulator, set success rules, collect episodes, and save reproducible labels.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install dreamlabs
```

OpenYAM scene loading also needs Node.js on `PATH`. The first OpenYAM run downloads the publicly served Sandbox MuJoCo package into a temporary cache; later runs reuse it. The web build's catalog URL is not a documented API and may change.

Get an API key at https://platform.dreamlabs.ai. The website keeps the key you enter in server memory and clears it when you press **Clear** or restart the server.

## Run

```bash
python app.py
```

Open http://localhost:8000.

1. Paste your API key → **Use key**. It is kept in memory only; **Clear** removes it.
2. Choose YAM or OpenYAM and a room under **Simulation setup**. The room styles, layout, and lighting affect the next run.
3. With YAM, use **Scene** to choose objects and targets and drag them on the table map. OpenYAM currently uses its fixed Sandbox object layout.
4. Type a task such as `put apple into the bowl` → **Run**.
5. Set the distance, release, and still-time rules. Existing runs are re-evaluated immediately.
6. Select a run to inspect and replay it. **Save labels** writes the current server-side verdicts to `label.json`.

## Simulator selection

Dream Labs Sandbox currently lists YAM, Trossen WidowX AI, MakerMods Metal, SO-101, OpenYAM, AgileX PiPER, Universal Robots UR5e, Seeed reBot B601, and MakerMods Maker Arm. Its rooms are Kitchen, Sunny apartment, Cottage, Walnut study, and Evening den. Objects can be selected from Food, Toys, Kitchen, Home and tools, Containers, and Pictures, up to eight on the table. The public Playground's **SIM 1–4** buttons switch between shared live rooms; they are not local scene presets.

The bundled `dream_examples/yam_mujoco` package provides YAM. OpenYAM uses the public Sandbox `fruit_table` MuJoCo robot, scene, camera, and object packages, with a local Dream-1 control adapter. **Run** supports YAM and OpenYAM in the tabletop and five simplified local room styles. The room styles use publicly served Sandbox desk textures and color/light descriptors at runtime; the room walls are local geometry because the Sandbox's baked room meshes render in Three.js, outside MuJoCo. No Dream Labs asset is committed to this repository. Other robots are marked **unavailable** until their distinct action mapping and gripper contact rules are validated. Lighting changes rendering. Each run records its robot, room, layout, lighting, and seed in `meta.json`.

The run list also shows episodes collected from the command line, including after a page reload.

Select a run and press **Play** to replay it. New runs save camera frames at 5 fps and support scrubbing. Earlier runs save only object coordinates and the last camera image, so they replay as a top-down object path without robot motion. The full camera replay is not recoverable for earlier runs.

To collect 30 episodes across the three sample tasks with ten seeded starting layouts:

```bash
python batch.py --repeats 10 --seconds 60 --budget-minutes 60
```

The batch uses the selected robot and room, the key in `~/.dreamlabs/credentials` (`DREAM_API_KEY=...`), and a snapshot of the rules saved in the UI. Batch runs use the simulator's default object layout. It writes each episode to `runs/<id>/` and stops on an invalid run, such as an API authentication error. Check the API key and endpoint before starting a large batch. `--tasks`, `--seed`, `--repeats`, and `--seconds` are adjustable.

## Team collection for fine-tuning candidates

Each teammate installs the repo and their own Dream Labs API key, then chooses a unique member number (1–6) and one object. Run this on **each laptop**; the example makes 40 attempts for member 1:

```bash
python collect_success.py --member 1 --object apple --attempts 40 --seconds 60
```

Change `--member` and `--object` on each laptop. Supported objects are `apple`, `orange`, `banana`, and `bottle`; six people will need to share some objects. The exact instruction is `pick <object> into the bowl`. Each member gets a disjoint, reproducible seed range. Use `--attempts 30` for 30 attempts. The default time budget is 120 minutes; change it with `--budget-minutes`. Run the **same command again** to resume unfinished attempts. Keep the same rules, robot, room, lighting, and seconds during a collection. All laptops read the committed `collection_rules.json`, so agree on that file before starting; changing it mid-collection is rejected. The UI's machine-local rules do not affect team collection.

Every attempt is saved under `runs/<id>/` with `meta.json`, `trajectory.json`, camera frames, `frames.json` (frame timestamps), and `label.json`. The trajectory now includes time-aligned left and right robot actions. Only judged successes with complete actions and frames are copied to `batches/member-N-object/success/`; `manifest.json` records all completed attempts and failure reasons. Invalid episodes stop the collection for inspection. Copy each teammate's entire `batches/member-N-object/` directory to a shared drive, then combine the `success/` directories. `runs/` and `batches/` are ignored by Git because they may be large.

These exports are **raw fine-tuning candidates**, not a Dream Labs upload file. The replay images are a combined top and wrist view, and the action vectors are simulator commands. Confirm Dream Labs' required training schema, camera separation, action representation, and data rights before uploading. Review a sample of automatically labeled successes manually; the coordinate-based label may not match visual task quality.

## Rules

| Rule | Meaning |
| --- | --- |
| Distance | Final horizontal distance from the named object to the bowl/plate center. The object's center must also be in the target's fixed height band. |
| Release | Whether the named object must stop touching a gripper fingertip. |
| Still time | How long the final placement and release must persist. Maximum object speed in that final window must stay below 0.05 m/s. |

The current UI exposes distance from 1 to 15 cm and still time from 0 to 5 seconds. The backend accepts 0.5 to 30 cm and 0 to 10 seconds. Target height bands and the 0.05 m/s speed limit are currently fixed. An execution error, user stop, missing model response, or incomplete trajectory is labeled **invalid**, never **failure**. Saved labels include the rules snapshot so later rule changes do not silently rewrite them.
