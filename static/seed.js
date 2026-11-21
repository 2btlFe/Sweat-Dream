// The seed on the page (scene.py): a Seed field with a new-seed button, and a "random" box on each scene setting.
// Ticked settings are drawn from the seed by the server, so the same seed gives the same start again.
// index.html calls initSeed() once the scene settings form is built, sends seeding() with every preview and run,
// and passes each preview response to afterPreview(). ood.js is not changed.

const $ = id => document.getElementById(id);
const NOT_RANDOM = new Set(["camera_target"]);  // chooses which camera to edit; not a scene value
const CAMERA_FIELDS = new Set(["camera_x_cm", "camera_y_cm", "camera_z_cm", "camera_pitch_deg", "camera_yaw_deg",
  "camera_roll_deg", "camera_fov_offset_deg"]);  // random ones apply to the workspace (top) camera

const STYLE = `
  .seed-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: end; margin-top: 12px; }
  .seed-note { font-size: 11px; margin: 6px 0 0; }
  .ood-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
  .ood-random { display: inline-flex; align-items: center; gap: 4px; margin: 0; font-size: 11px; font-weight: 500;
    white-space: nowrap; cursor: pointer; }
  .ood-random input { width: auto; margin: 0; accent-color: var(--accent); }`;

export function newSeed() {
  $("seed").value = Math.floor(Math.random() * 1000000);  // a train seed: 0..999999
}

/** The seed and the ticked settings, for /api/preview and /api/run. An empty seed gets a new one first. */
export function seeding() {
  if (!$("seed")) return {};  // the settings form did not load: sim.py picks a seed and prints it
  if (!$("seed").value.trim()) newSeed();
  return {
    seed: $("seed").value.trim(),
    ood_random: Array.from(document.querySelectorAll("#ood [data-random]:checked"), box => box.dataset.random),
  };
}

/** Add the Seed field and the random boxes. @param onChange Called when either changes (to refresh the preview). */
export function initSeed(onChange) {
  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.append(style);

  const row = document.createElement("div");
  row.innerHTML = `
    <div class="seed-row">
      <div><label for="seed">Seed</label><input id="seed" type="number" min="0" step="1" placeholder="new on Run"></div>
      <button id="newSeed" type="button" title="New random seed" aria-label="New random seed">↻</button>
    </div>
    <p class="muted seed-note">Same seed, same start: the object layout and every setting marked <em>random</em>
      come from it (random camera settings apply to the workspace camera).</p>`;
  $("examples").before(...row.children);
  newSeed();
  $("seed").addEventListener("input", onChange);
  $("newSeed").addEventListener("click", () => { newSeed(); onChange(); });

  const form = $("ood");
  for (const fieldRow of form.querySelectorAll(".ood-field")) {
    const control = fieldRow.querySelector("select[name], input[type=number][name]");
    if (!control || NOT_RANDOM.has(control.name)) continue;
    const label = fieldRow.querySelector("label");
    const dice = document.createElement("label");
    dice.className = "ood-random";
    const box = Object.assign(document.createElement("input"), { type: "checkbox", title: "Draw this value from the seed" });
    box.dataset.random = control.name;
    dice.append(box, " random");
    const head = document.createElement("div");
    head.className = "ood-head";
    label.replaceWith(head);
    head.append(label, dice);
    box.addEventListener("change", () => { lock(fieldRow); onChange(); });
  }
  form.addEventListener("reset", () => setTimeout(() => form.querySelectorAll(".ood-field").forEach(lock)));
}

// A ticked setting is drawn by the server: its own inputs are locked and show the drawn value.
function lock(fieldRow) {
  const box = fieldRow.querySelector("[data-random]");
  if (!box) return;
  fieldRow.querySelectorAll("input:not([data-random]), select").forEach(input => { input.disabled = box.checked; });
}

/** Show the values the seed drew (the preview's X-Scene-Controls header) in the ticked settings. */
export function afterPreview(response) {
  let drawn;
  try { drawn = JSON.parse(response.headers.get("X-Scene-Controls") || "{}"); } catch { return; }
  const cameraShown = ($("ood-camera_target")?.value ?? "top") === "top";
  for (const box of document.querySelectorAll("#ood [data-random]:checked")) {
    const name = box.dataset.random;
    if (CAMERA_FIELDS.has(name) && !cameraShown) continue;  // the inputs show another camera now
    const control = $("ood-" + name);
    let value = CAMERA_FIELDS.has(name) ? drawn.camera_views?.top?.[name] : drawn[name];
    if (value === undefined) {  // not among the changes: the seed drew the default value
      value = control.tagName === "SELECT"
        ? Array.from(control.options).find(option => option.defaultSelected).value : control.defaultValue;
    }
    control.value = value;
    const slider = control.parentElement.querySelector("input[type=range]");
    if (slider) slider.value = value;
  }
}
