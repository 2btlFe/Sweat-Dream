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
  .seed-advance { display: flex; gap: 6px; align-items: flex-start; font-size: 11px; color: var(--muted); margin: 6px 0 0; }
  .seed-advance input { width: auto; margin: 1px 0 0; accent-color: var(--accent); }
  .ood-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
  .ood-random { display: inline-flex; align-items: center; gap: 4px; margin: 0; font-size: 11px; font-weight: 500;
    white-space: nowrap; cursor: pointer; }
  .ood-random input { width: auto; margin: 0; accent-color: var(--accent); }
  .random-all { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 12px; margin: 0 0 10px; }
  .random-all button { padding: 3px 10px; font-size: 12px; }
  legend .ood-random { margin-left: 8px; font-weight: 400; }`;

export function newSeed() {
  $("seed").value = Math.floor(Math.random() * 1000000);  // a train seed: 0..999999
}

/** After a run (or a batch of n episodes, seeds seed … seed + n - 1) the Seed field moves on to the next unused
 *  seed, when "Next seed after each run" is ticked. @return Whether it moved. */
export function advanceSeed(n = 1) {
  if (!$("seedAdvance")?.checked || !$("seed").value.trim()) return false;
  $("seed").value = (Number($("seed").value) + n) % 1000000;  // train seeds are 0..999999
  return true;
}

/** Bring back a run's start: its seed and the settings it drew from the seed (the others are the page's). */
export function useSeed(seed, randomNames = []) {
  $("seed").value = seed;
  for (const box of document.querySelectorAll("#ood [data-random]")) {
    box.checked = randomNames.includes(box.dataset.random);
    lock(box.closest(".ood-field"));
  }
  syncGroups();
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
      come from it (random camera settings apply to the workspace camera).</p>
    <label class="seed-advance"><input type="checkbox" id="seedAdvance" checked> Next seed after each run or batch,
      so every start differs (each run keeps its seed: <em>Same start</em> in its detail brings it back)</label>`;
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
  form.addEventListener("reset", () => setTimeout(() => { form.querySelectorAll(".ood-field").forEach(lock); syncGroups(); }));

  // Tick many at once: every setting (All / None), or every setting of one group (the box in its title).
  const setBoxes = (boxes, on) => {
    boxes.forEach(box => { box.checked = on; lock(box.closest(".ood-field")); });
    syncGroups();
    onChange();
  };
  const bar = document.createElement("div");
  bar.className = "random-all";
  bar.innerHTML = `<span class="muted">Random from the seed:</span>
    <button type="button" class="ghost" id="randomAll">All</button><button type="button" class="ghost" id="randomNone">None</button>`;
  form.before(bar);
  $("randomAll").onclick = () => setBoxes(form.querySelectorAll("[data-random]"), true);
  $("randomNone").onclick = () => setBoxes(form.querySelectorAll("[data-random]"), false);
  for (const group of form.querySelectorAll("fieldset")) {
    const boxes = group.querySelectorAll("[data-random]");
    if (!boxes.length) continue;
    const dice = document.createElement("label");
    dice.className = "ood-random";
    const all = Object.assign(document.createElement("input"), { type: "checkbox", title: "Draw every setting of this group from the seed" });
    all.dataset.randomGroup = "";
    dice.append(all, " all random");
    group.querySelector("legend").append(dice);
    all.addEventListener("change", () => setBoxes(boxes, all.checked));
    boxes.forEach(box => box.addEventListener("change", syncGroups));
  }
  syncGroups();
}

// A group's "all random" box: ticked when all its settings are, half-ticked when some are.
function syncGroups() {
  for (const all of document.querySelectorAll("#ood [data-random-group]")) {
    const boxes = [...all.closest("fieldset").querySelectorAll("[data-random]")];
    const on = boxes.filter(box => box.checked).length;
    all.checked = on === boxes.length;
    all.indeterminate = on > 0 && on < boxes.length;
  }
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
