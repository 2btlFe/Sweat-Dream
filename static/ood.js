const $ = id => document.getElementById(id);

export async function loadOOD(api) {
  const form = $("ood");
  form.addEventListener("submit", event => event.preventDefault());
  try {
    const groups = await api("/api/ood");
    const supported = ["Camera view", "Image effects", "Lighting & surface"];
    if (Object.keys(groups).length !== supported.length || supported.some(group => !(group in groups))) {
      throw new Error("Outdated scene settings");
    }
    for (const [title, fields] of Object.entries(groups)) {
      const group = document.createElement("fieldset");
      const legend = document.createElement("legend");
      legend.textContent = title;
      group.append(legend);
      for (const [name, field] of Object.entries(fields)) {
        const fieldRow = document.createElement("div");
        fieldRow.className = "ood-field";
        const label = document.createElement("label");
        label.textContent = field.label;
        const control = document.createElement(field.choices ? "select" : "input");
        control.name = name;
        control.id = "ood-" + name;
        label.htmlFor = control.id;
        // "random": the value is drawn from the run's seed (scene.py draw_controls), so it is reproducible.
        const dice = document.createElement("label");
        dice.className = "ood-random";
        const box = document.createElement("input");
        Object.assign(box, { type: "checkbox", title: "Draw this value from the seed" });
        box.dataset.random = name;
        dice.append(box, " random");
        const head = document.createElement("div");
        head.className = "ood-head";
        head.append(label, dice);
        fieldRow.append(head);
        if (field.choices) {
          for (const [value, text] of Object.entries(field.choices)) {
            control.add(new Option(text, value, value === field.default));
          }
          fieldRow.append(control);
        } else {
          const bounds = { min: field.min, max: field.max, step: field.step, defaultValue: field.default };
          Object.assign(control, { type: "number", ...bounds, required: true });
          const slider = document.createElement("input");
          Object.assign(slider, { type: "range", ...bounds });
          slider.setAttribute("aria-label", field.label + " slider");
          slider.addEventListener("input", () => { control.value = slider.value; });
          control.addEventListener("input", () => {
            if (control.validity.valid) slider.value = control.value;
          });
          const pair = document.createElement("div");
          pair.className = "ood-slider";
          pair.append(slider, control);
          fieldRow.append(pair);
        }
        box.addEventListener("change", () => syncRandom(fieldRow));
        group.append(fieldRow);
      }
      form.append(group);
    }
    form.addEventListener("reset", () => setTimeout(() => form.querySelectorAll(".ood-field").forEach(syncRandom)));
    $("oodReset").hidden = false;
    return true;
  } catch (err) {
    $("oodError").textContent = "Could not load scene settings. Restart app.py and reload the page.";
    $("oodError").hidden = false;
    return false;
  }
}

// A control marked random is drawn by the server from the seed: its own inputs are locked and show the drawn value.
function syncRandom(fieldRow) {
  const random = fieldRow.querySelector("[data-random]").checked;
  fieldRow.querySelectorAll("input:not([data-random]), select").forEach(input => { input.disabled = random; });
}

/** The names of the scene controls marked random. */
export function readRandom() {
  return Array.from($("ood").querySelectorAll("[data-random]:checked"), box => box.dataset.random);
}

/** Show the values the seed drew for the random controls (from the preview's X-Scene-Controls header). */
export function showDrawn(values) {
  for (const name of readRandom()) {
    const control = $("ood-" + name);
    let value = values[name];
    if (value === undefined) {  // not in the changes: the seed drew the default value
      value = control.tagName === "SELECT"
        ? Array.from(control.options).find(option => option.defaultSelected).value : control.defaultValue;
    }
    control.value = value;
    const slider = control.parentElement.querySelector("input[type=range]");
    if (slider) slider.value = value;
  }
}

export function readOOD(report = true) {
  const form = $("ood");
  if (!(report ? form.reportValidity() : form.checkValidity())) return null;
  return Object.fromEntries(Array.from(form.querySelectorAll("input[name], select[name]"), control => [
    control.name, control.type === "number" ? control.valueAsNumber : control.value,
  ]));
}
