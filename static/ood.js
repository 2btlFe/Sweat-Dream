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
        fieldRow.append(label);
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
        group.append(fieldRow);
      }
      form.append(group);
    }
    $("oodReset").hidden = false;
    return true;
  } catch (err) {
    $("oodError").textContent = "Could not load scene settings. Restart app.py and reload the page.";
    $("oodError").hidden = false;
    return false;
  }
}

export function readOOD(report = true) {
  const form = $("ood");
  if (!(report ? form.reportValidity() : form.checkValidity())) return null;
  return Object.fromEntries(Array.from(form.querySelectorAll("input[name], select[name]"), control => [
    control.name, control.type === "number" ? control.valueAsNumber : control.value,
  ]));
}
