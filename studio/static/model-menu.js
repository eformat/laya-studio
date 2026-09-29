/* Model catalog popover for the Laya Decision Studio. */

export async function loadModels() {
  const res = await fetch("/v1/models");
  if (!res.ok) throw new Error("/v1/models failed");
  return res.json();
}

export async function loadStatus() {
  const res = await fetch("/api/status");
  if (!res.ok) throw new Error("/api/status failed");
  return res.json();
}

/* Render the model options into the popover. `loaded` maps checkpoint id ->
   boolean (from /api/status); the dot shows resident checkpoints. */
export function renderModelOptions(container, models, loaded, current, onSelect) {
  container.innerHTML = "";
  for (const model of models) {
    const btn = document.createElement("button");
    btn.className = "opt" + (model.id === current ? " selected" : "");
    btn.setAttribute("role", "option");
    btn.setAttribute("aria-selected", model.id === current ? "true" : "false");

    const head = document.createElement("span");
    head.className = "opt-head";
    const dot = document.createElement("span");
    dot.className = "dot" + (model.auto ? " on" : (loaded?.[model.id] ? " on" : " off"));
    head.appendChild(dot);
    const name = document.createElement("span");
    name.textContent = model.label;
    head.appendChild(name);

    const blurb = document.createElement("span");
    blurb.className = "blurb";
    blurb.textContent = model.blurb || "";

    const meta = document.createElement("span");
    meta.className = "meta";
    const bits = [];
    if (model.auto) bits.push("routing");
    if (model.encoder) bits.push(model.encoder);
    if (model.params) bits.push(model.params + " params");
    if (model.context) bits.push("ctx " + model.context);
    if (model.repo_id && !model.auto) bits.push(model.repo_id);
    meta.textContent = bits.join(" · ");

    btn.append(head, blurb, meta);
    btn.addEventListener("click", () => {
      onSelect(model.id);
    });
    container.appendChild(btn);
  }
}
