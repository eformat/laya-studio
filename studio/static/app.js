/* Laya Decision Studio — workbench logic.
   Modes (LLM routing / One context / Batch contexts), example scenarios,
   editable question sets, the model popover, and result rendering with every
   candidate probability. */

import {
  buildBatch, buildSingle, clone, emptyQuestion, fmt, nextContextId,
  optionRows, parseStatesJson, questionOrder, statesToJson, topAnswer,
} from "./contract.js";
import { loadModels, loadStatus, renderModelOptions } from "./model-menu.js";

const MODES = {
  routing: {
    heading: "User prompt",
    description: "Route each prompt to the right LLM before calling it.",
    summary: (n) => `1 prompt · ${n} routing decisions`,
    batch: false,
  },
  single: {
    heading: "Context",
    description: "Inspect several aspects of one case.",
    summary: (n) => `1 context · ${n} decisions`,
    batch: false,
  },
  batch: {
    heading: "Contexts",
    description: "Apply shared questions to a list of inputs in one batched pass.",
    summary: (n, states) => `${states} contexts · ${n} shared question${n === 1 ? "" : "s"}`,
    batch: true,
  },
};

const state = {
  mode: "routing",
  examples: { routing: [], single: [], batch: [] },
  pristine: { routing: [], single: [], batch: [] },
  current: { routing: 0, single: 0, batch: 0 },
  drafts: { routing: null, single: null, batch: null },
  stateView: "text", // text | json
  batchView: "list", // list | json
  model: "auto",
  models: [],
  loaded: {},
  ready: null,
  lazy: false,
  autorun: false,
  running: false,
  results: { routing: null, single: null, batch: null },
  timings: { routing: null, single: null, batch: null },
  openRow: null,
  sharedOpen: true,
};

const $ = (id) => document.getElementById(id);
const els = {};

let autoTimer = null;
let pollTimer = null;

/* ------------------------------- init -------------------------------- */

async function init() {
  cacheEls();
  bindChrome();
  try {
    const [modelsDoc, status, examples] = await Promise.all([
      loadModels(),
      loadStatus(),
      fetch("/api/examples").then((r) => {
        if (!r.ok) throw new Error("/api/examples failed");
        return r.json();
      }),
    ]);
    state.models = modelsDoc.models || [];
    state.loaded = status.loaded || {};
    state.ready = status.ready;
    state.lazy = !!status.lazy;
    for (const ex of examples) {
      const surface = ex.surface in state.examples ? ex.surface : "single";
      state.examples[surface].push(ex);
      state.pristine[surface].push(clone(ex));
    }
  } catch (err) {
    console.error(err);
    toast("Could not reach the studio API");
    state.ready = false;
  }
  selectMode("routing");
  applyStatus();
  schedulePoll();
}

function cacheEls() {
  for (const id of [
    "connection", "context-mode-routing", "context-mode-single", "context-mode-batch",
    "mode-description", "scenario-summary", "examples", "scenario-category", "scenario-title",
    "reset", "run-top", "offline-note", "context-heading", "context-count", "single-format",
    "batch-format", "mode-text", "mode-json", "batch-view-list", "batch-view-json",
    "single-context-editor", "state-input", "state-count", "batch-context-editor",
    "batch-list-view", "context-list", "context-limit", "add-context", "batch-json-view",
    "states-json", "batch-json-error", "shared-toggle", "questions-heading", "question-count",
    "add-toggle", "add-options", "shared-note", "questions", "auto-run", "run", "run-label",
    "run-shortcut", "input-budget", "model-card", "model-trigger", "model-name", "model-version",
    "model-spec", "model-state", "results-title", "result-count", "run-status", "results",
    "timing", "model-popover", "model-options", "toast", "questions-heading",
  ]) els[id] = $(id);
  const missing = Object.keys(els).filter((id) => !els[id]);
  if (missing.length) console.error("cacheEls missing element ids:", missing);
}

/* ------------------------------ chrome ------------------------------- */

function bindChrome() {
  $("context-mode-routing").addEventListener("click", () => selectMode("routing"));
  $("context-mode-single").addEventListener("click", () => selectMode("single"));
  $("context-mode-batch").addEventListener("click", () => selectMode("batch"));

  $("mode-text").addEventListener("click", () => setStateView("text"));
  $("mode-json").addEventListener("click", () => setStateView("json"));
  $("batch-view-list").addEventListener("click", () => setBatchView("list"));
  $("batch-view-json").addEventListener("click", () => setBatchView("json"));

  $("reset").addEventListener("click", () => {
    resetDraft();
    renderAll();
    toast("Draft reset");
  });

  $("add-context").addEventListener("click", () => {
    const draft = draftFor();
    if (!draft) return;
    draft.states.push({ id: nextContextId(draft.states), state: "" });
    renderEditor();
    scheduleAuto();
  });

  $("add-toggle").addEventListener("click", () => {
    const open = els["add-options"].hidden;
    els["add-options"].hidden = !open;
    els["add-toggle"].setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", (ev) => {
    if (!ev.target.closest(".add-menu") && !els["add-options"].hidden) {
      els["add-options"].hidden = true;
      els["add-toggle"].setAttribute("aria-expanded", "false");
    }
    if (!ev.target.closest("#model-trigger") && !ev.target.closest("#model-popover")
        && !els["model-popover"].hidden) {
      closeModelPopover();
    }
  });
  for (const btn of els["add-options"].querySelectorAll("[data-add]")) {
    btn.addEventListener("click", () => {
      const draft = draftFor();
      if (!draft) return;
      const type = btn.dataset.add;
      let id = type;
      let n = 1;
      while (id in draft.questions) id = `${type}_${++n}`;
      draft.questions[id] = emptyQuestion(type);
      renderQuestions();
      renderSummary();
      scheduleAuto();
    });
  }

  els["shared-toggle"].addEventListener("click", () => {
    state.sharedOpen = !state.sharedOpen;
    els["shared-toggle"].setAttribute("aria-expanded", String(state.sharedOpen));
    els.questions.hidden = !state.sharedOpen;
    if (els["shared-note"]) els["shared-note"].hidden = !state.sharedOpen || state.mode !== "batch";
  });

  els["auto-run"].addEventListener("change", () => {
    state.autorun = els["auto-run"].checked;
    if (state.autorun) scheduleAuto();
    else if (autoTimer) { clearTimeout(autoTimer); autoTimer = null; }
  });

  for (const btn of [els.run, els["run-top"]]) {
    btn.addEventListener("click", run);
  }
  document.addEventListener("keydown", (ev) => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
      ev.preventDefault();
      run();
    }
  });

  els["model-trigger"].addEventListener("click", openModelPopover);
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeModelPopover();
  });
}

/* ------------------------------- modes ------------------------------- */

function selectMode(mode) {
  state.mode = mode;
  for (const [key, id] of [
    ["routing", "context-mode-routing"], ["single", "context-mode-single"], ["batch", "context-mode-batch"],
  ]) {
    $(id).setAttribute("aria-pressed", key === mode ? "true" : "false");
  }
  els["mode-description"].textContent = MODES[mode].description;
  if (!state.drafts[mode]) resetDraft();
  renderAll();
}

function draftFor() {
  return state.drafts[state.mode];
}

function resetDraft() {
  const mode = state.mode;
  const list = state.examples[mode];
  const index = Math.min(state.current[mode], Math.max(0, list.length - 1));
  const ex = list[index];
  state.results[mode] = null;
  state.timings[mode] = null;
  state.openRow = null;
  state.drafts[mode] = ex ? exampleDraft(ex, state.model) : emptyDraft(mode);
}

function emptyDraft(mode) {
  if (MODES[mode].batch) {
    return { states: [{ id: "T-001", state: "" }], questions: { intent: emptyQuestion("choice") } };
  }
  return { state: "", questions: { decision: emptyQuestion("choice") } };
}

function exampleDraft(ex, model) {
  // Pristine clone, then apply the variant for the selected checkpoint, if any.
  const draft = clone(ex);
  delete draft.id; delete draft.title; delete draft.category;
  delete draft.description; delete draft.surface; delete draft.model_variants;
  if (model && model !== "auto" && ex.model_variants && ex.model_variants[model]) {
    const variant = ex.model_variants[model];
    if (variant.state !== undefined) draft.state = clone(variant.state);
    if (variant.states !== undefined) draft.states = clone(variant.states);
    if (variant.questions) {
      for (const [qid, q] of Object.entries(variant.questions)) {
        draft.questions[qid] = Object.assign(draft.questions[qid] || {}, clone(q));
      }
    }
  }
  return draft;
}

function selectExample(mode, index) {
  state.current[mode] = index;
  state.results[mode] = null;
  state.timings[mode] = null;
  state.openRow = null;
  state.drafts[mode] = exampleDraft(state.examples[mode][index], state.model);
  renderAll();
}

/* ------------------------------ renders ------------------------------ */

function renderAll() {
  renderScenarioBar();
  renderHead();
  renderEditor();
  renderQuestions();
  renderModelCard();
  renderResults();
  renderRunState();
}

function renderSummary() {
  const mode = state.mode;
  const draft = draftFor();
  const n = draft ? questionOrder(draft.questions).length : 0;
  if (MODES[mode].batch) {
    els["scenario-summary"].textContent = MODES[mode].summary(n, draft ? draft.states.length : 0);
  } else {
    els["scenario-summary"].textContent = MODES[mode].summary(n);
  }
}

function renderScenarioBar() {
  const list = state.examples[state.mode];
  const wrap = els.examples;
  wrap.innerHTML = "";
  list.forEach((ex, i) => {
    const btn = document.createElement("button");
    btn.textContent = ex.title;
    btn.title = ex.description || ex.title;
    btn.className = i === state.current[state.mode] ? "active" : "";
    btn.addEventListener("click", () => selectExample(state.mode, i));
    wrap.appendChild(btn);
  });
  renderSummary();
}

function renderHead() {
  const list = state.examples[state.mode];
  const ex = list[state.current[state.mode]];
  if (ex) {
    els["scenario-category"].textContent = (ex.category || "").toUpperCase();
    els["scenario-title"].textContent = ex.title;
  } else {
    els["scenario-category"].textContent = "FREEFORM";
    els["scenario-title"].textContent = MODES[state.mode].heading;
  }
  els["context-heading"].textContent = MODES[state.mode].heading;
  els["questions-heading"].textContent = state.mode === "routing" ? "Routing decisions" : "Decisions";
  els["run-label"].textContent = "Run decisions";
  els["input-budget"].textContent = MODES[state.mode].batch
    ? "Shared questions per state. Contexts are batched into shared forward passes."
    : "Complete input per question. Nothing is truncated.";
  if (els["shared-note"]) {
    els["shared-note"].hidden = !state.sharedOpen || state.mode !== "batch";
  }
}

function setStateView(view) {
  state.stateView = view;
  renderEditor();
}

function setBatchView(view) {
  state.batchView = view;
  renderEditor();
}

function renderEditor() {
  const draft = draftFor();
  const batch = MODES[state.mode].batch;

  els["single-format"].hidden = batch;
  els["batch-format"].hidden = !batch;
  els["single-context-editor"].hidden = batch;
  els["batch-context-editor"].hidden = !batch;
  els["context-count"].hidden = !batch;
  if (batch && draft) els["context-count"].textContent = String(draft.states.length);

  $("mode-text").classList.toggle("selected", state.stateView === "text");
  $("mode-json").classList.toggle("selected", state.stateView === "json");
  $("batch-view-list").classList.toggle("selected", state.batchView === "list");
  $("batch-view-json").classList.toggle("selected", state.batchView === "json");

  if (!draft) return;

  if (!batch) {
    els["batch-context-editor"].hidden = true;
    els["single-context-editor"].hidden = false;
    if (state.stateView === "json") {
      els["state-input"].classList.add("json");
      const isObj = typeof draft.state === "object";
      if (document.activeElement !== els["state-input"]) {
        els["state-input"].value = isObj ? JSON.stringify(draft.state, null, 2) : JSON.stringify(draft.state);
      }
    } else {
      els["state-input"].classList.remove("json");
      if (document.activeElement !== els["state-input"]) {
        els["state-input"].value = typeof draft.state === "string"
          ? draft.state
          : JSON.stringify(draft.state, null, 2);
      }
    }
    els["state-count"].textContent = `${JSON.stringify(draft.state || "").length} chars`;
    els["state-input"].oninput = () => {
      const text = els["state-input"].value;
      if (state.stateView === "json") {
        try {
          draft.state = JSON.parse(text);
          els["state-input"].classList.remove("bad");
        } catch { els["state-input"].classList.add("bad"); }
      } else {
        draft.state = text;
      }
      renderSummary();
      scheduleAuto();
    };
    els["state-input"].onblur = () => renderEditor();
    return;
  }

  // batch
  if (state.batchView === "json") {
    els["batch-list-view"].hidden = true;
    els["batch-json-view"].hidden = false;
    if (document.activeElement !== els["states-json"]) {
      els["states-json"].value = statesToJson(draft.states);
      els["batch-json-error"].hidden = true;
    }
    els["states-json"].oninput = () => {
      try {
        draft.states = parseStatesJson(els["states-json"].value);
        els["batch-json-error"].hidden = true;
        renderSummary();
        scheduleAuto();
      } catch (err) {
        els["batch-json-error"].textContent = String(err.message || err);
        els["batch-json-error"].hidden = false;
      }
    };
    els["states-json"].onblur = () => renderEditor();
  } else {
    els["batch-json-view"].hidden = true;
    els["batch-list-view"].hidden = false;
    renderContextList(draft);
  }
  renderSummary();
}

function renderContextList(draft) {
  const wrap = els["context-list"];
  wrap.innerHTML = "";
  draft.states.forEach((item, i) => {
    const row = document.createElement("div");
    row.className = "context-row";

    const id = document.createElement("input");
    id.className = "ctx-id";
    id.value = item.id;
    id.setAttribute("aria-label", "Context id");
    id.addEventListener("change", () => {
      draft.states[i].id = id.value.trim() || item.id;
      renderContextList(draft);
    });

    const ta = document.createElement("textarea");
    ta.value = item.state;
    ta.rows = 2;
    ta.placeholder = "Context text…";
    ta.addEventListener("input", () => {
      draft.states[i].state = ta.value;
      scheduleAuto();
    });

    const rm = document.createElement("button");
    rm.className = "ctx-remove";
    rm.title = "Remove context";
    rm.textContent = "×";
    rm.addEventListener("click", () => {
      draft.states.splice(i, 1);
      renderEditor();
      scheduleAuto();
    });

    row.append(id, ta, rm);
    wrap.appendChild(row);
  });
}

function renderQuestions() {
  const draft = draftFor();
  els.questions.innerHTML = "";
  if (!draft) return;
  const ids = questionOrder(draft.questions);
  els["question-count"].textContent = String(ids.length);
  for (const qid of ids) {
    els.questions.appendChild(questionCard(qid, draft));
  }
}

function questionCard(qid, draft) {
  const q = draft.questions[qid];
  const card = document.createElement("div");
  card.className = "question-card";

  const head = document.createElement("div");
  head.className = "question-head";
  const badge = document.createElement("span");
  badge.className = `type-badge ${q.type}`;
  badge.textContent = q.type === "choice" ? "C" : q.type === "noul" ? "N" : "S";
  const idInput = document.createElement("input");
  idInput.className = "qid";
  idInput.value = qid;
  idInput.setAttribute("aria-label", "Question id");
  idInput.addEventListener("change", () => {
    const next = idInput.value.trim() || qid;
    if (next === qid) return;
    // Re-key preserving order.
    const entries = Object.entries(draft.questions).map(([k, v]) => (k === qid ? [next, v] : [k, v]));
    draft.questions = Object.fromEntries(entries);
    renderQuestions();
    scheduleAuto();
  });
  const rm = document.createElement("button");
  rm.className = "q-remove";
  rm.title = "Remove question";
  rm.textContent = "×";
  rm.addEventListener("click", () => {
    delete draft.questions[qid];
    renderQuestions();
    renderSummary();
    scheduleAuto();
  });
  head.append(badge, idInput, rm);

  const ins = document.createElement("textarea");
  ins.className = "instructions";
  ins.value = q.instructions || "";
  ins.placeholder = "The text the model should answer…";
  ins.addEventListener("input", () => {
    draft.questions[qid].instructions = ins.value;
    scheduleAuto();
  });

  card.append(head, ins);

  if (q.type === "choice" || q.type === "score") {
    const list = document.createElement("div");
    list.className = "criteria-list";
    const entries = q.type === "choice"
      ? Object.entries(q.criteria || {})
      : (q.criteria || []).map((v, i) => [String(i), v]);
    entries.forEach(([key, val], idx) => {
      list.appendChild(criterionRow(qid, draft, q, key, val, idx));
    });
    const add = document.createElement("button");
    add.className = "criteria-add";
    add.textContent = "＋ Add option";
    add.addEventListener("click", () => {
      if (q.type === "choice") {
        let key = "option";
        let n = Object.keys(q.criteria).length + 1;
        while (key + n in q.criteria) n += 1;
        q.criteria[key + n] = "";
      } else {
        q.criteria.push("");
      }
      renderQuestions();
      scheduleAuto();
    });
    card.append(list, add);
  } else if (q.type === "noul" && q.criteria) {
    const list = document.createElement("div");
    list.className = "criteria-list";
    for (const key of ["true", "false"]) {
      if (key in q.criteria) list.appendChild(criterionRow(qid, draft, q, key, q.criteria[key]));
    }
    card.append(list);
  }
  return card;
}

function criterionRow(qid, draft, q, key, val, idx) {
  const row = document.createElement("div");
  row.className = "criteria-row";

  if (q.type === "choice") {
    const keyInput = document.createElement("input");
    keyInput.className = "c-key";
    keyInput.value = key;
    keyInput.setAttribute("aria-label", "Option label");
    keyInput.addEventListener("change", () => {
      const next = keyInput.value.trim() || key;
      if (next === key) return;
      const entries = Object.entries(q.criteria).map(([k, v]) => (k === key ? [next, v] : [k, v]));
      q.criteria = Object.fromEntries(entries);
      renderQuestions();
      scheduleAuto();
    });
    row.appendChild(keyInput);
  } else {
    const idxLabel = document.createElement("span");
    idxLabel.className = "c-key";
    idxLabel.style.borderStyle = "solid";
    idxLabel.style.background = "var(--panel-soft)";
    idxLabel.textContent = String(idx ?? key);
    row.appendChild(idxLabel);
  }

  const valInput = document.createElement("textarea");
  valInput.className = "c-val";
  valInput.rows = 1;
  valInput.value = val ?? "";
  valInput.placeholder = "Describe this option for the model…";
  valInput.addEventListener("input", () => {
    if (q.type === "choice") q.criteria[key] = valInput.value;
    else if (q.type === "score") q.criteria[idx] = valInput.value;
    else q.criteria[key] = valInput.value;
    scheduleAuto();
  });

  const rm = document.createElement("button");
  rm.className = "c-remove";
  rm.title = "Remove option";
  rm.textContent = "×";
  rm.addEventListener("click", () => {
    if (q.type === "choice") delete q.criteria[key];
    else if (q.type === "score") q.criteria.splice(idx, 1);
    else delete q.criteria[key];
    if (q.type === "score" && (!q.criteria || q.criteria.length === 0)) q.criteria = [""];
    renderQuestions();
    scheduleAuto();
  });

  row.append(valInput, rm);
  return row;
}

/* ---------------------------- model card ----------------------------- */

function renderModelCard() {
  const model = state.models.find((m) => m.id === state.model);
  els["model-name"].textContent = model ? model.label : "Laya";
  els["model-version"].textContent = model && model.repo_id ? model.repo_id : "";
  const spec = els["model-spec"];
  spec.innerHTML = "";
  const rows = [];
  if (model) {
    if (model.encoder) rows.push(["Encoder", model.encoder]);
    if (model.params) rows.push(["Params", model.params]);
    if (model.context) rows.push(["Context", model.context]);
    if (state.lazy) rows.push(["Loading", "lazy (on demand)"]);
  }
  for (const [k, v] of rows) {
    const line = document.createElement("div");
    line.className = "spec-line";
    const b = document.createElement("b");
    b.textContent = k;
    const s = document.createElement("span");
    s.textContent = String(v);
    line.append(b, s);
    spec.appendChild(line);
  }
  const ready = state.ready === true;
  const cardState = state.ready === null ? "checking" : ready ? "ready" : "down";
  els["model-card"].dataset.state = cardState;
  els["model-card"].setAttribute("aria-busy", String(cardState === "checking"));
  els["model-state"].innerHTML = cardState === "ready"
    ? "<i></i>Ready"
    : cardState === "down"
      ? "<i></i>" + (state.lazy ? "Loads on first run" : "Checkpoints loading…")
      : "<i></i>Checking model";
  els["model-trigger"].disabled = state.models.length === 0;
}

function openModelPopover() {
  const pop = els["model-popover"];
  pop.hidden = false;
  const rect = els["model-trigger"].getBoundingClientRect();
  pop.style.left = Math.min(window.innerWidth - 340, rect.left) + "px";
  pop.style.top = Math.min(window.innerHeight - pop.offsetHeight - 20, rect.bottom + 8) + "px";
  renderModelOptions(els["model-options"], state.models, state.loaded, state.model, (id) => {
    if (id !== state.model) {
      state.model = id;
      // Switching clears old results and shows the actual prompt sent to that model.
      state.results[state.mode] = null;
      state.timings[state.mode] = null;
      state.openRow = null;
      state.drafts[state.mode] = exampleDraft(
        state.examples[state.mode][state.current[state.mode]], state.model);
      renderAll();
      closeModelPopover();
      scheduleAuto();
    } else {
      closeModelPopover();
    }
  });
}

function closeModelPopover() {
  els["model-popover"].hidden = true;
}

/* ------------------------------- run --------------------------------- */

function renderRunState() {
  const canRun = state.ready === true || state.lazy;
  for (const btn of [els.run, els["run-top"]]) btn.disabled = !canRun || state.running;
  els["offline-note"].hidden = state.ready === true || state.ready === null;
}

function payloadFor() {
  const draft = draftFor();
  if (!draft) return null;
  if (MODES[state.mode].batch) {
    if (!draft.states.length) {
      toast("Add at least one context");
      return null;
    }
    return buildBatch(draft.states, draft.questions, state.model);
  }
  if (draft.state === "" || draft.state == null) {
    toast("Add a prompt or context first");
    return null;
  }
  return buildSingle(draft.state, draft.questions, state.model);
}

async function run() {
  if (state.running) return;
  const payload = payloadFor();
  if (!payload) return;
  const endpoint = MODES[state.mode].batch ? "/v1/systemone/batches" : "/v1/systemone";
  state.running = true;
  els["run-status"].dataset.state = "running";
  els["run-status"].innerHTML = "<span class=\"status-led\"></span>Running decisions…";
  renderRunState();
  const t0 = performance.now();
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { detail = (await res.json()).detail || detail; } catch { /* keep */ }
      throw new Error(detail);
    }
    const body = await res.json();
    state.results[state.mode] = body;
    const inferMs = Number(res.headers.get("X-Inference-Time-Ms")) || (performance.now() - t0);
    state.timings[state.mode] = { inferMs };
    if (state.mode === "batch") state.openRow = null;
    renderResults();
    els["run-status"].dataset.state = "done";
    els["run-status"].innerHTML = "<span class=\"status-led\"></span>Done";
  } catch (err) {
    console.error(err);
    toast(String(err.message || err), true);
    els["run-status"].dataset.state = "error";
    els["run-status"].innerHTML = "<span class=\"status-led\"></span>" + String(err.message || err);
  } finally {
    state.running = false;
    renderRunState();
  }
}

function scheduleAuto() {
  if (!state.autorun) return;
  if (autoTimer) clearTimeout(autoTimer);
  autoTimer = setTimeout(() => {
    autoTimer = null;
    if (state.autorun && !state.running) run();
  }, 900);
}

/* ----------------------------- results ------------------------------- */

function renderResults() {
  const result = state.results[state.mode];
  const draft = draftFor();
  const wrap = els.results;
  wrap.innerHTML = "";
  const count = els["result-count"];

  if (!result) {
    count.textContent = "—";
    wrap.appendChild(emptyResults());
    els.timing.hidden = true;
    return;
  }

  if (state.mode === "batch") {
    count.textContent = `${result.count} × ${questionOrder(draft.questions).length}`;
    wrap.appendChild(batchMatrix(result, draft));
    renderTiming(result);
    return;
  }

  const answers = result.answers || {};
  const ids = questionOrder(draft.questions);
  count.textContent = String(ids.length);
  const list = document.createElement("div");
  list.className = "answer-list";
  for (const qid of ids) {
    list.appendChild(answerCard(qid, answers[qid], result));
  }
  wrap.appendChild(list);
  renderTiming(result);
}

function renderTiming(result) {
  const t = state.timings[state.mode];
  const usage = result.usage || {};
  const bits = [];
  if (t && t.inferMs) bits.push(`inference ${t.inferMs.toFixed(1)} ms`);
  if (usage.input_tokens !== undefined) bits.push(`in ${usage.input_tokens} tok`);
  if (usage.output_tokens !== undefined) bits.push(`out ${usage.output_tokens} tok`);
  if (result.routing && result.routing.model) bits.push(`checkpoint ${result.routing.model}`);
  else if (result.model) bits.push(`checkpoint ${result.model}`);
  els.timing.hidden = bits.length === 0;
  els.timing.textContent = bits.join(" · ");
}

function emptyResults() {
  const div = document.createElement("div");
  div.className = "empty-results";
  div.innerHTML = `
    <span class="empty-symbol" aria-hidden="true">✧</span>
    <h4>Context becomes<br><em>perspective.</em></h4>
    <p>Run your decisions. Compare every answer<br>and the alternatives behind it.</p>
    <div class="empty-types"><span><b>C</b> Choose</span><span><b>N</b> Yes / no</span><span><b>S</b> Score</span></div>`;
  return div;
}

function answerCard(qid, answer, result, routing) {
  const card = document.createElement("div");
  card.className = "answer-card";
  if (!answer) {
    card.innerHTML = `<div class="answer-head"><span class="a-qid">${escapeHtml(qid)}</span><span class="a-value">—</span></div>`;
    return card;
  }
  const conf = Number(answer.answer_confidence);
  const head = document.createElement("div");
  head.className = "answer-head";
  const badge = document.createElement("span");
  badge.className = `type-badge ${answer.type}`;
  badge.textContent = answer.type === "choice" ? "C" : answer.type === "noul" ? "N" : "S";
  const qidEl = document.createElement("span");
  qidEl.className = "a-qid";
  qidEl.textContent = qid;
  const value = document.createElement("span");
  value.className = "a-value";
  if (answer.type === "choice") value.textContent = String(answer.choice);
  else if (answer.type === "score") value.textContent = `${fmt(answer.score)} / ${(answer.legend ? Object.keys(answer.legend).length - 1 : "n")}`;
  else value.textContent = `yes @ ${fmt(answer.noul)}`;
  const confPill = document.createElement("span");
  confPill.className = "conf" + (Number.isFinite(conf) && conf < 0.6 ? " low" : "");
  confPill.title = "answer_confidence — the probability of the reported answer";
  confPill.textContent = `conf ${fmt(answer.answer_confidence)}`;
  head.append(badge, qidEl, value, confPill);
  card.appendChild(head);

  if (routing && routing.reason) {
    const r = document.createElement("div");
    r.className = "answer-routing";
    r.innerHTML = `<b>${escapeHtml(routing.model || result.model || "auto")}</b> · ${escapeHtml(routing.reason)}`;
    card.appendChild(r);
  } else if (result && result.routing && result.routing.reason) {
    const r = document.createElement("div");
    r.className = "answer-routing";
    r.innerHTML = `<b>${escapeHtml(result.routing.model || result.model || "auto")}</b> · ${escapeHtml(result.routing.reason)}`;
    card.appendChild(r);
  }

  const rows = optionRows(answer);
  if (rows.length) {
    const list = document.createElement("div");
    list.className = "prob-list";
    const best = rows.length ? Math.max(...rows.map(([, p]) => p)) : 0;
    for (const [label, p] of rows) {
      const row = document.createElement("div");
      row.className = "prob-row" + (p >= best - 1e-9 ? " top" : "");
      const lab = document.createElement("span");
      lab.className = "p-label";
      lab.title = label;
      lab.textContent = label;
      const track = document.createElement("span");
      track.className = "p-track";
      const fill = document.createElement("span");
      fill.className = "p-fill";
      fill.style.width = `${Math.max(2, Math.min(100, p * 100))}%`;
      track.appendChild(fill);
      const val = document.createElement("span");
      val.className = "p-val";
      val.textContent = fmt(p);
      row.append(lab, track, val);
      list.appendChild(row);
    }
    card.appendChild(list);
  }
  if (answer.type === "score") {
    const note = document.createElement("p");
    note.className = "score-note";
    note.textContent = `Expected score over ${Object.keys(answer.probabilities || {}).length} levels (legend index 0 first).`;
    card.appendChild(note);
  }
  return card;
}

function batchMatrix(result, draft) {
  const qids = questionOrder(draft.questions);
  const wrap = document.createElement("div");
  const table = document.createElement("div");
  table.className = "matrix-wrap";
  const tbl = document.createElement("table");
  tbl.className = "matrix";
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const label of ["Id", ...qids, ...(state.model === "auto" ? ["Route"] : [])]) {
    const th = document.createElement("th");
    th.textContent = label;
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  tbl.appendChild(thead);
  const tbody = document.createElement("tbody");
  for (const row of result.results || []) {
    const tr = document.createElement("tr");
    tr.dataset.id = row.id;
    if (state.openRow === row.id) tr.classList.add("open");
    const idCell = document.createElement("td");
    idCell.className = "m-id";
    idCell.textContent = row.id;
    tr.appendChild(idCell);
    for (const qid of qids) {
      const td = document.createElement("td");
      const [label, sub] = topAnswer((row.answers || {})[qid]);
      const top = document.createElement("span");
      top.className = "cell-top";
      top.textContent = label;
      td.appendChild(top);
      if (sub) {
        const s = document.createElement("span");
        s.className = "cell-sub";
        s.textContent = sub;
        td.appendChild(s);
      }
      tr.appendChild(td);
    }
    if (state.model === "auto") {
      const td = document.createElement("td");
      td.className = "m-route";
      td.textContent = (row.routing && row.routing.model) || row.model || "";
      tr.appendChild(td);
    }
    tr.addEventListener("click", () => {
      state.openRow = state.openRow === row.id ? null : row.id;
      renderResults();
    });
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);
  table.appendChild(tbl);

  wrap.appendChild(table);
  const foot = document.createElement("p");
  foot.className = "matrix-foot";
  foot.textContent = "Open a row for the full distribution.";
  wrap.appendChild(foot);

  if (state.openRow) {
    const row = (result.results || []).find((r) => r.id === state.openRow);
    if (row) {
      const detail = document.createElement("div");
      detail.className = "batch-detail";
      const h = document.createElement("h4");
      h.textContent = row.id;
      detail.appendChild(h);
      if (row.routing && row.routing.reason) {
        const r = document.createElement("p");
        r.className = "d-route";
        r.textContent = `${row.routing.model || ""} · ${row.routing.reason}`;
        detail.appendChild(r);
      }
      const list = document.createElement("div");
      list.className = "answer-list";
      for (const qid of qids) {
        list.appendChild(answerCard(qid, (row.answers || {})[qid], row, row.routing));
      }
      detail.appendChild(list);
      wrap.appendChild(detail);
    }
  }
  return wrap;
}

/* ------------------------------ status ------------------------------- */

function applyStatus() {
  const conn = els.connection;
  if (state.ready === true) {
    conn.className = "connection ok";
    conn.innerHTML = "<i></i> Connected";
  } else if (state.ready === false) {
    conn.className = "connection err";
    conn.innerHTML = "<i></i> Loading checkpoints";
  } else {
    conn.className = "connection";
    conn.innerHTML = "<i></i> Checking connection";
  }
  renderModelCard();
  renderRunState();
}

function schedulePoll() {
  if (state.ready === true) return;
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch("/api/ready");
      const body = await res.json().catch(() => ({}));
      const wasReady = state.ready;
      state.ready = res.ok && body.status === "ready";
      if (body.models) state.loaded = body.models;
      applyStatus();
      if (state.ready && wasReady !== true) {
        clearInterval(pollTimer);
        toast("Checkpoints ready");
      }
    } catch {
      state.ready = false;
      applyStatus();
    }
  }, 6000);
}

/* ------------------------------ misc --------------------------------- */

function toast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.classList.toggle("error", isError);
  els.toast.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { els.toast.hidden = true; }, 3200);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

init();
