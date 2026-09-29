/* Payload building and light client-side validation for the Laya Decision Studio.
   The server (studio/contract.py) is the source of truth; this mirrors its shapes. */

export function clone(value) {
  return structuredClone(value);
}

const TEMPLATES = {
  choice: () => ({
    type: "choice",
    instructions: "Which option best describes this?",
    criteria: { option_a: "Describe the first option", option_b: "Describe the second option" },
  }),
  noul: () => ({
    type: "noul",
    instructions: "Is this true of the context? Ask a yes / no question.",
  }),
  score: () => ({
    type: "score",
    instructions: "Rate this on an ordered scale, index 0 first.",
    criteria: ["Low", "Medium", "High"],
  }),
};

export function emptyQuestion(type) {
  const make = TEMPLATES[type];
  if (!make) throw new Error("unknown question type " + type);
  return make();
}

/* ------------------------------ single ------------------------------ */

export function buildSingle(state, questions, model) {
  const payload = { state, questions };
  if (model && model !== "auto") payload.model = model;
  return payload;
}

/* ------------------------------- batch ------------------------------ */

export function buildBatch(states, questions, model) {
  const payload = {
    states: states.map((s) => ({ id: s.id, state: s.state })),
    questions,
  };
  if (model && model !== "auto") payload.model = model;
  return payload;
}

export function parseStatesJson(text) {
  let doc;
  try {
    doc = JSON.parse(text);
  } catch (err) {
    throw new Error("Not valid JSON: " + err.message);
  }
  if (!Array.isArray(doc) || doc.length === 0) {
    throw new Error("Paste a non-empty array of { id, state } objects.");
  }
  const seen = new Set();
  return doc.map((item, i) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new Error(`Item ${i}: must be an object with 'id' and 'state'.`);
    }
    if (typeof item.id !== "string" || !item.id) {
      throw new Error(`Item ${i}: 'id' must be a non-empty string.`);
    }
    if (item.state === undefined || item.state === null) {
      throw new Error(`Item ${item.id}: 'state' is required.`);
    }
    if (seen.has(item.id)) throw new Error(`Duplicate id ${item.id}; every state needs its own id.`);
    seen.add(item.id);
    return { id: item.id, state: item.state };
  });
}

export function statesToJson(states) {
  return JSON.stringify(
    states.map((s) => ({ id: s.id, state: s.state })),
    null,
    2,
  );
}

export function nextContextId(states) {
  let n = 1;
  const used = new Set(states.map((s) => s.id));
  while (used.has("T-" + String(n).padStart(3, "0"))) n += 1;
  return "T-" + String(n).padStart(3, "0");
}

/* ------------------------------ helpers ------------------------------ */

export function questionOrder(questions) {
  return Object.keys(questions || {});
}

export function topAnswer(answer) {
  // A compact one-line summary for a matrix cell: [label, sub].
  if (!answer) return ["—", ""];
  if (answer.type === "choice") {
    return [String(answer.choice), fmt(answer.probabilities?.[answer.choice])];
  }
  if (answer.type === "score") {
    return [String(answer.score), fmt(answer.answer_confidence)];
  }
  if (answer.type === "noul") {
    const p = Number(answer.noul);
    return [p >= 0.5 ? "yes" : "no", fmt(p)];
  }
  return ["—", ""];
}

export function fmt(x) {
  if (x === undefined || x === null) return "";
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(3) : String(x);
}

export function optionRows(answer) {
  // Ordered [label, probability] rows for one answer, best first.
  if (!answer) return [];
  if (answer.type === "noul") {
    const p = Number(answer.noul) || 0;
    return [
      ["yes", p],
      ["no", 1 - p],
    ];
  }
  const probs = answer.probabilities || {};
  if (answer.type === "score" && answer.legend) {
    const rows = Object.keys(probs).map((k) => [`${k} · ${answer.legend[k] ?? ""}`, Number(probs[k])]);
    rows.sort((a, b) => b[1] - a[1]);
    return rows;
  }
  const rows = Object.entries(probs).map(([label, p]) => [label, Number(p)]);
  rows.sort((a, b) => b[1] - a[1]);
  return rows;
}
