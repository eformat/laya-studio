# Laya Decision Studio

A [Decision Studio](https://huggingface.co/spaces/llm-semantic-router/decision-studio)-style workbench for [laya](https://github.com/NandhaKishorM/laya), running entirely in one local process: a FastAPI app with laya's `Router` embedded, and a static vanilla-JS workbench — no framework, no build step.

**One prompt. Several decisions.** Explore laya's three checkpoints — `english`, `multilingual`, `typed-decisions` — or let the Router pick. Ask several related questions about one case, apply shared questions to a batch of contexts, or route a prompt before calling an LLM. Every answer shows its full probability distribution, the calibrated `answer_confidence`, and (under Auto) the Router's routing reason.

The question schema is laya's own — `choice` / `score` / `noul` — which is wire-compatible with the Decision Studio's example format, so the studio's curated scenarios port unchanged.

## Run it

```bash
make serve          # http://127.0.0.1:7860 (creates .venv on first use)

python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # laya[serve], pytest, httpx
.venv/bin/python -m studio.app                        # http://127.0.0.1:7860
```

`make test` runs the API tests; `make clean` removes caches. Port and host are overridable: `make serve PORT=8000 HOST=0.0.0.0 EXTRA=--no-preload`.

The first run downloads the laya checkpoints (~2–3 GB total). Checkpoints load up front by default; the studio shows an offline note until they are resident.

```bash
.venv/bin/python -m studio.app --no-preload      # load lazily, on first run
.venv/bin/python -m studio.app --device cuda     # or cpu / mps
.venv/bin/python -m studio.app --port 8000
```

Environment variables follow `laya-serve`: `LAYA_DEVICE`, `LAYA_PRELOAD`, `LAYA_MODELS` (comma list), `LAYA_THREADS`, `LAYA_AUTO_TASK`, `LAYA_MAX_LOADED`.

## Host it in Kubernetes

Build the image (checkpoints baked in, ~3–4 GB: pods start with no Hugging Face egress and no cache volume), push to Quay, and install the chart:

```bash
make image                  # podman build -t quay.io/eformat/laya-decision-studio:latest
make push                   # podman push
helm install laya-studio chart/ -n laya-studio --create-namespace
```

The chart ships a Deployment, Service, ServiceAccount, and an OpenShift Route (TLS edge). Readiness is `/api/ready` (503 until every checkpoint is resident); liveness is `/api/status`.

**GPU / MIG slice:** enable the GPU toggle and pick the resource name — `nvidia.com/gpu` or an MIG slice (`nvidia.com/mig-1g.18gb`, `nvidia.com/mig-2g.35gb`, `nvidia.com/mig-3g.71gb`):

```bash
helm install laya-studio chart/ -n laya-studio --create-namespace \
  --set studio.gpu.enabled=true \
  --set studio.gpu.resourceName=nvidia.com/mig-1g.18gb
```

`LAYA_DEVICE` defaults to `cuda` with the GPU on, `cpu` without; anything set in `studio.env` wins (`--set "studio.env.LAYA_MODELS=english\,multilingual"`).

Other knobs: `studio.replicas`, `studio.resources` (three resident checkpoints take ~2–3 GiB of RAM), `studio.route.host`, `studio.route.enabled=false` for plain Kubernetes (then front the Service with your own Ingress).

## The workbench

- **LLM routing** — route a prompt to the right downstream LLM (fast / reasoning / code / vision / multilingual) before calling it, with complexity, latency priority, jailbreak check, fact-check, and tool decisions.
- **One context** — inspect several aspects of one case (a ticket, a review, a policy case) in a single forward pass.
- **Batch contexts** — apply shared questions to a list of `{id, state}` contexts in one batched request; each row shows its top answer and (under Auto) the checkpoint the Router picked per row.

Shared question sets are editable: add a question (`C`hoice / `N`oul / `S`core), rename ids, edit instructions and criteria. Drafts are per mode; `Ctrl/Cmd+Enter` runs; auto-run sends edits after a pause.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/examples` | the curated scenario catalog |
| `GET /v1/models` | model catalog + request limits |
| `GET /api/status` | per-checkpoint loaded state, device, revisions |
| `GET /api/ready` | 200/503 — ready when every configured checkpoint is resident |
| `POST /v1/systemone` | one decision: `{"model"?, "state", "questions", "max_len"?, "head_max_len"?}` |
| `POST /v1/systemone/batches` | batch: `{"model"?, "states": [{"id", "state"}], "questions"}` (up to 1,024 states) |
| `POST /api/evaluate` | loose endpoint: accepts either shape, dispatches on `states` |
| `GET /docs` under `/api/docs` | OpenAPI |

```bash
curl -s localhost:7860/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"body": "We were billed twice for March. Please refund it today or we will cancel."},
  "questions": {
    "department": {"type": "choice", "instructions": "Which department?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"}
  }
}' | python -m json.tool
```

Responses are laya's own result shape: `answers` (per question: the typed answer, every candidate probability, entropy `confidence`, calibrated `answer_confidence`), `usage`, and `routing` (`model`, `repo`, `reason`).

## Model semantics

- `"auto"` (default) — the Router detects script/language and picks `english` or `multilingual`; every answer carries the routing `reason`.
- `english` / `multilingual` / `typed-decisions` (or any laya alias: `en`, `ml`, `typed`…) — pinned; `max_len` / `head_max_len` budgets pass through.
- **Fail closed:** an unknown model answers 422 — it never silently falls back to another checkpoint.

## Layout

```
studio/
├── app.py          # FastAPI app; Router embedded; static mount
├── contract.py     # strict request contract; laya-serve limit guards
├── registry.py     # model catalog (auto + 3 checkpoints)
├── examples.json   # curated scenarios (adapted from the Decision Studio)
├── static/         # index.html, app.js, contract.js, model-menu.js, styles.css
└── tests/          # API + contract tests against a stubbed Router
```

## Tests

```bash
.venv/bin/python -m pytest studio/tests/ -q
```

The tests stub the Router, so they need no checkpoint download and no GPU.

## Credits

Workbench inspired by the [llm-semantic-router Decision Studio](https://huggingface.co/spaces/llm-semantic-router/decision-studio) (MIT); example scenarios adapted from its curated catalog. Decisions run on [laya](https://github.com/NandhaKishorM/laya) (Apache-2.0). This repo contains no model weights and calls no external API.
