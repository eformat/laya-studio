"""Tests for the Laya Decision Studio API, against a stubbed Router.

No checkpoint is downloaded and no forward pass runs: the stub answers
deterministically and records what the app passed to it, so these tests pin
the HTTP contract (shapes, limits, fail-closed model resolution, batch
alignment) rather than laya's inference.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from studio.app import create_app

QUESTIONS = {
    "destination": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments", "technical": "bugs", "other": "everything else"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["not urgent", "soon", "blocking"],
    },
    "churn": {"type": "noul", "instructions": "Does the user threaten to leave?"},
}

SINGLE_REQUEST = {"state": "Please refund the duplicate charge.", "questions": QUESTIONS}


class FakeAgent:
    """Just enough Agent for the device report: a plain-string device."""

    device = "cpu"


class StubRouter:
    """A Router that answers deterministically without any checkpoint."""

    def __init__(self, loaded=("english", "multilingual", "typed-decisions")):
        self._loaded = list(loaded)
        self._agents = {name: FakeAgent() for name in self._loaded}
        self.predict_calls = []
        self.batch_calls = []

    @property
    def loaded(self):
        return list(self._loaded)

    def preload(self, names=None):
        self._loaded = list(names) if names is not None else [
            "english", "multilingual", "typed-decisions"]
        self._agents = {name: FakeAgent() for name in self._loaded}

    def predict(self, state, questions, model=None, **kwargs):
        self.predict_calls.append(
            {"state": state, "questions": questions, "model": model, **kwargs})
        return {
            "model": model or "english",
            "answers": answers_for(questions),
            "usage": {"input_tokens": 42, "output_tokens": 0},
            "routing": {"model": model or "english",
                        "repo": "convaiinnovations/laya", "reason": "Latin script; test route"},
        }

    def predict_batch(self, requests, **kwargs):
        self.batch_calls.append(requests)
        return [self.predict(r["state"], r["questions"], model=r.get("model"), **{})
                for r in requests]


def answers_for(questions):
    answers = {}
    for qid, q in questions.items():
        t = q.get("type")
        if t == "choice":
            labels = list(q["criteria"])
            n = len(labels)
            answers[qid] = {
                "type": "choice",
                "choice": labels[0],
                "probabilities": {label: round(1.0 / n, 4) for label in labels},
                "confidence": 0.05,
                "answer_confidence": round(1.0 / n, 4),
                "action": {"act_probability": 0.9},
            }
        elif t == "score":
            n = len(q["criteria"])
            answers[qid] = {
                "type": "score",
                "score": round((n - 1) / 2, 4),
                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                "probabilities": {str(i): round(1.0 / n, 4) for i in range(n)},
                "confidence": 0.05,
                "answer_confidence": round(1.0 / n, 4),
                "action": {"act_probability": 0.9},
            }
        else:
            answers[qid] = {
                "type": "noul", "noul": 0.8, "confidence": 0.8,
                "answer_confidence": 0.8, "action": {"act_probability": 0.9},
            }
    return answers


@pytest.fixture()
def stub():
    return StubRouter()


@pytest.fixture()
def client(stub, monkeypatch):
    monkeypatch.setenv("LAYA_PRELOAD", "1")
    monkeypatch.delenv("LAYA_MODELS", raising=False)
    monkeypatch.delenv("LAYA_DEVICE", raising=False)
    return TestClient(create_app(router=stub))


# ------------------------------- catalog -------------------------------

def test_examples_served(client):
    res = client.get("/api/examples")
    assert res.status_code == 200
    examples = res.json()
    assert len(examples) >= 10
    surfaces = {ex["surface"] for ex in examples}
    assert {"routing", "single", "batch"} <= surfaces
    for ex in examples:
        assert {"id", "title", "state" in ex or "states" in ex, "questions"}
        # Every question is laya-shaped; the batch shape needs {id, state}.
        for q in ex["questions"].values():
            assert q["type"] in ("choice", "score", "noul")


def test_models_catalog(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    body = res.json()
    assert body["default"] == "auto"
    ids = [m["id"] for m in body["models"]]
    assert ids[0] == "auto"
    assert {"english", "multilingual", "typed-decisions"} <= set(ids)


def test_status_shape(client, stub):
    res = client.get("/api/status")
    assert res.status_code == 200
    body = res.json()
    assert body["backend"] == "native"
    assert body["ready"] is True
    assert body["loaded"]["english"] is True
    assert body["checkpoint_devices"]["english"] == "cpu"


def test_ready_200(client):
    res = client.get("/api/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"


def test_ready_503_while_loading(monkeypatch):
    monkeypatch.setenv("LAYA_PRELOAD", "1")
    client = TestClient(create_app(router=StubRouter(loaded=())))
    res = client.get("/api/ready")
    assert res.status_code == 503
    assert res.json()["status"] == "unavailable"


# ------------------------------- single --------------------------------

def test_single_happy_path(client, stub):
    res = client.post("/v1/systemone", json=SINGLE_REQUEST)
    assert res.status_code == 200
    body = res.json()
    assert body["model"] == "english"
    assert body["answers"]["destination"]["choice"] == "billing"
    assert "probabilities" in body["answers"]["destination"]
    assert body["routing"]["reason"]
    assert res.headers.get("X-Inference-Time-Ms")
    # auto -> no explicit model reaches the Router
    assert stub.predict_calls[0]["model"] is None


def test_single_pinned_model(client, stub):
    res = client.post("/v1/systemone", json=dict(SINGLE_REQUEST, model="ml"))
    assert res.status_code == 200
    assert res.json()["model"] == "multilingual"
    assert stub.predict_calls[0]["model"] == "multilingual"


def test_single_auto_keyword(client, stub):
    res = client.post("/v1/systemone", json=dict(SINGLE_REQUEST, model="auto"))
    assert res.status_code == 200
    assert stub.predict_calls[0]["model"] is None


def test_unknown_model_422_fail_closed(client, stub):
    res = client.post("/v1/systemone", json=dict(SINGLE_REQUEST, model="decision-lux"))
    assert res.status_code == 422
    assert "not available" in res.json()["detail"]
    assert stub.predict_calls == []


def test_missing_state_400(client):
    res = client.post("/v1/systemone", json={"questions": QUESTIONS})
    assert res.status_code == 400


def test_state_null_400(client):
    res = client.post("/v1/systemone", json={"state": None, "questions": QUESTIONS})
    assert res.status_code == 400


def test_missing_questions_422(client):
    res = client.post("/v1/systemone", json={"state": "hello"})
    assert res.status_code == 422


def test_unknown_field_422(client):
    res = client.post("/v1/systemone", json=dict(SINGLE_REQUEST, extra=1))
    assert res.status_code == 422


def test_unknown_question_type_422(client):
    bad = {"state": "x", "questions": {"q": {"type": "vector", "instructions": "?"}}}
    res = client.post("/v1/systemone", json=bad)
    assert res.status_code == 422


def test_bad_json_422(client):
    res = client.post("/v1/systemone", content=b"{not json",
                      headers={"content-type": "application/json"})
    assert res.status_code == 422


def test_duplicate_json_keys_422(client):
    raw = json.dumps(SINGLE_REQUEST)[:-1] + ', "state": "second"}'
    res = client.post("/v1/systemone", content=raw.encode(),
                      headers={"content-type": "application/json"})
    assert res.status_code == 422
    assert "Duplicate" in res.json()["detail"]


def test_nonfinite_json_422(client):
    raw = '{"state": "x", "questions": {}, "max_len": NaN}'
    res = client.post("/v1/systemone", content=raw.encode(),
                      headers={"content-type": "application/json"})
    assert res.status_code == 422


def test_content_type_enforced(client):
    res = client.post("/v1/systemone", content=b"{}",
                      headers={"content-type": "text/plain"})
    assert res.status_code == 415


def test_too_many_questions_413(client):
    questions = {f"q{i}": {"type": "noul", "instructions": "yes?"} for i in range(65)}
    res = client.post("/v1/systemone", json={"state": "x", "questions": questions})
    assert res.status_code == 413


def test_too_many_choice_options_413(client):
    questions = {"q": {"type": "choice", "instructions": "pick",
                       "criteria": {f"o{i}": "d" for i in range(101)}}}
    res = client.post("/v1/systemone", json={"state": "x", "questions": questions})
    assert res.status_code == 413


def test_state_too_large_413(client):
    res = client.post("/v1/systemone", json={"state": "x" * 50001, "questions": QUESTIONS})
    assert res.status_code == 413


def test_budget_params_forwarded(client, stub):
    res = client.post("/v1/systemone",
                      json=dict(SINGLE_REQUEST, max_len=512, head_max_len=256))
    assert res.status_code == 200
    call = stub.predict_calls[0]
    assert call["max_len"] == 512
    assert call["head_max_len"] == 256


def test_bad_budget_422(client):
    res = client.post("/v1/systemone", json=dict(SINGLE_REQUEST, max_len=-1))
    assert res.status_code == 422


# -------------------------------- batch --------------------------------

BATCH_REQUEST = {
    "states": [
        {"id": "T-1", "state": "refund please"},
        {"id": "T-2", "state": "the app crashes"},
    ],
    "questions": QUESTIONS,
}


def test_batch_happy_path(client, stub):
    res = client.post("/v1/systemone/batches", json=BATCH_REQUEST)
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 2
    assert [r["id"] for r in body["results"]] == ["T-1", "T-2"]
    assert body["usage"]["input_tokens"] == 84
    # Every request shares the questions and carries none of its own model pin.
    assert len(stub.batch_calls) == 1
    sent = stub.batch_calls[0]
    assert len(sent) == 2
    for req in sent:
        assert req["questions"] == QUESTIONS
        assert "model" not in req


def test_batch_pinned_model_reaches_every_request(client, stub):
    res = client.post("/v1/systemone/batches", json=dict(BATCH_REQUEST, model="english"))
    assert res.status_code == 200
    assert all(r["model"] == "english" for r in stub.batch_calls[0])


def test_batch_state_missing_id_422(client):
    bad = {"states": [{"state": "no id"}], "questions": QUESTIONS}
    res = client.post("/v1/systemone/batches", json=bad)
    assert res.status_code == 422


def test_batch_duplicate_ids_422(client):
    bad = {"states": [{"id": "A", "state": "x"}, {"id": "A", "state": "y"}],
           "questions": QUESTIONS}
    res = client.post("/v1/systemone/batches", json=bad)
    assert res.status_code == 422
    assert "duplicate" in res.json()["detail"].lower()


def test_batch_empty_422(client):
    res = client.post("/v1/systemone/batches", json={"states": [], "questions": QUESTIONS})
    assert res.status_code == 422


def test_batch_too_many_states_413(client):
    states = [{"id": f"S-{i}", "state": "x"} for i in range(1025)]
    res = client.post("/v1/systemone/batches", json={"states": states, "questions": QUESTIONS})
    assert res.status_code == 413


# ------------------------------ evaluate -------------------------------

def test_evaluate_dispatches_single(client, stub):
    res = client.post("/api/evaluate", json=SINGLE_REQUEST)
    assert res.status_code == 200
    assert "answers" in res.json()


def test_evaluate_dispatches_batch(client, stub):
    res = client.post("/api/evaluate", json=BATCH_REQUEST)
    assert res.status_code == 200
    assert "results" in res.json()


# ------------------------- static + security ---------------------------

def test_static_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Laya Decision Studio" in res.text


def test_security_headers(client):
    res = client.get("/api/examples")
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert res.headers["Referrer-Policy"] == "no-referrer"
    assert res.headers["Cache-Control"] == "no-store"


# ------------------------------- metrics -------------------------------

def _sample(name, **labels):
    value = REGISTRY.get_sample_value(name, labels)
    return value or 0.0


def test_metrics_endpoint_served(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    assert b"laya_studio_http_requests_total" in res.content


def test_metrics_disabled_404(monkeypatch, stub):
    monkeypatch.setenv("LAYA_PRELOAD", "1")
    monkeypatch.setenv("LAYA_METRICS", "0")
    client = TestClient(create_app(router=stub))
    assert client.get("/metrics").status_code == 404


def test_metrics_single_request(client, stub):
    name = "laya_studio_http_requests_total"
    requests_before = _sample(name, method="POST", endpoint="/v1/systemone", status="200")
    tokens_before = _sample("laya_studio_inference_input_tokens_total", checkpoint="english")
    out_before = _sample("laya_studio_inference_output_tokens_total", checkpoint="english")
    decisions_before = _sample("laya_studio_decisions_total", checkpoint="english")
    passes_before = _sample("laya_studio_inference_seconds_count", checkpoint="english")

    assert client.post("/v1/systemone", json=SINGLE_REQUEST).status_code == 200

    assert _sample(name, method="POST", endpoint="/v1/systemone", status="200") == requests_before + 1
    # The stub's usage: 42 input tokens, 0 output (no generation).
    assert _sample("laya_studio_inference_input_tokens_total", checkpoint="english") == tokens_before + 42
    assert _sample("laya_studio_inference_output_tokens_total", checkpoint="english") == out_before
    assert _sample("laya_studio_decisions_total", checkpoint="english") == decisions_before + 1
    assert _sample("laya_studio_inference_seconds_count", checkpoint="english") == passes_before + 1
    # The gate allows one pass at a time; after the response, none are running.
    assert _sample("laya_studio_inference_in_flight") == 0


def test_metrics_batch_tokens_summed(client, stub):
    decisions_before = _sample("laya_studio_decisions_total", checkpoint="english")
    tokens_before = _sample("laya_studio_inference_input_tokens_total", checkpoint="english")

    assert client.post("/v1/systemone/batches", json=BATCH_REQUEST).status_code == 200

    # One decision per result, tokens summed across the two states.
    assert _sample("laya_studio_decisions_total", checkpoint="english") == decisions_before + 2
    assert _sample("laya_studio_inference_input_tokens_total", checkpoint="english") == tokens_before + 84


def test_metrics_error_status_labelled(client):
    name = "laya_studio_http_requests_total"
    before = _sample(name, method="POST", endpoint="/v1/systemone", status="422")
    assert client.post("/v1/systemone", json={"state": "hello"}).status_code == 422
    assert _sample(name, method="POST", endpoint="/v1/systemone", status="422") == before + 1


def test_metrics_endpoint_labels(client):
    name = "laya_studio_http_requests_total"
    static_before = _sample(name, method="GET", endpoint="static", status="200")
    assert client.get("/").status_code == 200
    assert _sample(name, method="GET", endpoint="static", status="200") == static_before + 1

    unmatched_before = _sample(name, method="GET", endpoint="unmatched", status="404")
    assert client.get("/api/nope").status_code == 404
    assert _sample(name, method="GET", endpoint="unmatched", status="404") == unmatched_before + 1


def test_metrics_auto_attributed_to_answering_checkpoint(client, stub):
    # auto -> the stub answers with the english checkpoint, so that is the
    # label the tokens are counted under, not "auto".
    tokens_before = _sample("laya_studio_inference_input_tokens_total", checkpoint="english")
    assert client.post("/v1/systemone", json=SINGLE_REQUEST).status_code == 200
    assert _sample("laya_studio_inference_input_tokens_total", checkpoint="english") == tokens_before + 42
