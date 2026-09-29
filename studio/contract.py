"""Strict request contract for the Laya Decision Studio.

The studio accepts two public shapes:

    POST /v1/systemone          {"model"?, "state", "questions", "max_len"?, "head_max_len"?}
    POST /v1/systemone/batches  {"model"?, "states": [{"id", "state"}...], "questions", ...}

The question schema is laya's own -- ``choice`` / ``score`` / ``noul`` -- which is
byte-compatible with the Decision Studio's ``examples.json`` format (a noul's
``criteria`` is keyed only ``true``/``false`` on both sides), so the studio's
curated scenarios port unchanged.

What this module adds over laya's own validation:

* the HTTP amplification guards laya-serve enforces (question count, state size,
  option counts, body size), applied *before* inference so a bad request never
  reaches the model;
* the batch shape: ids associate inputs with outputs, so an id is required and
  must be unique;
* fail-closed errors (``ContractError``) that carry an HTTP status.

Deeper checks -- duplicate labels, null levels, unknown question types -- stay in
laya's ``Router``, which raises ``ValueError`` with a message naming the question;
``app.py`` maps those to 422. This module never duplicates laya's wording; it
only rejects what laya cannot see before inference.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# The same bounds laya-serve enforces on its shipped HTTP surface, read from it
# rather than restated so the studio cannot drift from the server it embeds.
# Importing laya.serve is cheap: torch, fastapi and the Router are deferred
# inside its functions.
import laya.serve as _laya_serve

MAX_QUESTIONS = getattr(_laya_serve, "MAX_QUESTIONS", 64)
MAX_STATE_CHARS = getattr(_laya_serve, "MAX_STATE_CHARS", 50_000)
MAX_BODY_BYTES = getattr(_laya_serve, "MAX_BODY_BYTES", 2 * 1024 * 1024)
MAX_CHOICE_OPTIONS = getattr(_laya_serve, "MAX_CHOICE_OPTIONS", 100)
MAX_SCORE_LEVELS = getattr(_laya_serve, "MAX_SCORE_LEVELS", 32)
MAX_TOTAL_OPTIONS = getattr(_laya_serve, "MAX_TOTAL_OPTIONS", 512)

# The studio accepts up to 1,024 state/question decisions per batch, matching the
# Decision Studio's documented limit. Live microbatch capacity stays laya's own.
MAX_BATCH_STATES = 1024

_QUESTION_TYPES = ("choice", "score", "noul")
_BUDGET_KEYS = ("max_len", "head_max_len")


class ContractError(Exception):
    """A request the studio refuses before inference. Carries an HTTP status."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.message = message
        self.status = status


def _state_chars(state: Any) -> int:
    try:
        return len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))
    except (TypeError, ValueError, RecursionError):
        return MAX_STATE_CHARS + 1


def _check_question(qid: str, question: Any) -> None:
    """Shape-check one question and enforce the per-question option limits."""
    if not isinstance(question, dict):
        raise ContractError("question %r: definition must be an object" % qid)
    qtype = question.get("type")
    if qtype not in _QUESTION_TYPES:
        raise ContractError(
            "question %r: unknown type %r; use one of %s" % (qid, qtype, list(_QUESTION_TYPES)))
    if not isinstance(question.get("instructions"), str) or not question.get("instructions"):
        raise ContractError("question %r: 'instructions' must be a non-empty string" % qid)
    crit = question.get("criteria")
    if qtype == "choice":
        if not isinstance(crit, (dict, list)) or not crit:
            raise ContractError(
                "question %r: a choice question takes 'criteria' as an object of "
                "label -> description, or a list of labels" % qid)
        if len(crit) > MAX_CHOICE_OPTIONS:
            raise ContractError(
                "too many choice options for %r (%d > %d)" % (qid, len(crit), MAX_CHOICE_OPTIONS),
                413)
    elif qtype == "score":
        if not isinstance(crit, list) or not crit:
            raise ContractError(
                "question %r: a score question takes 'criteria' as a list of level "
                "descriptions, index 0 first" % qid)
        if len(crit) > MAX_SCORE_LEVELS:
            raise ContractError(
                "too many score levels for %r (%d > %d)" % (qid, len(crit), MAX_SCORE_LEVELS),
                413)
    elif crit is not None and not isinstance(crit, dict):
        raise ContractError(
            "question %r: a noul question takes 'criteria' as an object with optional "
            "'true'/'false' descriptions, or omits it" % qid)


def validate_questions(questions: Any) -> None:
    """Shape-check the question map and enforce the studio's option budgets."""
    if not isinstance(questions, dict):
        raise ContractError("'questions' must be an object")
    if len(questions) > MAX_QUESTIONS:
        raise ContractError("too many questions (%d > %d)" % (len(questions), MAX_QUESTIONS), 413)
    total_options = 0
    for qid, question in questions.items():
        if not isinstance(question, dict):
            continue
        crit = question.get("criteria")
        if question.get("type") in ("choice", "score") and isinstance(crit, (dict, list)):
            total_options += len(crit)
        _check_question(qid, question)
    if total_options > MAX_TOTAL_OPTIONS:
        raise ContractError(
            "too many answer options across questions (%d > %d)"
            % (total_options, MAX_TOTAL_OPTIONS), 413)


def _check_budgets(payload: Dict[str, Any]) -> Dict[str, int]:
    """Validate optional max_len / head_max_len overrides (422, like laya-serve)."""
    budgets: Dict[str, int] = {}
    for key in _BUDGET_KEYS:
        if key not in payload or payload[key] is None:
            continue
        val = payload[key]
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            raise ContractError("%s must be a positive integer" % key)
        budgets[key] = val
    return budgets


def parse_single(payload: Any) -> Dict[str, Any]:
    """Validate a single decision request and return the inference inputs.

    Returns ``{"state", "questions", "model", "max_len", "head_max_len"}``;
    ``model`` is the raw requested name or ``"auto"``.
    """
    if not isinstance(payload, dict):
        raise ContractError("request body must be an object")
    unknown = set(payload) - {"model", "state", "questions", *_BUDGET_KEYS}
    if unknown:
        raise ContractError("Provide exactly model, questions, state; unknown field(s): %s"
                            % ", ".join(sorted(unknown)))
    if "questions" not in payload:
        raise ContractError("Provide exactly model, questions, state.")
    if "state" not in payload or payload["state"] is None:
        raise ContractError("'state' is required", 400)
    questions = payload["questions"]
    validate_questions(questions)
    state = payload["state"]
    if _state_chars(state) > MAX_STATE_CHARS:
        raise ContractError("state too large (%d > %d chars)"
                            % (_state_chars(state), MAX_STATE_CHARS), 413)
    budgets = _check_budgets(payload)
    return {
        "state": state,
        "questions": questions,
        "model": payload.get("model") or "auto",
        "max_len": budgets.get("max_len"),
        "head_max_len": budgets.get("head_max_len"),
    }


def parse_batch(payload: Any) -> Dict[str, Any]:
    """Validate a batch decision request.

    Returns ``{"states": [{"id", "state"}], "questions", "model", "max_len", "head_max_len"}``.
    Ids associate inputs with outputs, so every state carries one and they are unique.
    """
    if not isinstance(payload, dict):
        raise ContractError("request body must be an object")
    unknown = set(payload) - {"model", "states", "questions", *_BUDGET_KEYS}
    if unknown:
        raise ContractError("Provide exactly model, questions, states; unknown field(s): %s"
                            % ", ".join(sorted(unknown)))
    if "questions" not in payload:
        raise ContractError("Provide exactly model, questions, states.")
    if "states" not in payload:
        raise ContractError("Provide exactly model, questions, states.")
    validate_questions(payload["questions"])
    states = payload["states"]
    if not isinstance(states, list) or not states:
        raise ContractError("'states' must be a non-empty list of {id, state} objects")
    if len(states) > MAX_BATCH_STATES:
        raise ContractError(
            "too many states (%d > %d)" % (len(states), MAX_BATCH_STATES), 413)
    seen = set()
    for i, item in enumerate(states):
        if not isinstance(item, dict):
            raise ContractError("state %d: must be an object with 'id' and 'state'" % i)
        if "id" not in item or "state" not in item:
            raise ContractError("state %d: must have both 'id' and 'state'" % i)
        sid = item["id"]
        if not isinstance(sid, str) or not sid:
            raise ContractError("state %d: 'id' must be a non-empty string" % i)
        if sid in seen:
            raise ContractError("duplicate state id %r; every state needs its own id" % sid)
        seen.add(sid)
        if item["state"] is None:
            raise ContractError("state %r: 'state' is required" % sid, 400)
        chars = _state_chars(item["state"])
        if chars > MAX_STATE_CHARS:
            raise ContractError("state %r too large (%d > %d chars)"
                                % (sid, chars, MAX_STATE_CHARS), 413)
    budgets = _check_budgets(payload)
    return {
        "states": [{"id": s["id"], "state": s["state"]} for s in states],
        "questions": payload["questions"],
        "model": payload.get("model") or "auto",
        "max_len": budgets.get("max_len"),
        "head_max_len": budgets.get("head_max_len"),
    }


def build_single_request(parsed: Dict[str, Any], model: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Build the kwargs for ``Router.predict`` from a parsed single request."""
    kwargs: Dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    for key in _BUDGET_KEYS:
        if parsed.get(key) is not None:
            kwargs[key] = parsed[key]
    return parsed["state"], kwargs


def build_batch_requests(parsed: Dict[str, Any],
                         model: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build the request list for ``Router.predict_batch`` plus its call kwargs.

    Every request carries the shared questions; a pinned model reaches each
    request (the Router's batch form takes per-request overrides).
    """
    requests: List[Dict[str, Any]] = []
    for s in parsed["states"]:
        req: Dict[str, Any] = {"state": s["state"], "questions": parsed["questions"]}
        if model is not None:
            req["model"] = model
        for key in _BUDGET_KEYS:
            if parsed.get(key) is not None:
                req[key] = parsed[key]
        requests.append(req)
    kwargs: Dict[str, Any] = {}
    for key in _BUDGET_KEYS:
        if parsed.get(key) is not None:
            kwargs[key] = parsed[key]
    return requests, kwargs
