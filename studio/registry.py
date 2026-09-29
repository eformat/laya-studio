"""Model catalog for the Laya Decision Studio.

Laya replaces the Decision Studio's six remote runtimes with three in-process
checkpoints plus the Router's automatic choice. The catalog is the source of
truth for ``GET /v1/models``; ``resolve_model`` is fail-closed: a model that is
not in the catalog answers 422 instead of silently falling back to another
checkpoint, so a misrouted request is never attributed to the wrong model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .contract import ContractError

# laya's DEFAULT_MODELS keys, in display order. "auto" is not a laya checkpoint:
# it means "let the Router choose", the default the SDK itself documents.
DEFAULT_MODEL = "auto"

_CHECKPOINTS: List[Dict[str, Any]] = [
    {
        "id": "english",
        "label": "Laya English",
        "repo_id": "convaiinnovations/laya",
        "encoder": "ModernBERT-large",
        "params": "421M",
        "context": 512,
        "blurb": "English-first checkpoint",
    },
    {
        "id": "multilingual",
        "label": "Laya Multilingual",
        "repo_id": "convaiinnovations/laya-multilingual",
        "encoder": "mmBERT-base",
        "params": "322M",
        "context": 1024,
        "blurb": "100+ languages, 2x faster",
    },
    {
        "id": "typed-decisions",
        "label": "Laya Typed Decisions",
        "repo_id": "convaiinnovations/laya-typed-decisions",
        "encoder": "ModernBERT-large",
        "params": "421M",
        "context": 1024,
        "blurb": "Fine-tuned for the typed-decisions workflows",
    },
]


def checkpoints() -> List[Dict[str, Any]]:
    """The catalog's laya checkpoints, in display order."""
    return [dict(item) for item in _CHECKPOINTS]


def is_checkpoint(name: str) -> bool:
    return any(item["id"] == name for item in _CHECKPOINTS)


def public_models() -> List[Dict[str, Any]]:
    """The response body for ``GET /v1/models``: auto first, then the checkpoints."""
    models: List[Dict[str, Any]] = [{
        "id": DEFAULT_MODEL,
        "label": "Auto (Router)",
        "repo_id": "convaiinnovations/laya",
        "blurb": "The Router detects the language and picks the checkpoint",
        "auto": True,
    }]
    models.extend(dict(item, auto=False) for item in _CHECKPOINTS)
    return models


def resolve_model(model: Optional[str]) -> Optional[str]:
    """Map a request's ``model`` field onto a laya checkpoint, or ``None`` for auto.

    ``None`` / ``""`` / ``"auto"`` mean the Router chooses. Anything else must name
    a catalog checkpoint (laya's own aliases resolve through the SDK, so ``en``,
    ``ml``, ``typed`` and friends all work); an unknown id raises 422 rather than
    silently falling back.
    """
    if model is None or not str(model).strip() or str(model).strip().lower() == DEFAULT_MODEL:
        return None
    from laya.router import normalise_name

    try:
        key = normalise_name(model)
    except ValueError:
        raise ContractError(
            "This model is not available in this Studio: %r. "
            "Choose one of: auto, %s." % (model, ", ".join(item["id"] for item in _CHECKPOINTS)),
        ) from None
    if not is_checkpoint(key):
        raise ContractError(
            "This model is not available in this Studio: %r. "
            "Choose one of: auto, %s." % (model, ", ".join(item["id"] for item in _CHECKPOINTS)),
        )
    return key
