"""Same-origin Laya Decision Studio: FastAPI app with the Router embedded.

Replicates the core of llm-semantic-router's Decision Studio against laya:

* ``GET /api/examples`` — the curated scenario catalog (``examples.json``);
* ``GET /v1/models`` — the model catalog and request limits;
* ``GET /api/status`` · ``GET /api/ready`` — per-checkpoint readiness (503 while
  the configured checkpoints are still loading);
* ``POST /v1/systemone`` · ``POST /v1/systemone/batches`` — single and batch
  decisions over laya's own question schema (``choice`` / ``score`` / ``noul``);
* ``POST /api/evaluate`` — a loose internal endpoint that accepts either shape.

One process embeds the Router directly (the studio's native mode). Inference is
synchronous torch, so it runs on a single-worker threadpool behind an asyncio
gate — one forward pass at a time, the same shape laya-serve uses — with an
admission bound held through body read and inference, so concurrent near-cap
requests cannot starve ``/api/ready``. Requests and results are not
application-logged.

Configuration is by environment variable, read by ``laya.serve.build_router``:
``LAYA_DEVICE``, ``LAYA_PRELOAD`` (default 1), ``LAYA_MODELS``, ``LAYA_THREADS``,
``LAYA_AUTO_TASK``, ``LAYA_MAX_LOADED``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .contract import (ContractError, MAX_BODY_BYTES, build_batch_requests,
                       parse_batch, parse_single)
from .registry import DEFAULT_MODEL, checkpoints, public_models, resolve_model

ROOT = Path(__file__).resolve().parent
_log = logging.getLogger("laya.studio")

# Admission bound: each request can buffer up to MAX_BODY_BYTES before inference,
# so without a bound many concurrent near-cap requests would OOM the worker.
DEFAULT_MAX_CONCURRENT = 16


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


async def read_json(request: Request, limit: int = MAX_BODY_BYTES, *, with_size: bool = False):
    """Read a strict JSON body: capped, no duplicate keys, no nonfinite values."""
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "Use Content-Type: application/json")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(413, "Request is too large. No field is truncated.")
    try:
        document = json.loads(
            body,
            object_pairs_hook=unique_object,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON value")),
        )
        return (document, len(body)) if with_size else document
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise HTTPException(422, str(exc)) from None


def build_router():
    """Build the Router from the environment (device, preload, models, threads)."""
    from laya.serve import build_router as _laya_build_router

    return _laya_build_router()


def create_app(router: Optional[Any] = None):
    """Build the FastAPI app. Pass a Router to inject one (tests); otherwise one
    is built from the environment (and preloaded) at app-creation time."""
    from laya.mcp.device import agent_device, resolve_device, router_agent

    if router is None:
        router = build_router()

    # One inference worker, because one forward pass at a time is what a single
    # CPU or GPU Agent wants (the Router already guards checkpoint lifecycle).
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-studio")
    # Created on first request, not here: an asyncio.Lock binds to the running
    # loop when first awaited, and create_app may run before that loop exists.
    gate: Optional[asyncio.Lock] = None
    admission: Optional[asyncio.Semaphore] = None
    max_concurrent = int(os.environ.get("LAYA_MAX_CONCURRENT") or DEFAULT_MAX_CONCURRENT)
    if max_concurrent <= 0:
        max_concurrent = DEFAULT_MAX_CONCURRENT

    # The checkpoints readiness is measured against: LAYA_MODELS (the preloaded
    # set) or every catalog checkpoint. Under LAYA_PRELOAD=0 the Router still
    # answers (it builds on demand); the readiness response says so.
    models_env = os.environ.get("LAYA_MODELS", "").strip()
    configured = [m.strip() for m in models_env.split(",") if m.strip()] \
        or [item["id"] for item in checkpoints()]
    lazy = os.environ.get("LAYA_PRELOAD", "1").strip().lower() in ("0", "false", "no", "off")

    @asynccontextmanager
    async def lifespan(_api: FastAPI):
        try:
            yield
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    api = FastAPI(
        title="Laya Decision Studio",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    api.state.router = router

    def device_report() -> Dict[str, Any]:
        """Where a resident checkpoint really computes, not what was asked for."""
        checkpoint_devices: Dict[str, str] = {}
        for name in (router.loaded or []):
            device = agent_device(router_agent(router, name))
            if device:
                checkpoint_devices[name] = device
        actual = next(iter(checkpoint_devices.values()), None)
        return {
            "device": actual or resolve_device(),
            "device_is_preference": actual is None,
            "checkpoint_devices": checkpoint_devices,
        }

    def loaded_map() -> Dict[str, bool]:
        loaded = set(router.loaded or [])
        return {item["id"]: item["id"] in loaded for item in checkpoints()}

    @api.get("/api/status")
    def status() -> Dict[str, Any]:
        loaded = loaded_map()
        return {
            "status": "ok",
            "backend": "native",
            "lazy": lazy,
            "configured": configured,
            "loaded": loaded,
            "ready": all(loaded[name] for name in configured),
            "revisions": getattr(router, "loaded_revisions", {}),
            **device_report(),
        }

    @api.get("/api/ready")
    def ready():
        """503 until every configured checkpoint is resident, so the studio can
        show the offline note while the model is loading."""
        loaded = loaded_map()
        healthy = all(loaded[name] for name in configured)
        return JSONResponse(
            {
                "backend": "native",
                "lazy": lazy,
                "status": "ready" if healthy else "unavailable",
                "models": loaded,
            },
            status_code=200 if healthy else 503,
        )

    @api.get("/api/examples")
    def examples():
        return json.loads((ROOT / "examples.json").read_text(encoding="utf-8"))

    @api.get("/v1/models")
    def models() -> Dict[str, Any]:
        return {
            "default": DEFAULT_MODEL,
            "models": public_models(),
            "limits": {
                "request_bytes": MAX_BODY_BYTES,
                "questions": None,
                "contexts": None,
                "states": 1024,
                "synchronous_wait_seconds": 45,
            },
        }

    @api.exception_handler(ContractError)
    async def contract_error(request: Request, exc: ContractError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    def _ensure_gates():
        nonlocal gate, admission
        if gate is None:
            gate = asyncio.Lock()
        if admission is None:
            admission = asyncio.Semaphore(max_concurrent)

    def _predict_kwargs(model: Optional[str], parsed: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        for key in ("max_len", "head_max_len"):
            if parsed.get(key) is not None:
                kwargs[key] = parsed[key]
        return kwargs

    async def _handle_single(payload: Any) -> JSONResponse:
        parsed = parse_single(payload)
        model = resolve_model(parsed["model"])
        kwargs = _predict_kwargs(model, parsed)
        try:
            async with gate:
                loop = asyncio.get_running_loop()
                t0 = time.perf_counter()
                result = await loop.run_in_executor(
                    pool, lambda: router.predict(
                        parsed["state"], parsed["questions"], **kwargs))
                infer_ms = (time.perf_counter() - t0) * 1000.0
        except HTTPException:
            raise
        except ValueError as e:
            # laya's question validation names the question and what to fix.
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:  # noqa: BLE001 -- never leak paths/weights/OOM text to clients
            _log.exception("inference failed")
            raise HTTPException(status_code=500, detail="inference failed")
        return JSONResponse(result, headers={
            "Server-Timing": f"inference;dur={infer_ms:.2f}",
            "X-Inference-Time-Ms": f"{infer_ms:.2f}",
        })

    async def _handle_batch(payload: Any) -> JSONResponse:
        parsed = parse_batch(payload)
        model = resolve_model(parsed["model"])
        kwargs = _predict_kwargs(model, parsed)
        requests, batch_kwargs = build_batch_requests(parsed, model)
        try:
            async with gate:
                loop = asyncio.get_running_loop()
                t0 = time.perf_counter()
                results = await loop.run_in_executor(
                    pool, lambda: router.predict_batch(requests, **batch_kwargs))
                infer_ms = (time.perf_counter() - t0) * 1000.0
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:  # noqa: BLE001
            _log.exception("batch inference failed")
            raise HTTPException(status_code=500, detail="inference failed")
        body = {
            "model": parsed["model"],
            "count": len(results),
            "results": [
                {
                    "id": s["id"],
                    "model": r.get("model"),
                    "answers": r.get("answers"),
                    "usage": r.get("usage"),
                    "routing": r.get("routing"),
                }
                for s, r in zip(parsed["states"], results)
            ],
            "usage": {
                "input_tokens": sum((r.get("usage") or {}).get("input_tokens", 0) for r in results),
                "output_tokens": sum((r.get("usage") or {}).get("output_tokens", 0) for r in results),
            },
        }
        return JSONResponse(body, headers={
            "Server-Timing": f"inference;dur={infer_ms:.2f}",
            "X-Inference-Time-Ms": f"{infer_ms:.2f}",
        })

    @api.post("/v1/systemone")
    async def systemone(request: Request):
        _ensure_gates()
        if admission.locked():
            raise HTTPException(status_code=503, detail="server busy, try again later",
                                headers={"Retry-After": "1"})
        await admission.acquire()
        try:
            payload = await read_json(request)
            return await _handle_single(payload)
        finally:
            admission.release()

    @api.post("/v1/systemone/batches")
    async def systemone_batches(request: Request):
        _ensure_gates()
        if admission.locked():
            raise HTTPException(status_code=503, detail="server busy, try again later",
                                headers={"Retry-After": "1"})
        await admission.acquire()
        try:
            payload = await read_json(request)
            return await _handle_batch(payload)
        finally:
            admission.release()

    @api.post("/api/evaluate")
    async def evaluate(request: Request):
        """Loose internal endpoint: dispatches on the presence of `states`."""
        _ensure_gates()
        if admission.locked():
            raise HTTPException(status_code=503, detail="server busy, try again later",
                                headers={"Retry-After": "1"})
        await admission.acquire()
        try:
            payload = await read_json(request)
            if isinstance(payload, dict) and "states" in payload:
                return await _handle_batch(payload)
            return await _handle_single(payload)
        finally:
            admission.release()

    @api.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith(("/api/", "/v1/")):
            response.headers["Cache-Control"] = "no-store"
        return response

    api.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="studio")
    return api


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Laya Decision Studio")
    parser.add_argument("--host", default=os.environ.get("LAYA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("LAYA_PORT", "7860") or 7860))
    parser.add_argument("--no-preload", action="store_true",
                        help="load checkpoints lazily instead of all up front")
    parser.add_argument("--device", default=None, help="cuda | cpu | mps")
    args = parser.parse_args()
    if args.no_preload:
        os.environ["LAYA_PRELOAD"] = "0"
    if args.device:
        os.environ["LAYA_DEVICE"] = args.device
    uvicorn.run(create_app(), host=args.host, port=args.port,
                log_level=os.environ.get("LAYA_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
