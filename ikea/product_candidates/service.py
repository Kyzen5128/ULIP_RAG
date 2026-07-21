"""FastAPI transport boundary for the Product Candidates v1 runtime."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .contracts import (
    READYZ_VALIDATOR,
    REQUEST_VALIDATOR,
    RESPONSE_VALIDATOR,
    first_validation_error,
)
from .runtime import (
    READY_CHECK_KEYS,
    ProductCandidateRuntime,
    RequestPolicyError,
    RuntimeUnavailableError,
)


LOGGER = logging.getLogger("ulip.product_candidates")
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class ServiceOverloadedError(RuntimeError):
    pass


class _RejectDuplicateKeys(dict):
    def __init__(self, pairs: Any) -> None:
        values: Dict[str, Any] = {}
        for key, value in pairs:
            if key in values:
                raise ValueError("duplicate JSON object key")
            values[key] = value
        super().__init__(values)


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError("non-finite JSON number: %s" % value)


def _has_duplicate_category_term(wire: Any) -> bool:
    """Detect exact/case-only duplicates before generic schema reporting."""
    if not isinstance(wire, Mapping):
        return False
    primary = wire.get("categories_primary")
    secondary = wire.get("categories_secondary")
    if not isinstance(primary, list) or not isinstance(secondary, list):
        return False
    if any(not isinstance(value, str) for value in primary + secondary):
        return False
    normalized = [value.strip().casefold() for value in primary + secondary]
    return len(normalized) != len(set(normalized))


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    retry_after: Optional[int] = None,
) -> JSONResponse:
    headers = {"Cache-Control": "no-store"}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    if status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


class InferenceGate:
    """One active inference and at most one queued request."""

    def __init__(self, max_queue: int = 1) -> None:
        if isinstance(max_queue, bool) or not isinstance(max_queue, int) or max_queue < 0:
            raise ValueError("max_queue must be a non-negative integer")
        # asyncio primitives are bound to the loop on which they are used in
        # the Python version deployed on edge05.  ``create_app`` runs at module
        # import time, before Uvicorn owns a running loop, so bind lazily rather
        # than creating a semaphore/lock here.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._state_lock: Optional[asyncio.Lock] = None
        self._queued = 0
        self._max_queue = max_queue

    def bind_to_current_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is loop:
            return
        if self._loop is not None and (
            self._queued
            or (self._semaphore is not None and self._semaphore.locked())
        ):
            raise RuntimeError("cannot rebind a busy inference gate")
        self._loop = loop
        self._semaphore = asyncio.Semaphore(1)
        self._state_lock = asyncio.Lock()
        self._queued = 0

    async def acquire(self) -> None:
        self.bind_to_current_loop()
        assert self._semaphore is not None
        assert self._state_lock is not None
        queued_slot = False
        async with self._state_lock:
            if not self._semaphore.locked():
                # No await is possible while the semaphore value is positive, so
                # reserve the active slot before releasing the state lock.
                await self._semaphore.acquire()
                return
            if self._queued >= self._max_queue:
                raise ServiceOverloadedError("GPU inference queue is full")
            self._queued += 1
            queued_slot = True
        try:
            await self._semaphore.acquire()
        finally:
            if queued_slot:
                async with self._state_lock:
                    self._queued -= 1

    def release(self) -> None:
        if self._semaphore is None:
            raise RuntimeError("inference gate is not bound to an event loop")
        self._semaphore.release()


@dataclass
class ServiceState:
    runtime: Optional[ProductCandidateRuntime]
    runtime_loader: Optional[Callable[[], ProductCandidateRuntime]]
    bearer_token: Optional[str]
    require_strong_token: bool
    hard_timeout_s: float
    gate: InferenceGate
    startup_failure: Optional[str] = None

    def token_valid(self) -> bool:
        if not isinstance(self.bearer_token, str) or not self.bearer_token:
            return False
        if self.require_strong_token and len(self.bearer_token.encode("utf-8")) < 32:
            return False
        return True

    async def startup(self) -> None:
        self.gate.bind_to_current_loop()
        if not self.token_valid():
            self.startup_failure = "AUTH_CONFIGURATION_INVALID"
            return
        if self.runtime is not None:
            if not self.runtime.ready:
                self.startup_failure = "DEPLOYMENT_PROFILE_NOT_READY"
            return
        if self.runtime_loader is None:
            self.startup_failure = "RUNTIME_LOADER_MISSING"
            return
        try:
            runtime = await asyncio.to_thread(self.runtime_loader)
            if not isinstance(runtime, ProductCandidateRuntime):
                raise TypeError("runtime_loader returned the wrong type")
            self.runtime = runtime
            if not runtime.ready:
                self.startup_failure = "DEPLOYMENT_PROFILE_NOT_READY"
        except Exception:
            LOGGER.exception("Product Candidates runtime startup failed")
            self.startup_failure = "RUNTIME_INITIALIZATION_FAILED"

    def readiness_payload(self) -> Dict[str, Any]:
        if (
            self.startup_failure is None
            and self.token_valid()
            and self.runtime is not None
            and self.runtime.ready
        ):
            return self.runtime.readiness_payload()
        checks = {key: False for key in READY_CHECK_KEYS}
        inventory = {
            "vector_shape": [0, 1],
            "vector_rows": 0,
            "catalog_rows": 0,
            "dimension_profile_rows": 0,
        }
        if (
            self.runtime is not None
            and self.token_valid()
            and self.startup_failure in {None, "DEPLOYMENT_PROFILE_NOT_READY"}
        ):
            checks = dict(self.runtime.checks)
            inventory = dict(self.runtime.inventory)
        elif self.runtime is not None:
            # The readyz schema requires a not_ready service to expose at least
            # one failed fixed check.  Service configuration failure means the
            # verified assets are not an active deployment, so do not advertise
            # their all-true check tuple.
            inventory = dict(self.runtime.inventory)
        failures = []
        if (
            self.runtime is not None
            and self.token_valid()
            and self.startup_failure in {None, "DEPLOYMENT_PROFILE_NOT_READY"}
        ):
            failures.extend(
                "%s_FAILED" % key.upper()
                for key, passed in checks.items() if not passed
            )
            if self.runtime.integrity_failure is not None:
                failures.append(self.runtime.integrity_failure)
        if self.startup_failure:
            failures.append(self.startup_failure)
        if not self.token_valid():
            failures.append("AUTH_CONFIGURATION_INVALID")
        if not failures:
            failures.append("SERVICE_NOT_INITIALIZED")
        return {
            "schema_version": "1.0",
            "service": "ulip-product-candidates",
            "status": "not_ready",
            "deployment_profile_id": None,
            "endpoint_schema_version": "1.0",
            "provenance": None,
            "checks": checks,
            "inventory": inventory,
            "failures": sorted(set(failures)),
        }


async def _release_after_work(task: "asyncio.Task[Any]", gate: InferenceGate) -> None:
    try:
        await task
    except Exception:
        pass
    finally:
        gate.release()


async def _run_runtime_with_limit(
    state: ServiceState,
    wire: Mapping[str, Any],
) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        await asyncio.wait_for(state.gate.acquire(), timeout=state.hard_timeout_s)
    except asyncio.TimeoutError as exc:
        raise RuntimeUnavailableError(
            "INFERENCE_TIMEOUT", "request exceeded the server hard timeout",
        ) from exc
    except ServiceOverloadedError:
        raise
    remaining = state.hard_timeout_s - (time.monotonic() - started)
    if remaining <= 0:
        state.gate.release()
        raise RuntimeUnavailableError(
            "INFERENCE_TIMEOUT", "request exceeded the server hard timeout",
        )
    assert state.runtime is not None
    work = asyncio.create_task(asyncio.to_thread(state.runtime.retrieve, wire))
    try:
        result = await asyncio.wait_for(asyncio.shield(work), timeout=remaining)
    except asyncio.TimeoutError as exc:
        # Python cannot cancel an already-running CUDA thread.  Keep the single
        # inference lease until that work actually exits, even after returning 503.
        asyncio.create_task(_release_after_work(work, state.gate))
        raise RuntimeUnavailableError(
            "INFERENCE_TIMEOUT", "request exceeded the server hard timeout",
        ) from exc
    except asyncio.CancelledError:
        # A disconnected caller must not leak the sole inference lease.  The
        # worker thread itself cannot be cancelled safely, so retain the lease
        # until it exits just as we do for a hard timeout.
        asyncio.create_task(_release_after_work(work, state.gate))
        raise
    except Exception:
        state.gate.release()
        raise
    state.gate.release()
    return result


def create_app(
    *,
    runtime: Optional[ProductCandidateRuntime] = None,
    runtime_loader: Optional[Callable[[], ProductCandidateRuntime]] = None,
    bearer_token: Optional[str] = None,
    require_strong_token: bool = False,
    hard_timeout_s: float = 25.0,
    max_queue: int = 1,
) -> FastAPI:
    """Build an app without loading CUDA; production assets load in lifespan."""
    if (
        isinstance(hard_timeout_s, bool)
        or not isinstance(hard_timeout_s, (int, float))
        or not math_is_finite_positive(float(hard_timeout_s))
    ):
        raise ValueError("hard_timeout_s must be positive and finite")
    state = ServiceState(
        runtime=runtime,
        runtime_loader=runtime_loader,
        bearer_token=bearer_token,
        require_strong_token=require_strong_token,
        hard_timeout_s=float(hard_timeout_s),
        gate=InferenceGate(max_queue=max_queue),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        await state.startup()
        yield

    app = FastAPI(
        title="ULIP Product Candidates",
        version="1.0",
        lifespan=lifespan,
    )
    app.state.product_candidates = state

    @app.get("/healthz")
    async def healthz() -> Response:
        return JSONResponse(
            content={
                "schema_version": "1.0",
                "service": "ulip-product-candidates",
                "status": "ok",
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/readyz")
    async def readyz() -> Response:
        if state.runtime is not None:
            # Re-run the offline bundle verifier for every readiness probe.
            # This re-hashes manifests and members and rechecks row joins; it
            # neither queries MongoDB nor reloads the GPU model.
            await asyncio.to_thread(state.runtime.revalidate_integrity)
        payload = state.readiness_payload()
        error = first_validation_error(READYZ_VALIDATOR, payload)
        if error is not None:
            LOGGER.error("Internal readyz payload failed its schema: %s", error.message)
            payload = {
                "schema_version": "1.0",
                "service": "ulip-product-candidates",
                "status": "not_ready",
                "deployment_profile_id": None,
                "endpoint_schema_version": "1.0",
                "provenance": None,
                "checks": {key: False for key in READY_CHECK_KEYS},
                "inventory": {
                    "vector_shape": [0, 1],
                    "vector_rows": 0,
                    "catalog_rows": 0,
                    "dimension_profile_rows": 0,
                },
                "failures": ["READINESS_PAYLOAD_INVALID"],
            }
        return JSONResponse(
            status_code=200 if payload["status"] == "ready" else 503,
            content=payload,
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/v2/product-candidates")
    async def product_candidates(request: Request) -> Response:
        if not state.token_valid() or state.runtime is None or not state.runtime.ready:
            return _error_response(
                503, "SERVICE_NOT_READY", "immutable ULIP deployment is not ready",
                retry_after=5,
            )
        authorization = request.headers.get("authorization", "")
        scheme, separator, supplied_token = authorization.partition(" ")
        if (
            not separator
            or scheme.casefold() != "bearer"
            or not supplied_token
            or not hmac.compare_digest(
                supplied_token.encode("utf-8"),
                (state.bearer_token or "").encode("utf-8"),
            )
        ):
            return _error_response(401, "AUTHENTICATION_REQUIRED", "valid Bearer token required")
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
        if media_type != "application/json":
            return _error_response(415, "CONTENT_TYPE_UNSUPPORTED", "Content-Type must be application/json")
        accept = request.headers.get("accept", "*/*").casefold()
        if not any(value in accept for value in ("application/json", "application/*", "*/*")):
            return _error_response(406, "ACCEPT_UNSUPPORTED", "Accept must allow application/json")
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_BYTES:
                    return _error_response(413, "REQUEST_TOO_LARGE", "request body exceeds 1 MiB")
            except ValueError:
                return _error_response(422, "REQUEST_SCHEMA_INVALID", "invalid Content-Length")
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > MAX_REQUEST_BYTES:
                return _error_response(413, "REQUEST_TOO_LARGE", "request body exceeds 1 MiB")
        body = bytes(chunks)
        try:
            text = body.decode("utf-8")
            wire = json.loads(
                text,
                parse_constant=_reject_nonfinite_constant,
                object_pairs_hook=_RejectDuplicateKeys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return _error_response(422, "REQUEST_JSON_INVALID", "body must be strict UTF-8 JSON")
        if _has_duplicate_category_term(wire):
            return _error_response(
                422,
                "CATEGORY_TERM_DUPLICATE",
                "category terms must be case-insensitively unique across tiers",
            )
        error = first_validation_error(REQUEST_VALIDATOR, wire)
        if error is not None:
            path = "/".join(str(part) for part in error.absolute_path) or "<root>"
            return _error_response(
                422, "REQUEST_SCHEMA_INVALID", "request schema validation failed at %s" % path,
            )
        try:
            response_payload = await _run_runtime_with_limit(state, wire)
            response_error = first_validation_error(RESPONSE_VALIDATOR, response_payload)
            if response_error is not None:
                raise RuntimeUnavailableError(
                    "RESPONSE_SCHEMA_INVALID", "runtime produced an invalid response",
                )
            encoded = json.dumps(
                response_payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(encoded) > MAX_RESPONSE_BYTES:
                raise RuntimeUnavailableError(
                    "RESPONSE_TOO_LARGE", "response exceeds the 4 MiB consumer limit",
                )
            return Response(
                content=encoded,
                status_code=200,
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        except RequestPolicyError as exc:
            return _error_response(422, exc.code, exc.message)
        except ServiceOverloadedError:
            return _error_response(
                503, "INFERENCE_QUEUE_FULL", "GPU inference queue is full",
                retry_after=1,
            )
        except RuntimeUnavailableError as exc:
            LOGGER.warning("Product Candidates request failed closed: %s", exc.code)
            return _error_response(503, exc.code, exc.message, retry_after=1)
        except Exception:
            LOGGER.exception("Unexpected Product Candidates failure")
            return _error_response(
                503, "RUNTIME_FAILURE", "vanilla ULIP retrieval is unavailable",
                retry_after=1,
            )

    return app


def math_is_finite_positive(value: float) -> bool:
    # Kept local so importing the transport boundary never imports torch.
    import math
    return math.isfinite(value) and value > 0
