"""API key authentication middleware for NeuralMemory server.

Reads NEURAL_MEMORY_API_KEY from environment. When set, every request to
API routes must supply the key via one of:
  - Authorization: Bearer <key>
  - X-API-Key: <key>

Public paths (health, docs, UI, assets) are always allowed through.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths that bypass API key check
_PUBLIC_PREFIXES: tuple[str, ...] = (
    "/health",
    "/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/ui",
    "/dashboard",
    "/assets",
    "/favicon.ico",
)


def _get_configured_key() -> str:
    """Return API key from environment, stripped of whitespace."""
    return os.environ.get("NEURAL_MEMORY_API_KEY", "").strip()


def _is_public_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in _PUBLIC_PREFIXES)


def _extract_key(request: Request) -> str | None:
    """Extract API key from Authorization or X-API-Key header."""
    # X-API-Key header (preferred for programmatic access)
    key: str = str(request.headers.get("X-API-Key", "")).strip()
    if key:
        return key

    # Authorization: Bearer <key>
    auth: str = str(request.headers.get("Authorization", "")).strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()

    return None


# BaseHTTPMiddleware is typed as Any in starlette stubs, suppress mypy
_Base: Any = None
try:
    from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

    _Base = BaseHTTPMiddleware
except ImportError:
    pass


class APIKeyMiddleware(_Base):  # type: ignore[misc]
    """Middleware that enforces API key authentication when configured.

    If NEURAL_MEMORY_API_KEY is not set, all requests are allowed through
    with a warning logged at startup (to alert operators).
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        configured = _get_configured_key()
        if not configured:
            logger.warning(
                "NEURAL_MEMORY_API_KEY is not set — server is running WITHOUT authentication. "
                "Set this env var to secure the API."
            )
        else:
            logger.info("API key authentication is enabled.")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        # Always allow public paths
        if _is_public_path(path):
            return await call_next(request)

        configured_key = _get_configured_key()

        # No key configured → allow through (but already warned at startup)
        if not configured_key:
            return await call_next(request)

        provided_key = _extract_key(request)

        if not provided_key:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Missing API key. Provide it via 'X-API-Key' header or "
                    "'Authorization: Bearer <key>'."
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(configured_key, provided_key):
            logger.warning(
                "Invalid API key attempt from %s", request.client and request.client.host
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."},
            )

        return await call_next(request)
