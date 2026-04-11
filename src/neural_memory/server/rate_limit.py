"""Simple in-memory sliding-window rate limiter middleware.

Default: 100 requests per 60 seconds per client IP.
Configure via env vars:
  NEURAL_MEMORY_RATE_LIMIT      — max requests per window (0 = disabled, default 100)
  NEURAL_MEMORY_RATE_LIMIT_WINDOW — window in seconds (default 60)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths exempt from rate limiting (health checks should never be throttled)
_EXEMPT_PREFIXES: tuple[str, ...] = ("/health", "/ready")


def _get_limit() -> int:
    try:
        return max(0, int(os.environ.get("NEURAL_MEMORY_RATE_LIMIT", "100")))
    except ValueError:
        return 100


def _get_window() -> int:
    try:
        return max(1, int(os.environ.get("NEURAL_MEMORY_RATE_LIMIT_WINDOW", "60")))
    except ValueError:
        return 60


def _client_ip(request: Request) -> str:
    """Return best-effort client IP, considering X-Forwarded-For behind a proxy."""
    forwarded: str = str(request.headers.get("X-Forwarded-For", "")).strip()
    if forwarded:
        # Take the leftmost (originating) IP
        return forwarded.split(",")[0].strip()
    if request.client:
        return str(request.client.host)
    return "unknown"


# BaseHTTPMiddleware is typed as Any in starlette stubs, suppress mypy
_Base: Any = None
try:
    from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

    _Base = BaseHTTPMiddleware
except ImportError:
    pass


class RateLimitMiddleware(_Base):  # type: ignore[misc]
    """Per-IP sliding-window rate limiter.

    Uses an in-memory deque per IP to track request timestamps within
    the rolling window. This is sufficient for single-process deployments.
    For multi-process (gunicorn workers) use Redis-backed rate limiting (Phase 1+).
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        # ip → deque of request timestamps within current window
        self._counters: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()
        limit = _get_limit()
        window = _get_window()
        if limit == 0:
            logger.info("Rate limiting is DISABLED (NEURAL_MEMORY_RATE_LIMIT=0).")
        else:
            logger.info("Rate limiting enabled: %d requests per %ds per IP.", limit, window)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        limit = _get_limit()
        if limit == 0:
            return await call_next(request)

        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in _EXEMPT_PREFIXES):
            return await call_next(request)

        window = _get_window()
        ip = _client_ip(request)
        now = time.monotonic()
        cutoff = now - window

        async with self._lock:
            if ip not in self._counters:
                self._counters[ip] = deque()
            q = self._counters[ip]

            # Evict timestamps outside the window
            while q and q[0] < cutoff:
                q.popleft()

            if len(q) >= limit:
                oldest = q[0]
                retry_after = int(window - (now - oldest)) + 1
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Rate limit exceeded. Max {limit} requests per {window}s. "
                        f"Retry after {retry_after}s."
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            q.append(now)

        return await call_next(request)
