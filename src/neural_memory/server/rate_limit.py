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
from collections import OrderedDict, deque
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


# Hard cap on the number of distinct IP buckets retained. Prevents unbounded
# dict growth (memory-exhaustion DoS) even after the empty-deque reaper runs.
_MAX_TRACKED_IPS = 10_000


def _get_trusted_proxies() -> frozenset[str]:
    """Return the set of direct-peer IPs whose X-Forwarded-For we trust.

    Configure via NEURAL_MEMORY_TRUSTED_PROXIES (comma-separated IPs). When the
    direct peer is NOT in this set, X-Forwarded-For is ignored (it is trivially
    client-spoofable and would let a caller both bypass the limit and inflate
    the IP-bucket dict without bound). See audit #46.
    """
    raw = str(os.environ.get("NEURAL_MEMORY_TRUSTED_PROXIES", "")).strip()
    if not raw:
        return frozenset()
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def _client_ip(request: Request) -> str:
    """Return best-effort client IP.

    X-Forwarded-For is only honored when the direct TCP peer (``request.client``)
    is a configured trusted reverse proxy. Otherwise the header is ignored and
    the direct peer address is used as the rate-limit key — preventing XFF
    spoofing from defeating the limiter or exploding the bucket dict (audit #46).
    """
    peer = str(request.client.host) if request.client else "unknown"

    trusted = _get_trusted_proxies()
    if peer in trusted:
        forwarded: str = str(request.headers.get("X-Forwarded-For", "")).strip()
        if forwarded:
            # Take the leftmost (originating) client IP from the trusted chain.
            candidate = forwarded.split(",")[0].strip()
            if candidate:
                return candidate
    return peer


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
        # ip → deque of request timestamps within current window.
        # OrderedDict so we can LRU-evict the oldest bucket when at capacity.
        self._counters: OrderedDict[str, deque[float]] = OrderedDict()
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
                # LRU-cap: if we are at capacity, evict the least-recently-used
                # bucket before inserting a new one. Combined with the empty-deque
                # reaper below this bounds memory regardless of distinct-IP volume.
                if len(self._counters) >= _MAX_TRACKED_IPS:
                    self._counters.pop(next(iter(self._counters)), None)
                self._counters[ip] = deque()
            else:
                # Mark as recently used (move to end) so LRU eviction is meaningful.
                self._counters.move_to_end(ip)
            q = self._counters[ip]

            # Evict timestamps outside the window
            while q and q[0] < cutoff:
                q.popleft()

            # Reaper: drop the bucket entirely once it has no live timestamps.
            # The eviction loop above only popped stale entries; without this the
            # empty deque would persist forever, growing the dict unboundedly
            # under a stream of distinct IPs (audit #46).
            if not q:
                del self._counters[ip]
                q = self._counters[ip] = deque()
                self._counters.move_to_end(ip)

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
