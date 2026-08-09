"""SSL context helper for aiohttp — fixes macOS certificate issues.

macOS Python often ships without system CA certs linked, causing
``SSLCertVerificationError`` on HTTPS requests.  This module provides
a drop-in ``ssl.SSLContext`` that uses **certifi** (already an aiohttp
dependency) as the CA bundle.

Usage::

    from neural_memory.utils.ssl_helper import safe_client_session
    async with safe_client_session() as session:
        async with session.get(url) as resp:
            ...
"""

from __future__ import annotations

import ssl
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_ssl_context() -> ssl.SSLContext | bool:
    """Return an SSL context with certifi CA bundle.

    Falls back to ``True`` (default verification) if certifi
    is not installed — this preserves current behaviour on
    platforms where system certs work fine.
    """
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    except ImportError:
        return True


def safe_client_session(**kwargs: Any) -> Any:
    """Create an ``aiohttp.ClientSession`` with certifi SSL context.

    Drop-in replacement for ``aiohttp.ClientSession(**kwargs)``.
    Requires ``neural-memory[sync]`` (or ``[server]`` which includes aiohttp).
    """
    from neural_memory.utils.optional_dependencies import require_capability

    aiohttp = require_capability("aiohttp", "sync", "HTTP client session")

    ssl_ctx = get_ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    return aiohttp.ClientSession(connector=connector, **kwargs)
