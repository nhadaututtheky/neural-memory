"""Sandbox compatibility probe for ``aiosqlite``.

Some restricted environments (Codex sandbox, certain seccomp profiles,
locked-down containers) allow Python threads to run but block the
``loop.call_soon_threadsafe()`` wakeup that ``aiosqlite``'s worker thread
uses to deliver results back to the event loop. The symptom is a silent
hang: every storage-backed CLI command, MCP tool, or REST endpoint blocks
forever in ``selectors.select()`` waiting for a future that will never
resolve.

This module runs a 2-second probe — ``aiosqlite.connect(":memory:")``
wrapped in ``asyncio.wait_for`` — and caches the verdict. Entry points
(CLI dispatcher, MCP stdio loop, FastAPI lifespan) call the probe before
touching storage so they fail fast with an actionable error instead of
hanging.

See: https://github.com/nhadaututtheky/neural-memory/issues/151
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS: Final[float] = 2.0
BYPASS_ENV: Final[str] = "NMEM_SKIP_AIOSQLITE_PROBE"

SANDBOX_HINT: Final[str] = (
    "Detected a restricted environment where aiosqlite cannot complete a "
    "basic in-memory connect. This is typical for Codex-style sandboxes "
    "that allow Python threads but block asyncio cross-thread loop wakeups "
    "(call_soon_threadsafe).\n"
    "Workarounds:\n"
    "  - Run the command outside the sandbox\n"
    "  - Or grant the environment permission for thread → event-loop "
    "wakeups\n"
    "  - Set NMEM_SKIP_AIOSQLITE_PROBE=1 only if you understand the risk "
    "of hangs\n"
    "  - Track: https://github.com/nhadaututtheky/neural-memory/issues/151"
)


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of an ``aiosqlite`` compatibility probe."""

    ok: bool
    detail: str
    error_class: str | None = None


_cached_result: ProbeResult | None = None


def _is_bypassed() -> bool:
    """Allow tests / power users to skip the probe via env var."""
    return os.environ.get(BYPASS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


async def _probe_coro(timeout: float) -> ProbeResult:
    """Single probe attempt — must be cheap and side-effect free."""
    import aiosqlite  # local import keeps probe optional

    try:
        conn = await asyncio.wait_for(aiosqlite.connect(":memory:"), timeout=timeout)
    except TimeoutError:
        return ProbeResult(
            ok=False,
            detail=f"aiosqlite connect timed out after {timeout:.1f}s",
            error_class="TimeoutError",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ProbeResult(
            ok=False,
            detail=f"aiosqlite connect failed: {type(exc).__name__}: {exc}",
            error_class=type(exc).__name__,
        )

    try:
        await asyncio.wait_for(conn.close(), timeout=timeout)
    except Exception:  # pragma: no cover - close is best effort
        logger.debug("aiosqlite probe close failed", exc_info=True)

    return ProbeResult(ok=True, detail="aiosqlite probe succeeded")


def probe_aiosqlite_compat(
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    force: bool = False,
) -> ProbeResult:
    """Probe whether ``aiosqlite`` can run in the current environment.

    Result is cached process-wide. Pass ``force=True`` to re-run.
    Honours ``NMEM_SKIP_AIOSQLITE_PROBE`` for opt-out.
    """
    global _cached_result

    if _cached_result is not None and not force:
        return _cached_result

    if _is_bypassed():
        _cached_result = ProbeResult(ok=True, detail="probe bypassed via env var")
        return _cached_result

    try:
        result = asyncio.run(_probe_coro(timeout))
    except RuntimeError as exc:
        # Called from inside a running event loop (rare — library misuse).
        # We can't safely probe; default to OK and let the caller surface
        # any actual aiosqlite error itself.
        logger.debug("Cannot probe aiosqlite from running loop: %s", exc)
        result = ProbeResult(ok=True, detail="probe skipped (already in event loop)")

    _cached_result = result
    return result


class SandboxIncompatibleError(RuntimeError):
    """Raised when the environment cannot run ``aiosqlite``."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def ensure_aiosqlite_or_exit_cli() -> None:
    """Probe; on failure print a friendly error to stderr and ``typer.Exit(2)``.

    Used by CLI entry points so storage-backed commands fail fast instead
    of hanging on ``aiosqlite.connect``.
    """
    result = probe_aiosqlite_compat()
    if result.ok:
        return

    import typer

    typer.secho(
        "ERROR: Neural Memory cannot run in this environment.",
        fg=typer.colors.RED,
        bold=True,
        err=True,
    )
    typer.secho(f"  Detail: {result.detail}", fg=typer.colors.RED, err=True)
    typer.echo("", err=True)
    for line in SANDBOX_HINT.splitlines():
        typer.secho(f"  {line}", err=True)
    raise typer.Exit(code=2)


def ensure_aiosqlite_or_raise() -> None:
    """Probe; on failure raise ``SandboxIncompatibleError``.

    Used by long-running servers (FastAPI lifespan) where typer.Exit is
    inappropriate — the caller decides how to surface the failure.
    """
    result = probe_aiosqlite_compat()
    if not result.ok:
        raise SandboxIncompatibleError(result.detail)


def reset_probe_cache() -> None:
    """Clear the cached probe result. Test-only helper."""
    global _cached_result
    _cached_result = None
