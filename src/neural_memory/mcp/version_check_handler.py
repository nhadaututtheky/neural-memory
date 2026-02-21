"""Background version check handler for MCP server.

Periodically checks PyPI for newer versions of neural-memory and caches
the result. Surfaces an ``update_hint`` in tool responses when an update
is available.

Starts via ``maybe_start_version_check()`` at server startup.
Cancelled via ``cancel_version_check()`` in server shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory import __version__
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.unified_config import MaintenanceConfig, UnifiedConfig

logger = logging.getLogger(__name__)

_PYPI_URL = "https://pypi.org/pypi/neural-memory/json"
_REQUEST_TIMEOUT = 10  # seconds


@dataclass(frozen=True)
class VersionInfo:
    """Cached version check result."""

    current: str
    latest: str
    update_available: bool
    checked_at: datetime


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a version string into a comparable tuple.

    Handles pre-release suffixes by stripping them for comparison.
    E.g. "2.4.0" -> (2, 4, 0), "2.5.0a1" -> (2, 5, 0).
    """
    # Strip common pre-release suffixes
    clean = version.split("a")[0].split("b")[0].split("rc")[0].split("dev")[0]
    parts: list[int] = []
    for part in clean.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def _is_newer(latest: str, current: str) -> bool:
    """Check if latest version is newer than current."""
    return _parse_version(latest) > _parse_version(current)


class VersionCheckHandler:
    """Mixin: background version check for MCP server.

    Periodically queries PyPI to detect newer versions. Caches the
    result and exposes it via ``get_update_hint()`` for injection
    into tool responses.
    """

    _version_check_task: asyncio.Task[None] | None = None
    _version_info: VersionInfo | None = None

    if TYPE_CHECKING:
        config: UnifiedConfig

    async def maybe_start_version_check(self) -> asyncio.Task[None] | None:
        """Start the background version check loop if configured.

        Called once at server startup. Guards against double-start.
        Returns the background task or None.
        """
        if self._version_check_task is not None and not self._version_check_task.done():
            return self._version_check_task

        cfg: MaintenanceConfig = self.config.maintenance
        if not cfg.enabled or not cfg.version_check_enabled:
            return None

        task = asyncio.create_task(self._version_check_loop(cfg))
        task.add_done_callback(_log_version_check_exception)
        self._version_check_task = task
        return self._version_check_task

    async def _version_check_loop(self, cfg: MaintenanceConfig) -> None:
        """Background loop: check PyPI on startup then every interval.

        First check runs after a short delay (30s) to avoid blocking
        server startup. Subsequent checks wait the full interval.
        """
        # Short initial delay — let the server finish starting up
        await asyncio.sleep(30)
        await self._check_latest_version()

        interval_seconds = cfg.version_check_interval_hours * 3600
        while True:
            await asyncio.sleep(interval_seconds)
            await self._check_latest_version()

    async def _check_latest_version(self) -> None:
        """Query PyPI for the latest version and cache the result."""
        try:
            latest = await _fetch_latest_version()
            if latest is None:
                return

            update_available = _is_newer(latest, __version__)
            self._version_info = VersionInfo(
                current=__version__,
                latest=latest,
                update_available=update_available,
                checked_at=utcnow(),
            )

            if update_available:
                logger.info(
                    "NeuralMemory update available: %s -> %s",
                    __version__,
                    latest,
                )
        except Exception:
            logger.debug("Version check failed", exc_info=True)

    def get_update_hint(self) -> dict[str, Any] | None:
        """Return update hint dict if a newer version is available.

        Returns None if no update is available or check hasn't run yet.
        Suitable for injection into tool response dicts.
        """
        if self._version_info is None or not self._version_info.update_available:
            return None

        return {
            "message": (
                f"NeuralMemory v{self._version_info.latest} is available "
                f"(current: v{self._version_info.current}). "
                f"Run: pip install --upgrade neural-memory"
            ),
            "current_version": self._version_info.current,
            "latest_version": self._version_info.latest,
        }

    def cancel_version_check(self) -> None:
        """Cancel the background version check task if running."""
        if self._version_check_task is not None and not self._version_check_task.done():
            self._version_check_task.cancel()
            logger.debug("Version check task cancelled")


async def _fetch_latest_version() -> str | None:
    """Fetch the latest version from PyPI.

    Uses urllib to avoid adding httpx/aiohttp as a dependency.
    Runs the blocking HTTP call in a thread executor.
    """
    import json
    import urllib.request

    def _blocking_fetch() -> str | None:
        try:
            if not _PYPI_URL.startswith("https://"):
                return None
            req = urllib.request.Request(  # noqa: S310
                _PYPI_URL,
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
                version: str | None = data.get("info", {}).get("version")
                return version
        except Exception:
            return None

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _blocking_fetch)


def _log_version_check_exception(task: asyncio.Task[None]) -> None:
    """Log unhandled exceptions from the version check task."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Version check task raised unhandled exception: %s", exc)
