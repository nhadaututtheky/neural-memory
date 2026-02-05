"""Non-blocking update checker for the CLI.

Checks PyPI for newer versions of neural-memory, cached for 24 hours.
Runs in background thread to avoid slowing down CLI commands.

Cache location: ~/.neuralmemory/.update_check
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

PYPI_URL = "https://pypi.org/pypi/neural-memory/json"
CHECK_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours
REQUEST_TIMEOUT_SECONDS = 3


def _get_cache_path() -> Path:
    """Get path to update check cache file."""
    data_dir = os.environ.get("NEURALMEMORY_DIR") or str(
        Path.home() / ".neuralmemory"
    )
    return Path(data_dir) / ".update_check"


def _read_cache() -> dict[str, Any]:
    """Read cached update check result."""
    try:
        cache_path = _get_cache_path()
        if not cache_path.exists():
            return {}
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_cache(data: dict[str, Any]) -> None:
    """Write update check result to cache."""
    try:
        cache_path = _get_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse version string into comparable tuple."""
    parts: list[int] = []
    for part in version_str.split(".")[:3]:
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def _is_newer(remote: str, local: str) -> bool:
    """Check if remote version is newer than local."""
    return _parse_version(remote) > _parse_version(local)


def _fetch_latest_version() -> str | None:
    """Fetch latest version from PyPI. Returns None on failure."""
    try:
        from urllib.request import Request, urlopen

        req = Request(
            PYPI_URL,
            headers={"Accept": "application/json", "User-Agent": "neural-memory-cli"},
        )
        with urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read())
            return data.get("info", {}).get("version")
    except Exception:
        return None


def _check_and_notify() -> None:
    """Background worker: check PyPI and print update notice if needed."""
    try:
        from neural_memory import __version__

        cache = _read_cache()
        last_check = cache.get("last_check", 0)

        # Throttle: skip if checked recently
        if time.time() - last_check < CHECK_INTERVAL_SECONDS:
            # Still show cached notification if not dismissed
            cached_version = cache.get("latest_version")
            if (
                cached_version
                and _is_newer(cached_version, __version__)
                and not cache.get("dismissed")
            ):
                _print_update_notice(__version__, cached_version)
            return

        latest = _fetch_latest_version()
        if not latest:
            return

        # Update cache
        new_cache: dict[str, Any] = {
            "last_check": time.time(),
            "latest_version": latest,
            "dismissed": False,
        }
        _write_cache(new_cache)

        if _is_newer(latest, __version__):
            _print_update_notice(__version__, latest)
    except Exception:
        pass


def _print_update_notice(current: str, latest: str) -> None:
    """Print a styled update notice to stderr (doesn't pollute stdout)."""
    import sys

    notice = (
        f"\n  Update available: neural-memory {current} → {latest}\n"
        f"  Run: pip install -U neural-memory\n"
    )
    # Use stderr so it doesn't break piped output (e.g. nmem recall ... | jq)
    print(notice, file=sys.stderr, flush=True)


def run_update_check_background() -> None:
    """Launch update check in a daemon thread. Non-blocking, fire-and-forget."""
    thread = threading.Thread(target=_check_and_notify, daemon=True)
    thread.start()
