"""File-based consolidation lock for multi-agent safety.

Prevents concurrent consolidation runs from corrupting the brain.
Uses atomic file creation (O_CREAT|O_EXCL) to prevent TOCTOU races.
Cross-platform PID checking (Windows + Unix).
Per-brain lock files to avoid cross-brain contention.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_STALE_SECONDS = 3600  # 1 hour


def _lock_path(brain_id: str = "") -> Path:
    """Get the lock file path, optionally per-brain."""
    base = Path.home() / ".neuralmemory"
    base.mkdir(parents=True, exist_ok=True)
    safe_name = brain_id.replace("/", "_").replace("\\", "_") if brain_id else ""
    filename = f"consolidation-{safe_name}.lock" if safe_name else "consolidation.lock"
    return base / filename


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with given PID is still running (cross-platform)."""
    if pid <= 0:
        return False

    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except (OSError, AttributeError):
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def _try_clear_stale(lock_file: Path) -> bool:
    """Check if existing lock is stale and clear it if so.

    Returns True if lock was cleared (caller should retry acquire).
    Returns False if lock is actively held.
    """
    try:
        data = json.loads(lock_file.read_text())
        pid = data.get("pid", 0)
        ts = data.get("timestamp", 0)
        age = time.time() - ts

        if _is_pid_alive(pid) and age < _STALE_SECONDS:
            logger.debug("Consolidation lock held by PID %d (age %.0fs)", pid, age)
            return False

        logger.info("Clearing stale consolidation lock (PID %d, age %.0fs)", pid, age)
    except (json.JSONDecodeError, OSError):
        logger.debug("Clearing corrupt consolidation lock")

    try:
        lock_file.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def acquire_consolidation_lock(brain_id: str = "") -> bool:
    """Try to acquire the consolidation lock atomically.

    Uses O_CREAT|O_EXCL for atomic file creation (no TOCTOU race).
    Returns True if lock acquired, False if held by active process.
    Auto-clears stale locks (dead PID or older than 1 hour).
    """
    lock_file = _lock_path(brain_id)
    payload = json.dumps({"pid": os.getpid(), "timestamp": time.time()})

    # Try atomic create first
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, payload.encode())
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        pass
    except OSError:
        logger.error("Failed to create consolidation lock")
        return False

    # Lock exists — check if stale
    if not _try_clear_stale(lock_file):
        return False

    # Stale lock cleared — retry atomic create
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, payload.encode())
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        # Another agent grabbed it between our clear and retry
        return False
    except OSError:
        logger.error("Failed to create consolidation lock on retry")
        return False


def release_consolidation_lock(brain_id: str = "") -> None:
    """Release the consolidation lock."""
    lock_file = _lock_path(brain_id)
    try:
        if lock_file.exists():
            data = json.loads(lock_file.read_text())
            if data.get("pid") == os.getpid():
                lock_file.unlink()
    except (json.JSONDecodeError, OSError):
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass
