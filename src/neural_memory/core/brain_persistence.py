"""File-based persistence for eternal context state.

Manages a directory structure at ~/.neuralmemory/eternal/<brain_id>/
containing JSON state files for the 3-tier eternal context system.

Directory layout:
    brain.json         Tier 1 — Critical (never deleted)
    session.json       Tier 2 — Important (session-scoped)
    context.json       Tier 3 — Context (temporary)
    session_log.txt    Append-only session log
    snapshots/         Timestamped snapshots (7-day retention default)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neural_memory.unified_config import get_neuralmemory_dir

if TYPE_CHECKING:
    from neural_memory.core.eternal_context import BrainState, ContextSnapshot, SessionState

logger = logging.getLogger(__name__)

# Maximum log file size before rotation (1 MB)
_MAX_LOG_SIZE = 1_000_000


class BrainPersistence:
    """File-based persistence for eternal context state.

    Provides JSON read/write operations for 3-tier state files,
    append-only session logging, and timestamped snapshot management.
    """

    def __init__(self, brain_id: str, base_dir: Path | None = None) -> None:
        self._brain_id = brain_id
        root = base_dir or get_neuralmemory_dir()
        self._dir = root / "eternal" / brain_id
        self._snapshots_dir = self._dir / "snapshots"

    @property
    def directory(self) -> Path:
        """Root directory for this brain's eternal context."""
        return self._dir

    # ──────────────────── Directory setup ────────────────────

    def ensure_dirs(self) -> None:
        """Create directory structure if it doesn't exist."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────── Tier 1: brain.json ────────────────────

    def save_brain_state(self, state: BrainState) -> None:
        """Persist Tier 1 (Critical) state to brain.json."""
        self.ensure_dirs()
        data = _brain_state_to_dict(state)
        self._write_json(self._dir / "brain.json", data)

    def load_brain_state(self) -> BrainState:
        """Load Tier 1 state from brain.json, or return defaults."""
        from neural_memory.core.eternal_context import BrainState

        data = self._read_json(self._dir / "brain.json")
        if data is None:
            return BrainState()
        return _dict_to_brain_state(data)

    # ──────────────────── Tier 2: session.json ────────────────────

    def save_session_state(self, state: SessionState) -> None:
        """Persist Tier 2 (Important) state to session.json."""
        self.ensure_dirs()
        data = _session_state_to_dict(state)
        self._write_json(self._dir / "session.json", data)

    def load_session_state(self) -> SessionState:
        """Load Tier 2 state from session.json, or return defaults."""
        from neural_memory.core.eternal_context import SessionState

        data = self._read_json(self._dir / "session.json")
        if data is None:
            return SessionState()
        return _dict_to_session_state(data)

    # ──────────────────── Tier 3: context.json ────────────────────

    def save_context(self, snapshot: ContextSnapshot) -> None:
        """Persist Tier 3 (Context) state to context.json."""
        self.ensure_dirs()
        data = _context_snapshot_to_dict(snapshot)
        self._write_json(self._dir / "context.json", data)

    def load_context(self) -> ContextSnapshot:
        """Load Tier 3 state from context.json, or return defaults."""
        from neural_memory.core.eternal_context import ContextSnapshot

        data = self._read_json(self._dir / "context.json")
        if data is None:
            return ContextSnapshot()
        return _dict_to_context_snapshot(data)

    # ──────────────────── Session log ────────────────────

    def append_log(self, entry: str) -> None:
        """Append a timestamped entry to session_log.txt."""
        self.ensure_dirs()
        log_path = self._dir / "session_log.txt"

        # Rotate if too large
        if log_path.exists() and log_path.stat().st_size > _MAX_LOG_SIZE:
            rotated = self._dir / "session_log.old.txt"
            if rotated.exists():
                rotated.unlink()
            log_path.rename(rotated)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {entry}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def read_log(self, tail: int = 50) -> list[str]:
        """Read the last N lines from session_log.txt."""
        log_path = self._dir / "session_log.txt"
        if not log_path.exists():
            return []
        lines = log_path.read_text(encoding="utf-8").splitlines()
        return lines[-tail:]

    # ──────────────────── Snapshots ────────────────────

    def create_snapshot(self, brain: BrainState, session: SessionState) -> Path:
        """Create a timestamped snapshot of Tier 1 + 2."""
        self.ensure_dirs()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        snapshot_path = self._snapshots_dir / f"{timestamp}.json"

        data = {
            "created_at": datetime.now().isoformat(),
            "brain_id": self._brain_id,
            "brain": _brain_state_to_dict(brain),
            "session": _session_state_to_dict(session),
        }
        self._write_json(snapshot_path, data)
        return snapshot_path

    def list_snapshots(self) -> list[Path]:
        """List snapshot files sorted by name (oldest first)."""
        if not self._snapshots_dir.exists():
            return []
        return sorted(self._snapshots_dir.glob("*.json"))

    def cleanup_snapshots(self, retention_days: int = 7) -> int:
        """Remove snapshots older than retention_days. Returns count deleted."""
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted = 0
        for path in self.list_snapshots():
            try:
                data = self._read_json(path)
                if data and "created_at" in data:
                    created = datetime.fromisoformat(data["created_at"])
                    if created < cutoff:
                        path.unlink()
                        deleted += 1
                elif path.stat().st_mtime < cutoff.timestamp():
                    path.unlink()
                    deleted += 1
            except (ValueError, OSError) as e:
                logger.warning("Failed to process snapshot %s: %s", path, e)
        return deleted

    def load_snapshot(self, path: Path) -> tuple[BrainState, SessionState]:
        """Load a snapshot file and return (BrainState, SessionState)."""
        from neural_memory.core.eternal_context import BrainState, SessionState

        data = self._read_json(path)
        if data is None:
            return BrainState(), SessionState()

        brain = _dict_to_brain_state(data.get("brain", {}))
        session = _dict_to_session_state(data.get("session", {}))
        return brain, session

    # ──────────────────── Internal helpers ────────────────────

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write dict to JSON file atomically."""
        tmp_path = path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                f.write("\n")
            tmp_path.replace(path)
        except OSError as e:
            logger.error("Failed to write %s: %s", path, e)
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        """Read JSON file, returning None if missing or corrupted."""
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", path, e)
            return None


# ──────────────────── Serialization helpers ────────────────────


def _brain_state_to_dict(state: BrainState) -> dict[str, Any]:
    """Serialize BrainState to dict."""
    return {
        "project_name": state.project_name,
        "tech_stack": list(state.tech_stack),
        "key_decisions": [dict(d) for d in state.key_decisions],
        "instructions": list(state.instructions),
        "version": state.version,
    }


def _dict_to_brain_state(data: dict[str, Any]) -> BrainState:
    """Deserialize dict to BrainState."""
    from neural_memory.core.eternal_context import BrainState

    return BrainState(
        project_name=data.get("project_name", ""),
        tech_stack=tuple(data.get("tech_stack", ())),
        key_decisions=tuple(
            {str(k): str(v) for k, v in d.items()} for d in data.get("key_decisions", ())
        ),
        instructions=tuple(data.get("instructions", ())),
        version=data.get("version", "1.0"),
    )


def _session_state_to_dict(state: SessionState) -> dict[str, Any]:
    """Serialize SessionState to dict."""
    return {
        "feature": state.feature,
        "task": state.task,
        "progress": state.progress,
        "errors_history": [dict(e) for e in state.errors_history],
        "pending_tasks": list(state.pending_tasks),
        "branch": state.branch,
        "commit": state.commit,
        "started_at": state.started_at,
        "updated_at": state.updated_at,
    }


def _dict_to_session_state(data: dict[str, Any]) -> SessionState:
    """Deserialize dict to SessionState."""
    from neural_memory.core.eternal_context import SessionState

    return SessionState(
        feature=data.get("feature", ""),
        task=data.get("task", ""),
        progress=data.get("progress", 0.0),
        errors_history=tuple(data.get("errors_history", ())),
        pending_tasks=tuple(data.get("pending_tasks", ())),
        branch=data.get("branch", ""),
        commit=data.get("commit", ""),
        started_at=data.get("started_at", ""),
        updated_at=data.get("updated_at", ""),
    )


def _context_snapshot_to_dict(snapshot: ContextSnapshot) -> dict[str, Any]:
    """Serialize ContextSnapshot to dict."""
    return {
        "conversation_summary": list(snapshot.conversation_summary),
        "recent_files": list(snapshot.recent_files),
        "recent_queries": list(snapshot.recent_queries),
        "message_count": snapshot.message_count,
        "token_estimate": snapshot.token_estimate,
    }


def _dict_to_context_snapshot(data: dict[str, Any]) -> ContextSnapshot:
    """Deserialize dict to ContextSnapshot."""
    from neural_memory.core.eternal_context import ContextSnapshot

    return ContextSnapshot(
        conversation_summary=tuple(data.get("conversation_summary", ())),
        recent_files=tuple(data.get("recent_files", ())),
        recent_queries=tuple(data.get("recent_queries", ())),
        message_count=data.get("message_count", 0),
        token_estimate=data.get("token_estimate", 0),
    )
