"""Session management handler for MCP server."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.git_context import detect_git_context

if TYPE_CHECKING:
    from neural_memory.storage.sqlite_store import SQLiteStorage

logger = logging.getLogger(__name__)


class SessionHandler:
    """Mixin: session tracking tool handlers."""

    async def _get_active_session(self, storage: SQLiteStorage) -> dict[str, Any] | None:
        """Get active session metadata, or None if no active session."""
        try:
            sessions = await storage.find_typed_memories(
                memory_type=MemoryType.CONTEXT,
                tags={"session_state"},
                limit=1,
            )
            if sessions and sessions[0].metadata.get("active", True):
                return sessions[0].metadata
        except Exception:
            logger.debug("Failed to get active session", exc_info=True)
        return None

    async def _session(self, args: dict[str, Any]) -> dict[str, Any]:
        """Track current working session state."""
        action = args.get("action", "get")
        storage = await self.get_storage()

        if action == "get":
            return await self._session_get(storage)
        elif action == "set":
            return await self._session_set(args, storage)
        elif action == "end":
            return await self._session_end(storage)
        return {"error": f"Unknown session action: {action}"}

    # ── GET ──

    async def _session_get(self, storage: SQLiteStorage) -> dict[str, Any]:
        """Return current session state."""
        session = await self._find_current_session(storage)
        if not session or not session.metadata.get("active", True):
            return {"active": False, "message": "No active session"}

        meta = session.metadata
        return {
            "active": True,
            "feature": meta.get("feature", ""),
            "task": meta.get("task", ""),
            "progress": meta.get("progress", 0.0),
            "started_at": meta.get("started_at", ""),
            "notes": meta.get("notes", ""),
            "branch": meta.get("branch", ""),
            "commit": meta.get("commit", ""),
            "repo": meta.get("repo", ""),
        }

    # ── SET ──

    async def _session_set(self, args: dict[str, Any], storage: SQLiteStorage) -> dict[str, Any]:
        """Update session state with new metadata."""
        now = datetime.now()
        existing = await self._find_current_session(storage)
        git_ctx = detect_git_context()

        metadata = self._build_session_metadata(args, existing, git_ctx, now)
        content = self._format_session_content(metadata)

        session_tags: set[str] = {"session_state"}
        if git_ctx:
            session_tags.add(f"branch:{git_ctx.branch}")

        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        encoder = MemoryEncoder(storage, brain.config)
        storage.disable_auto_save()

        result = await encoder.encode(content=content, timestamp=now, tags=session_tags)
        typed_mem = TypedMemory.create(
            fiber_id=result.fiber.id,
            memory_type=MemoryType.CONTEXT,
            priority=Priority.from_int(7),
            source="mcp_session",
            expires_in_days=1,
            tags=session_tags,
            metadata=metadata,
        )
        await storage.add_typed_memory(typed_mem)
        await storage.batch_save()

        return {
            "active": True,
            "feature": metadata["feature"],
            "task": metadata["task"],
            "progress": metadata["progress"],
            "started_at": metadata["started_at"],
            "notes": metadata["notes"],
            "branch": metadata.get("branch", ""),
            "commit": metadata.get("commit", ""),
            "repo": metadata.get("repo", ""),
            "message": "Session state updated",
        }

    # ── END ──

    async def _session_end(self, storage: SQLiteStorage) -> dict[str, Any]:
        """End current session and save summary."""
        existing = await self._find_current_session(storage)
        if not existing or not existing.metadata.get("active", True):
            return {"active": False, "message": "No active session to end"}

        feature = existing.metadata.get("feature", "unknown")
        task = existing.metadata.get("task", "")
        progress = existing.metadata.get("progress", 0.0)

        summary = f"Session ended: worked on {feature}"
        if task:
            summary += f", task: {task}"
        summary += f", progress: {int(progress * 100)}%"

        brain = await storage.get_brain(storage._current_brain_id)
        if not brain:
            return {"error": "No brain configured"}

        encoder = MemoryEncoder(storage, brain.config)
        storage.disable_auto_save()
        now = datetime.now()

        # Tombstone so GET returns inactive
        tombstone_result = await encoder.encode(
            content=summary, timestamp=now, tags={"session_state"}
        )
        tombstone_mem = TypedMemory.create(
            fiber_id=tombstone_result.fiber.id,
            memory_type=MemoryType.CONTEXT,
            priority=Priority.from_int(7),
            source="mcp_session",
            expires_in_days=1,
            tags={"session_state"},
            metadata={"active": False, "ended_at": now.isoformat()},
        )
        await storage.add_typed_memory(tombstone_mem)

        # Longer-lived summary for future recall
        summary_result = await encoder.encode(
            content=summary, timestamp=now, tags={"session_summary"}
        )
        summary_mem = TypedMemory.create(
            fiber_id=summary_result.fiber.id,
            memory_type=MemoryType.CONTEXT,
            priority=Priority.from_int(5),
            source="mcp_session",
            expires_in_days=7,
            tags={"session_summary"},
        )
        await storage.add_typed_memory(summary_mem)
        await storage.batch_save()

        return {"active": False, "summary": summary, "message": "Session ended and summary saved"}

    # ── Helpers ──

    async def _find_current_session(self, storage: SQLiteStorage) -> TypedMemory | None:
        """Find the most recent session_state TypedMemory."""
        sessions = await storage.find_typed_memories(
            memory_type=MemoryType.CONTEXT,
            tags={"session_state"},
            limit=1,
        )
        return sessions[0] if sessions else None

    @staticmethod
    def _build_session_metadata(
        args: dict[str, Any],
        existing: TypedMemory | None,
        git_ctx: Any,
        now: datetime,
    ) -> dict[str, Any]:
        """Build session metadata dict from args + existing + git."""
        prev = existing.metadata if existing else {}
        metadata: dict[str, Any] = {
            "active": True,
            "feature": args.get("feature", prev.get("feature", "")),
            "task": args.get("task", prev.get("task", "")),
            "progress": args.get("progress", prev.get("progress", 0.0)),
            "notes": args.get("notes", prev.get("notes", "")),
            "started_at": prev.get("started_at", now.isoformat()),
            "updated_at": now.isoformat(),
        }
        if git_ctx:
            metadata["branch"] = git_ctx.branch
            metadata["commit"] = git_ctx.commit
            metadata["repo"] = git_ctx.repo_name
        return metadata

    @staticmethod
    def _format_session_content(metadata: dict[str, Any]) -> str:
        """Format session metadata into a human-readable summary."""
        content = f"Session: {metadata['feature']}"
        if metadata["task"]:
            content += f" — {metadata['task']}"
        if metadata["progress"]:
            content += f" ({int(metadata['progress'] * 100)}%)"
        return content
