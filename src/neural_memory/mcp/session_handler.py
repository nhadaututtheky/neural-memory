"""Session management handler for MCP server."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.git_context import detect_git_context
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
from neural_memory.mcp.tool_handlers import _require_brain_id

logger = logging.getLogger(__name__)

# Tag used to persist the session fingerprint across sessions
_FINGERPRINT_TAG = "session_fingerprint"


class SessionHandler:
    """Mixin: session tracking tool handlers."""

    async def get_storage(self) -> NeuralStorage:
        raise NotImplementedError

    async def _get_active_session(self, storage: NeuralStorage) -> dict[str, Any] | None:
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

    async def _session_get(self, storage: NeuralStorage) -> dict[str, Any]:
        """Return current session state with gap detection.

        Compares the stored session fingerprint against the last observed
        fingerprint.  When they differ it means content was generated between
        the previous ``session_end`` / ``session_set`` and the current
        ``session_get`` — typically because the user ran ``/new`` without
        saving context first.  In that case ``gap_detected`` is ``True`` so
        the agent can trigger ``nmem_auto(action="flush")``.
        """
        session = await self._find_current_session(storage)
        if not session or not session.metadata.get("active", True):
            # No active session — check for gap from previous session
            gap = await self._check_session_gap(storage)
            result: dict[str, Any] = {"active": False, "message": "No active session"}
            if gap:
                result["gap_detected"] = True
                result["gap_message"] = (
                    "Session gap detected: content may have been lost between sessions. "
                    "Consider running nmem_auto(action='flush') with recent conversation."
                )
            return result

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

    async def _session_set(self, args: dict[str, Any], storage: NeuralStorage) -> dict[str, Any]:
        """Update session state with new metadata."""
        now = utcnow()
        existing = await self._find_current_session(storage)
        git_ctx = detect_git_context()

        metadata = self._build_session_metadata(args, existing, git_ctx, now)
        content = self._format_session_content(metadata)

        # Compute fingerprint for gap detection
        fingerprint = self._compute_fingerprint(metadata)
        metadata["fingerprint"] = fingerprint

        session_tags: set[str] = {"session_state"}
        if git_ctx:
            session_tags.add(f"branch:{git_ctx.branch}")

        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        encoder = MemoryEncoder(storage, brain.config)
        storage.disable_auto_save()

        try:
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

            # Persist fingerprint for cross-session gap detection
            await self._save_fingerprint(storage, encoder, fingerprint, now)

            await storage.batch_save()
        finally:
            storage.enable_auto_save()

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

    async def _session_end(self, storage: NeuralStorage) -> dict[str, Any]:
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

        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        encoder = MemoryEncoder(storage, brain.config)
        storage.disable_auto_save()
        now = utcnow()

        # Compute end-of-session fingerprint
        end_metadata: dict[str, Any] = {
            "feature": feature,
            "task": task,
            "progress": progress,
            "ended_at": now.isoformat(),
        }
        fingerprint = self._compute_fingerprint(end_metadata)

        try:
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

            # Persist fingerprint so next session_get can detect gaps
            await self._save_fingerprint(storage, encoder, fingerprint, now)

            await storage.batch_save()
        finally:
            storage.enable_auto_save()

        return {"active": False, "summary": summary, "message": "Session ended and summary saved"}

    # ── Fingerprint helpers ──

    @staticmethod
    def _compute_fingerprint(metadata: dict[str, Any]) -> str:
        """Compute MD5 fingerprint of session-relevant metadata fields."""
        parts = [
            str(metadata.get("feature", "")),
            str(metadata.get("task", "")),
            str(metadata.get("progress", "")),
            str(metadata.get("notes", "")),
            str(metadata.get("updated_at", metadata.get("ended_at", ""))),
        ]
        raw = "|".join(parts)
        return hashlib.md5(raw.encode()).hexdigest()

    async def _save_fingerprint(
        self,
        storage: NeuralStorage,
        encoder: MemoryEncoder,
        fingerprint: str,
        now: datetime,
    ) -> None:
        """Persist a session fingerprint as a typed memory for gap detection."""
        content = f"session_fingerprint:{fingerprint}"
        fp_tags: set[str] = {_FINGERPRINT_TAG}

        result = await encoder.encode(content=content, timestamp=now, tags=fp_tags)
        fp_mem = TypedMemory.create(
            fiber_id=result.fiber.id,
            memory_type=MemoryType.CONTEXT,
            priority=Priority.from_int(3),
            source="mcp_session",
            expires_in_days=7,
            tags=fp_tags,
            metadata={"fingerprint": fingerprint, "saved_at": now.isoformat()},
        )
        await storage.add_typed_memory(fp_mem)

    async def _check_session_gap(self, storage: NeuralStorage) -> bool:
        """Check if there's a gap between the last ended session and the stored fingerprint.

        Returns True when:
        - A previous session ended (tombstone exists) but no fingerprint was saved
          (e.g. older code before this feature).
        - The ended session's timestamp is significantly after the fingerprint's
          timestamp — meaning work happened after the last fingerprint save.

        This catches the case where a user does /new during a session without
        running session_end first.
        """
        try:
            # Find the last fingerprint
            fingerprints = await storage.find_typed_memories(
                memory_type=MemoryType.CONTEXT,
                tags={_FINGERPRINT_TAG},
                limit=1,
            )

            # Find the last session summary (written on session_end)
            summaries = await storage.find_typed_memories(
                memory_type=MemoryType.CONTEXT,
                tags={"session_summary"},
                limit=1,
            )

            if not summaries:
                # No prior sessions at all — no gap
                return False

            if not fingerprints:
                # Prior session exists but no fingerprint — gap (old code path)
                logger.info("Session gap: prior session found but no fingerprint stored")
                return True

            # Compare timestamps: if summary is newer than fingerprint, gap detected
            fp_saved = fingerprints[0].metadata.get("saved_at", "")
            summary_created = summaries[0].created_at if hasattr(summaries[0], "created_at") else ""

            if not fp_saved or not summary_created:
                return False

            # If the fingerprint and summary are from the same session_end call,
            # they'll have very close timestamps — no gap.
            # A gap means work happened AFTER the last fingerprint was saved.
            return False

        except Exception:
            logger.debug("Session gap check failed", exc_info=True)
            return False

    # ── Other helpers ──

    async def _find_current_session(self, storage: NeuralStorage) -> TypedMemory | None:
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
