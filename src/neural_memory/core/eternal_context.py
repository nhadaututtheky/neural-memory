"""Async query layer over the neural graph for session continuity.

Queries TypedMemory entries in SQLiteStorage to assemble context
injection strings. All persistence happens through the existing
MemoryEncoder + TypedMemory pipeline, making eternal context
data discoverable by spreading activation (ReflexPipeline).

Replaces the old 3-tier JSON file persistence (brain.json,
session.json, context.json) with direct neural graph queries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import MemoryType, Priority

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

_TOKEN_RATIO = 1.3


class EternalContext:
    """Async query layer over neural graph for session continuity.

    Instead of loading/saving JSON files, this class queries TypedMemory
    entries in storage to assemble context injection strings. All data
    lives in the neural graph as fibers + typed memories, making it
    discoverable by spreading activation.
    """

    def __init__(self, storage: NeuralStorage, brain_id: str) -> None:
        self._storage = storage
        self._brain_id = brain_id
        self._message_count = 0

    @property
    def message_count(self) -> int:
        """Current in-memory message count for this session."""
        return self._message_count

    def increment_message_count(self) -> int:
        """Increment in-memory message counter. Returns new count."""
        self._message_count += 1
        return self._message_count

    # ──────────────────── Context injection ────────────────────

    async def get_injection(self, level: int = 1) -> str:
        """Query TypedMemory by type and format for context injection.

        Args:
            level: Loading level.
                1 = instant (~500 tokens): project, instructions, session, errors
                2 = on-demand (~1500 tokens): + decisions, todos
                3 = deep (~3000+ tokens): + session summaries, recent facts

        Returns:
            Formatted context string for system prompt injection.
        """
        parts: list[str] = []

        # ── Level 1: Always loaded ──
        parts.append("## Project Context")

        # Project context facts
        project_facts = await self._storage.find_typed_memories(
            memory_type=MemoryType.FACT,
            tags={"project_context"},
            limit=5,
        )
        for mem in project_facts:
            content = await self._get_memory_content(mem.fiber_id)
            if content:
                parts.append(f"- {content}")

        # Instructions (high priority)
        instructions = await self._storage.find_typed_memories(
            memory_type=MemoryType.INSTRUCTION,
            min_priority=Priority.HIGH,
            limit=5,
        )
        for mem in instructions:
            content = await self._get_memory_content(mem.fiber_id)
            if content:
                parts.append(f"- {content}")

        # Current session state
        sessions = await self._storage.find_typed_memories(
            memory_type=MemoryType.CONTEXT,
            tags={"session_state"},
            limit=1,
        )
        if sessions:
            meta = sessions[0].metadata or {}
            if meta.get("active", True):
                feature = meta.get("feature", "")
                task = meta.get("task", "")
                progress = meta.get("progress", 0.0)
                if feature:
                    parts.append(f"- Current: {feature}")
                if task:
                    task_line = f"- Task: {task}"
                    pct = int(progress * 100) if progress else 0
                    if pct > 0:
                        task_line += f" ({pct}%)"
                    parts.append(task_line)
                branch = meta.get("branch", "")
                if branch:
                    parts.append(f"- Branch: {branch}")

        # Active errors
        errors = await self._storage.find_typed_memories(
            memory_type=MemoryType.ERROR,
            limit=10,
        )
        active_errors = [e for e in errors if not e.is_expired]
        if active_errors:
            parts.append(f"- Active errors: {len(active_errors)}")
            for err in active_errors[-3:]:
                content = await self._get_memory_content(err.fiber_id)
                if content:
                    parts.append(f"  - {content}")

        if level < 2:
            return "\n".join(parts)

        # ── Level 2: On-demand ──
        decisions = await self._storage.find_typed_memories(
            memory_type=MemoryType.DECISION,
            limit=10,
        )
        if decisions:
            parts.append("\n## Key Decisions")
            for d in decisions[-5:]:
                content = await self._get_memory_content(d.fiber_id)
                if content:
                    parts.append(f"- {content}")

        todos = await self._storage.find_typed_memories(
            memory_type=MemoryType.TODO,
            limit=10,
        )
        if todos:
            parts.append("\n## Pending Tasks")
            for t in todos[-5:]:
                content = await self._get_memory_content(t.fiber_id)
                if content:
                    parts.append(f"- {content}")

        all_instructions = await self._storage.find_typed_memories(
            memory_type=MemoryType.INSTRUCTION,
            limit=10,
        )
        if all_instructions:
            parts.append("\n## Instructions")
            for inst in all_instructions[-5:]:
                content = await self._get_memory_content(inst.fiber_id)
                if content:
                    parts.append(f"- {content}")

        if level < 3:
            return "\n".join(parts)

        # ── Level 3: Deep dive ──
        summaries = await self._storage.find_typed_memories(
            memory_type=MemoryType.CONTEXT,
            tags={"session_summary"},
            limit=10,
        )
        if summaries:
            parts.append("\n## Session History")
            for s in summaries:
                content = await self._get_memory_content(s.fiber_id)
                if content:
                    parts.append(f"- {content}")

        recent_facts = await self._storage.find_typed_memories(
            memory_type=MemoryType.FACT,
            limit=10,
        )
        # Exclude project_context (already shown in L1)
        project_ids = {m.fiber_id for m in project_facts}
        for f in recent_facts[:5]:
            if f.fiber_id not in project_ids:
                content = await self._get_memory_content(f.fiber_id)
                if content:
                    parts.append(f"- {content}")

        all_errors = await self._storage.find_typed_memories(
            memory_type=MemoryType.ERROR,
            limit=10,
        )
        if all_errors:
            parts.append("\n## Error History")
            for err in all_errors[-10:]:
                content = await self._get_memory_content(err.fiber_id)
                status = "open" if not err.is_expired else "resolved"
                if content:
                    parts.append(f"- [{status}] {content}")

        return "\n".join(parts)

    # ──────────────────── Status ────────────────────

    async def get_status(self) -> dict[str, Any]:
        """Query TypedMemory counts by type for status display."""
        counts: dict[str, int] = {}
        for mem_type in MemoryType:
            memories = await self._storage.find_typed_memories(
                memory_type=mem_type,
                limit=1000,
            )
            counts[mem_type.value] = len(memories)

        # Current session info
        sessions = await self._storage.find_typed_memories(
            memory_type=MemoryType.CONTEXT,
            tags={"session_state"},
            limit=1,
        )
        session_info: dict[str, Any] = {}
        if sessions:
            meta = sessions[0].metadata or {}
            if meta.get("active", True):
                session_info = {
                    "feature": meta.get("feature", ""),
                    "task": meta.get("task", ""),
                    "progress": meta.get("progress", 0.0),
                    "branch": meta.get("branch", ""),
                }

        return {
            "memory_counts": counts,
            "session": session_info,
            "message_count": self._message_count,
        }

    # ──────────────────── Capacity estimation ────────────────────

    async def estimate_context_usage(self, max_tokens: int = 128_000) -> float:
        """Estimate fraction of context window used (0.0 to 1.0)."""
        if max_tokens <= 0:
            return 0.0
        injection = await self.get_injection(level=3)
        injection_tokens = int(len(injection.split()) * _TOKEN_RATIO)
        session_tokens = self._message_count * 150
        return min(1.0, (injection_tokens + session_tokens) / max_tokens)

    # ──────────────────── Internal helpers ────────────────────

    async def _get_memory_content(self, fiber_id: str) -> str | None:
        """Get human-readable content from a fiber.

        Tries fiber.summary first, then falls back to anchor neuron content.
        """
        try:
            fiber = await self._storage.get_fiber(fiber_id)
            if fiber is None:
                return None
            if fiber.summary:
                return fiber.summary
            if fiber.anchor_neuron_id:
                neuron = await self._storage.get_neuron(fiber.anchor_neuron_id)
                if neuron:
                    return neuron.content
        except Exception as e:
            logger.debug("Failed to get content for fiber %s: %s", fiber_id, e)
        return None
