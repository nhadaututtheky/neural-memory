"""Drop-in replacement for Nanobot's MemoryStore, backed by NeuralMemory.

Same public interface as ``nanobot.agent.memory.MemoryStore``::

    get_today_file() -> Path
    read_today() -> str
    append_today(content: str)
    read_long_term() -> str
    write_long_term(content: str)
    get_recent_memories(days=7) -> str
    list_memory_files() -> list[Path]
    get_memory_context() -> str

All methods are async. The original Nanobot MemoryStore is sync —
callers must await these methods when using NMMemoryStore.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.integrations.nanobot.context import NMContext

logger = logging.getLogger(__name__)


class NMMemoryStore:
    """NeuralMemory-backed memory store for Nanobot.

    Args:
        ctx: Shared NMContext with initialized storage and brain.
        workspace: Nanobot workspace directory.
    """

    def __init__(self, ctx: NMContext, workspace: Path) -> None:
        self._ctx = ctx
        self._workspace = workspace

    def get_today_file(self) -> Path:
        """Return synthetic path for today's memory file."""
        today = utcnow().strftime("%Y-%m-%d")
        return self._workspace / "memory" / f"{today}.md"

    async def read_today(self) -> str:
        """Read today's memories as timestamped markdown lines."""
        today_start = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        fibers = await self._ctx.storage.get_fibers(limit=200)

        today_fibers = [f for f in fibers if f.created_at >= today_start]
        if not today_fibers:
            return ""

        parts: list[str] = []
        for fiber in today_fibers:
            content = await self._get_fiber_content(fiber)
            if content:
                ts = fiber.created_at.strftime("%H:%M")
                parts.append(f"- [{ts}] {content}")

        return "\n".join(parts)

    async def append_today(self, content: str) -> None:
        """Store content as a new memory in the neural graph."""
        from neural_memory.core.memory_types import (
            Priority,
            TypedMemory,
            suggest_memory_type,
        )
        from neural_memory.engine.encoder import MemoryEncoder

        storage = self._ctx.storage
        encoder = MemoryEncoder(storage, self._ctx.config)
        auto_save_disabled = False
        if hasattr(storage, "disable_auto_save"):
            storage.disable_auto_save()
            auto_save_disabled = True

        try:
            result = await encoder.encode(
                content=content,
                timestamp=utcnow(),
            )

            mem_type = suggest_memory_type(content)
            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=mem_type,
                priority=Priority.NORMAL,
                source="nanobot_store",
            )
            await storage.add_typed_memory(typed_mem)
            if hasattr(storage, "batch_save"):
                await storage.batch_save()
        finally:
            if auto_save_disabled and hasattr(storage, "enable_auto_save"):
                storage.enable_auto_save()

    async def read_long_term(self) -> str:
        """Read long-term memories via spreading activation recall."""
        from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline

        pipeline = ReflexPipeline(self._ctx.storage, self._ctx.config)
        result = await pipeline.query(
            query="important long-term knowledge, key decisions, persistent instructions",
            depth=DepthLevel.CONTEXT,
            max_tokens=self._ctx.config.max_context_tokens,
            reference_time=utcnow(),
        )
        return result.context or ""

    async def write_long_term(self, content: str) -> None:
        """Store content as high-priority persistent memory."""
        from neural_memory.core.memory_types import (
            MemoryType,
            Priority,
            TypedMemory,
        )
        from neural_memory.engine.encoder import MemoryEncoder

        storage = self._ctx.storage
        encoder = MemoryEncoder(storage, self._ctx.config)
        auto_save_disabled = False
        if hasattr(storage, "disable_auto_save"):
            storage.disable_auto_save()
            auto_save_disabled = True

        try:
            result = await encoder.encode(
                content=content,
                timestamp=utcnow(),
                tags={"long-term", "persistent"},
            )

            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=MemoryType.FACT,
                priority=Priority.HIGH,
                source="nanobot_store",
                tags={"long-term", "persistent"},
            )
            await storage.add_typed_memory(typed_mem)
            if hasattr(storage, "batch_save"):
                await storage.batch_save()
        finally:
            if auto_save_disabled and hasattr(storage, "enable_auto_save"):
                storage.enable_auto_save()

    async def get_recent_memories(self, days: int = 7) -> str:
        """Get memories from the last N days as markdown."""
        cutoff = utcnow() - timedelta(days=days)
        fibers = await self._ctx.storage.get_fibers(limit=200)

        recent = [f for f in fibers if f.created_at >= cutoff]
        if not recent:
            return ""

        parts: list[str] = []
        for fiber in recent:
            content = await self._get_fiber_content(fiber)
            if content:
                date_str = fiber.created_at.strftime("%Y-%m-%d %H:%M")
                parts.append(f"- [{date_str}] {content}")

        return "\n".join(parts)

    def list_memory_files(self) -> list[Path]:
        """Return empty list — NeuralMemory uses SQLite, not files."""
        return []

    async def get_memory_context(self) -> str:
        """Get formatted context for system prompt injection.

        Called by Nanobot's ContextBuilder to inject memory into
        the system prompt alongside identity and skills.
        """
        sections: list[str] = []

        recent = await self.get_recent_memories(days=3)
        if recent:
            sections.append(f"## Recent Memories\n{recent}")

        long_term = await self.read_long_term()
        if long_term:
            sections.append(f"## Key Knowledge\n{long_term}")

        if not sections:
            return ""

        return "# NeuralMemory Context\n\n" + "\n\n".join(sections)

    async def _get_fiber_content(self, fiber: object) -> str | None:
        """Extract display content from a fiber."""
        content = getattr(fiber, "summary", None)
        if not content:
            anchor_id = getattr(fiber, "anchor_neuron_id", None)
            if anchor_id:
                anchor = await self._ctx.storage.get_neuron(anchor_id)
                if anchor:
                    content = anchor.content
        return content
