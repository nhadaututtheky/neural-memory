"""Eternal context handler for MCP server."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from neural_memory.core.eternal_context import EternalContext
from neural_memory.core.memory_types import MemoryType
from neural_memory.core.trigger_engine import check_triggers
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig
from neural_memory.mcp.tool_handler_utils import _require_brain_id

logger = logging.getLogger(__name__)


class EternalHandler:
    """Mixin: eternal context + recap tool handlers."""

    _eternal_ctx: EternalContext | None
    config: UnifiedConfig
    _remember: Any

    async def get_storage(self) -> NeuralStorage:
        raise NotImplementedError

    async def get_eternal_context(self) -> EternalContext:
        """Get or create the eternal context query layer.

        Re-creates the context if the active brain has changed since
        the last call, so brain switches are reflected immediately.
        """
        current_brain = self.config.current_brain
        if self._eternal_ctx is None or self._eternal_ctx._brain_id != current_brain:
            storage = await self.get_storage()
            self._eternal_ctx = EternalContext(storage, current_brain)
        return self._eternal_ctx

    async def _eternal(self, args: dict[str, Any]) -> dict[str, Any]:
        """Manage eternal context — backed by neural graph."""
        action = args.get("action", "status")
        ctx = await self.get_eternal_context()

        if action == "status":
            status = await ctx.get_status()
            usage = await ctx.estimate_context_usage(self.config.eternal.max_context_tokens)
            return {
                "enabled": self.config.eternal.enabled,
                "memory_counts": status["memory_counts"],
                "session": status["session"],
                "message_count": status["message_count"],
                "context_usage": round(usage, 3),
            }

        elif action == "save":
            return await self._eternal_save(args)

        return {"error": f"Unknown eternal action: {action}"}

    async def _eternal_save(self, args: dict[str, Any]) -> dict[str, Any]:
        """Save project context, decisions, instructions into neural graph."""
        storage = await self.get_storage()
        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        saved_items: list[str] = []

        # Project context: dedup by deleting old, then encode new
        if "project_name" in args or "tech_stack" in args:
            old_facts = await storage.find_typed_memories(
                memory_type=MemoryType.FACT, tags={"project_context"}, limit=100
            )
            if old_facts:
                await asyncio.gather(
                    *[storage.delete_typed_memory(old.fiber_id) for old in old_facts]
                )

            parts: list[str] = []
            if args.get("project_name"):
                parts.append(f"Project: {args['project_name']}")
            if args.get("tech_stack"):
                parts.append(f"Tech stack: {', '.join(args['tech_stack'])}")
            if parts:
                await self._remember(
                    {
                        "content": ". ".join(parts),
                        "type": "fact",
                        "priority": 10,
                        "tags": ["project_context", "eternal"],
                    }
                )
                saved_items.append("project_context")

        if "decision" in args:
            content = f"Decision: {args['decision']}"
            reason = args.get("reason", "")
            if reason:
                content += f" — Reason: {reason}"
            await self._remember(
                {
                    "content": content,
                    "type": "decision",
                    "priority": 7,
                    "tags": ["eternal"],
                }
            )
            saved_items.append("decision")

        if "instruction" in args:
            await self._remember(
                {
                    "content": args["instruction"],
                    "type": "instruction",
                    "priority": 9,
                    "tags": ["eternal"],
                }
            )
            saved_items.append("instruction")

        return {
            "saved": True,
            "items": saved_items,
            "message": f"Saved {', '.join(saved_items)}." if saved_items else "No changes.",
        }

    async def _recap(self, args: dict[str, Any]) -> dict[str, Any]:
        """Load saved context for session resumption."""
        ctx = await self.get_eternal_context()
        topic = args.get("topic")

        if topic:
            return await self._recap_topic(ctx, topic)
        return await self._recap_level(ctx, args.get("level", 1))

    async def _recap_topic(self, ctx: EternalContext, topic: str) -> dict[str, Any]:
        """Recap with topic-based search via ReflexPipeline."""
        storage = await self.get_storage()
        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=topic, depth=DepthLevel(1), max_tokens=500, reference_time=utcnow()
        )
        context_text = await ctx.get_injection(level=1)
        if result.context:
            context_text += f"\n\n## Topic: {topic}\n{result.context}"

        # Layered recall: include global brain preferences/instructions
        global_section = await self._get_global_recap_context(topic)
        if global_section:
            context_text += f"\n\n{global_section}"

        response: dict[str, Any] = {
            "context": context_text,
            "topic": topic,
            "confidence": result.confidence,
            "level": 1,
            "message": f"Recap for topic: {topic}",
        }

        # Proactive hints from priming data
        if self.config.proactive.enabled:
            try:
                hints = await self._get_recap_proactive_hints(result, storage)
                if hints:
                    response["proactive_hints"] = hints
            except Exception:
                logger.debug("Proactive hints in recap failed (non-critical)", exc_info=True)

        return response

    async def _recap_level(self, ctx: EternalContext, level: int) -> dict[str, Any]:
        """Recap at a specific detail level (1-3)."""
        level = max(1, min(3, level))
        context_text = await ctx.get_injection(level=level)

        # Layered recall: include global preferences/instructions in recap
        global_section = await self._get_global_recap_context()
        if global_section:
            context_text += f"\n\n{global_section}"

        token_est = int(len(context_text.split()) * 1.3)

        has_feature = False
        try:
            status = await ctx.get_status()
            has_feature = bool(status.get("session", {}).get("feature"))
        except Exception:
            logger.debug("Failed to check session for welcome message", exc_info=True)

        response: dict[str, Any] = {
            "context": context_text,
            "level": level,
            "tokens_used": token_est,
            "message": f"Level {level} recap loaded." + (" Welcome back!" if has_feature else ""),
        }

        # Proactive: predict topics from habit patterns at session start
        if self.config.proactive.enabled:
            try:
                predictions = await self._get_session_predictions()
                if predictions:
                    response["predicted_topics"] = predictions
            except Exception:
                logger.debug("Session predictions failed (non-critical)", exc_info=True)

        return response

    def _fire_eternal_trigger(self, text: str) -> None:
        """Fire-and-forget: check auto-save triggers.

        Data is already persisted in SQLite by _remember()/_recall().
        Triggers now only track message count and log events.
        """
        if not self.config.eternal.enabled:
            return
        try:
            if self._eternal_ctx is None:
                return
            msg_count = self._eternal_ctx.increment_message_count()

            check_triggers(
                text=text,
                message_count=msg_count,
                token_estimate=0,
                max_tokens=self.config.eternal.max_context_tokens,
                checkpoint_interval=self.config.eternal.auto_save_interval,
                warning_threshold=self.config.eternal.context_warning_threshold,
            )
        except Exception:
            logger.debug("Eternal trigger check failed", exc_info=True)

    async def _get_recap_proactive_hints(
        self,
        result: Any,
        storage: Any,
    ) -> list[dict[str, Any]] | None:
        """Extract proactive hints from priming data in recap result."""
        from neural_memory.engine.proactive import (
            format_hints_for_response,
            select_proactive_hints,
        )

        metadata = getattr(result, "metadata", None) or {}
        priming_result = metadata.get("_priming_result")
        if priming_result is None:
            return None

        hints = await select_proactive_hints(
            priming_result=priming_result,
            storage=storage,
            max_hints=self.config.proactive.max_hints,
            max_chars=self.config.proactive.max_hint_chars,
            min_activation=self.config.proactive.min_activation,
        )
        if not hints:
            return None
        return format_hints_for_response(hints)

    async def _get_session_predictions(self) -> list[str] | None:
        """Predict topics for session start using habit patterns.

        Uses BEFORE synapses (topic co-occurrence) to suggest what
        the user might work on, based on historical query patterns.
        Returns topic names or None if insufficient data.
        """
        from neural_memory.engine.session_state import SessionManager

        # Use same session_id format as recall_handler (mcp-{id})
        session_id = f"mcp-{id(self)}"
        session_state = SessionManager.get_instance().get(session_id)

        if session_state is None or session_state.query_count < 3:
            return None

        storage = await self.get_storage()

        from neural_memory.engine.priming import prime_from_habits

        habit_boosts = await prime_from_habits(storage, session_state)
        if not habit_boosts:
            return None

        # Get the top predicted neuron contents
        predictions: list[str] = []
        sorted_boosts = sorted(habit_boosts.items(), key=lambda x: x[1], reverse=True)
        for nid, _level in sorted_boosts[:3]:
            try:
                neuron = await storage.get_neuron(nid)
                if neuron and neuron.content:
                    content = neuron.content[:100]
                    if content not in predictions:
                        predictions.append(content)
            except Exception:
                continue

        return predictions if predictions else None

    async def _get_global_recap_context(self, topic: str | None = None) -> str | None:
        """Fetch global brain preferences/instructions for recap context.

        Queries the _global brain for cross-project preferences and instructions,
        returning a formatted section to append to recap context.

        Args:
            topic: Optional topic to focus the global query on.

        Returns:
            Formatted string with global preferences, or None if no global brain
            or no relevant global memories found.
        """
        try:
            from neural_memory.engine.cross_brain import _query_single_brain
            from neural_memory.unified_config import GLOBAL_BRAIN_NAME

            global_db = self.config.get_brain_db_path(GLOBAL_BRAIN_NAME)
            if not global_db.exists():
                return None

            query = topic or "preferences instructions workflow"
            _, _, _, context = await _query_single_brain(
                db_path=global_db,
                brain_name=GLOBAL_BRAIN_NAME,
                query=query,
                depth=DepthLevel(1),
                max_tokens=200,
            )
            if not context:
                return None

            return f"## [global] Cross-project preferences\n{context}"
        except Exception:
            logger.debug("Global recap context failed (non-critical)", exc_info=True)
            return None
