"""Auto-capture handler for MCP server."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.auto_capture import analyze_text_for_memories
from neural_memory.mcp.constants import MAX_CONTENT_LENGTH

if TYPE_CHECKING:
    from neural_memory.engine.session_reflection import SessionReflection
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

# Tools whose output is worth passively capturing
_CAPTURABLE_TOOLS: frozenset[str] = frozenset(
    {"nmem_recall", "nmem_context", "nmem_recap", "nmem_explain"}
)
# Rate limit: max passive saves per window
_PASSIVE_CAPTURE_MAX_PER_WINDOW = 3
_PASSIVE_CAPTURE_WINDOW_SECS = 60.0


class AutoHandler:
    """Mixin: auto-capture tool handlers."""

    config: UnifiedConfig
    _remember: Any
    _passive_capture_timestamps: list[float]
    _session_memories: list[dict[str, Any]]  # Accumulated auto-captured memories

    async def get_storage(self) -> NeuralStorage:
        raise NotImplementedError

    def _get_session_id(self) -> str:
        """Get current session ID (consistent with recall_handler)."""
        return f"mcp-{id(self)}"

    async def _auto(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle auto-capture settings and analysis."""
        action = args.get("action", "status")

        if action == "status":
            return self._auto_status()
        elif action == "enable":
            return self._auto_toggle(enabled=True)
        elif action == "disable":
            return self._auto_toggle(enabled=False)
        elif action == "analyze":
            return await self._auto_analyze(args, save=args.get("save", False))
        elif action == "process":
            return await self._auto_process(args)
        elif action == "flush":
            return await self._auto_flush(args)
        return {"error": f"Unknown action: {action}"}

    def _auto_status(self) -> dict[str, Any]:
        """Return current auto-capture settings."""
        return {
            "enabled": self.config.auto.enabled,
            "capture_decisions": self.config.auto.capture_decisions,
            "capture_errors": self.config.auto.capture_errors,
            "capture_todos": self.config.auto.capture_todos,
            "capture_facts": self.config.auto.capture_facts,
            "capture_insights": self.config.auto.capture_insights,
            "capture_preferences": self.config.auto.capture_preferences,
            "min_confidence": self.config.auto.min_confidence,
        }

    def _auto_toggle(self, *, enabled: bool) -> dict[str, Any]:
        """Enable or disable auto-capture."""
        from dataclasses import replace

        self.config = replace(self.config, auto=replace(self.config.auto, enabled=enabled))
        self.config.save()
        return {
            "enabled": enabled,
            "message": f"Auto-capture {'enabled' if enabled else 'disabled'}",
        }

    async def _auto_analyze(self, args: dict[str, Any], *, save: bool) -> dict[str, Any]:
        """Analyze text for capturable patterns."""
        text = args.get("text", "")
        if not text:
            return {"error": "Text required for analyze action"}
        if len(text) > MAX_CONTENT_LENGTH:
            return {"error": f"Text too long ({len(text)} chars). Max: {MAX_CONTENT_LENGTH}."}

        from neural_memory.safety.input_firewall import check_content

        fw = check_content(text)
        if fw.blocked:
            logger.debug("Auto-analyze: input firewall blocked — %s", fw.reason)
            return {"detected": [], "message": f"Input blocked: {fw.reason}"}
        if fw.sanitized:
            text = fw.sanitized

        detected = self._run_detection(text)
        if not detected:
            return {"detected": [], "message": "No memorable content detected"}

        if save:
            saved = await self._save_detected_memories(detected)
            return {
                "detected": detected,
                "saved": saved,
                "message": f"Analyzed and saved {len(saved)} memories",
            }
        return {
            "detected": detected,
            "message": f"Detected {len(detected)} potential memories (not saved)",
        }

    async def _auto_process(self, args: dict[str, Any]) -> dict[str, Any]:
        """Process text and auto-save detected memories."""
        if not self.config.auto.enabled:
            return {
                "saved": 0,
                "message": "Auto-capture is disabled. Use nmem_auto(action='enable') to enable.",
            }

        text = args.get("text", "")
        if not text:
            return {"error": "Text required for process action"}
        if len(text) > MAX_CONTENT_LENGTH:
            return {"error": f"Text too long ({len(text)} chars). Max: {MAX_CONTENT_LENGTH}."}

        from neural_memory.safety.input_firewall import check_content

        fw = check_content(text)
        if fw.blocked:
            logger.debug("Auto-process: input firewall blocked — %s", fw.reason)
            return {"saved": 0, "message": f"Input blocked: {fw.reason}"}
        if fw.sanitized:
            text = fw.sanitized

        detected = self._run_detection(text)
        if not detected:
            return {"saved": 0, "message": "No memorable content detected"}

        saved = await self._save_detected_memories(detected)

        # Cleanup expired ephemeral neurons at session end
        ephemeral_cleaned = 0
        try:
            storage = await self.get_storage()
            result = await storage.cleanup_ephemeral_neurons(max_age_hours=24.0)
            ephemeral_cleaned = int(result) if isinstance(result, (int, float)) else 0
            if ephemeral_cleaned > 0:
                logger.info("Cleaned up %d expired ephemeral neurons", ephemeral_cleaned)
        except Exception:
            logger.debug("Ephemeral cleanup skipped (non-critical)", exc_info=True)

        # Session-end reflection: analyze session memories for patterns
        reflection_result = None
        try:
            reflection_result = await self._run_session_reflection()
        except Exception:
            logger.debug("Session reflection skipped (non-critical)", exc_info=True)

        # Regenerate Knowledge Surface after processing session summary
        surface_regenerated = False
        try:
            await self._regenerate_surface_if_available()
            surface_regenerated = True
        except Exception:
            logger.debug("Surface regeneration skipped (non-critical)", exc_info=True)

        response: dict[str, Any] = {
            "saved": len(saved),
            "memories": saved,
            "surface_regenerated": surface_regenerated,
            "message": f"Auto-captured {len(saved)} memories"
            if saved
            else "No memories met confidence threshold",
        }
        if ephemeral_cleaned > 0:
            response["ephemeral_cleaned"] = ephemeral_cleaned
        if reflection_result and reflection_result.patterns_found > 0:
            response["reflection"] = {
                "summary": reflection_result.summary,
                "patterns_found": reflection_result.patterns_found,
                "insights_saved": len(reflection_result.pattern_neurons),
                "session_stats": reflection_result.session_stats,
            }
        return response

    async def _run_session_reflection(self) -> SessionReflection | None:
        """Run session-end reflection and save pattern neurons.

        Analyzes in-memory session memories, detects patterns, and saves
        insight/workflow/decision neurons.

        Returns:
            SessionReflection result, or None if reflection was skipped.
        """
        from neural_memory.engine.session_reflection import reflect_on_session
        from neural_memory.engine.session_state import SessionManager

        if not hasattr(self, "_session_memories"):
            self._session_memories = []

        session_id = self._get_session_id()

        # Get session state for topics and query count
        session_state = SessionManager.get_instance().get(session_id)
        session_topics: list[str] = []
        query_count = 0
        if session_state is not None:
            session_topics = session_state.get_top_topics(limit=5)
            query_count = session_state.query_count

        reflection = reflect_on_session(
            memories=self._session_memories,
            session_topics=session_topics,
            query_count=query_count,
        )

        if not reflection.pattern_neurons:
            return reflection

        # Save reflection summary neuron
        session_tag = f"session:{session_id}"
        await self._remember(
            {
                "content": reflection.summary,
                "type": "workflow",
                "priority": 6,
                "tags": [session_tag, "reflection"],
                "_auto_capture": True,
            }
        )

        # Save pattern neurons (insights, decisions from detected patterns)
        for pn in reflection.pattern_neurons:
            await self._remember(
                {
                    "content": pn["content"],
                    "type": pn["type"],
                    "priority": pn["priority"],
                    "tags": [session_tag, "reflection"],
                    "_auto_capture": True,
                }
            )

        logger.info(
            "Session reflection: %d patterns found, %d neurons saved",
            reflection.patterns_found,
            len(reflection.pattern_neurons) + 1,
        )

        # Clear session memories after reflection to prevent unbounded growth
        # and duplicate reflections on subsequent process calls
        self._session_memories = []

        return reflection

    async def _regenerate_surface_if_available(self) -> None:
        """Regenerate Knowledge Surface if the feature is available."""
        from neural_memory.surface.lifecycle import regenerate_surface

        storage = await self.get_storage()
        brain_id = getattr(storage, "brain_id", None)
        if not brain_id:
            return

        brain = await storage.get_brain(brain_id)
        brain_name = brain.name if brain else "default"

        await regenerate_surface(storage=storage, brain_name=brain_name)

        # Invalidate cached surface so next session picks up changes
        if hasattr(self, "_surface_text"):
            self._surface_text = ""
            self._surface_brain = ""

    async def _auto_flush(self, args: dict[str, Any]) -> dict[str, Any]:
        """Emergency flush: aggressive capture before context is lost.

        Designed to be called before compaction or session end.
        Skips dedup, lowers confidence threshold, captures all memory types.
        """
        text = args.get("text", "")
        if not text:
            return {"error": "Text required for flush action"}
        if len(text) > MAX_CONTENT_LENGTH:
            return {"error": f"Text too long ({len(text)} chars). Max: {MAX_CONTENT_LENGTH}."}

        from neural_memory.safety.input_firewall import check_content

        fw = check_content(text)
        if fw.blocked:
            logger.debug("Auto-flush: input firewall blocked — %s", fw.reason)
            return {"saved": 0, "message": f"Input blocked: {fw.reason}"}
        if fw.sanitized:
            text = fw.sanitized

        # Run detection with ALL types enabled regardless of config
        detected = analyze_text_for_memories(
            text,
            capture_decisions=True,
            capture_errors=True,
            capture_todos=True,
            capture_facts=True,
            capture_insights=True,
            capture_preferences=True,
        )

        if not detected:
            return {"saved": 0, "message": "No memorable content detected in flush"}

        # Emergency mode: lower confidence threshold to 0.5 (vs normal min_confidence)
        emergency_threshold = 0.5
        eligible = [item for item in detected if item["confidence"] >= emergency_threshold]

        if not eligible:
            return {"saved": 0, "message": "No memories met emergency threshold (0.5)"}

        # Boost priority for emergency-captured memories
        boosted = [{**item, "priority": min(item.get("priority", 5) + 2, 10)} for item in eligible]

        saved = await self._save_detected_memories_no_dedup(boosted)
        return {
            "saved": len(saved),
            "memories": saved,
            "mode": "emergency_flush",
            "threshold": emergency_threshold,
            "message": f"Emergency flush: captured {len(saved)} memories"
            if saved
            else "No memories saved",
        }

    async def _save_detected_memories_no_dedup(self, detected: list[dict[str, Any]]) -> list[str]:
        """Save detected memories WITHOUT dedup checks. For emergency flush only.

        Auto-redacts high-severity sensitive content before saving.
        """
        from neural_memory.safety.sensitive import auto_redact_content

        auto_redact_severity = self.config.safety.auto_redact_min_severity
        redacted: list[dict[str, Any]] = []
        for item in detected:
            content = item["content"]
            redacted_content, matches, _ = auto_redact_content(
                content, min_severity=auto_redact_severity
            )
            if matches:
                logger.debug("Auto-redacted %d matches in flush memory", len(matches))
            redacted.append({**item, "content": redacted_content})

        # Tag with session ID for session-end reflection
        session_tag = f"session:{self._get_session_id()}"

        results = await asyncio.gather(
            *[
                self._remember(
                    {
                        "content": item["content"],
                        "type": item["type"],
                        "priority": item.get("priority", 5),
                        "tags": ["emergency_flush", session_tag],
                        "_auto_capture": True,
                    }
                )
                for item in redacted
            ]
        )
        # Track saved memories for session-end reflection
        if not hasattr(self, "_session_memories"):
            self._session_memories = []
        for item, result in zip(redacted, results, strict=False):
            if "error" not in result:
                self._session_memories.append(item)

        return [
            item["content"][:50]
            for item, result in zip(redacted, results, strict=False)
            if "error" not in result
        ]

    async def _passive_capture(self, text: str) -> None:
        """Silently analyze text and capture high-confidence memories."""
        try:
            detected = self._run_detection(text)
            if detected:
                type_thresholds = {
                    "error": 0.7,
                    "decision": 0.75,
                    "insight": 0.75,
                    "preference": 0.75,
                }
                high_confidence = [
                    item
                    for item in detected
                    if item["confidence"]
                    >= max(self.config.auto.min_confidence, type_thresholds.get(item["type"], 0.8))
                ]
                if high_confidence:
                    await self._save_detected_memories(high_confidence)
        except Exception:
            logger.debug("Passive capture failed", exc_info=True)

    async def _post_tool_capture(
        self, tool_name: str, args: dict[str, Any], result_text: str
    ) -> None:
        """Silently capture memories from capturable tool call results.

        Called after tool execution for tools in ``_CAPTURABLE_TOOLS``.
        Rate-limited to ``_PASSIVE_CAPTURE_MAX_PER_WINDOW`` saves per minute
        to avoid noise.

        Args:
            tool_name: Name of the tool that was called.
            args: The tool's input arguments.
            result_text: The JSON-serialized tool result string.
        """
        if tool_name not in _CAPTURABLE_TOOLS:
            return
        if not isinstance(result_text, str) or len(result_text) < 50:
            return

        # Rate limit check
        now = time.monotonic()
        if not hasattr(self, "_passive_capture_timestamps"):
            self._passive_capture_timestamps = []
        # Prune old timestamps outside the window
        self._passive_capture_timestamps = [
            ts for ts in self._passive_capture_timestamps if now - ts < _PASSIVE_CAPTURE_WINDOW_SECS
        ]
        if len(self._passive_capture_timestamps) >= _PASSIVE_CAPTURE_MAX_PER_WINDOW:
            return

        try:
            # Capture from query text (agent often embeds decisions in queries)
            query_text = args.get("query", "") or args.get("text", "") or args.get("topic", "")
            texts_to_analyze: list[str] = []
            if isinstance(query_text, str) and len(query_text) >= 30:
                texts_to_analyze.append(query_text)
            # Capture from result (truncated to avoid huge payloads)
            truncated_result = result_text[:MAX_CONTENT_LENGTH]
            if len(truncated_result) >= 50:
                texts_to_analyze.append(truncated_result)

            from neural_memory.safety.input_firewall import check_content

            for text in texts_to_analyze:
                fw = check_content(text)
                if fw.blocked:
                    logger.debug("Passive capture: firewall blocked — %s", fw.reason)
                    continue
                if fw.sanitized:
                    text = fw.sanitized
                detected = self._run_detection(text)
                if not detected:
                    continue
                type_thresholds = {
                    "error": 0.7,
                    "decision": 0.75,
                    "insight": 0.75,
                    "preference": 0.75,
                }
                high_confidence = [
                    {**item, "tags": ["passive_capture"]}
                    for item in detected
                    if item["confidence"]
                    >= max(
                        self.config.auto.min_confidence,
                        type_thresholds.get(item["type"], 0.8),
                    )
                ]
                if high_confidence:
                    await self._save_detected_memories(high_confidence)
                    self._passive_capture_timestamps.append(now)
                    logger.debug(
                        "Post-tool passive capture: saved %d memories from %s",
                        len(high_confidence),
                        tool_name,
                    )
                    return  # One save per call is enough
        except Exception:
            logger.debug("Post-tool passive capture failed", exc_info=True)

    async def _save_detected_memories(self, detected: list[dict[str, Any]]) -> list[str]:
        """Save detected memories that meet confidence threshold.

        Auto-redacts high-severity sensitive content before saving.
        """
        eligible = [
            item for item in detected if item["confidence"] >= self.config.auto.min_confidence
        ]
        if not eligible:
            return []

        # Auto-redact before saving
        from neural_memory.safety.sensitive import auto_redact_content

        auto_redact_severity = self.config.safety.auto_redact_min_severity
        redacted_eligible: list[dict[str, Any]] = []
        for item in eligible:
            content = item["content"]
            redacted, matches, _ = auto_redact_content(content, min_severity=auto_redact_severity)
            if matches:
                logger.debug("Auto-redacted %d matches in auto-captured memory", len(matches))
            redacted_eligible.append({**item, "content": redacted})

        # Significance scoring (amygdala boost) — adjust priorities before saving
        if self.config.proactive.significance_enabled:
            redacted_eligible = await self._apply_significance(redacted_eligible)

        # Tag with session ID for session-end reflection
        session_tag = f"session:{self._get_session_id()}"

        results = await asyncio.gather(
            *[
                self._remember(
                    {
                        "content": item["content"],
                        "type": item["type"],
                        "priority": item.get("priority", 5),
                        "tags": [session_tag],
                        "_auto_capture": True,
                    }
                )
                for item in redacted_eligible
            ]
        )

        # Track saved memories for session-end reflection
        if not hasattr(self, "_session_memories"):
            self._session_memories = []
        for item, result in zip(redacted_eligible, results, strict=False):
            if "error" not in result:
                self._session_memories.append(item)

        return [
            item["content"][:50]
            for item, result in zip(redacted_eligible, results, strict=False)
            if "error" not in result
        ]

    async def _apply_significance(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score items for significance and adjust priorities.

        Near-duplicate items (surprise=0.0) are filtered out entirely.
        Other items get priority boosts based on significance signals.
        """
        from neural_memory.engine.significance import score_significance

        try:
            storage = await self.get_storage()
            brain_id = storage.brain_id
            if not brain_id:
                return items
            brain = await storage.get_brain(brain_id)
            if not brain:
                return items
        except Exception:
            logger.debug("Significance: storage unavailable, skipping scoring", exc_info=True)
            return items

        config = self.config.proactive
        scored: list[dict[str, Any]] = []
        for item in items:
            try:
                sig = await score_significance(
                    content=item["content"],
                    detected_type=item["type"],
                    base_priority=item.get("priority", 5),
                    storage=storage,
                    brain_config=brain.config,
                    correction_boost=config.correction_boost,
                    contradiction_boost=config.contradiction_boost,
                    novelty_boost=config.novelty_boost,
                )

                # Skip near-duplicates (adjusted priority <= 2 after penalty)
                if sig.adjusted_priority <= 2:
                    logger.debug(
                        "Significance: skipping near-duplicate: %s",
                        item["content"][:50],
                    )
                    continue

                new_item = {
                    **item,
                    "priority": sig.adjusted_priority,
                    "_significance": sig.to_metadata(),
                }

                # Tag contradictions for proactive surfacing
                if sig.is_contradiction:
                    existing_tags = list(item.get("tags", []))
                    existing_tags.append("contradiction")
                    new_item["tags"] = existing_tags

                scored.append(new_item)
            except Exception:
                logger.debug("Significance scoring failed for item", exc_info=True)
                scored.append(item)  # keep original on failure

        return scored

    def _run_detection(self, text: str) -> list[dict[str, Any]]:
        """Run pattern detection with current config."""
        return analyze_text_for_memories(
            text,
            capture_decisions=self.config.auto.capture_decisions,
            capture_errors=self.config.auto.capture_errors,
            capture_todos=self.config.auto.capture_todos,
            capture_facts=self.config.auto.capture_facts,
            capture_insights=self.config.auto.capture_insights,
            capture_preferences=self.config.auto.capture_preferences,
        )
