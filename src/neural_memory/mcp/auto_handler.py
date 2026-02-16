"""Auto-capture handler for MCP server."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.auto_capture import analyze_text_for_memories
from neural_memory.mcp.constants import MAX_CONTENT_LENGTH

if TYPE_CHECKING:
    from neural_memory.storage.sqlite_store import SQLiteStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class AutoHandler:
    """Mixin: auto-capture tool handlers."""

    config: UnifiedConfig
    _remember: Any

    async def get_storage(self) -> SQLiteStorage:
        raise NotImplementedError

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

        detected = self._run_detection(text)
        if not detected:
            return {"saved": 0, "message": "No memorable content detected"}

        saved = await self._save_detected_memories(detected)
        return {
            "saved": len(saved),
            "memories": saved,
            "message": f"Auto-captured {len(saved)} memories"
            if saved
            else "No memories met confidence threshold",
        }

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

        results = await asyncio.gather(
            *[
                self._remember(
                    {
                        "content": item["content"],
                        "type": item["type"],
                        "priority": item.get("priority", 5),
                    }
                )
                for item in redacted_eligible
            ]
        )
        return [
            item["content"][:50]
            for item, result in zip(redacted_eligible, results, strict=False)
            if "error" not in result
        ]

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
