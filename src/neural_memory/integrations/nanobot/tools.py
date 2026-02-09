"""NeuralMemory tools for Nanobot's ToolRegistry.

Four tools conforming to Nanobot's Tool ABC interface:
- nmem_remember — store a memory
- nmem_recall — query with spreading activation
- nmem_context — get recent context
- nmem_health — brain diagnostics
"""

from __future__ import annotations

from typing import Any

from neural_memory.integrations.nanobot.base_tool import BaseNMTool


class NMRememberTool(BaseNMTool):
    """Store a memory in NeuralMemory's neural graph."""

    @property
    def name(self) -> str:
        return "nmem_remember"

    @property
    def description(self) -> str:
        return (
            "Store a memory in NeuralMemory. Use this to remember facts, "
            "decisions, insights, todos, errors, and other information "
            "that should persist across sessions."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to remember",
                },
                "type": {
                    "type": "string",
                    "enum": [
                        "fact",
                        "decision",
                        "preference",
                        "todo",
                        "insight",
                        "context",
                        "instruction",
                        "error",
                        "workflow",
                        "reference",
                    ],
                    "description": "Memory type (auto-detected if not specified)",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Priority 0-10 (5=normal, 10=critical)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
                "expires_days": {
                    "type": "integer",
                    "description": "Days until memory expires",
                },
            },
            "required": ["content"],
        }

    async def execute(self, **kwargs: Any) -> str:
        from neural_memory.core.memory_types import (
            MemoryType,
            Priority,
            TypedMemory,
            get_decay_rate,
            suggest_memory_type,
        )
        from neural_memory.core.neuron import NeuronState
        from neural_memory.engine.encoder import MemoryEncoder
        from neural_memory.safety.sensitive import check_sensitive_content
        from neural_memory.utils.timeutils import utcnow

        content = kwargs.get("content", "")
        if not content:
            return self._json({"error": "Content is required"})

        if len(content) > 100_000:
            return self._json({"error": f"Content too long ({len(content)} chars). Max: 100000."})

        sensitive = check_sensitive_content(content, min_severity=2)
        if sensitive:
            types_found = sorted({m.type.value for m in sensitive})
            return self._json(
                {"error": "Sensitive content detected", "sensitive_types": types_found}
            )

        if "type" in kwargs:
            try:
                mem_type = MemoryType(kwargs["type"])
            except ValueError:
                return self._json({"error": f"Invalid memory type: {kwargs['type']}"})
        else:
            mem_type = suggest_memory_type(content)

        priority = Priority.from_int(kwargs.get("priority", 5))
        tags = set(kwargs.get("tags", []))

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
                tags=tags if tags else None,
            )

            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=mem_type,
                priority=priority,
                source="nanobot_tool",
                expires_in_days=kwargs.get("expires_days"),
                tags=tags if tags else None,
            )
            await storage.add_typed_memory(typed_mem)

            type_decay_rate = get_decay_rate(mem_type.value)
            for neuron in result.neurons_created:
                state = await storage.get_neuron_state(neuron.id)
                if state and state.decay_rate != type_decay_rate:
                    updated = NeuronState(
                        neuron_id=state.neuron_id,
                        activation_level=state.activation_level,
                        access_frequency=state.access_frequency,
                        last_activated=state.last_activated,
                        decay_rate=type_decay_rate,
                        created_at=state.created_at,
                    )
                    await storage.update_neuron_state(updated)

            if hasattr(storage, "batch_save"):
                await storage.batch_save()
        finally:
            if auto_save_disabled and hasattr(storage, "enable_auto_save"):
                storage.enable_auto_save()

        return self._json(
            {
                "success": True,
                "fiber_id": result.fiber.id,
                "memory_type": mem_type.value,
                "neurons_created": len(result.neurons_created),
                "synapses_created": len(result.synapses_created),
                "message": f"Remembered: {content[:80]}{'...' if len(content) > 80 else ''}",
            }
        )


class NMRecallTool(BaseNMTool):
    """Query memories via spreading activation."""

    @property
    def name(self) -> str:
        return "nmem_recall"

    @property
    def description(self) -> str:
        return (
            "Query memories from NeuralMemory using spreading activation. "
            "Use this to recall past information, decisions, patterns, or "
            "context relevant to the current task."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search memories",
                },
                "depth": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                    "description": "Search depth: 0=instant, 1=context, 2=habit, 3=deep",
                },
                "max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10000,
                    "description": "Maximum tokens in response (default: 500)",
                },
                "min_confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Minimum confidence threshold",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
        from neural_memory.utils.timeutils import utcnow

        query = kwargs.get("query", "")
        if not query:
            return self._json({"error": "Query is required"})

        try:
            depth = DepthLevel(kwargs.get("depth", 1))
        except ValueError:
            return self._json({"error": f"Invalid depth: {kwargs.get('depth')}. Must be 0-3."})

        max_tokens = min(kwargs.get("max_tokens", 500), 10_000)
        min_confidence = kwargs.get("min_confidence", 0.0)

        pipeline = ReflexPipeline(self._ctx.storage, self._ctx.config)
        result = await pipeline.query(
            query=query,
            depth=depth,
            max_tokens=max_tokens,
            reference_time=utcnow(),
        )

        if result.confidence < min_confidence:
            return self._json(
                {
                    "answer": None,
                    "message": f"No memories found above confidence {min_confidence}",
                    "confidence": result.confidence,
                }
            )

        return self._json(
            {
                "answer": result.context or "No relevant memories found.",
                "confidence": result.confidence,
                "neurons_activated": result.neurons_activated,
                "fibers_matched": result.fibers_matched,
                "depth_used": result.depth_used.value if result.depth_used else depth.value,
                "tokens_used": result.tokens_used,
            }
        )


class NMContextTool(BaseNMTool):
    """Get recent context from NeuralMemory."""

    @property
    def name(self) -> str:
        return "nmem_context"

    @property
    def description(self) -> str:
        return (
            "Get recent context from NeuralMemory. Use at the start of "
            "tasks to inject relevant recent memories."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent memories (default: 10)",
                },
                "fresh_only": {
                    "type": "boolean",
                    "description": "Only include memories < 30 days old",
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        limit = kwargs.get("limit", 10)
        fresh_only = kwargs.get("fresh_only", False)

        fibers = await self._ctx.storage.get_fibers(
            limit=limit * 2 if fresh_only else limit,
        )

        if not fibers:
            return self._json({"context": "No memories stored yet.", "count": 0})

        if fresh_only:
            from neural_memory.safety.freshness import FreshnessLevel, evaluate_freshness
            from neural_memory.utils.timeutils import utcnow

            now = utcnow()
            fibers = [
                f
                for f in fibers
                if evaluate_freshness(f.created_at, now).level
                in (FreshnessLevel.FRESH, FreshnessLevel.RECENT)
            ][:limit]

        context_parts: list[str] = []
        for fiber in fibers:
            content = fiber.summary
            if not content and fiber.anchor_neuron_id:
                anchor = await self._ctx.storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    content = anchor.content
            if content:
                context_parts.append(f"- {content}")

        context_text = "\n".join(context_parts) if context_parts else "No context available."

        return self._json(
            {
                "context": context_text,
                "count": len(context_parts),
                "tokens_used": len(context_text.split()),
            }
        )


class NMHealthTool(BaseNMTool):
    """Get brain health diagnostics."""

    @property
    def name(self) -> str:
        return "nmem_health"

    @property
    def description(self) -> str:
        return (
            "Get brain health diagnostics including purity score, grade, and actionable warnings."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        from neural_memory.engine.diagnostics import DiagnosticsEngine

        engine = DiagnosticsEngine(self._ctx.storage)
        report = await engine.analyze(self._ctx.brain.id)

        return self._json(
            {
                "brain": self._ctx.brain.name,
                "grade": report.grade,
                "purity_score": round(report.purity_score, 1),
                "connectivity": round(report.connectivity, 3),
                "diversity": round(report.diversity, 3),
                "freshness": round(report.freshness, 3),
                "consolidation_ratio": round(report.consolidation_ratio, 3),
                "orphan_rate": round(report.orphan_rate, 3),
                "warnings": [
                    {"severity": w.severity.value, "code": w.code, "message": w.message}
                    for w in report.warnings
                ],
                "recommendations": list(report.recommendations),
            }
        )
