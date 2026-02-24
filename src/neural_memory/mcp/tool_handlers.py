"""MCP tool handler implementations.

Extracted from server.py to keep file sizes manageable.
Each method handles one MCP tool call (nmem_*).

The ToolHandler mixin is inherited by MCPServer in server.py.
All methods access storage/config via self.get_storage() and self.config
from the MCPServer base class.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import (
    MemoryType,
    Priority,
    TypedMemory,
    get_decay_rate,
    suggest_memory_type,
)
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.hooks import HookEvent
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.mcp.constants import MAX_CONTENT_LENGTH
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.engine.hooks import HookRegistry
    from neural_memory.mcp.maintenance_handler import HealthPulse
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


def _require_brain_id(storage: NeuralStorage) -> str:
    """Return the current brain ID or raise ValueError if not set."""
    brain_id = storage._current_brain_id
    if not brain_id:
        raise ValueError("No brain context set")
    return brain_id


class ToolHandler:
    """Mixin providing all MCP tool handler implementations.

    Protocol stubs for attributes/methods used from MCPServer.
    """

    if TYPE_CHECKING:
        config: UnifiedConfig
        hooks: HookRegistry

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

        def _fire_eternal_trigger(self, content: str) -> None:
            raise NotImplementedError

        async def _check_maintenance(self) -> HealthPulse | None:
            raise NotImplementedError

        def _get_maintenance_hint(self, pulse: HealthPulse | None) -> str | None:
            raise NotImplementedError

        async def _passive_capture(self, text: str) -> None:
            raise NotImplementedError

        async def _get_active_session(self, storage: NeuralStorage) -> dict[str, Any] | None:
            raise NotImplementedError

        async def _check_onboarding(self) -> dict[str, Any] | None:
            raise NotImplementedError

        def get_update_hint(self) -> dict[str, Any] | None:
            raise NotImplementedError

    # ──────────────────── Core tool handlers ────────────────────

    async def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
        """Store a memory in the neural graph."""
        storage = await self.get_storage()
        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        content = args["content"]
        if len(content) > MAX_CONTENT_LENGTH:
            return {"error": f"Content too long ({len(content)} chars). Max: {MAX_CONTENT_LENGTH}."}

        # Check for sensitive content with selective auto-redaction
        from neural_memory.safety.sensitive import auto_redact_content, check_sensitive_content

        try:
            auto_redact_severity = int(self.config.safety.auto_redact_min_severity)
        except (TypeError, ValueError, AttributeError):
            auto_redact_severity = 3
        redacted_content, redacted_matches, content_hash = auto_redact_content(
            content, min_severity=auto_redact_severity
        )

        if redacted_matches:
            # Content was auto-redacted — use redacted version
            content = redacted_content
            logger.info(
                "Auto-redacted %d sensitive matches (severity >= %d)",
                len(redacted_matches),
                auto_redact_severity,
            )

        # Check for remaining sensitive content below auto-redact threshold
        remaining_matches = check_sensitive_content(content, min_severity=2)
        # Filter out matches that were already redacted
        remaining_matches = [m for m in remaining_matches if m.severity < auto_redact_severity]
        if remaining_matches:
            types_found = sorted({m.type.value for m in remaining_matches})
            return {
                "error": "Sensitive content detected",
                "sensitive_types": types_found,
                "message": "Content contains potentially sensitive information. "
                "Remove secrets before storing.",
            }

        # Determine if content should be encrypted
        should_encrypt = args.get("encrypted", False)
        encrypted_content: str | None = None
        encryption_meta: dict[str, Any] = {}

        try:
            encryption_cfg = self.config.encryption
            encryption_enabled = encryption_cfg.enabled
        except AttributeError:
            encryption_enabled = False

        if encryption_enabled:
            # Auto-encrypt if sensitive content was detected in original input
            if not should_encrypt and getattr(encryption_cfg, "auto_encrypt_sensitive", True):
                from neural_memory.safety.sensitive import (
                    check_sensitive_content as _check_sensitive,
                )

                original_matches = _check_sensitive(args["content"], min_severity=2)
                if original_matches:
                    should_encrypt = True

            if should_encrypt:
                try:
                    from pathlib import Path

                    from neural_memory.safety.encryption import MemoryEncryptor

                    brain_id = _require_brain_id(storage)
                    keys_dir_str = getattr(encryption_cfg, "keys_dir", "")
                    keys_dir = (
                        Path(keys_dir_str) if keys_dir_str else (self.config.data_dir / "keys")
                    )

                    encryptor = MemoryEncryptor(keys_dir=keys_dir)
                    enc_result = encryptor.encrypt(content, brain_id)
                    encrypted_content = enc_result.ciphertext
                    encryption_meta = {
                        "encrypted": True,
                        "key_id": enc_result.key_id,
                        "algorithm": enc_result.algorithm,
                    }
                    logger.info("Encrypted memory content for brain %s", brain_id)
                except Exception:
                    logger.warning("Encryption failed, storing plaintext", exc_info=True)

        # Determine memory type
        if "type" in args:
            try:
                mem_type = MemoryType(args["type"])
            except ValueError:
                return {"error": f"Invalid memory type: {args['type']}"}
        else:
            mem_type = suggest_memory_type(content)

        priority = Priority.from_int(args.get("priority", 5))

        # Build dedup pipeline if enabled
        dedup_pipeline = None
        try:
            dedup_settings = self.config.dedup
            if isinstance(dedup_settings.enabled, bool) and dedup_settings.enabled:
                from neural_memory.engine.dedup.config import DedupConfig
                from neural_memory.engine.dedup.pipeline import DedupPipeline

                dedup_cfg = DedupConfig(
                    enabled=True,
                    simhash_threshold=int(dedup_settings.simhash_threshold),
                    embedding_threshold=float(dedup_settings.embedding_threshold),
                    embedding_ambiguous_low=float(dedup_settings.embedding_ambiguous_low),
                    llm_enabled=bool(dedup_settings.llm_enabled),
                    llm_provider=str(dedup_settings.llm_provider),
                    llm_model=str(dedup_settings.llm_model),
                    llm_max_pairs_per_encode=int(dedup_settings.llm_max_pairs_per_encode),
                    merge_strategy=str(dedup_settings.merge_strategy),
                )

                # Create LLM judge if enabled
                llm_judge = None
                if dedup_cfg.llm_enabled and dedup_cfg.llm_provider != "none":
                    from neural_memory.engine.dedup.llm_judge import create_judge

                    llm_judge = create_judge(dedup_cfg.llm_provider, dedup_cfg.llm_model)

                dedup_pipeline = DedupPipeline(
                    config=dedup_cfg,
                    storage=storage,
                    llm_judge=llm_judge,
                )
        except (AttributeError, TypeError, ValueError):
            dedup_pipeline = None

        encoder = MemoryEncoder(storage, brain.config, dedup_pipeline=dedup_pipeline)

        await self.hooks.emit(HookEvent.PRE_REMEMBER, {"content": content, "type": mem_type.value})

        storage.disable_auto_save()

        try:
            raw_tags = args.get("tags", [])
            if len(raw_tags) > 50:
                return {"error": f"Too many tags ({len(raw_tags)}). Max: 50."}
            tags = set()
            for t in raw_tags:
                if isinstance(t, str) and len(t) <= 100:
                    tags.add(t)
            encode_content = encrypted_content if encrypted_content is not None else content
            result = await encoder.encode(
                content=encode_content, timestamp=utcnow(), tags=tags if tags else None
            )

            # Attach encryption metadata to fiber
            if encryption_meta:
                from dataclasses import replace as dc_replace

                updated_meta = {**result.fiber.metadata, **encryption_meta}
                updated_fiber = dc_replace(result.fiber, metadata=updated_meta)
                result = dc_replace(result, fiber=updated_fiber)

            import os

            _source = os.environ.get("NEURALMEMORY_SOURCE", "")[:256]
            mcp_source = f"mcp:{_source}" if _source else "mcp_tool"

            expiry_days = args.get("expires_days")
            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=mem_type,
                priority=priority,
                source=mcp_source,
                expires_in_days=expiry_days,
                tags=tags if tags else None,
            )
            await storage.add_typed_memory(typed_mem)

            # Set type-specific decay rate on neuron states
            type_decay_rate = get_decay_rate(mem_type.value)
            for neuron in result.neurons_created:
                state = await storage.get_neuron_state(neuron.id)
                if state and state.decay_rate != type_decay_rate:
                    from neural_memory.core.neuron import NeuronState

                    updated_state = NeuronState(
                        neuron_id=state.neuron_id,
                        activation_level=state.activation_level,
                        access_frequency=state.access_frequency,
                        last_activated=state.last_activated,
                        decay_rate=type_decay_rate,
                        created_at=state.created_at,
                    )
                    await storage.update_neuron_state(updated_state)

            await storage.batch_save()
        finally:
            storage.enable_auto_save()

        # Auto-schedule high-priority fibers for spaced repetition
        if priority.value >= 7:
            try:
                from neural_memory.engine.spaced_repetition import SpacedRepetitionEngine

                sr_engine = SpacedRepetitionEngine(storage, brain.config)
                await sr_engine.auto_schedule_fiber(result.fiber.id, brain.id)
            except Exception:
                logger.debug("Auto-schedule for review failed (non-critical)", exc_info=True)

        self._fire_eternal_trigger(content)

        await self._record_tool_action("remember", content[:100])

        pulse = await self._check_maintenance()

        await self.hooks.emit(
            HookEvent.POST_REMEMBER,
            {
                "fiber_id": result.fiber.id,
                "content": content,
                "type": mem_type.value,
                "neurons_created": len(result.neurons_created),
                "conflicts_detected": result.conflicts_detected,
            },
        )

        response: dict[str, Any] = {
            "success": True,
            "fiber_id": result.fiber.id,
            "memory_type": mem_type.value,
            "neurons_created": len(result.neurons_created),
            "message": f"Remembered: {content[:50]}{'...' if len(content) > 50 else ''}",
        }

        if redacted_matches:
            response["auto_redacted"] = True
            response["auto_redacted_count"] = len(redacted_matches)

        if encryption_meta:
            response["encrypted"] = True

        if expiry_days is not None:
            response["expires_in_days"] = expiry_days

        try:
            conflicts_detected = int(result.conflicts_detected)
        except (TypeError, ValueError, AttributeError):
            conflicts_detected = 0
        if conflicts_detected > 0:
            response["conflicts_detected"] = conflicts_detected
            response["message"] += f" ({conflicts_detected} conflict(s) detected)"

        hint = self._get_maintenance_hint(pulse)
        if hint:
            response["maintenance_hint"] = hint

        update_hint = self.get_update_hint()
        if update_hint:
            response["update_hint"] = update_hint

        # Related memory discovery via 2-hop spreading activation
        try:
            anchor_id = result.fiber.anchor_neuron_id
            if anchor_id:
                from neural_memory.engine.activation import SpreadingActivation

                activator = SpreadingActivation(storage, brain.config)
                activations = await activator.activate(
                    anchor_neurons=[anchor_id],
                    max_hops=2,
                    min_activation=0.05,
                )

                # Pre-filter: only keep hop>0 candidates, sort by activation
                # descending, cap to top 20 to limit I/O from get_neurons_batch
                candidates = sorted(
                    (
                        ar
                        for ar in activations.values()
                        if ar.hop_distance > 0 and ar.neuron_id != anchor_id
                    ),
                    key=lambda ar: ar.activation_level,
                    reverse=True,
                )[:20]

                candidate_ids = [c.neuron_id for c in candidates]

                if candidate_ids:
                    related_neurons = await storage.get_neurons_batch(candidate_ids)
                    anchor_neurons = {
                        nid: n for nid, n in related_neurons.items() if n.metadata.get("is_anchor")
                    }

                    if anchor_neurons:
                        # Take top 3 anchor neurons by activation level
                        sorted_anchors = sorted(
                            anchor_neurons.keys(),
                            key=lambda nid: activations[nid].activation_level,
                            reverse=True,
                        )[:3]

                        # Map anchor neurons to their fibers
                        fibers = await storage.find_fibers_batch(sorted_anchors)
                        fiber_by_anchor: dict[str, Any] = {}
                        for fiber in fibers:
                            if (
                                fiber.anchor_neuron_id in anchor_neurons
                                and fiber.id != result.fiber.id
                            ):
                                fiber_by_anchor.setdefault(fiber.anchor_neuron_id, fiber)

                        related_memories = []
                        for nid in sorted_anchors:
                            related_fiber = fiber_by_anchor.get(nid)
                            if related_fiber:
                                preview = (
                                    related_fiber.summary or anchor_neurons[nid].content or ""
                                )[:100]
                                related_memories.append(
                                    {
                                        "fiber_id": related_fiber.id,
                                        "preview": preview,
                                        "similarity": round(activations[nid].activation_level, 2),
                                    }
                                )

                        if related_memories:
                            response["related_memories"] = related_memories
        except Exception:
            logger.warning("Related memory discovery failed (non-critical)", exc_info=True)

        # Onboarding hint for fresh brains
        onboarding = await self._check_onboarding()
        if onboarding:
            response["onboarding"] = onboarding

        # Surface pending alerts count
        alert_info = await self._surface_pending_alerts()  # type: ignore[attr-defined]
        if alert_info:
            response.update(alert_info)

        return response

    async def _recall(self, args: dict[str, Any]) -> dict[str, Any]:
        """Query memories via spreading activation."""
        # Cross-brain recall: early return if brains parameter is provided
        brain_names = args.get("brains")
        if brain_names and isinstance(brain_names, list) and len(brain_names) > 0:
            return await self._cross_brain_recall(args, brain_names)

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        query = args["query"]
        try:
            depth = DepthLevel(args.get("depth", 1))
        except ValueError:
            return {"error": f"Invalid depth level: {args.get('depth')}. Must be 0-3."}
        max_tokens = min(args.get("max_tokens", 500), 10_000)
        min_confidence = args.get("min_confidence", 0.0)

        # Inject session context for richer recall on vague queries
        effective_query = query
        try:
            session = await self._get_active_session(storage)
            if session and isinstance(session, dict):
                session_terms: list[str] = []
                feature = session.get("feature", "")
                task = session.get("task", "")
                if isinstance(feature, str) and feature:
                    session_terms.append(feature)
                if isinstance(task, str) and task:
                    session_terms.append(task)
                if session_terms and len(query.split()) < 8:
                    effective_query = f"{query} [context: {', '.join(session_terms)}]"
        except Exception:
            logger.debug("Session context injection failed", exc_info=True)

        # Parse optional temporal filter
        valid_at = None
        if "valid_at" in args:
            try:
                valid_at = datetime.fromisoformat(args["valid_at"])
            except (ValueError, TypeError):
                return {"error": f"Invalid valid_at datetime: {args['valid_at']}"}

        await self.hooks.emit(HookEvent.PRE_RECALL, {"query": query, "depth": depth.value})

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=effective_query,
            depth=depth,
            max_tokens=max_tokens,
            reference_time=utcnow(),
            valid_at=valid_at,
        )

        # Passive auto-capture on long queries
        if self.config.auto.enabled and len(query) >= 50:
            await self._passive_capture(query)

        self._fire_eternal_trigger(query)

        if result.confidence < min_confidence:
            return {
                "answer": None,
                "message": f"No memories found with confidence >= {min_confidence}",
                "confidence": result.confidence,
            }

        response: dict[str, Any] = {
            "answer": result.context or "No relevant memories found.",
            "confidence": result.confidence,
            "neurons_activated": result.neurons_activated,
            "fibers_matched": result.fibers_matched,
            "depth_used": result.depth_used.value,
            "tokens_used": result.tokens_used,
        }

        if result.score_breakdown is not None:
            response["score_breakdown"] = {
                "base_activation": round(result.score_breakdown.base_activation, 4),
                "intersection_boost": round(result.score_breakdown.intersection_boost, 4),
                "freshness_boost": round(result.score_breakdown.freshness_boost, 4),
                "frequency_boost": round(result.score_breakdown.frequency_boost, 4),
            }

        # Surface conflict info from retrieval
        disputed_ids: list[str] = (result.metadata or {}).get("disputed_ids", [])
        if disputed_ids:
            response["has_conflicts"] = True
            response["conflict_count"] = len(disputed_ids)

            # Full conflict details only when opt-in
            if args.get("include_conflicts"):
                neurons_map = await storage.get_neurons_batch(disputed_ids)
                response["conflicts"] = [
                    {
                        "existing_neuron_id": nid,
                        "content": n.content[:200] if n else "",
                        "status": "superseded"
                        if n and n.metadata.get("_superseded")
                        else "disputed",
                    }
                    for nid, n in neurons_map.items()
                    if n is not None
                ]

        # Expiry warnings (opt-in)
        warn_expiry_days = args.get("warn_expiry_days")
        if warn_expiry_days is not None and result.fibers_matched:
            try:
                expiring = await storage.get_expiring_memories_for_fibers(
                    fiber_ids=result.fibers_matched,
                    within_days=int(warn_expiry_days),
                )
                if expiring:
                    response["expiry_warnings"] = [
                        {
                            "fiber_id": tm.fiber_id,
                            "memory_type": tm.memory_type.value,
                            "days_until_expiry": tm.days_until_expiry,
                            "priority": tm.priority.value,
                            "suggestion": "Re-store this memory if still relevant, or set a new expires_days.",
                        }
                        for tm in expiring
                    ]
            except Exception:
                logger.debug("Expiry warning check failed", exc_info=True)

        await self._record_tool_action("recall", query[:100])

        pulse = await self._check_maintenance()
        hint = self._get_maintenance_hint(pulse)
        if hint:
            response["maintenance_hint"] = hint

        update_hint = self.get_update_hint()
        if update_hint:
            response["update_hint"] = update_hint

        await self.hooks.emit(
            HookEvent.POST_RECALL,
            {
                "query": query,
                "confidence": result.confidence,
                "neurons_activated": result.neurons_activated,
                "fibers_matched": result.fibers_matched,
            },
        )

        # Suggest related queries from learned patterns
        try:
            from neural_memory.engine.query_pattern_mining import (
                extract_topics,
                suggest_follow_up_queries,
            )

            topics = extract_topics(query)
            if topics:
                related = await suggest_follow_up_queries(storage, topics, brain.config)
                if related:
                    response["related_queries"] = related
        except Exception:
            logger.debug("Query pattern suggestion failed", exc_info=True)

        # Onboarding hint for fresh brains
        onboarding = await self._check_onboarding()
        if onboarding:
            response["onboarding"] = onboarding

        # Surface pending alerts count
        alert_info = await self._surface_pending_alerts()  # type: ignore[attr-defined]
        if alert_info:
            response.update(alert_info)

        return response

    async def _cross_brain_recall(
        self, args: dict[str, Any], brain_names: list[str]
    ) -> dict[str, Any]:
        """Handle cross-brain recall by querying multiple brains in parallel."""
        from neural_memory.engine.cross_brain import cross_brain_recall

        query = args.get("query", "")
        if not query:
            return {"error": "query is required"}

        # Cap at 5 brains
        brain_names = brain_names[:5]
        depth = args.get("depth", 1)
        max_tokens = min(args.get("max_tokens", 500), 10_000)

        try:
            result = await cross_brain_recall(
                config=self.config,
                brain_names=brain_names,
                query=query,
                depth=depth,
                max_tokens=max_tokens,
            )
        except Exception:
            logger.error("Cross-brain recall failed", exc_info=True)
            return {"error": "Cross-brain recall failed"}

        fibers_out = [
            {
                "fiber_id": f.fiber_id,
                "source_brain": f.source_brain,
                "summary": f.summary,
                "confidence": f.confidence,
            }
            for f in result.fibers
        ]

        return {
            "answer": result.merged_context,
            "brains_queried": result.brains_queried,
            "total_neurons_activated": result.total_neurons_activated,
            "fibers": fibers_out,
            "cross_brain": True,
        }

    async def _context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get recent context."""
        storage = await self.get_storage()

        limit = min(args.get("limit", 10), 200)
        fresh_only = args.get("fresh_only", False)

        fibers = await storage.get_fibers(limit=limit * 2 if fresh_only else limit)
        if not fibers:
            result: dict[str, Any] = {"context": "No memories stored yet.", "count": 0}
            onboarding = await self._check_onboarding()
            if onboarding:
                result["onboarding"] = onboarding
            return result

        if fresh_only:
            from neural_memory.safety.freshness import FreshnessLevel, evaluate_freshness

            now = utcnow()
            fresh_fibers = [
                f
                for f in fibers
                if evaluate_freshness(f.created_at, now).level
                in (FreshnessLevel.FRESH, FreshnessLevel.RECENT)
            ]
            fibers = fresh_fibers[:limit]

        # Smart context optimization: score, dedup, budget
        from neural_memory.engine.context_optimizer import optimize_context

        try:
            max_tokens = int(self.config.brain.max_context_tokens)
            if max_tokens < 100:
                max_tokens = 4000
        except (TypeError, ValueError, AttributeError):
            max_tokens = 4000
        plan = await optimize_context(storage, fibers, max_tokens)

        if plan.items:
            context_parts = [f"- {item.content}" for item in plan.items]
            context_text = "\n".join(context_parts)
        else:
            context_text = "No context available."

        await self._record_tool_action("context")

        response: dict[str, Any] = {
            "context": context_text,
            "count": len(plan.items),
            "tokens_used": plan.total_tokens,
        }

        if plan.dropped_count > 0:
            response["optimization_stats"] = {
                "items_dropped": plan.dropped_count,
                "top_score": round(plan.items[0].score, 4) if plan.items else 0.0,
            }

        # Expiry warnings (opt-in)
        warn_expiry_days = args.get("warn_expiry_days")
        if warn_expiry_days is not None and fibers:
            try:
                fiber_ids = [f.id for f in fibers]
                expiring = await storage.get_expiring_memories_for_fibers(
                    fiber_ids=fiber_ids,
                    within_days=int(warn_expiry_days),
                )
                if expiring:
                    response["expiry_warnings"] = [
                        {
                            "fiber_id": tm.fiber_id,
                            "memory_type": tm.memory_type.value,
                            "days_until_expiry": tm.days_until_expiry,
                            "priority": tm.priority.value,
                            "suggestion": "Re-store this memory if still relevant, or set a new expires_days.",
                        }
                        for tm in expiring
                    ]
            except Exception:
                logger.debug("Expiry warning check failed", exc_info=True)

        # Surface pending alerts count
        alert_info = await self._surface_pending_alerts()  # type: ignore[attr-defined]
        if alert_info:
            response.update(alert_info)

        return response

    async def _todo(self, args: dict[str, Any]) -> dict[str, Any]:
        """Add a TODO."""
        return await self._remember(
            {
                "content": args["task"],
                "type": "todo",
                "priority": args.get("priority", 5),
                "expires_days": 30,
            }
        )

    async def _stats(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get brain statistics."""
        storage = await self.get_storage()
        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        stats = await storage.get_enhanced_stats(brain.id)

        # Count active conflicts (unresolved CONTRADICTS synapses)
        conflicts_active = 0
        try:
            from neural_memory.core.synapse import SynapseType

            contradicts_synapses = await storage.get_synapses(type=SynapseType.CONTRADICTS)
            conflicts_active = sum(
                1 for s in contradicts_synapses if not s.metadata.get("_resolved")
            )
        except Exception:
            logger.debug("Conflict count failed (non-critical)", exc_info=True)

        response = {
            "brain": brain.name,
            "neuron_count": stats["neuron_count"],
            "synapse_count": stats["synapse_count"],
            "fiber_count": stats["fiber_count"],
            "db_size_bytes": stats.get("db_size_bytes", 0),
            "today_fibers_count": stats.get("today_fibers_count", 0),
            "hot_neurons": stats.get("hot_neurons", []),
            "newest_memory": stats.get("newest_memory"),
            "conflicts_active": conflicts_active,
        }

        # Onboarding hint for fresh brains
        onboarding = await self._check_onboarding()
        if onboarding:
            response["onboarding"] = onboarding

        update_hint = self.get_update_hint()
        if update_hint:
            response["update_hint"] = update_hint

        return response

    async def _health(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run brain health diagnostics."""
        storage = await self.get_storage()
        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        from neural_memory.engine.diagnostics import DiagnosticsEngine

        engine = DiagnosticsEngine(storage)
        report = await engine.analyze(brain.id)

        return {
            "brain": brain.name,
            "grade": report.grade,
            "purity_score": report.purity_score,
            "connectivity": report.connectivity,
            "diversity": report.diversity,
            "freshness": report.freshness,
            "consolidation_ratio": report.consolidation_ratio,
            "orphan_rate": report.orphan_rate,
            "activation_efficiency": report.activation_efficiency,
            "recall_confidence": report.recall_confidence,
            "neuron_count": report.neuron_count,
            "synapse_count": report.synapse_count,
            "fiber_count": report.fiber_count,
            "warnings": [
                {"severity": w.severity.value, "code": w.code, "message": w.message}
                for w in report.warnings
            ],
            "recommendations": list(report.recommendations),
            "top_penalties": [
                {
                    "component": p.component,
                    "current_score": p.current_score,
                    "weight": p.weight,
                    "penalty_points": p.penalty_points,
                    "estimated_gain": p.estimated_gain,
                    "action": p.action,
                }
                for p in report.top_penalties
            ],
        }

    async def _evolution(self, args: dict[str, Any]) -> dict[str, Any]:
        """Measure brain evolution dynamics."""
        storage = await self.get_storage()
        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        from neural_memory.engine.brain_evolution import EvolutionEngine

        try:
            engine = EvolutionEngine(storage)
            evo = await engine.analyze(brain.id)
        except Exception:
            logger.error("Evolution analysis failed", exc_info=True)
            return {"error": "Evolution analysis failed"}

        result: dict[str, Any] = {
            "brain": evo.brain_name,
            "proficiency_level": evo.proficiency_level.value,
            "proficiency_index": evo.proficiency_index,
            "maturity_level": evo.maturity_level,
            "plasticity": evo.plasticity,
            "density": evo.density,
            "activity_score": evo.activity_score,
            "semantic_ratio": evo.semantic_ratio,
            "reinforcement_days": evo.reinforcement_days,
            "topology_coherence": evo.topology_coherence,
            "plasticity_index": evo.plasticity_index,
            "knowledge_density": evo.knowledge_density,
            "total_neurons": evo.total_neurons,
            "total_synapses": evo.total_synapses,
            "total_fibers": evo.total_fibers,
            "fibers_at_semantic": evo.fibers_at_semantic,
            "fibers_at_episodic": evo.fibers_at_episodic,
        }

        if evo.stage_distribution is not None:
            result["stage_distribution"] = {
                "short_term": evo.stage_distribution.short_term,
                "working": evo.stage_distribution.working,
                "episodic": evo.stage_distribution.episodic,
                "semantic": evo.stage_distribution.semantic,
                "total": evo.stage_distribution.total,
            }

        if evo.closest_to_semantic:
            result["closest_to_semantic"] = [
                {
                    "fiber_id": p.fiber_id,
                    "stage": p.stage,
                    "days_in_stage": p.days_in_stage,
                    "days_required": p.days_required,
                    "reinforcement_days": p.reinforcement_days,
                    "reinforcement_required": p.reinforcement_required,
                    "progress_pct": p.progress_pct,
                    "next_step": p.next_step,
                }
                for p in evo.closest_to_semantic
            ]

        return result

    async def _suggest(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get prefix-based autocomplete suggestions."""
        storage = await self.get_storage()
        prefix = args.get("prefix", "")
        if not prefix.strip():
            return {"suggestions": [], "count": 0}

        limit = min(args.get("limit", 5), 20)
        type_filter = None
        if "type_filter" in args:
            from neural_memory.core.neuron import NeuronType

            try:
                type_filter = NeuronType(args["type_filter"])
            except ValueError:
                return {"error": "Invalid type_filter value"}

        suggestions = await storage.suggest_neurons(
            prefix=prefix, type_filter=type_filter, limit=limit
        )
        formatted = [
            {
                "content": s["content"],
                "type": s["type"],
                "neuron_id": s["neuron_id"],
                "score": s["score"],
            }
            for s in suggestions
        ]
        return {
            "suggestions": formatted,
            "count": len(formatted),
            "tokens_used": sum(len(s["content"].split()) for s in formatted),
        }

    async def _habits(self, args: dict[str, Any]) -> dict[str, Any]:
        """Manage learned workflow habits."""
        storage = await self.get_storage()
        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        action = args.get("action", "list")

        if action == "suggest":
            current_action = args.get("current_action", "")
            if not current_action:
                return {"error": "current_action is required for suggest"}

            from neural_memory.engine.workflow_suggest import suggest_next_action

            suggestions = await suggest_next_action(storage, current_action, brain.config)
            return {
                "suggestions": [
                    {
                        "action": s.action_type,
                        "confidence": round(s.confidence, 4),
                        "source_habit": s.source_habit,
                        "sequential_count": s.sequential_count,
                    }
                    for s in suggestions
                ],
                "count": len(suggestions),
            }

        elif action == "list":
            # TODO: filter by metadata in query instead of fetching all fibers
            fibers = await storage.get_fibers(limit=1000)
            habits = [f for f in fibers if f.metadata.get("_habit_pattern")]
            return {
                "habits": [
                    {
                        "name": h.summary or "unnamed",
                        "steps": h.metadata.get("_workflow_actions", []),
                        "frequency": h.metadata.get("_habit_frequency", 0),
                        "confidence": h.metadata.get("_habit_confidence", 0.0),
                        "fiber_id": h.id,
                    }
                    for h in habits
                ],
                "count": len(habits),
            }

        elif action == "clear":
            fibers = await storage.get_fibers(limit=1000)
            habits = [f for f in fibers if f.metadata.get("_habit_pattern")]
            if habits:
                await asyncio.gather(*[storage.delete_fiber(h.id) for h in habits])
            cleared = len(habits)
            return {"cleared": cleared, "message": f"Cleared {cleared} learned habits"}

        return {"error": f"Unknown action: {action}"}

    async def _version(self, args: dict[str, Any]) -> dict[str, Any]:
        """Brain version control operations."""
        storage = await self.get_storage()
        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        from neural_memory.engine.brain_versioning import VersioningEngine

        engine = VersioningEngine(storage)
        action = args.get("action", "list")

        if action == "create":
            name = args.get("name")
            if not name:
                return {"error": "Version name is required for create"}
            description = args.get("description", "")
            try:
                version = await engine.create_version(brain.id, name, description)
            except ValueError:
                return {"error": "Failed to create version: invalid parameters"}
            return {
                "success": True,
                "version_id": version.id,
                "version_name": version.version_name,
                "version_number": version.version_number,
                "neuron_count": version.neuron_count,
                "synapse_count": version.synapse_count,
                "fiber_count": version.fiber_count,
                "message": f"Created version '{name}' (#{version.version_number})",
            }

        elif action == "list":
            limit = min(args.get("limit", 20), 100)
            versions = await engine.list_versions(brain.id, limit=limit)
            return {
                "versions": [
                    {
                        "id": v.id,
                        "name": v.version_name,
                        "number": v.version_number,
                        "description": v.description,
                        "neuron_count": v.neuron_count,
                        "synapse_count": v.synapse_count,
                        "fiber_count": v.fiber_count,
                        "created_at": v.created_at.isoformat(),
                    }
                    for v in versions
                ],
                "count": len(versions),
            }

        elif action == "rollback":
            version_id = args.get("version_id")
            if not version_id:
                return {"error": "version_id is required for rollback"}
            try:
                rollback_v = await engine.rollback(brain.id, version_id)
            except ValueError:
                return {"error": "Rollback failed: version not found or invalid"}
            return {
                "success": True,
                "rollback_version_id": rollback_v.id,
                "rollback_version_name": rollback_v.version_name,
                "neuron_count": rollback_v.neuron_count,
                "synapse_count": rollback_v.synapse_count,
                "fiber_count": rollback_v.fiber_count,
                "message": f"Rolled back to '{rollback_v.version_name}'",
            }

        elif action == "diff":
            from_id = args.get("from_version")
            to_id = args.get("to_version")
            if not from_id or not to_id:
                return {"error": "from_version and to_version are required for diff"}
            try:
                diff = await engine.diff(brain.id, from_id, to_id)
            except ValueError:
                return {"error": "Diff failed: one or both versions not found"}
            return {
                "summary": diff.summary,
                "neurons_added": len(diff.neurons_added),
                "neurons_removed": len(diff.neurons_removed),
                "neurons_modified": len(diff.neurons_modified),
                "synapses_added": len(diff.synapses_added),
                "synapses_removed": len(diff.synapses_removed),
                "synapses_weight_changed": len(diff.synapses_weight_changed),
                "fibers_added": len(diff.fibers_added),
                "fibers_removed": len(diff.fibers_removed),
            }

        return {"error": f"Unknown action: {action}"}

    async def _transplant(self, args: dict[str, Any]) -> dict[str, Any]:
        """Transplant memories from another brain."""
        from neural_memory.unified_config import get_shared_storage

        target_storage = await self.get_storage()
        target_brain_id = target_storage._current_brain_id
        if not target_brain_id:
            return {"error": "No brain configured"}

        target_brain = await target_storage.get_brain(target_brain_id)
        if not target_brain:
            return {"error": "No brain configured"}

        source_brain_name = args.get("source_brain")
        if not source_brain_name:
            return {"error": "source_brain is required"}

        if source_brain_name == target_brain.name:
            return {
                "error": "Source brain and target brain are the same. "
                "Transplanting a brain into itself is a destructive no-op."
            }

        # Open a separate storage for the source brain (.db file)
        try:
            source_storage = await get_shared_storage(brain_name=source_brain_name)
        except Exception:
            logger.error("Failed to open source brain storage", exc_info=True)
            return {"error": "Source brain not found"}

        source_brain_id = source_storage._current_brain_id
        if not source_brain_id:
            return {"error": "Source brain not found"}

        source_brain = await source_storage.get_brain(source_brain_id)
        if source_brain is None:
            return {"error": "Source brain not found"}

        from neural_memory.engine.brain_transplant import TransplantFilter, transplant
        from neural_memory.engine.merge import ConflictStrategy

        tags = args.get("tags")
        memory_types = args.get("memory_types")
        strategy_str = args.get("strategy", "prefer_local")

        try:
            strategy = ConflictStrategy(strategy_str)
        except ValueError:
            return {"error": f"Invalid strategy: {strategy_str}"}

        filt = TransplantFilter(
            tags=frozenset(tags) if tags else None,
            memory_types=frozenset(memory_types) if memory_types else None,
        )

        try:
            result = await transplant(
                source_storage=source_storage,
                target_storage=target_storage,
                source_brain_id=source_brain_id,
                target_brain_id=target_brain_id,
                filt=filt,
                strategy=strategy,
            )
        except ValueError as exc:
            logger.error("Transplant failed: %s", exc)
            return {"error": "Transplant failed"}

        return {
            "success": True,
            "fibers_transplanted": result.fibers_transplanted,
            "neurons_transplanted": result.neurons_transplanted,
            "synapses_transplanted": result.synapses_transplanted,
            "merge_summary": result.merge_report.summary(),
            "message": f"Transplanted from '{source_brain_name}': {result.fibers_transplanted} fibers",
        }

    async def _record_tool_action(self, action_type: str, context: str = "") -> None:
        """Record an action event for habit learning (fire-and-forget)."""
        try:
            import os

            source = os.environ.get("NEURALMEMORY_SOURCE", "mcp")[:256]
            storage = await self.get_storage()
            await storage.record_action(
                action_type=action_type,
                action_context=context[:200] if context else "",
                session_id=f"{source}-{id(self)}",
            )
        except Exception:
            logger.debug("Action recording failed (non-critical)", exc_info=True)
