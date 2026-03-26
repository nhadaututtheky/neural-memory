"""MCP handler mixin for recall and context tools."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.engine.hooks import HookEvent
from neural_memory.engine.retrieval import DepthLevel
from neural_memory.mcp.constants import MAX_TOKEN_BUDGET
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.engine.hooks import HookRegistry
    from neural_memory.mcp.maintenance_handler import HealthPulse
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class RecallHandler:
    """Mixin providing recall and context MCP tool handlers."""

    if TYPE_CHECKING:
        config: UnifiedConfig
        hooks: HookRegistry
        _surface_text: str
        _surface_brain: str

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

        async def _check_cross_language_hint(
            self,
            query: str,
            result: Any,
            config: Any,
        ) -> str | None:
            raise NotImplementedError

        async def _surface_pending_alerts(self) -> dict[str, int] | None:
            raise NotImplementedError

        async def _record_tool_action(self, action_type: str, context: str = "") -> None:
            raise NotImplementedError

    # ──────────────────── Surface Depth Routing ────────────────────

    def _check_surface_depth(
        self,
        query: str,
    ) -> tuple[dict[str, Any] | None, int | None]:
        """Check the Knowledge Surface DEPTH MAP for recall routing.

        If the query matches a SUFFICIENT entity, returns surface context
        directly (no brain.db query needed). For NEEDS_DETAIL or NEEDS_DEEP,
        returns a suggested depth override.

        Args:
            query: The recall query string.

        Returns:
            Tuple of (surface_response_or_None, depth_override_or_None).
            If surface_response is not None, caller should return it immediately.
        """
        if not hasattr(self, "_surface_text") or not self._surface_text:
            return None, None

        try:
            from neural_memory.surface.models import DepthLevel
            from neural_memory.surface.parser import parse

            surface = parse(self._surface_text)
        except Exception:
            return None, None

        # Normalize query for matching
        query_lower = query.lower().strip()

        # Find matching entity in graph nodes
        for entry in surface.graph:
            node = entry.node
            if query_lower in node.content.lower() or node.content.lower() in query_lower:
                depth_level = surface.get_depth_hint(node.id)
                if depth_level == DepthLevel.SUFFICIENT:
                    # Build context from surface graph
                    context_parts = [f"[{node.id}] {node.content} ({node.node_type})"]
                    for edge in entry.edges:
                        if edge.target_id:
                            context_parts.append(
                                f"  →{edge.edge_type}→ [{edge.target_id}] {edge.target_text}"
                            )
                        else:
                            context_parts.append(f"  →{edge.edge_type}→ {edge.target_text}")

                    # Add cluster context if available
                    for cluster in surface.clusters:
                        if node.id in cluster.node_ids:
                            context_parts.append(f"  @{cluster.name}: {cluster.description}")

                    return {
                        "answer": "\n".join(context_parts),
                        "confidence": 0.8,
                        "source": "knowledge_surface",
                        "depth_hint": "SUFFICIENT",
                        "message": "Answered from Knowledge Surface (no brain.db query needed)",
                    }, None

                elif depth_level == DepthLevel.NEEDS_DEEP:
                    return None, 2

                elif depth_level == DepthLevel.NEEDS_DETAIL:
                    return None, 1

        return None, None

    async def _recall(self, args: dict[str, Any]) -> dict[str, Any]:
        """Query memories via spreading activation."""
        from neural_memory.mcp.tool_handlers import (
            _build_citation_audit,
            _parse_tags,
            _require_brain_id,
        )

        # Cross-brain recall: early return if brains parameter is provided
        brain_names = args.get("brains")
        if brain_names and isinstance(brain_names, list) and len(brain_names) > 0:
            return await self._cross_brain_recall(args, brain_names)

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for recall")
            return {"error": "No brain configured"}
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        query = args.get("query")
        if not query or not isinstance(query, str):
            return {"error": "query is required and must be a string"}

        # Ghost recall key: exact fiber lookup via "fiber:{id}" or "recall:fiber:{id}"
        fiber_key = None
        if query.startswith("recall:fiber:"):
            fiber_key = query[len("recall:fiber:") :]
        elif query.startswith("fiber:"):
            fiber_key = query[len("fiber:") :]

        if fiber_key:
            fiber_key = fiber_key.strip()
            if not fiber_key:
                return {"error": "Invalid recall key: empty fiber ID"}
            fiber = await storage.get_fiber(fiber_key)
            if not fiber:
                return {"error": f"No fiber found with ID: {fiber_key}"}
            anchor = (
                await storage.get_neuron(fiber.anchor_neuron_id) if fiber.anchor_neuron_id else None
            )
            content = (anchor.content if anchor else None) or fiber.summary or ""
            fiber_tags = sorted(fiber.tags)[:5]
            return {
                "answer": content,
                "fiber_id": fiber.id,
                "summary": fiber.summary,
                "tags": fiber_tags,
                "confidence": 1.0,
                "recall_type": "exact_fiber",
            }

        try:
            depth = DepthLevel(args.get("depth", 1))
        except ValueError:
            return {"error": f"Invalid depth level: {args.get('depth')}. Must be 0-3."}
        max_tokens = min(args.get("max_tokens", 500), 10_000)
        min_confidence = args.get("min_confidence", 0.0)
        recall_mode = args.get("mode", "associative")
        if recall_mode not in ("associative", "exact"):
            return {"error": f"Invalid mode: {recall_mode}. Must be 'associative' or 'exact'."}
        tags = _parse_tags(args)
        include_citations = args.get("include_citations", True)
        min_trust: float | None = None
        raw_min_trust = args.get("min_trust")
        if raw_min_trust is not None:
            try:
                min_trust = float(raw_min_trust)
            except (TypeError, ValueError):
                return {"error": f"Invalid min_trust: {raw_min_trust}"}

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
                # Convert to UTC before stripping timezone
                if valid_at.tzinfo is not None:
                    from datetime import UTC

                    valid_at = valid_at.astimezone(UTC).replace(tzinfo=None)
            except (ValueError, TypeError):
                return {"error": f"Invalid valid_at datetime: {args['valid_at']}"}

        await self.hooks.emit(HookEvent.PRE_RECALL, {"query": query, "depth": depth.value})

        # Surface depth routing: SUFFICIENT → answer from surface, skip brain.db
        if not args.get("depth"):  # Only route when user didn't specify depth
            surface_response, depth_override = self._check_surface_depth(query)
            if surface_response is not None:
                return surface_response
            if depth_override is not None:
                try:
                    depth = DepthLevel(depth_override)
                except ValueError:
                    pass

        permanent_only = bool(args.get("permanent_only", False))

        from neural_memory.mcp.tool_handlers import ReflexPipeline  # type: ignore[attr-defined]

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=effective_query,
            depth=depth,
            max_tokens=max_tokens,
            reference_time=utcnow(),
            valid_at=valid_at,
            tags=tags,
            session_id=f"mcp-{id(self)}",
            exclude_ephemeral=permanent_only,
        )

        # Passive auto-capture on long queries
        if self.config.auto.enabled and len(query) >= 50:
            await self._passive_capture(query)

        self._fire_eternal_trigger(query)

        # Budget-aware context re-formatting (opt-in via recall_token_budget param)
        budget_stats: dict[str, Any] | None = None
        raw_recall_budget = args.get("recall_token_budget")
        if raw_recall_budget is not None and result.fibers_matched:
            try:
                recall_budget = min(int(raw_recall_budget), MAX_TOKEN_BUDGET)
                from neural_memory.engine.retrieval_context import format_context_budgeted
                from neural_memory.engine.token_budget import BudgetConfig

                budget_cfg = BudgetConfig(
                    system_overhead_tokens=self.config.budget.system_overhead,
                    per_fiber_overhead=self.config.budget.per_fiber_overhead,
                )

                # Fetch fiber objects for matched fibers
                candidate_fibers = []
                for fid in result.fibers_matched:
                    f = await storage.get_fiber(fid)
                    if f:
                        candidate_fibers.append(f)

                # Build a minimal activations map from co_activations and neurons
                from neural_memory.engine.activation import ActivationResult

                dummy_activations: dict[str, ActivationResult] = {}
                for co in result.co_activations:
                    for nid in co.neuron_ids:
                        if nid not in dummy_activations:
                            dummy_activations[nid] = ActivationResult(
                                neuron_id=nid,
                                activation_level=co.binding_strength,
                                hop_distance=0,
                                path=[nid],
                                source_anchor=nid,
                            )

                if candidate_fibers:
                    # Get encryptor if encryption is enabled
                    encryptor_obj = None
                    try:
                        if self.config.encryption.enabled:
                            from pathlib import Path as _Path

                            from neural_memory.safety.encryption import MemoryEncryptor

                            keys_dir_str = getattr(self.config.encryption, "keys_dir", "")
                            keys_dir = (
                                _Path(keys_dir_str)
                                if keys_dir_str
                                else (self.config.data_dir / "keys")
                            )
                            encryptor_obj = MemoryEncryptor(keys_dir=keys_dir)
                    except Exception:
                        pass

                    budgeted_ctx, _, allocation = await format_context_budgeted(
                        storage=storage,
                        activations=dummy_activations,
                        fibers=candidate_fibers,
                        max_tokens=recall_budget,
                        encryptor=encryptor_obj,
                        brain_id=brain_id,
                        budget_config=budget_cfg,
                    )

                    from dataclasses import replace as _dc_replace

                    from neural_memory.engine.token_budget import format_budget_report

                    budget_stats = format_budget_report(allocation)
                    # Replace the pipeline-generated context with budget-aware context
                    result = _dc_replace(result, context=budgeted_ctx)
            except Exception:
                logger.debug(
                    "Budget-aware recall failed (non-critical), using standard context",
                    exc_info=True,
                )

        if result.confidence < min_confidence:
            return {
                "answer": None,
                "message": f"No memories found with confidence >= {min_confidence}",
                "confidence": result.confidence,
            }

        # Post-filter by trust_score if min_trust is specified
        if min_trust is not None and result.fibers_matched:
            try:
                trusted_fiber_ids: set[str] = set()
                for fid in result.fibers_matched:
                    tm = await storage.get_typed_memory(fid)
                    if tm is None:
                        trusted_fiber_ids.add(fid)  # No typed_memory = include by default
                    elif tm.trust_score is None:
                        trusted_fiber_ids.add(fid)  # Unscored = include by default
                    elif tm.trust_score >= min_trust:
                        trusted_fiber_ids.add(fid)
                filtered_fibers = [f for f in result.fibers_matched if f in trusted_fiber_ids]
                result = (
                    result._replace(fibers_matched=filtered_fibers)
                    if hasattr(result, "_replace")
                    else result
                )
            except Exception:
                logger.debug("Trust filter failed (non-critical)", exc_info=True)

        # Exact mode: return raw neuron contents without truncation
        if recall_mode == "exact" and result.fibers_matched:
            exact_items: list[dict[str, Any]] = []
            for fid in result.fibers_matched:
                fiber = await storage.get_fiber(fid)
                if not fiber:
                    continue
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if not anchor:
                    continue
                content = anchor.content
                # Decrypt if needed
                if fiber.metadata.get("encrypted"):
                    try:
                        from pathlib import Path

                        from neural_memory.safety.encryption import MemoryEncryptor

                        keys_dir_str = getattr(self.config.encryption, "keys_dir", "")
                        keys_dir = (
                            Path(keys_dir_str) if keys_dir_str else (self.config.data_dir / "keys")
                        )
                        encryptor = MemoryEncryptor(keys_dir=keys_dir)
                        bid = storage.brain_id or ""
                        content = encryptor.decrypt(content, bid)
                    except Exception:
                        logger.debug("Decryption failed in exact recall", exc_info=True)
                tm = await storage.get_typed_memory(fid)
                item: dict[str, Any] = {
                    "fiber_id": fid,
                    "content": content,
                    "memory_type": tm.memory_type.value if tm else None,
                    "priority": tm.priority.value if tm else None,
                    "tags": list(tm.tags) if tm and tm.tags else [],
                    "created_at": fiber.created_at.isoformat() if fiber.created_at else None,
                }
                # Include structure metadata if present
                structure = anchor.metadata.get("_structure") if anchor.metadata else None
                if structure:
                    item["structure"] = structure

                # Citation + audit trail (Phase 4)
                citation_audit = await _build_citation_audit(storage, anchor.id, include_citations)
                if citation_audit.get("citation"):
                    item["citation"] = citation_audit["citation"]
                if citation_audit.get("audit"):
                    item["audit"] = citation_audit["audit"]

                exact_items.append(item)

            response: dict[str, Any] = {
                "mode": "exact",
                "memories": exact_items,
                "confidence": result.confidence,
                "neurons_activated": result.neurons_activated,
                "fibers_matched": result.fibers_matched,
                "depth_used": result.depth_used.value,
            }
        else:
            response = {
                "answer": result.context or "No relevant memories found.",
                "confidence": result.confidence,
                "neurons_activated": result.neurons_activated,
                "fibers_matched": result.fibers_matched,
                "depth_used": result.depth_used.value,
                "tokens_used": result.tokens_used,
            }

        if budget_stats is not None:
            response["budget_stats"] = budget_stats

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

        # Enrich results with source metadata from typed_memory.source
        try:
            if result.fibers_matched:
                source_map: dict[str, dict[str, Any]] = {}
                for fid in result.fibers_matched:
                    tm = await storage.get_typed_memory(fid)
                    if not tm or not tm.source or not tm.source.startswith("source:"):
                        continue
                    src_id = tm.source[len("source:") :]
                    src = await storage.get_source(src_id)
                    if src:
                        source_map[fid] = {
                            "source_id": src.id,
                            "name": src.name,
                            "source_type": src.source_type.value,
                            "version": src.version,
                            "status": src.status.value,
                        }
                if source_map:
                    response["sources"] = source_map
        except Exception:
            logger.debug("Source enrichment failed (non-critical)", exc_info=True)

        # Cognitive chunking: group results when many fibers matched
        fibers_count_for_chunking = (
            len(result.fibers_matched)
            if isinstance(result.fibers_matched, list)
            else result.fibers_matched
        )
        if (
            getattr(brain.config, "chunking_enabled", True)
            and fibers_count_for_chunking
            and fibers_count_for_chunking > 7
        ):
            try:
                from neural_memory.engine.chunking import chunk_retrieval_results

                chunk_neuron_ids = list(result.subgraph.neuron_ids) if result.subgraph else []
                stored_levels = (result.metadata or {}).get("activation_levels", {})
                chunk_activations = {nid: stored_levels.get(nid, 0.5) for nid in chunk_neuron_ids}
                synapse_pairs: list[tuple[str, str, float]] = []
                for nid in chunk_neuron_ids[:20]:
                    syns = await storage.get_synapses(source_id=nid)
                    for s in syns:
                        synapse_pairs.append((s.source_id, s.target_id, s.weight))

                max_chunks = (
                    int(getattr(brain.config, "max_chunks", 5))
                    if isinstance(getattr(brain.config, "max_chunks", 5), (int, float))
                    else 5
                )
                chunks = chunk_retrieval_results(
                    neuron_ids=chunk_neuron_ids,
                    activation_levels=chunk_activations,
                    synapse_pairs=synapse_pairs,
                    max_chunks=max_chunks,
                )
                if chunks:
                    response["cognitive_chunks"] = [
                        {
                            "label": c.label,
                            "neuron_ids": list(c.neuron_ids),
                            "coherence": c.coherence,
                            "relevance": c.relevance,
                        }
                        for c in chunks
                    ]
            except Exception:
                logger.debug("Cognitive chunking failed (non-critical)", exc_info=True)

        # Session intelligence: attach topic context
        session_topics = (result.metadata or {}).get("session_topics")
        if session_topics:
            response["session_topics"] = session_topics
            response["session_query_count"] = (result.metadata or {}).get("session_query_count", 0)

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

        # Cross-language hint: suggest embedding when recall misses due to language mismatch
        cross_lang_hint = await self._check_cross_language_hint(
            query,
            result,
            brain.config,
        )
        if cross_lang_hint:
            response["cross_language_hint"] = cross_lang_hint

        # Pro hint: when many fibers matched but results were truncated
        fibers_count = getattr(result, "fibers_matched", 0)
        if isinstance(fibers_count, int) and fibers_count > 10:
            pro_hints = response.get("pro_hints", [])
            pro_hints.append(
                f"Showing top results from {fibers_count} matches. "
                "Pro: Cone queries return ALL relevant memories for exhaustive recall."
            )
            response["pro_hints"] = pro_hints

        # Surface pending alerts count
        alert_info = await self._surface_pending_alerts()
        if alert_info:
            response.update(alert_info)

        return response

    async def _cross_brain_recall(
        self, args: dict[str, Any], brain_names: list[str]
    ) -> dict[str, Any]:
        """Handle cross-brain recall by querying multiple brains in parallel."""
        from neural_memory.engine.cross_brain import cross_brain_recall
        from neural_memory.mcp.tool_handlers import _parse_tags

        query = args.get("query", "")
        if not query:
            return {"error": "query is required"}

        # Validate and cap at 5 brains
        import re

        _brain_pattern = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
        brain_names = [n for n in brain_names[:5] if isinstance(n, str) and _brain_pattern.match(n)]
        if not brain_names:
            return {"error": "No valid brain names provided"}
        try:
            depth = int(args.get("depth", 1))
            depth = max(0, min(depth, 3))
        except (TypeError, ValueError):
            depth = 1
        max_tokens = min(int(args.get("max_tokens", 500)), 10_000)

        tags = _parse_tags(args)

        try:
            result = await cross_brain_recall(
                config=self.config,
                brain_names=brain_names,
                query=query,
                depth=depth,
                max_tokens=max_tokens,
                tags=tags,
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
        # Pass fidelity config — fetch from storage brain (BrainConfig has fidelity fields)
        # BrainSettings (self.config.brain) does NOT have fidelity fields
        brain_obj = await storage.get_brain(storage.brain_id) if storage.brain_id else None
        brain_config = brain_obj.config if brain_obj else None

        # Build embed_fn for anisotropic compression (if embedding enabled)
        embed_fn = None
        if brain_config and brain_config.embedding_enabled:
            try:
                from neural_memory.engine.semantic_discovery import _create_provider

                provider = _create_provider(brain_config)
                embed_fn = provider.embed
            except Exception:
                logger.debug("Embedding provider unavailable for anisotropic compression")

        plan = await optimize_context(
            storage,
            fibers,
            max_tokens,
            fidelity_enabled=brain_config.fidelity_enabled if brain_config else True,
            fidelity_full_threshold=brain_config.fidelity_full_threshold if brain_config else 0.6,
            fidelity_summary_threshold=brain_config.fidelity_summary_threshold
            if brain_config
            else 0.3,
            fidelity_essence_threshold=brain_config.fidelity_essence_threshold
            if brain_config
            else 0.1,
            decay_rate=brain_config.decay_rate if brain_config else 0.1,
            decay_floor=brain_config.decay_floor if brain_config else 0.05,
            embed_fn=embed_fn,
        )

        include_ghosts = args.get("include_ghosts", True)

        if plan.items:
            # Separate ghost items from non-ghost items
            non_ghost = [item for item in plan.items if item.fidelity_level != "ghost"]
            ghost_items = [item for item in plan.items if item.fidelity_level == "ghost"]

            context_parts = [f"- {item.content}" for item in non_ghost]
            context_text = "\n".join(context_parts) if context_parts else ""

            # Append ghost section if enabled and ghosts exist
            if include_ghosts and ghost_items:
                ghost_parts = [f"- {item.content}" for item in ghost_items]
                ghost_section = "\n--- faded memories (use recall key to restore) ---\n"
                ghost_section += "\n".join(ghost_parts)
                context_text = (
                    (context_text + "\n" + ghost_section) if context_text else ghost_section
                )

            if not context_text:
                context_text = "No context available."
        else:
            context_text = "No context available."

        # Track ghost shown timestamps (only if ghosts were actually shown)
        if include_ghosts and plan.ghost_fiber_ids:
            try:
                from neural_memory.utils.timeutils import utcnow as _utcnow

                now = _utcnow()
                # Batch update: single SQL for all ghost fibers (avoids N round-trips)
                await storage.batch_update_ghost_shown(plan.ghost_fiber_ids, now)
            except Exception:
                logger.debug("Ghost tracking update failed", exc_info=True)

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

        # Fidelity stats — always include when fidelity is enabled so callers
        # can distinguish "fidelity off" from "all items at FULL"
        fidelity_on = brain_config.fidelity_enabled if brain_config else True
        fs = plan.fidelity_stats
        if fidelity_on:
            response["fidelity_stats"] = {
                "full": fs.full,
                "summary": fs.summary,
                "essence": fs.essence,
                "ghost": fs.ghost,
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
        alert_info = await self._surface_pending_alerts()
        if alert_info:
            response.update(alert_info)

        return response
