"""MCP handler mixin for recall and context tools."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import MemoryType
from neural_memory.engine.hooks import HookEvent
from neural_memory.engine.retrieval import DepthLevel
from neural_memory.mcp.constants import MAX_HOT_CONTEXT_MEMORIES, MAX_TOKEN_BUDGET
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
        from neural_memory.mcp.tool_handler_utils import (
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
        tag_mode = args.get("tag_mode", "and")
        if tag_mode not in ("and", "or"):
            return {"error": f"Invalid tag_mode: {tag_mode}. Must be 'and' or 'or'."}
        include_citations = args.get("include_citations", True)
        clean_for_prompt = bool(args.get("clean_for_prompt", False))
        min_trust: float | None = None
        raw_min_trust = args.get("min_trust")
        if raw_min_trust is not None:
            try:
                min_trust = float(raw_min_trust)
            except (TypeError, ValueError):
                return {"error": f"Invalid min_trust: {raw_min_trust}"}

        min_arousal: float | None = None
        raw_min_arousal = args.get("min_arousal")
        if raw_min_arousal is not None:
            try:
                min_arousal = float(raw_min_arousal)
            except (TypeError, ValueError):
                return {"error": f"Invalid min_arousal: {raw_min_arousal}"}

        valence_filter: str | None = args.get("valence")
        if valence_filter is not None:
            valence_filter = str(valence_filter).lower().strip()
            if valence_filter not in ("positive", "negative", "neutral"):
                return {"error": f"Invalid valence: {valence_filter}. Must be positive/negative/neutral."}

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

        # Parse optional time-travel filter
        as_of = None
        if "as_of" in args:
            try:
                as_of = datetime.fromisoformat(args["as_of"])
                if as_of.tzinfo is not None:
                    from datetime import UTC

                    as_of = as_of.astimezone(UTC).replace(tzinfo=None)
            except (ValueError, TypeError):
                return {"error": f"Invalid as_of datetime: {args['as_of']}"}

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

        from neural_memory.engine.retrieval import ReflexPipeline

        # Parse optional per-query simhash threshold override
        simhash_threshold: int | None = None
        if "simhash_threshold" in args:
            try:
                simhash_threshold = int(args["simhash_threshold"])
                if not 0 <= simhash_threshold <= 64:
                    return {"error": "simhash_threshold must be 0-64"}
            except (ValueError, TypeError):
                return {"error": f"Invalid simhash_threshold: {args['simhash_threshold']}"}

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
            tag_mode=tag_mode,
            as_of=as_of,
            simhash_threshold=simhash_threshold,
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
                        clean_for_prompt=clean_for_prompt,
                    )

                    from dataclasses import replace as _dc_replace

                    from neural_memory.engine.token_budget import format_budget_report

                    budget_stats = format_budget_report(allocation)
                    # Replace the pipeline-generated context with budget-aware context
                    result = _dc_replace(result, context=budgeted_ctx)
            except Exception:
                logger.warning(
                    "Budget-aware recall failed, falling back to standard context",
                    exc_info=True,
                )
                budget_stats = {"budget_applied": False, "reason": "budget formatting failed"}

        if result.confidence < min_confidence:
            return {
                "answer": None,
                "message": f"No memories found with confidence >= {min_confidence}",
                "confidence": result.confidence,
            }

        # Post-filter by trust_score and/or tier (single pass to avoid redundant DB lookups).
        # Tier semantics: fibers without a typed_memory row are treated as "warm" (the default).
        # - tier="warm" → includes un-typed fibers (they default to warm)
        # - tier="hot"/"cold" → excludes un-typed fibers (only explicit tier matches)
        recall_tier = args.get("tier")
        if recall_tier is not None:
            recall_tier = str(recall_tier).lower().strip()
        needs_post_filter = (min_trust is not None or recall_tier) and result.fibers_matched
        if needs_post_filter:
            try:
                passing_ids: set[str] = set()
                for fid in result.fibers_matched:
                    tm = await storage.get_typed_memory(fid)

                    # Trust filter
                    if min_trust is not None:
                        if tm is not None and tm.trust_score is not None:
                            if tm.trust_score < min_trust:
                                continue

                    # Tier filter
                    if recall_tier:
                        if tm is None:
                            if recall_tier != "warm":
                                continue
                        elif getattr(tm, "tier", "warm") != recall_tier:
                            continue

                    passing_ids.add(fid)

                filtered_fibers = [f for f in result.fibers_matched if f in passing_ids]
                result = (
                    result._replace(fibers_matched=filtered_fibers)
                    if hasattr(result, "_replace")
                    else result
                )
            except Exception:
                logger.warning(
                    "Post-filter (trust/tier) failed, returning unfiltered results", exc_info=True
                )

        # Post-filter by arousal and/or valence (stored in fiber metadata)
        needs_emotion_filter = (min_arousal is not None or valence_filter is not None)
        if needs_emotion_filter and result.fibers_matched:
            try:
                emotion_passing: set[str] = set()
                for fid in result.fibers_matched:
                    fiber = await storage.get_fiber(fid)
                    if not fiber:
                        emotion_passing.add(fid)  # Include if fiber not found
                        continue

                    # Arousal filter
                    if min_arousal is not None:
                        fiber_arousal = fiber.metadata.get("_arousal", 0.0)
                        if not (isinstance(fiber_arousal, (int, float)) and fiber_arousal >= min_arousal):
                            continue

                    # Valence filter
                    if valence_filter is not None:
                        fiber_valence = fiber.metadata.get("_valence", "")
                        if fiber_valence != valence_filter:
                            continue

                    emotion_passing.add(fid)

                filtered_fibers = [f for f in result.fibers_matched if f in emotion_passing]
                result = (
                    result._replace(fibers_matched=filtered_fibers)
                    if hasattr(result, "_replace")
                    else result
                )
            except Exception:
                logger.warning(
                    "Post-filter (arousal/valence) failed, returning unfiltered results",
                    exc_info=True,
                )

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

        # Unified confidence score (metacognitive assessment)
        if result.confidence_score is not None:
            cs = result.confidence_score
            response["confidence_score"] = {
                "overall": cs.overall,
                "retrieval": cs.retrieval,
                "content_quality": cs.content_quality,
                "fidelity": cs.fidelity,
                "freshness": cs.freshness,
                "familiarity_penalty": cs.familiarity_penalty,
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
        from neural_memory.mcp.tool_handler_utils import _parse_tags

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
        tag_mode = args.get("tag_mode", "and")
        if tag_mode not in ("and", "or"):
            return {"error": f"Invalid tag_mode: {tag_mode}. Must be 'and' or 'or'."}

        try:
            result = await cross_brain_recall(
                config=self.config,
                brain_names=brain_names,
                query=query,
                depth=depth,
                max_tokens=max_tokens,
                tags=tags,
                tag_mode=tag_mode,
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
        """Get recent context.

        Note: HOT-tier memories are always injected regardless of fresh_only.
        This is intentional — HOT memories represent always-in-context data
        (safety boundaries, pinned knowledge) that should never be excluded.
        """
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

        # Inject HOT tier memories — always in context regardless of recency
        # When domain is specified, boundary memories are filtered:
        # - Boundaries with matching domain:{value} tag → included
        # - Boundaries with NO domain: tag (global) → included
        # - Boundaries with a DIFFERENT domain: tag → excluded
        # Non-boundary HOT memories are always included regardless of domain.
        existing_ids = {f.id for f in fibers}
        context_domain = args.get("domain")
        if context_domain and isinstance(context_domain, str):
            context_domain = context_domain.lower().strip()[:50]
        try:
            hot_memories = await storage.find_typed_memories(
                tier="hot", limit=MAX_HOT_CONTEXT_MEMORIES
            )
            for tm in hot_memories:
                if tm.fiber_id in existing_ids:
                    continue
                # Domain filter: only applies to boundary memories
                if context_domain and tm.memory_type == MemoryType.BOUNDARY:
                    domain_tags = {t for t in tm.tags if t.startswith("domain:")}
                    if domain_tags and f"domain:{context_domain}" not in domain_tags:
                        continue  # has domain tags but none match → skip
                hot_fiber = await storage.get_fiber(tm.fiber_id)
                if hot_fiber:
                    fibers.append(hot_fiber)
                    existing_ids.add(tm.fiber_id)
            if len(hot_memories) >= MAX_HOT_CONTEXT_MEMORIES:
                logger.warning(
                    "HOT memory limit reached (%d) — some HOT memories may be excluded from context",
                    MAX_HOT_CONTEXT_MEMORIES,
                )
        except Exception:
            logger.warning("HOT memory injection failed — tier filter unavailable", exc_info=True)

        # Cross-session correction injection: recent error→fix pairs
        corrections_text = ""
        try:
            corrections_text = await self._inject_recent_corrections(storage)
        except Exception:
            logger.debug("Correction injection failed (non-critical)", exc_info=True)

        # A8 Phase 2: Proactive context — surface signals + cluster injection + session summary
        signals_text = ""
        topic_context_text = ""
        session_summary_text = ""
        try:
            proactive = await self._build_proactive_context(storage, existing_ids)
            signals_text = proactive.get("signals", "")
            topic_context_text = proactive.get("topic_context", "")
            session_summary_text = proactive.get("session_summary", "")
        except Exception:
            logger.debug("Proactive context failed (non-critical)", exc_info=True)

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

            context_parts = []
            for item in non_ghost:
                prefix = ""
                if item.tier == "hot":
                    prefix = "[HOT] "
                if item.unreliable:
                    prefix += "[UNRELIABLE] "
                suffix = ""
                if item.confidence is not None and item.confidence < 1.0:
                    suffix = f" (confidence: {item.confidence:.0%})"
                context_parts.append(f"- {prefix}{item.content}{suffix}")
            context_text = "\n".join(context_parts) if context_parts else ""

            # Append ghost section if enabled and ghosts exist
            if include_ghosts and ghost_items:
                ghost_parts = []
                for item in ghost_items:
                    if item.superseded_by:
                        ghost_parts.append(
                            f"- [OUTDATED] {item.content} → See: {item.superseded_by}"
                        )
                    else:
                        ghost_parts.append(f"- {item.content}")
                ghost_section = "\n--- faded memories (use recall key to restore) ---\n"
                ghost_section += "\n".join(ghost_parts)
                context_text = (
                    (context_text + "\n" + ghost_section) if context_text else ghost_section
                )

            # Prepend corrections section (cross-session learning)
            if corrections_text:
                context_text = (
                    corrections_text + "\n" + context_text if context_text else corrections_text
                )

            # A8 Phase 2: Prepend proactive context sections
            # Order: session summary → signals → topic context → corrections → main context
            if topic_context_text:
                context_text = (
                    topic_context_text + "\n" + context_text if context_text else topic_context_text
                )
            if signals_text:
                context_text = signals_text + "\n" + context_text if context_text else signals_text
            if session_summary_text:
                context_text = (
                    session_summary_text + "\n" + context_text
                    if context_text
                    else session_summary_text
                )

            if not context_text:
                context_text = "No context available."
        else:
            parts = [
                p
                for p in [session_summary_text, signals_text, topic_context_text, corrections_text]
                if p
            ]
            context_text = "\n".join(parts) if parts else "No context available."

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

        # Count HOT memories in final output
        hot_count = sum(1 for item in plan.items if item.tier == "hot")

        response: dict[str, Any] = {
            "context": context_text,
            "count": len(plan.items),
            "tokens_used": plan.total_tokens,
            "token_budget": max_tokens,
            "hot_memories_injected": hot_count,
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

    async def _inject_recent_corrections(
        self,
        storage: NeuralStorage,
        max_corrections: int = 5,
        max_age_days: int = 7,
    ) -> str:
        """Fetch recent error→fix corrections for cross-session learning.

        Looks for RESOLVED_BY synapses created within the last N days,
        then formats them as a corrections section for context injection.

        Args:
            storage: Storage backend
            max_corrections: Maximum corrections to inject
            max_age_days: Only include corrections from the last N days

        Returns:
            Formatted corrections text, or empty string if none found
        """
        from datetime import timedelta

        from neural_memory.core.synapse import SynapseType

        now = utcnow()
        cutoff = now - timedelta(days=max_age_days)

        # Fetch all RESOLVED_BY synapses in this brain
        resolved_synapses = await storage.get_synapses(type=SynapseType.RESOLVED_BY)

        # Filter to recent ones
        recent = [s for s in resolved_synapses if s.created_at >= cutoff]
        if not recent:
            return ""

        # Sort by recency, take top N
        recent.sort(key=lambda s: s.created_at, reverse=True)
        recent = recent[:max_corrections]

        # Batch-fetch source (error) and target (fix) neurons
        all_ids = list({s.source_id for s in recent} | {s.target_id for s in recent})
        neurons = await storage.get_neurons_batch(all_ids)

        correction_lines: list[str] = []
        for syn in recent:
            error_neuron = neurons.get(syn.source_id)
            fix_neuron = neurons.get(syn.target_id)
            if not error_neuron or not fix_neuron:
                continue
            error_content = error_neuron.content[:100]
            fix_content = fix_neuron.content[:100]
            correction_lines.append(f"- {error_content} -> {fix_content}")

        if not correction_lines:
            return ""

        header = "--- corrections from recent sessions ---"
        return header + "\n" + "\n".join(correction_lines)

    async def _build_proactive_context(
        self,
        storage: NeuralStorage,
        existing_fiber_ids: set[str],
    ) -> dict[str, str]:
        """Build proactive context sections from surface + session state.

        A8 Phase 2: Auto-inject relevant memories without agent asking.
        Uses surface CLUSTERS for topic-aware injection, SIGNALS for alerts,
        and session EMA for meta-summary.

        Returns:
            Dict with keys: signals, topic_context, session_summary (all strings, empty if N/A)
        """
        result: dict[str, str] = {"signals": "", "topic_context": "", "session_summary": ""}

        # Parse surface (if available)
        surface = None
        if hasattr(self, "_surface_text") and self._surface_text:
            try:
                from neural_memory.surface.parser import parse

                surface = parse(self._surface_text)
            except Exception:
                logger.debug("Surface parse failed in proactive context", exc_info=True)

        # Get session state (if available)
        session_state = None
        try:
            import os

            from neural_memory.engine.session_state import SessionManager

            source = os.environ.get("NEURALMEMORY_SOURCE", "mcp")[:256]
            session_id = f"{source}-{id(self)}"
            session_state = SessionManager.get_instance().get(session_id)
        except Exception:
            logger.debug("Session state lookup failed in proactive context", exc_info=True)

        # T2.2: Surface SIGNALS as proactive alerts
        if surface and surface.signals:
            from neural_memory.surface.models import SignalLevel

            signal_lines: list[str] = []
            for sig in surface.signals:
                if sig.level == SignalLevel.URGENT:
                    signal_lines.append(f"! {sig.text}")
                elif sig.level == SignalLevel.WATCHING:
                    signal_lines.append(f"~ {sig.text}")
            if signal_lines:
                result["signals"] = "--- active signals ---\n" + "\n".join(signal_lines)

        # T2.1: Surface cluster injection — match clusters to session topics
        # M4 fix: work on a local copy to avoid mutating caller's set
        local_seen = set(existing_fiber_ids)
        if surface and surface.clusters and session_state:
            topic_weights = session_state.get_topic_weights(limit=5)
            active_topics = {t for t, w in topic_weights.items() if w >= 0.3}

            if active_topics:
                matched_fiber_ids: list[str] = []
                for cluster in surface.clusters:
                    if len(matched_fiber_ids) >= 9:
                        break  # M5 fix: early exit when cap reached
                    # Match cluster name or description against session topics
                    cluster_terms = set(cluster.name.lower().split())
                    if cluster.description:
                        cluster_terms |= set(cluster.description.lower().split())
                    if active_topics & cluster_terms:
                        # H1 fix: search neurons by content, not by tag lookup
                        for node_id in cluster.node_ids[:3]:  # Max 3 per cluster
                            if len(matched_fiber_ids) >= 9:
                                break  # M5 fix: early exit inner loop
                            node = surface.get_node(node_id)
                            if not node or not node.content.strip():
                                continue
                            try:
                                # Find neurons whose content matches the surface node
                                # H2 fix: truncate to first 80 chars for reliable substring match
                                search_text = node.content.strip()[:80]
                                neurons = await storage.find_neurons(
                                    content_contains=search_text,
                                    limit=2,
                                )
                                for neuron in neurons:
                                    # Find fibers containing this neuron
                                    fibers = await storage.find_fibers(
                                        contains_neuron=neuron.id,
                                        limit=1,
                                    )
                                    for fiber in fibers:
                                        if (
                                            fiber.id not in local_seen
                                            and len(matched_fiber_ids) < 9
                                        ):
                                            matched_fiber_ids.append(fiber.id)
                                            local_seen.add(fiber.id)
                            except Exception:
                                continue

                if matched_fiber_ids:
                    # Fetch fiber content for injection
                    topic_lines: list[str] = []
                    for fid in matched_fiber_ids[:9]:
                        try:
                            topic_fiber = await storage.get_fiber(fid)
                            if topic_fiber:
                                content = topic_fiber.summary or ""
                                if not content and topic_fiber.anchor_neuron_id:
                                    anchor = await storage.get_neuron(topic_fiber.anchor_neuron_id)
                                    content = anchor.content if anchor else ""
                                if content:
                                    topic_lines.append(f"- {content[:150]}")
                        except Exception:
                            continue
                    if topic_lines:
                        result["topic_context"] = "--- active topic context ---\n" + "\n".join(
                            topic_lines
                        )

        # T2.4: Session meta-summary
        summary_parts: list[str] = []
        if session_state:
            topic_weights = session_state.get_topic_weights(limit=3)
            topic_str = (
                ", ".join(f"{t} ({w:.1f})" for t, w in topic_weights.items())
                if topic_weights
                else "none"
            )
            summary_parts.append(
                f"Session: {session_state.query_count} queries | topics: {topic_str}"
            )

        if surface and surface.meta:
            coverage = surface.meta.coverage
            staleness = surface.meta.staleness
            if coverage is not None:
                summary_parts.append(f"surface: {coverage:.0%} coverage")
            if staleness is not None and staleness > 0.5:
                summary_parts.append(f"staleness: {staleness:.0%}")

        if summary_parts:
            result["session_summary"] = "--- " + " | ".join(summary_parts) + " ---"

        return result
