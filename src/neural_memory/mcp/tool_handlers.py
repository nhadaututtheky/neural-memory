"""MCP tool handler implementations.

Extracted from server.py to keep file sizes manageable.
Each method handles one MCP tool call (nmem_*).

The ToolHandler mixin is inherited by MCPServer in server.py.
All methods access storage/config via self.get_storage() and self.config
from the MCPServer base class.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory import __version__
from neural_memory.core.memory_types import (
    MemoryTier,
    MemoryType,
    Priority,
)
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.mcp.constants import MAX_CONTENT_LENGTH, MAX_TOKEN_BUDGET
from neural_memory.mcp.recall_handler import RecallHandler
from neural_memory.mcp.remember_handler import RememberHandler
from neural_memory.utils.timeutils import utcnow

# Max tags per recall query (remember allows 50 for storage, recall caps at 20 for filtering)
_MAX_RECALL_TAGS = 20
_MAX_TAG_LENGTH = 100


def _parse_tags(args: dict[str, Any], *, max_items: int = _MAX_RECALL_TAGS) -> set[str] | None:
    """Parse and validate tags from MCP tool arguments.

    Returns a set of valid tag strings, or None if no valid tags provided.
    """
    raw_tags = args.get("tags")
    if not raw_tags or not isinstance(raw_tags, list):
        return None
    tags = {t for t in raw_tags[:max_items] if isinstance(t, str) and 0 < len(t) <= _MAX_TAG_LENGTH}
    return tags or None


if TYPE_CHECKING:
    from neural_memory.engine.hooks import HookRegistry
    from neural_memory.mcp.maintenance_handler import HealthPulse
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

# Gromov delta cache: {brain_id: (timestamp, GromovResult)} — TTL 1 hour
_gromov_cache: dict[str, tuple[Any, Any]] = {}


def _require_brain_id(storage: NeuralStorage) -> str:
    """Return the current brain ID or raise ValueError if not set."""
    brain_id = storage.brain_id
    if not brain_id:
        raise ValueError("No brain context set")
    return brain_id


async def _get_brain_or_error(
    storage: NeuralStorage,
) -> tuple[Any, dict[str, Any] | None]:
    """Get brain object or return (None, error_dict)."""
    try:
        brain_id = _require_brain_id(storage)
    except ValueError:
        return None, {"error": "No brain configured"}
    brain = await storage.get_brain(brain_id)
    if not brain:
        return None, {"error": "No brain configured"}
    return brain, None


async def _build_citation_audit(
    storage: NeuralStorage,
    neuron_id: str,
    include_citations: bool = True,
) -> dict[str, Any]:
    """Build citation and audit trail for a neuron from its synapses.

    Looks up SOURCE_OF, STORED_BY, VERIFIED_AT, APPROVED_BY synapses
    connected to the neuron and returns citation + audit dicts.
    """
    from neural_memory.core.synapse import SynapseType

    result: dict[str, Any] = {}

    # Fetch incoming synapses for this neuron
    synapses = await storage.get_synapses(target_id=neuron_id)

    # Build citation from SOURCE_OF synapse
    if include_citations:
        source_synapses = [s for s in synapses if s.type == SynapseType.SOURCE_OF]
        if source_synapses:
            source_syn = source_synapses[0]
            try:
                source_obj = await storage.get_source(source_syn.source_id)
                if source_obj:
                    from neural_memory.engine.citation import (
                        CitationFormat,
                        CitationInput,
                        format_citation,
                    )

                    citation_input = CitationInput(
                        source_name=source_obj.name,
                        source_type=source_obj.source_type.value,
                        source_version=source_obj.version,
                        effective_date=(
                            source_obj.effective_date.isoformat()
                            if source_obj.effective_date
                            else None
                        ),
                        neuron_id=neuron_id,
                        metadata=source_obj.metadata,
                    )
                    result["citation"] = {
                        "inline": format_citation(citation_input, CitationFormat.INLINE),
                        "footnote": format_citation(citation_input, CitationFormat.FOOTNOTE),
                        "source_id": source_obj.id,
                        "source_name": source_obj.name,
                        "source_type": source_obj.source_type.value,
                    }
            except Exception:
                logger.debug("Citation generation failed", exc_info=True)

    # Build audit trail from STORED_BY, VERIFIED_AT, APPROVED_BY synapses
    stored_by_syns = [s for s in synapses if s.type == SynapseType.STORED_BY]
    verified_syns = [s for s in synapses if s.type == SynapseType.VERIFIED_AT]
    approved_syns = [s for s in synapses if s.type == SynapseType.APPROVED_BY]

    if stored_by_syns or verified_syns or approved_syns:
        audit: dict[str, Any] = {}
        if stored_by_syns:
            syn = stored_by_syns[0]
            audit["stored_by"] = syn.metadata.get("actor", syn.source_id)
            audit["stored_at"] = syn.created_at.isoformat() if syn.created_at else None
        if verified_syns:
            syn = verified_syns[0]
            audit["verified"] = True
            audit["verified_by"] = syn.metadata.get("actor", syn.source_id)
            audit["verified_at"] = syn.created_at.isoformat() if syn.created_at else None
        else:
            audit["verified"] = False
        if approved_syns:
            syn = approved_syns[0]
            audit["approved_by"] = syn.metadata.get("actor", syn.source_id)
            audit["approved_at"] = syn.created_at.isoformat() if syn.created_at else None
        result["audit"] = audit

    return result


class ToolHandler(RememberHandler, RecallHandler):
    """Mixin providing all MCP tool handler implementations.

    Protocol stubs for attributes/methods used from MCPServer.
    """

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

    # ──────────────────── Core tool handlers ────────────────────

    async def _todo(self, args: dict[str, Any]) -> dict[str, Any]:
        """Add a TODO."""
        task = args.get("task")
        if not task or not isinstance(task, str):
            return {"error": "task is required and must be a string"}
        return await self._remember(
            {
                "content": task,
                "type": "todo",
                "priority": args.get("priority", 5),
                "expires_days": 30,
            }
        )

    def _get_storage_info(self) -> dict[str, Any]:
        """Get current storage backend and Pro status."""
        from neural_memory.plugins import has_pro

        return {
            "backend": self.config.storage_backend,
            "pro_installed": has_pro(),
            "is_pro": self.config.is_pro(),
        }

    async def _stats(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get brain statistics."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

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

        # Tier distribution counts
        tier_distribution = {"hot": 0, "warm": 0, "cold": 0}
        try:
            for tier_name in ("hot", "warm", "cold"):
                tier_distribution[tier_name] = await storage.count_typed_memories(tier=tier_name)
        except Exception:
            logger.debug("Tier distribution count failed (non-critical)", exc_info=True)

        # Storage backend info
        storage_info = self._get_storage_info()

        response = {
            "version": __version__,
            "brain": brain.name,
            "storage_backend": storage_info["backend"],
            "pro_installed": storage_info["pro_installed"],
            "is_pro": storage_info["is_pro"],
            "neuron_count": stats["neuron_count"],
            "synapse_count": stats["synapse_count"],
            "fiber_count": stats["fiber_count"],
            "db_size_bytes": stats.get("db_size_bytes", 0),
            "today_fibers_count": stats.get("today_fibers_count", 0),
            "hot_neurons": stats.get("hot_neurons", []),
            "newest_memory": stats.get("newest_memory"),
            "conflicts_active": conflicts_active,
            "tier_distribution": tier_distribution,
        }

        # Upgrade URL for free users — agents use this to guide purchase
        if not storage_info["is_pro"]:
            from neural_memory.mcp.sync_handler import PRO_LANDING_URL

            response["upgrade_url"] = PRO_LANDING_URL

        # Actionable hints based on brain state
        hints = await self._generate_stats_hints(storage, brain.id, stats)
        if hints:
            response["hints"] = hints

        # Onboarding hint for fresh brains
        onboarding = await self._check_onboarding()
        if onboarding:
            response["onboarding"] = onboarding

        update_hint = self.get_update_hint()
        if update_hint:
            response["update_hint"] = update_hint

        return response

    async def _generate_stats_hints(
        self,
        storage: Any,
        brain_id: str,
        stats: dict[str, Any],
    ) -> list[str]:
        """Generate actionable hints based on brain state.

        Hints appear in stats output to guide users on what to do next.
        """
        hints: list[str] = []
        fiber_count = stats.get("fiber_count", 0)
        neuron_count = stats.get("neuron_count", 0)
        synapse_count = stats.get("synapse_count", 0)

        if fiber_count == 0:
            return hints

        # Consolidation hint: many memories but 0% consolidated
        try:
            from neural_memory.engine.memory_stages import MemoryStage

            semantic_records = await storage.find_maturations(stage=MemoryStage.SEMANTIC)
            semantic_count = len(semantic_records)
            consolidation_pct = (semantic_count / fiber_count * 100) if fiber_count else 0

            if fiber_count >= 50 and consolidation_pct == 0:
                hints.append(
                    f"You have {fiber_count} memories but 0% consolidated. "
                    "Run: nmem_auto action='process' or nmem consolidate --strategy mature "
                    "to advance memories from episodic to semantic stage."
                )
            elif fiber_count >= 100 and consolidation_pct < 10:
                hints.append(
                    f"{fiber_count} memories, only {consolidation_pct:.0f}% consolidated. "
                    "Recall topics you've stored to help memories mature, "
                    "then run consolidation."
                )
        except Exception:
            logger.debug("Maturation check failed (non-critical)", exc_info=True)

        # Low activation hint: many neurons but few activated
        try:
            states = await storage.get_all_neuron_states()
            activated = sum(1 for s in states if s.access_frequency > 0)
            activation_pct = (activated / neuron_count * 100) if neuron_count else 0

            if neuron_count >= 50 and activation_pct < 20:
                idle_count = neuron_count - activated
                hints.append(
                    f"{idle_count} neurons ({100 - activation_pct:.0f}%) never accessed. "
                    "Use nmem_recall with topics you've stored to activate them "
                    "and strengthen recall pathways."
                )
        except Exception:
            logger.debug("Activation check failed (non-critical)", exc_info=True)

        # Low connectivity hint
        if neuron_count > 0:
            connectivity = synapse_count / neuron_count
            if connectivity < 2.0 and neuron_count >= 20:
                hints.append(
                    f"Low connectivity ({connectivity:.1f} synapses/neuron, target: 3+). "
                    "Store memories with context like 'X because Y' to build richer links."
                )

        # Spaced repetition hint: if review system has due items
        try:
            from neural_memory.engine.spaced_repetition import SpacedRepetitionEngine

            brain = await storage.get_brain(brain_id)
            if brain:
                review_engine = SpacedRepetitionEngine(storage, brain.config)
                review_stats = await review_engine.get_stats()
                due_count = review_stats.get("due", 0)
                if due_count > 0:
                    hints.append(
                        f"{due_count} memories due for review. "
                        "Run nmem_review action='queue' to strengthen retention."
                    )
        except Exception:
            logger.debug("Review check failed (non-critical)", exc_info=True)

        # InfinityDB upgrade hint: Pro active but still on SQLite
        try:
            info = self._get_storage_info()
            if (
                info["is_pro"]
                and info["pro_installed"]
                and info["backend"] == "sqlite"
                and neuron_count >= 100
            ):
                hints.append(
                    "Pro is active but you're on SQLite. "
                    "InfinityDB offers HNSW indexing, tiered compression, and cone queries. "
                    "Run: nmem storage status → nmem migrate infinitydb → nmem storage switch infinitydb"
                )
        except Exception:
            logger.debug("InfinityDB hint check failed (non-critical)", exc_info=True)

        return hints

    async def _health(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run brain health diagnostics."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return {"error": "No brain configured"}

        from neural_memory.engine.diagnostics import DiagnosticsEngine

        engine = DiagnosticsEngine(storage)
        report = await engine.analyze(brain.id)

        result: dict[str, Any] = {
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
            "roadmap": self._build_health_roadmap(report),
        }

        # Deep analysis: Gromov delta-hyperbolicity (expensive, opt-in, cached 1h)
        if args.get("deep", False):
            try:
                from neural_memory.engine.gromov import estimate_gromov_delta

                cache_key = brain.id if brain else "_default"
                now = utcnow()
                cached = _gromov_cache.get(cache_key)
                if cached and (now - cached[0]).total_seconds() < 3600:
                    gromov = cached[1]
                else:
                    gromov = await estimate_gromov_delta(storage, sample_size=200)
                    _gromov_cache[cache_key] = (now, gromov)
                result["gromov"] = {
                    "delta": gromov.delta,
                    "normalized_delta": gromov.normalized_delta,
                    "structure_quality": gromov.structure_quality,
                    "sample_count": gromov.sample_count,
                    "tuple_count": gromov.tuple_count,
                    "diameter": gromov.diameter,
                }
            except Exception:
                logger.debug("Gromov delta estimation failed", exc_info=True)
                result["gromov"] = {"error": "estimation failed"}

        # Pro hint for large brains
        total_entities = (
            result.get("fiber_count", 0)
            + result.get("neuron_count", 0)
            + result.get("synapse_count", 0)
        )
        if total_entities > 500:
            result["pro_hints"] = [
                f"Brain has {total_entities} entities. "
                "Pro: Merkle delta sync keeps multi-device brains in sync ~95% faster."
            ]

        return result

    @staticmethod
    def _build_health_roadmap(report: Any) -> dict[str, Any]:
        """Build an actionable roadmap from current grade to next grade.

        Shows prioritized steps sorted by estimated_gain (biggest impact first),
        the points needed to reach the next grade, and specific commands to run.
        """
        grade_thresholds = {"F": 40, "D": 60, "C": 75, "B": 90, "A": 100}
        next_grade_map = {"F": "D", "D": "C", "C": "B", "B": "A", "A": "A"}

        current_grade = report.grade
        next_grade = next_grade_map.get(current_grade, "A")
        target_score = grade_thresholds.get(next_grade, 100)
        points_needed = max(0, target_score - report.purity_score)

        # Sort penalties by estimated gain (most impactful first)
        steps: list[dict[str, Any]] = []
        cumulative_gain = 0.0
        for p in sorted(report.top_penalties, key=lambda x: x.estimated_gain, reverse=True):
            if p.estimated_gain <= 0:
                continue
            cumulative_gain += p.estimated_gain
            steps.append(
                {
                    "priority": len(steps) + 1,
                    "component": p.component,
                    "current": f"{p.current_score:.0%}",
                    "action": p.action,
                    "estimated_gain": f"+{p.estimated_gain:.1f} pts",
                    "sufficient": cumulative_gain >= points_needed,
                }
            )

        # Estimate timeframe based on points needed
        if points_needed <= 0:
            timeframe = "Already achieved"
        elif points_needed <= 5:
            timeframe = "~1 week with daily use"
        elif points_needed <= 15:
            timeframe = "~2 weeks with regular use"
        elif points_needed <= 30:
            timeframe = "~1 month with consistent use"
        else:
            timeframe = "~2 months with dedicated effort"

        roadmap: dict[str, Any] = {
            "current_grade": current_grade,
            "current_score": report.purity_score,
            "next_grade": next_grade,
            "points_needed": round(points_needed, 1),
            "timeframe": timeframe,
            "steps": steps,
        }

        if current_grade == "A":
            roadmap["message"] = (
                "Excellent! Brain is at top grade. Maintain regular recall and storage."
            )
        elif points_needed <= sum(p.estimated_gain for p in report.top_penalties):
            roadmap["message"] = (
                f"Grade {current_grade} → {next_grade} is achievable in {timeframe} "
                f"by addressing the top {min(len(steps), 3)} actions below."
            )
        else:
            roadmap["message"] = (
                f"Grade {next_grade} requires {points_needed:.1f} more points ({timeframe}). "
                "Focus on the highest-impact actions and give the brain time to mature."
            )

        return roadmap

    async def _evolution(self, args: dict[str, Any]) -> dict[str, Any]:
        """Measure brain evolution dynamics."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

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
        """Get prefix-based autocomplete suggestions or idle neuron reinforcement hints."""
        storage = await self.get_storage()
        prefix = args.get("prefix", "")
        limit = min(args.get("limit", 5), 20)

        # When no prefix: return idle neurons that need reinforcement
        if not prefix.strip():
            return await self._suggest_idle_neurons(storage, limit)

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

    async def _suggest_idle_neurons(self, storage: Any, limit: int) -> dict[str, Any]:
        """Return neurons that have never been accessed — candidates for reinforcement.

        Sorted by creation age (oldest idle neurons first) to prioritize
        long-neglected knowledge.
        """
        try:
            states = await storage.get_all_neuron_states()
            idle_states = [s for s in states if s.access_frequency == 0]

            # Sort by creation time ascending (oldest first)
            idle_states.sort(key=lambda s: s.created_at or "")

            suggestions = []
            for state in idle_states[:limit]:
                neuron = await storage.get_neuron(state.neuron_id)
                if neuron is None:
                    continue
                content_preview = neuron.content[:200] if neuron.content else ""
                suggestions.append(
                    {
                        "content": content_preview,
                        "type": neuron.type.value if neuron.type else "unknown",
                        "neuron_id": neuron.id,
                        "score": 0.0,
                        "idle": True,
                    }
                )

            total_idle = len(idle_states)
            hint = ""
            if total_idle > 0:
                hint = (
                    f"{total_idle} neurons never accessed. "
                    "Recall these topics with nmem_recall to activate them "
                    "and strengthen your memory graph."
                )

            return {
                "suggestions": suggestions,
                "count": len(suggestions),
                "total_idle": total_idle,
                "mode": "idle_reinforcement",
                "hint": hint,
                "tokens_used": sum(len(s["content"].split()) for s in suggestions),
            }
        except Exception:
            logger.debug("Idle neuron suggestion failed", exc_info=True)
            return {"suggestions": [], "count": 0}

    async def _habits(self, args: dict[str, Any]) -> dict[str, Any]:
        """Manage learned workflow habits."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

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
            habits = await storage.find_fibers(metadata_key="_habit_pattern", limit=1000)
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
            habits = await storage.find_fibers(metadata_key="_habit_pattern", limit=1000)
            cleared = 0
            # Delete sequentially to avoid overwhelming SQLite with concurrent writes
            for h in habits:
                await storage.delete_fiber(h.id)
                cleared += 1
            return {"cleared": cleared, "message": f"Cleared {cleared} learned habits"}

        return {"error": f"Unknown action: {action}"}

    async def _version(self, args: dict[str, Any]) -> dict[str, Any]:
        """Brain version control operations."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

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
                logger.error("Version create failed for brain '%s' name '%s'", brain.id, name)
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
                logger.error(
                    "Version rollback failed for brain '%s' version '%s'", brain.id, version_id
                )
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
                logger.error(
                    "Version diff failed for brain '%s' from '%s' to '%s'", brain.id, from_id, to_id
                )
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
        target_brain_id = target_storage.brain_id
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

        source_brain_id = source_storage.brain_id
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

    # ========== Source Registry ==========

    async def _source(self, args: dict[str, Any]) -> dict[str, Any]:
        """Manage memory sources (provenance registry)."""
        action = args.get("action", "")
        if not action:
            return {"error": "action is required"}

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for source action '%s'", action)
            return {"error": "No brain configured"}

        if action == "register":
            name = args.get("name")
            if not name or not isinstance(name, str):
                return {"error": "name is required for register"}

            from neural_memory.core.source import Source

            try:
                source = Source.create(
                    brain_id=brain_id,
                    name=name,
                    source_type=args.get("source_type", "document"),
                    version=args.get("version", ""),
                    file_hash=args.get("file_hash", ""),
                    metadata=args.get("metadata") or {},
                )
            except ValueError:
                return {"error": f"Invalid source_type: {args.get('source_type')}"}
            source_id = await storage.add_source(source)
            return {
                "source_id": source_id,
                "name": source.name,
                "source_type": source.source_type.value,
                "status": source.status.value,
            }

        if action == "list":
            sources = await storage.list_sources(
                source_type=args.get("source_type"),
                status=args.get("status"),
            )
            return {
                "sources": [
                    {
                        "source_id": s.id,
                        "name": s.name,
                        "source_type": s.source_type.value,
                        "version": s.version,
                        "status": s.status.value,
                        "created_at": s.created_at.isoformat(),
                    }
                    for s in sources
                ],
                "count": len(sources),
            }

        if action == "get":
            source_id = str(args.get("source_id") or "")
            if not source_id:
                return {"error": "source_id is required for get"}
            source = await storage.get_source(source_id)
            if source is None:
                return {"error": f"Source '{source_id}' not found"}
            neuron_count = await storage.count_neurons_for_source(source_id)
            return {
                "source_id": source.id,
                "name": source.name,
                "source_type": source.source_type.value,
                "version": source.version,
                "status": source.status.value,
                "file_hash": source.file_hash,
                "metadata": source.metadata,
                "linked_neuron_count": neuron_count,
                "created_at": source.created_at.isoformat(),
                "updated_at": source.updated_at.isoformat(),
            }

        if action == "update":
            source_id = str(args.get("source_id") or "")
            if not source_id:
                return {"error": "source_id is required for update"}
            updated = await storage.update_source(
                source_id,
                status=args.get("status"),
                version=args.get("version"),
                metadata=args.get("metadata"),
            )
            if not updated:
                return {"error": f"Source '{source_id}' not found"}
            return {"updated": True, "source_id": source_id}

        if action == "delete":
            source_id = str(args.get("source_id") or "")
            if not source_id:
                return {"error": "source_id is required for delete"}
            # Warn about linked neurons
            neuron_count = await storage.count_neurons_for_source(source_id)
            if neuron_count > 0:
                # Soft-delete: mark superseded instead of hard delete
                await storage.update_source(source_id, status="superseded")
                return {
                    "deleted": False,
                    "superseded": True,
                    "source_id": source_id,
                    "warning": f"Source has {neuron_count} linked neurons. "
                    "Marked as superseded instead of deleted.",
                }
            deleted = await storage.delete_source(source_id)
            if not deleted:
                return {"error": f"Source '{source_id}' not found"}
            return {"deleted": True, "source_id": source_id}

        return {"error": f"Unknown action: {action}"}

    # ========== Provenance ==========

    async def _provenance(self, args: dict[str, Any]) -> dict[str, Any]:
        """Trace provenance, verify, or approve a neuron."""
        action = args.get("action", "")
        if not action:
            return {"error": "action is required (trace, verify, approve)"}

        neuron_id = args.get("neuron_id")
        if not neuron_id or not isinstance(neuron_id, str):
            return {"error": "neuron_id is required"}

        storage = await self.get_storage()
        _require_brain_id(storage)

        # Verify neuron exists
        neuron = await storage.get_neuron(neuron_id)
        if neuron is None:
            return {"error": f"Neuron '{neuron_id}' not found"}

        if action == "trace":
            return await self._provenance_trace(storage, neuron_id)

        if action == "verify":
            actor = args.get("actor", "mcp_agent")
            return await self._provenance_add_audit(
                storage, neuron_id, SynapseType.VERIFIED_AT, actor
            )

        if action == "approve":
            actor = args.get("actor", "mcp_agent")
            return await self._provenance_add_audit(
                storage, neuron_id, SynapseType.APPROVED_BY, actor
            )

        return {"error": f"Unknown action: {action}. Use trace, verify, or approve."}

    async def _provenance_trace(self, storage: NeuralStorage, neuron_id: str) -> dict[str, Any]:
        """Trace full provenance chain for a neuron."""
        synapses = await storage.get_synapses(target_id=neuron_id)

        chain: list[dict[str, Any]] = []

        for syn in synapses:
            if syn.type == SynapseType.SOURCE_OF:
                source_obj = await storage.get_source(syn.source_id)
                chain.append(
                    {
                        "type": "source",
                        "source_id": syn.source_id,
                        "source_name": source_obj.name if source_obj else None,
                        "source_type": source_obj.source_type.value if source_obj else None,
                        "timestamp": syn.created_at.isoformat() if syn.created_at else None,
                    }
                )
            elif syn.type == SynapseType.STORED_BY:
                chain.append(
                    {
                        "type": "stored_by",
                        "actor": syn.metadata.get("actor", syn.source_id),
                        "tool": syn.metadata.get("tool"),
                        "timestamp": syn.created_at.isoformat() if syn.created_at else None,
                    }
                )
            elif syn.type == SynapseType.VERIFIED_AT:
                chain.append(
                    {
                        "type": "verified",
                        "actor": syn.metadata.get("actor", syn.source_id),
                        "timestamp": syn.created_at.isoformat() if syn.created_at else None,
                    }
                )
            elif syn.type == SynapseType.APPROVED_BY:
                chain.append(
                    {
                        "type": "approved",
                        "actor": syn.metadata.get("actor", syn.source_id),
                        "timestamp": syn.created_at.isoformat() if syn.created_at else None,
                    }
                )

        return {
            "neuron_id": neuron_id,
            "provenance": chain,
            "has_source": any(e["type"] == "source" for e in chain),
            "is_verified": any(e["type"] == "verified" for e in chain),
            "is_approved": any(e["type"] == "approved" for e in chain),
        }

    async def _provenance_add_audit(
        self,
        storage: NeuralStorage,
        neuron_id: str,
        synapse_type: SynapseType,
        actor: str,
    ) -> dict[str, Any]:
        """Add a VERIFIED_AT or APPROVED_BY audit synapse."""
        syn = Synapse.create(
            source_id=neuron_id,
            target_id=neuron_id,  # self-referencing audit
            type=synapse_type,
            weight=1.0,
            metadata={"actor": actor, "tool": "nmem_provenance"},
        )
        await storage.add_synapse(syn)
        return {
            "success": True,
            "neuron_id": neuron_id,
            "action": synapse_type.value,
            "actor": actor,
            "synapse_id": syn.id,
        }

    # ========== Show, Edit & Forget ==========

    async def _show(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get full verbatim content + metadata + synapses for a memory by ID."""
        memory_id = args.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            return {"error": "memory_id is required"}

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for show")
            return {"error": "No brain configured"}

        # Try as fiber_id first (typed memory), then as neuron_id
        typed_mem = await storage.get_typed_memory(memory_id)
        fiber = await storage.get_fiber(memory_id) if typed_mem else None

        if typed_mem and fiber:
            anchor = await storage.get_neuron(fiber.anchor_neuron_id)
            content = anchor.content if anchor else ""

            # Decrypt if needed
            if anchor and fiber.metadata.get("encrypted"):
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
                    logger.debug("Decryption failed in show", exc_info=True)

            # Get connected synapses
            synapses_out = await storage.get_synapses(source_id=fiber.anchor_neuron_id)
            synapses_in = await storage.get_synapses(target_id=fiber.anchor_neuron_id)
            synapse_list = [
                {
                    "type": s.type.value if hasattr(s.type, "value") else str(s.type),
                    "target_id": s.target_id,
                    "source_id": s.source_id,
                    "weight": s.weight,
                }
                for s in [*synapses_out, *synapses_in]
            ]

            return {
                "memory_id": memory_id,
                "content": content,
                "memory_type": typed_mem.memory_type.value,
                "priority": typed_mem.priority.value,
                "tags": list(typed_mem.tags) if typed_mem.tags else [],
                "created_at": fiber.created_at.isoformat() if fiber.created_at else None,
                "anchor_neuron_id": fiber.anchor_neuron_id,
                "neuron_count": fiber.neuron_count,
                "summary": fiber.summary,
                "metadata": fiber.metadata,
                "synapses": synapse_list,
                "trust_score": typed_mem.trust_score,
                "expires_at": typed_mem.expires_at.isoformat() if typed_mem.expires_at else None,
            }

        # Try as direct neuron_id
        neuron = await storage.get_neuron(memory_id)
        if neuron:
            synapses_out = await storage.get_synapses(source_id=memory_id)
            synapses_in = await storage.get_synapses(target_id=memory_id)
            synapse_list = [
                {
                    "type": s.type.value if hasattr(s.type, "value") else str(s.type),
                    "target_id": s.target_id,
                    "source_id": s.source_id,
                    "weight": s.weight,
                }
                for s in [*synapses_out, *synapses_in]
            ]

            return {
                "memory_id": memory_id,
                "content": neuron.content,
                "neuron_type": neuron.type.value
                if hasattr(neuron.type, "value")
                else str(neuron.type),
                "created_at": neuron.created_at.isoformat() if neuron.created_at else None,
                "metadata": neuron.metadata,
                "synapses": synapse_list,
            }

        return {"error": "Memory not found"}

    async def _edit(self, args: dict[str, Any]) -> dict[str, Any]:
        """Edit an existing memory's type, content, priority, or tier."""
        memory_id = args.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            return {"error": "memory_id is required"}

        new_type = args.get("type")
        new_content = args.get("content")
        new_priority = args.get("priority")
        new_tier = args.get("tier")
        if new_tier is not None:
            new_tier = str(new_tier).lower().strip()

        if new_type is None and new_content is None and new_priority is None and new_tier is None:
            return {"error": "At least one of type, content, priority, or tier must be provided"}

        if new_type is not None:
            try:
                MemoryType(new_type)
            except ValueError:
                return {"error": f"Invalid memory type: {new_type}"}

        if new_tier is not None:
            try:
                MemoryTier(new_tier)
            except ValueError:
                return {"error": f"Invalid tier: {new_tier}. Must be hot, warm, or cold."}

        if new_content is not None and len(new_content) > MAX_CONTENT_LENGTH:
            return {
                "error": f"Content too long ({len(new_content)} chars). Max: {MAX_CONTENT_LENGTH}."
            }

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for edit")
            return {"error": "No brain configured"}

        # Try as fiber_id first, then as neuron_id
        typed_mem = await storage.get_typed_memory(memory_id)
        fiber = await storage.get_fiber(memory_id) if typed_mem else None

        if typed_mem and fiber:
            # Edit via fiber path
            changes: list[str] = []

            # Update typed_memory (type, priority, tier)
            if new_type is not None or new_priority is not None or new_tier is not None:
                from dataclasses import replace as dc_replace

                updated_tm = typed_mem
                if new_type is not None:
                    updated_tm = dc_replace(updated_tm, memory_type=MemoryType(new_type))
                    changes.append(f"type: {typed_mem.memory_type.value} → {new_type}")
                    # Sync type into fiber.metadata to keep both stores consistent
                    updated_meta = {**fiber.metadata, "type": new_type}
                    fiber = dc_replace(fiber, metadata=updated_meta)
                    await storage.update_fiber(fiber)
                    # Enforce boundary invariant: boundaries are always HOT
                    if (
                        updated_tm.memory_type == MemoryType.BOUNDARY
                        and updated_tm.tier != MemoryTier.HOT
                    ):
                        old_tier = updated_tm.tier
                        updated_tm = updated_tm.with_tier(MemoryTier.HOT)
                        changes.append(f"tier: {old_tier} → hot (boundary auto-promote)")
                if new_priority is not None:
                    updated_tm = dc_replace(updated_tm, priority=Priority.from_int(new_priority))
                    changes.append(f"priority: {typed_mem.priority.value} → {new_priority}")
                if new_tier is not None:
                    old_tier = updated_tm.tier
                    updated_tm = updated_tm.with_tier(new_tier)
                    if updated_tm.tier != old_tier:
                        changes.append(f"tier: {old_tier} → {updated_tm.tier}")
                await storage.update_typed_memory(updated_tm)

            # Update anchor neuron content
            if new_content is not None:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    from dataclasses import replace as dc_replace

                    updated_neuron = dc_replace(anchor, content=new_content)
                    await storage.update_neuron(updated_neuron)
                    changes.append(f"content updated ({len(new_content)} chars)")

            return {
                "status": "edited",
                "memory_id": memory_id,
                "changes": changes,
            }

        # Try as direct neuron_id
        neuron = await storage.get_neuron(memory_id)
        if neuron:
            from dataclasses import replace as dc_replace

            changes = []
            if new_content is not None:
                neuron = dc_replace(neuron, content=new_content)
                changes.append(f"content updated ({len(new_content)} chars)")
            if new_type is not None:
                from neural_memory.core.neuron import NeuronType

                try:
                    neuron = dc_replace(neuron, type=NeuronType(new_type))
                    changes.append(f"neuron type → {new_type}")
                except ValueError:
                    pass  # NeuronType doesn't map 1:1 to MemoryType
            await storage.update_neuron(neuron)
            return {
                "status": "edited",
                "memory_id": memory_id,
                "changes": changes,
            }

        return {"error": "Memory not found"}

    async def _forget(self, args: dict[str, Any]) -> dict[str, Any]:
        """Explicitly delete or close a specific memory."""
        memory_id = args.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            return {"error": "memory_id is required"}

        hard = args.get("hard", False)
        reason = args.get("reason", "")

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for forget")
            return {"error": "No brain configured"}

        # Look up the memory
        typed_mem = await storage.get_typed_memory(memory_id)
        fiber = await storage.get_fiber(memory_id) if typed_mem else None

        if not typed_mem and not fiber:
            # Try as neuron_id — find its fiber
            neuron = await storage.get_neuron(memory_id)
            if not neuron:
                return {"error": "Memory not found"}
            # For neuron-only delete in hard mode
            if hard:
                await storage.delete_neuron(memory_id)
                return {
                    "status": "hard_deleted",
                    "memory_id": memory_id,
                    "message": "Neuron permanently deleted",
                }
            return {
                "error": f"No typed memory found for neuron {memory_id}. Use hard=true for neuron deletion."
            }

        if hard:
            # Permanent deletion: fiber + typed_memory + neurons
            storage.disable_auto_save()
            try:
                # Delete typed memory
                await storage.delete_typed_memory(memory_id)

                # Delete fiber (CASCADE handles fiber_neurons junction)
                if fiber:
                    await storage.delete_fiber(memory_id)

                await storage.batch_save()
            finally:
                storage.enable_auto_save()

            logger.info("Hard-deleted memory %s (reason: %s)", memory_id, reason or "none")
            return {
                "status": "hard_deleted",
                "memory_id": memory_id,
                "message": "Memory permanently deleted with cascade cleanup",
            }
        else:
            # Soft delete: expire immediately
            from dataclasses import replace as dc_replace

            assert typed_mem is not None  # guaranteed by early return above
            expired_tm = dc_replace(typed_mem, expires_at=utcnow())
            await storage.update_typed_memory(expired_tm)

            logger.info("Soft-deleted memory %s (reason: %s)", memory_id, reason or "none")
            return {
                "status": "soft_deleted",
                "memory_id": memory_id,
                "message": "Memory marked as expired (will be cleaned up on next consolidation)",
            }

    async def _consolidate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run memory consolidation on the current brain."""
        from neural_memory.engine.consolidation import (
            ConsolidationConfig,
            ConsolidationStrategy,
        )
        from neural_memory.engine.consolidation_delta import run_with_delta

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for consolidate")
            return {"error": "No brain configured"}

        # Parse strategy
        strategy_str = args.get("strategy", "all")
        try:
            strategy = ConsolidationStrategy(strategy_str)
        except ValueError:
            valid = [s.value for s in ConsolidationStrategy]
            return {"error": f"Invalid strategy: {strategy_str}. Valid: {valid}"}

        strategies = [strategy]
        dry_run = bool(args.get("dry_run", False))

        # Build config with optional overrides
        config_kwargs: dict[str, Any] = {}
        if "prune_weight_threshold" in args:
            val = args["prune_weight_threshold"]
            if isinstance(val, (int, float)):
                config_kwargs["prune_weight_threshold"] = float(val)
        if "merge_overlap_threshold" in args:
            val = args["merge_overlap_threshold"]
            if isinstance(val, (int, float)):
                config_kwargs["merge_overlap_threshold"] = float(val)
        if "prune_min_inactive_days" in args:
            val = args["prune_min_inactive_days"]
            if isinstance(val, (int, float)):
                config_kwargs["prune_min_inactive_days"] = float(val)

        config = ConsolidationConfig(**config_kwargs) if config_kwargs else None

        try:
            delta = await run_with_delta(
                storage,
                brain_id,
                strategies=strategies,
                dry_run=dry_run,
                config=config,
            )
        except Exception:
            logger.error("Consolidation failed", exc_info=True)
            return {"error": "Consolidation failed unexpectedly"}

        result = delta.to_dict()
        result["strategy"] = strategy_str
        result["dry_run"] = dry_run
        result["summary"] = delta.report.summary()
        return result

    async def _tool_stats(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get tool usage analytics."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        action = args.get("action", "summary")
        days = args.get("days", 30)
        limit = args.get("limit", 20)

        if action == "summary":
            result: dict[str, Any] = await storage.get_tool_stats(brain.id)  # type: ignore[attr-defined]
            return result
        elif action == "daily":
            daily = await storage.get_tool_stats_by_period(  # type: ignore[attr-defined]
                brain.id, days=days, limit=limit
            )
            return {"daily": daily, "days": days}
        else:
            return {"error": f"Unknown action: {action}"}

    async def _lifecycle(self, args: dict[str, Any]) -> dict[str, Any]:
        """Memory lifecycle management: status, recover, freeze, thaw."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        action = args.get("action", "status")
        neuron_id: str | None = args.get("id") or args.get("neuron_id")

        if action == "status":
            try:
                distribution = await storage.get_lifecycle_distribution()
            except Exception:
                logger.error("nmem_lifecycle status failed", exc_info=True)
                return {"error": "Failed to retrieve lifecycle distribution"}
            total = sum(distribution.values())
            return {
                "brain": brain.id,
                "distribution": distribution,
                "total_neurons": total,
            }

        if action in ("recover", "freeze", "thaw"):
            if not neuron_id:
                return {"error": f"action='{action}' requires 'id' (neuron_id)"}

            if action == "recover":
                # Find which fiber contains this neuron, then recover.
                from neural_memory.engine.compression import CompressionEngine

                fibers = await storage.find_fibers(contains_neuron=neuron_id)
                if not fibers:
                    # Try decompress by fiber_id directly (caller may pass fiber_id as id)
                    engine = CompressionEngine(storage)
                    success = await engine.recover_fiber(neuron_id)
                    if success:
                        return {"recovered": True, "fiber_id": neuron_id}
                    return {
                        "recovered": False,
                        "reason": "No fiber found for neuron and direct recovery failed",
                    }

                fiber = fibers[0]
                engine = CompressionEngine(storage)
                success = await engine.recover_fiber(fiber.id)
                return {
                    "recovered": success,
                    "fiber_id": fiber.id,
                    "neuron_id": neuron_id,
                }

            elif action == "freeze":
                try:
                    await storage.update_neuron_frozen(neuron_id, frozen=True)
                except Exception:
                    logger.error("nmem_lifecycle freeze failed for %s", neuron_id, exc_info=True)
                    return {"error": "Failed to freeze neuron"}
                return {"frozen": True, "neuron_id": neuron_id}

            elif action == "thaw":
                try:
                    await storage.update_neuron_frozen(neuron_id, frozen=False)
                except Exception:
                    logger.error("nmem_lifecycle thaw failed for %s", neuron_id, exc_info=True)
                    return {"error": "Failed to thaw neuron"}
                return {"frozen": False, "neuron_id": neuron_id}

        return {"error": f"Unknown action: {action}. Valid: status, recover, freeze, thaw"}

    # ──────────────────── Adaptive Instructions ────────────────────

    def _ensure_instruction_meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Backfill missing instruction metadata fields (non-destructive).

        Returns a new dict with all required instruction fields present.
        Existing keys in *meta* are preserved unchanged.
        """
        defaults: dict[str, Any] = {
            "version": 1,
            "execution_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": None,
            "last_executed_at": None,
            "failure_modes": [],
            "trigger_patterns": [],
            "refinement_history": [],
        }
        return {**defaults, **meta}

    async def _refine(self, args: dict[str, Any]) -> dict[str, Any]:
        """Refine an instruction or workflow memory.

        Increments the version counter, stores a snapshot in refinement_history,
        appends failure modes / trigger patterns, and persists the updated metadata.
        """
        neuron_id = args.get("neuron_id")
        if not neuron_id or not isinstance(neuron_id, str):
            return {"error": "neuron_id is required"}

        new_content = args.get("new_content")
        reason = args.get("reason", "")
        add_failure_mode = args.get("add_failure_mode")
        add_trigger = args.get("add_trigger")

        if new_content is None and not add_failure_mode and not add_trigger:
            return {
                "error": "At least one of new_content, add_failure_mode, or add_trigger is required"
            }

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for refine")
            return {"error": "No brain configured"}

        # Resolve fiber by ID
        typed_mem = await storage.get_typed_memory(neuron_id)
        fiber = await storage.get_fiber(neuron_id) if typed_mem else None

        if not typed_mem or not fiber:
            return {"error": "Memory not found"}

        # Validate instruction/workflow type
        mem_type_val = typed_mem.memory_type.value
        if mem_type_val not in ("instruction", "workflow"):
            return {
                "error": f"nmem_refine only supports instruction/workflow memories, got '{mem_type_val}'"
            }

        # Backfill metadata if old neuron lacks instruction fields
        meta = self._ensure_instruction_meta(dict(fiber.metadata))

        changes: list[str] = []

        if new_content is not None:
            if len(new_content) > MAX_CONTENT_LENGTH:
                return {
                    "error": f"Content too long ({len(new_content)} chars). Max: {MAX_CONTENT_LENGTH}."
                }
            # Fetch anchor to snapshot old content
            anchor = await storage.get_neuron(fiber.anchor_neuron_id)
            old_content = anchor.content if anchor else ""

            # Increment version and record history
            old_version = meta["version"]
            new_version = old_version + 1
            history_entry: dict[str, Any] = {
                "version": old_version,
                "changed_at": utcnow().isoformat(),
                "reason": reason,
                "old_content": old_content[:100],
            }
            refinement_history: list[dict[str, Any]] = list(meta["refinement_history"])
            if len(refinement_history) >= 10:
                refinement_history = refinement_history[-9:]
            refinement_history.append(history_entry)

            meta = {
                **meta,
                "version": new_version,
                "refinement_history": refinement_history,
            }

            # Update anchor neuron content
            if anchor:
                from dataclasses import replace as dc_replace

                updated_neuron = dc_replace(anchor, content=new_content)
                await storage.update_neuron(updated_neuron)
                changes.append(f"content updated (v{old_version}→v{new_version})")

        if add_failure_mode:
            failure_modes: list[str] = list(meta["failure_modes"])
            if add_failure_mode not in failure_modes:
                failure_modes = [*failure_modes, add_failure_mode]
            if len(failure_modes) > 20:
                failure_modes = failure_modes[-20:]
            meta = {**meta, "failure_modes": failure_modes}
            changes.append(f"failure_mode added: {add_failure_mode[:60]}")

        if add_trigger:
            trigger_patterns: list[str] = list(meta["trigger_patterns"])
            normalized_trigger = add_trigger.lower().strip()
            if normalized_trigger and normalized_trigger not in trigger_patterns:
                trigger_patterns = [*trigger_patterns, normalized_trigger]
            if len(trigger_patterns) > 10:
                trigger_patterns = trigger_patterns[-10:]
            meta = {**meta, "trigger_patterns": trigger_patterns}
            changes.append(f"trigger added: {normalized_trigger}")

        # Persist updated fiber metadata
        await storage.update_fiber_metadata(fiber.id, meta)

        return {
            "status": "refined",
            "memory_id": neuron_id,
            "changes": changes,
            "metadata": {
                "version": meta["version"],
                "execution_count": meta["execution_count"],
                "success_count": meta["success_count"],
                "failure_count": meta["failure_count"],
                "success_rate": meta["success_rate"],
                "last_executed_at": meta["last_executed_at"],
                "failure_modes": meta["failure_modes"],
                "trigger_patterns": meta["trigger_patterns"],
                "refinement_history": meta["refinement_history"],
            },
        }

    async def _report_outcome(self, args: dict[str, Any]) -> dict[str, Any]:
        """Report execution outcome for an instruction or workflow memory.

        Increments execution_count, success_count or failure_count, recomputes
        success_rate, updates last_executed_at, and optionally records failure modes.
        """
        neuron_id = args.get("neuron_id")
        if not neuron_id or not isinstance(neuron_id, str):
            return {"error": "neuron_id is required"}

        raw_success = args.get("success")
        if raw_success is None:
            return {"error": "success (bool) is required"}
        success = bool(raw_success)

        failure_description = args.get("failure_description")
        # context arg accepted but used only for logging / future linking
        _context = args.get("context")

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for report_outcome")
            return {"error": "No brain configured"}

        typed_mem = await storage.get_typed_memory(neuron_id)
        fiber = await storage.get_fiber(neuron_id) if typed_mem else None

        if not typed_mem or not fiber:
            return {"error": "Memory not found"}

        mem_type_val = typed_mem.memory_type.value
        if mem_type_val not in ("instruction", "workflow"):
            return {
                "error": f"nmem_report_outcome only supports instruction/workflow memories, got '{mem_type_val}'"
            }

        meta = self._ensure_instruction_meta(dict(fiber.metadata))

        # Increment counters
        new_exec_count = meta["execution_count"] + 1
        new_success_count = meta["success_count"] + (1 if success else 0)
        new_failure_count = meta["failure_count"] + (0 if success else 1)
        new_success_rate = new_success_count / new_exec_count
        new_last_executed = utcnow().isoformat()

        # Append failure mode if provided
        failure_modes: list[str] = list(meta["failure_modes"])
        if not success and failure_description:
            if failure_description not in failure_modes:
                failure_modes = [*failure_modes, failure_description]
            if len(failure_modes) > 20:
                failure_modes = failure_modes[-20:]

        updated_meta = {
            **meta,
            "execution_count": new_exec_count,
            "success_count": new_success_count,
            "failure_count": new_failure_count,
            "success_rate": round(new_success_rate, 4),
            "last_executed_at": new_last_executed,
            "failure_modes": failure_modes,
        }

        await storage.update_fiber_metadata(fiber.id, updated_meta)

        return {
            "status": "recorded",
            "memory_id": neuron_id,
            "success": success,
            "execution_count": new_exec_count,
            "success_count": new_success_count,
            "failure_count": new_failure_count,
            "success_rate": round(new_success_rate, 4),
            "last_executed_at": new_last_executed,
            "failure_modes": failure_modes,
        }

    async def _budget(self, args: dict[str, Any]) -> dict[str, Any]:
        """Token budget analysis for recall context allocation."""
        action = args.get("action", "")
        if action not in ("estimate", "analyze", "optimize"):
            return {
                "error": f"Invalid action: {action!r}. Must be 'estimate', 'analyze', or 'optimize'."
            }

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for budget analysis")
            return {"error": "No brain configured"}

        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        max_tokens = min(
            int(args.get("max_tokens", self.config.budget.default_tokens)), MAX_TOKEN_BUDGET
        )

        from neural_memory.engine.token_budget import (
            BudgetConfig,
            allocate_budget,
            compute_token_costs,
            format_budget_report,
        )

        budget_cfg = BudgetConfig(
            system_overhead_tokens=self.config.budget.system_overhead,
            per_fiber_overhead=self.config.budget.per_fiber_overhead,
        )

        if action == "estimate":
            query = args.get("query", "")
            if not query or not isinstance(query, str):
                return {"error": "query is required for action='estimate'"}

            try:
                pipeline = ReflexPipeline(storage, brain.config)
                result = await pipeline.query(
                    query=query,
                    depth=DepthLevel(0),  # Shallow — just activation, no heavy traversal
                    max_tokens=max_tokens,
                    reference_time=utcnow(),
                )
            except Exception:
                logger.error("Budget estimate pipeline failed", exc_info=True)
                return {"error": "Failed to run retrieval pipeline for estimate"}

            # Fetch fiber objects for the matched fibers
            fibers: list[Any] = []
            for fid in result.fibers_matched or []:
                fiber = await storage.get_fiber(fid)
                if fiber:
                    fibers.append(fiber)

            # Build activations map from co_activations (RetrievalResult has no .activations field)
            from neural_memory.engine.activation import ActivationResult

            estimate_activations: dict[str, ActivationResult] = {}
            for co in result.co_activations:
                for nid in co.neuron_ids:
                    if nid not in estimate_activations:
                        estimate_activations[nid] = ActivationResult(
                            neuron_id=nid,
                            activation_level=co.binding_strength,
                            hop_distance=0,
                            path=[nid],
                            source_anchor=nid,
                        )

            costs = compute_token_costs(fibers, estimate_activations, budget_cfg)
            allocation = allocate_budget(costs, max_tokens, budget_cfg)
            report = format_budget_report(allocation)

            return {
                "action": "estimate",
                "query": query,
                "max_tokens": max_tokens,
                "neurons_activated": result.neurons_activated,
                "fibers_found": len(fibers),
                "budget": report,
                "would_drop": allocation.fibers_dropped,
                "confidence": result.confidence,
            }

        elif action == "analyze":
            # Profile the brain's token cost distribution by memory type
            try:
                fibers_list = await storage.get_fibers(limit=min(200, 1000))
            except Exception:
                logger.error("Budget analyze: get_fibers failed", exc_info=True)
                return {"error": "Failed to list fibers for analysis"}

            if not fibers_list:
                return {
                    "action": "analyze",
                    "brain": brain_id,
                    "total_fibers": 0,
                    "message": "No fibers found in brain",
                }

            # Compute costs with uniform activation score (no query context)
            dummy_activations: dict[str, Any] = {}
            costs = compute_token_costs(fibers_list, dummy_activations, budget_cfg)

            if not costs:
                return {
                    "action": "analyze",
                    "brain": brain_id,
                    "total_fibers": len(fibers_list),
                    "message": "No cost data computed",
                }

            total_tokens = sum(c.total_tokens for c in costs)
            avg_tokens = total_tokens / len(costs) if costs else 0
            max_cost = max(c.total_tokens for c in costs)
            min_cost = min(c.total_tokens for c in costs)

            # Top 5 most expensive fibers
            top_expensive = sorted(costs, key=lambda c: c.total_tokens, reverse=True)[:5]

            return {
                "action": "analyze",
                "brain": brain_id,
                "total_fibers": len(fibers_list),
                "total_tokens_all_fibers": total_tokens,
                "avg_tokens_per_fiber": round(avg_tokens, 1),
                "max_fiber_tokens": max_cost,
                "min_fiber_tokens": min_cost,
                "estimated_full_recall_tokens": total_tokens + budget_cfg.system_overhead_tokens,
                "would_fit_in_4k": sum(1 for c in costs if c.total_tokens <= 4000),
                "top_expensive_fibers": [
                    {"fiber_id": c.fiber_id, "total_tokens": c.total_tokens} for c in top_expensive
                ],
            }

        else:  # optimize
            # Find low-value-per-token fibers that are compression candidates
            try:
                fibers_list = await storage.get_fibers(limit=min(200, 1000))
            except Exception:
                logger.error("Budget optimize: get_fibers failed", exc_info=True)
                return {"error": "Failed to list fibers for optimization"}

            if not fibers_list:
                return {
                    "action": "optimize",
                    "recommendations": [],
                    "message": "No fibers found",
                }

            dummy_activations = {}
            costs = compute_token_costs(fibers_list, dummy_activations, budget_cfg)

            # Fibers with zero or very low value_per_token and high cost are candidates
            candidates = [
                c
                for c in costs
                if c.total_tokens > 100  # Only fibers large enough to matter
            ]
            # Sort by worst efficiency (high cost, low value) — cost > 100, no activation context
            candidates_sorted = sorted(candidates, key=lambda c: c.total_tokens, reverse=True)[:10]

            recommendations = []
            for c in candidates_sorted:
                savings_estimate = max(0, c.total_tokens - budget_cfg.per_fiber_overhead - 20)
                recommendations.append(
                    {
                        "fiber_id": c.fiber_id,
                        "current_tokens": c.total_tokens,
                        "estimated_savings_if_compressed": savings_estimate,
                        "suggestion": "Consider running nmem_consolidate to compress this fiber's content into a summary.",
                    }
                )

            return {
                "action": "optimize",
                "fibers_analyzed": len(costs),
                "compression_candidates": len(recommendations),
                "recommendations": recommendations,
                "tip": "Run nmem_consolidate with strategy='mature' to auto-summarize old fibers.",
            }
