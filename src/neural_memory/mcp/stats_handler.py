"""MCP handler mixin for brain statistics and health diagnostics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory import __version__
from neural_memory.core.memory_types import MemoryTier
from neural_memory.mcp.tool_handler_utils import _get_brain_or_error
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

# Gromov delta cache: {brain_id: (timestamp, GromovResult)} — TTL 1 hour
_gromov_cache: dict[str, tuple[Any, Any]] = {}


class StatsHandler:
    """Mixin providing stats, health, and todo handler implementations."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

        async def _check_onboarding(self) -> dict[str, Any] | None:
            raise NotImplementedError

        def get_update_hint(self) -> dict[str, Any] | None:
            raise NotImplementedError

        async def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
            raise NotImplementedError

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
        tier_distribution = {MemoryTier.HOT: 0, MemoryTier.WARM: 0, MemoryTier.COLD: 0}
        try:
            for tier_name in (MemoryTier.HOT, MemoryTier.WARM, MemoryTier.COLD):
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
            return err

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
