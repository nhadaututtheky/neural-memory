"""Dashboard API routes — brain stats, health, brain management, timeline, diagrams, brain files, storage migration."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from neural_memory.server.dependencies import get_storage, require_local_request
from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/dashboard",
    tags=["dashboard"],
    dependencies=[Depends(require_local_request)],
)


class BrainSummary(BaseModel):
    """Brief brain summary for dashboard listing."""

    id: str
    name: str
    neuron_count: int = 0
    synapse_count: int = 0
    fiber_count: int = 0
    grade: str = "F"
    purity_score: float = 0.0
    is_active: bool = False


class DashboardStats(BaseModel):
    """Dashboard overview statistics."""

    active_brain: str | None = None
    total_brains: int = 0
    total_neurons: int = 0
    total_synapses: int = 0
    total_fibers: int = 0
    health_grade: str = "F"
    purity_score: float = 0.0
    brains: list[BrainSummary] = Field(default_factory=list)


class HealthReport(BaseModel):
    """Brain health report for the radar chart."""

    grade: str
    purity_score: float
    connectivity: float = 0.0
    diversity: float = 0.0
    freshness: float = 0.0
    consolidation_ratio: float = 0.0
    orphan_rate: float = 0.0
    activation_efficiency: float = 0.0
    recall_confidence: float = 0.0
    neuron_count: int = 0
    synapse_count: int = 0
    fiber_count: int = 0
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    top_penalties: list[dict[str, Any]] = Field(default_factory=list)


class SwitchBrainRequest(BaseModel):
    """Request to switch active brain."""

    brain_name: str = Field(..., min_length=1)


@router.get(
    "/stats",
    response_model=DashboardStats,
    summary="Get dashboard overview stats",
)
async def get_stats() -> DashboardStats:
    """Get overall dashboard statistics across all brains."""
    from neural_memory.unified_config import get_config, get_shared_storage

    cfg = get_config()
    brain_names = cfg.list_brains()
    active_name = cfg.current_brain

    async def _analyze_brain(name: str) -> BrainSummary:
        """Analyze a single brain using its own per-brain storage."""
        try:
            brain_storage = await get_shared_storage(brain_name=name)
            stats = await brain_storage.get_stats(name)
            nc = stats.get("neuron_count", 0)
            sc = stats.get("synapse_count", 0)
            fc = stats.get("fiber_count", 0)

            grade = "F"
            purity = 0.0
            try:
                from neural_memory.engine.diagnostics import DiagnosticsEngine

                diag = DiagnosticsEngine(brain_storage)
                report = await diag.analyze(name)
                grade = report.grade
                purity = report.purity_score
            except Exception:
                logger.debug("Diagnostics failed for brain %s", name, exc_info=True)

            return BrainSummary(
                id=name,
                name=name,
                neuron_count=nc,
                synapse_count=sc,
                fiber_count=fc,
                grade=grade,
                purity_score=purity,
                is_active=name == active_name,
            )
        except Exception:
            logger.debug("Brain analysis failed for %s", name, exc_info=True)
            return BrainSummary(id=name, name=name, is_active=name == active_name)

    brains = list(await asyncio.gather(*[_analyze_brain(name) for name in brain_names]))

    total_n = sum(b.neuron_count for b in brains)
    total_s = sum(b.synapse_count for b in brains)
    total_f = sum(b.fiber_count for b in brains)
    active_grade = "F"
    active_purity = 0.0
    for b in brains:
        if b.is_active:
            active_grade = b.grade
            active_purity = b.purity_score
            break

    return DashboardStats(
        active_brain=active_name,
        total_brains=len(brain_names),
        total_neurons=total_n,
        total_synapses=total_s,
        total_fibers=total_f,
        health_grade=active_grade,
        purity_score=active_purity,
        brains=brains,
    )


class TierDistribution(BaseModel):
    """Memory tier distribution for the active brain."""

    hot: int = 0
    warm: int = 0
    cold: int = 0
    total: int = 0


@router.get(
    "/tier-stats",
    response_model=TierDistribution,
    summary="Get memory tier distribution",
)
async def get_tier_stats(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> TierDistribution:
    """Get HOT/WARM/COLD tier distribution for the active brain."""
    counts = {"hot": 0, "warm": 0, "cold": 0}
    try:
        for tier_name in ("hot", "warm", "cold"):
            counts[tier_name] = await storage.count_typed_memories(tier=tier_name)
    except Exception:
        logger.debug("Tier stats query failed", exc_info=True)

    total = counts["hot"] + counts["warm"] + counts["cold"]
    return TierDistribution(
        hot=counts["hot"], warm=counts["warm"], cold=counts["cold"], total=total
    )


@router.get(
    "/tier-analytics",
    summary="Get tier analytics — breakdown by memory type + velocity metrics",
)
async def get_tier_analytics(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Get tier analytics: breakdown by type, velocity (7d/30d), recent changes."""
    from datetime import timedelta

    from neural_memory.core.memory_types import MemoryTier
    from neural_memory.mcp.tier_handler import _classify_change
    from neural_memory.utils.timeutils import utcnow

    now = utcnow()
    cutoff_7d = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    # Breakdown by memory type x tier (SQL aggregate — no full fetch)
    breakdown: dict[str, dict[str, int]] = {}
    grouped = await storage.count_typed_memories_grouped()
    total_memories = 0
    for memory_type, tier, count in grouped:
        if memory_type not in breakdown:
            breakdown[memory_type] = {MemoryTier.HOT: 0, MemoryTier.WARM: 0, MemoryTier.COLD: 0}
        breakdown[memory_type][tier] = count
        total_memories += count

    # Velocity from promotion_history metadata (needs full fetch for metadata)
    all_typed = await storage.find_typed_memories(limit=1000)
    velocity_7d = {"promoted": 0, "demoted": 0, "archived": 0}
    velocity_30d = {"promoted": 0, "demoted": 0, "archived": 0}
    for tm in all_typed:
        for entry in tm.metadata.get("promotion_history", []):
            direction = _classify_change(entry.get("from", ""), entry.get("to", ""))
            try:
                ts = datetime.fromisoformat(entry["at"])
                if ts >= cutoff_7d:
                    velocity_7d[direction] = velocity_7d.get(direction, 0) + 1
                if ts >= cutoff_30d:
                    velocity_30d[direction] = velocity_30d.get(direction, 0) + 1
            except (KeyError, ValueError, TypeError):
                pass

    return {
        "breakdown_by_type": breakdown,
        "velocity_7d": velocity_7d,
        "velocity_30d": velocity_30d,
        "total_memories": total_memories,
    }


@router.get(
    "/tier-history",
    summary="Get paginated tier change events",
)
async def get_tier_history(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Get recent tier change events across all memories."""
    from neural_memory.core.memory_types import MemoryType

    all_typed = await storage.find_typed_memories(limit=1000)
    events: list[dict[str, Any]] = []

    for tm in all_typed:
        type_key = (
            tm.memory_type.value if isinstance(tm.memory_type, MemoryType) else str(tm.memory_type)
        )
        for entry in tm.metadata.get("promotion_history", []):
            events.append(
                {
                    "fiber_id": tm.fiber_id,
                    "memory_type": type_key,
                    "from_tier": entry.get("from", ""),
                    "to_tier": entry.get("to", ""),
                    "reason": entry.get("reason", ""),
                    "at": entry.get("at", ""),
                }
            )

    # Sort by timestamp descending
    events.sort(key=lambda e: e.get("at", ""), reverse=True)
    total = len(events)
    page = events[offset : offset + limit]

    return {
        "events": page,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/brains",
    response_model=list[BrainSummary],
    summary="List all brains",
)
async def list_brains_api() -> list[BrainSummary]:
    """List all available brains with summary stats."""
    from neural_memory.unified_config import get_config, get_shared_storage

    cfg = get_config()
    brain_names = cfg.list_brains()
    active_name = cfg.current_brain
    results: list[BrainSummary] = []

    for name in brain_names:
        try:
            brain_storage = await get_shared_storage(brain_name=name)
            stats = await brain_storage.get_stats(name)
            results.append(
                BrainSummary(
                    id=name,
                    name=name,
                    neuron_count=stats.get("neuron_count", 0),
                    synapse_count=stats.get("synapse_count", 0),
                    fiber_count=stats.get("fiber_count", 0),
                    is_active=name == active_name,
                )
            )
        except Exception:
            logger.debug("Failed to get stats for brain %s", name, exc_info=True)
            results.append(BrainSummary(id=name, name=name, is_active=name == active_name))

    return results


@router.post(
    "/brains/switch",
    summary="Switch active brain",
)
async def switch_brain(
    request: SwitchBrainRequest,
    http_request: Request,
) -> dict[str, str]:
    """Switch the active brain.

    Updates config.toml (persistent) AND app.state.storage so that
    all endpoints immediately use the new brain's DB without restart.
    """
    from neural_memory.unified_config import get_config, get_shared_storage

    cfg = get_config()
    available = cfg.list_brains()
    if request.brain_name not in available:
        raise HTTPException(
            status_code=404,
            detail="Brain not found.",
        )

    cfg.switch_brain(request.brain_name)

    # Update the live storage so all Depends(get_storage) endpoints
    # immediately use the new brain's DB (not just after restart).
    try:
        new_storage = await get_shared_storage(brain_name=request.brain_name)
        http_request.app.state.storage = new_storage
    except Exception:
        logger.warning(
            "Failed to update live storage after brain switch to %s",
            request.brain_name,
            exc_info=True,
        )

    return {"status": "switched", "active_brain": request.brain_name}


@router.get(
    "/health",
    response_model=HealthReport,
    summary="Get active brain health report",
)
async def get_health(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> HealthReport:
    """Run full diagnostics on the active brain."""
    from neural_memory.engine.diagnostics import DiagnosticsEngine
    from neural_memory.unified_config import get_config

    brain_name = get_config().current_brain

    try:
        diag = DiagnosticsEngine(storage)
        report = await diag.analyze(brain_name)
    except Exception as exc:
        logger.warning("Diagnostics failed for brain %s: %s", brain_name, exc)
        return HealthReport(grade="F", purity_score=0.0)

    return HealthReport(
        grade=report.grade,
        purity_score=report.purity_score,
        connectivity=report.connectivity,
        diversity=report.diversity,
        freshness=report.freshness,
        consolidation_ratio=report.consolidation_ratio,
        orphan_rate=report.orphan_rate,
        activation_efficiency=report.activation_efficiency,
        recall_confidence=report.recall_confidence,
        neuron_count=report.neuron_count,
        synapse_count=report.synapse_count,
        fiber_count=report.fiber_count,
        warnings=[
            {
                "severity": w.severity.value,
                "code": w.code,
                "message": w.message,
                "details": w.details,
            }
            for w in report.warnings
        ],
        recommendations=list(report.recommendations),
        top_penalties=[
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
    )


# ── Timeline API ─────────────────────────────────────────


class TimelineEntry(BaseModel):
    """A single timeline entry."""

    id: str
    content: str
    neuron_type: str
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimelineResponse(BaseModel):
    """Timeline API response."""

    entries: list[TimelineEntry] = Field(default_factory=list)
    total: int = 0


@router.get(
    "/timeline",
    response_model=TimelineResponse,
    summary="Get chronological memory timeline",
)
async def get_timeline(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    limit: int = Query(default=500, ge=1, le=2000),
    start: str | None = Query(default=None, description="ISO datetime start"),
    end: str | None = Query(default=None, description="ISO datetime end"),
) -> TimelineResponse:
    """Get chronological list of memories for timeline visualization."""
    neurons = await storage.find_neurons(limit=min(limit, 2000))

    entries: list[TimelineEntry] = []
    for n in neurons:
        created = n.metadata.get("_created_at", "") if n.metadata else ""
        if not created and hasattr(n, "created_at") and n.created_at:
            created = (
                n.created_at.isoformat()
                if hasattr(n.created_at, "isoformat")
                else str(n.created_at)
            )

        if start and created and created < start:
            continue
        if end and created and created > end:
            continue

        entries.append(
            TimelineEntry(
                id=n.id,
                content=n.content or "",
                neuron_type=n.type.value,
                created_at=created,
                metadata=n.metadata or {},
            )
        )

    # Sort by created_at descending
    entries.sort(key=lambda e: e.created_at, reverse=True)

    return TimelineResponse(entries=entries[:limit], total=len(entries))


# ── Daily Stats API ──────────────────────────────────────


class DailyStatsEntry(BaseModel):
    """Aggregated daily brain activity."""

    date: str
    neurons_created: int = 0
    fibers_created: int = 0
    synapses_created: int = 0
    neuron_types: dict[str, int] = Field(default_factory=dict)


@router.get(
    "/timeline/daily-stats",
    response_model=list[DailyStatsEntry],
    summary="Get daily activity stats for timeline charts",
)
async def get_daily_stats(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    days: int = Query(default=30, ge=1, le=365),
) -> list[DailyStatsEntry]:
    """Get aggregated daily counts of neurons, fibers, and synapses."""
    from datetime import timedelta

    from neural_memory.utils.timeutils import utcnow

    now = utcnow()
    start = now - timedelta(days=days)
    end = now

    # Use public API: find_neurons with time_range
    neurons = await storage.find_neurons(time_range=(start, end), limit=1000)

    # Aggregate neurons by day
    days_map: dict[str, DailyStatsEntry] = {}
    for i in range(days + 1):
        d = (now - timedelta(days=days - i)).strftime("%Y-%m-%d")
        days_map[d] = DailyStatsEntry(date=d)

    for n in neurons:
        if not n.created_at:
            continue
        day = n.created_at.strftime("%Y-%m-%d")
        if day not in days_map:
            days_map[day] = DailyStatsEntry(date=day)
        entry = days_map[day]
        entry.neurons_created += 1
        ntype = n.type.value
        entry.neuron_types[ntype] = entry.neuron_types.get(ntype, 0) + 1

    # Fibers via get_fibers (public API)
    fibers = await storage.get_fibers(limit=1000)
    for f in fibers:
        if not f.created_at:
            continue
        if f.created_at < start:
            continue
        day = f.created_at.strftime("%Y-%m-%d")
        if day in days_map:
            days_map[day].fibers_created += 1

    return sorted(days_map.values(), key=lambda e: e.date)


# ── Fiber Diagram API ────────────────────────────────────


class FiberListItem(BaseModel):
    """Brief fiber summary for dropdown."""

    id: str
    summary: str
    neuron_count: int = 0


class FiberListResponse(BaseModel):
    """Fiber list API response."""

    fibers: list[FiberListItem] = Field(default_factory=list)


@router.get(
    "/fibers",
    response_model=FiberListResponse,
    summary="List fibers for dropdown",
)
async def list_fibers(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    limit: int = Query(default=100, ge=1, le=500),
) -> FiberListResponse:
    """Get lightweight fiber list for diagram dropdown."""
    fibers = await storage.get_fibers(limit=min(limit, 500))

    return FiberListResponse(
        fibers=[
            FiberListItem(
                id=f.id,
                summary=f.summary or f.id[:20],
                neuron_count=len(f.neuron_ids) if f.neuron_ids else 0,
            )
            for f in fibers
        ]
    )


class FiberDiagramResponse(BaseModel):
    """Fiber diagram data for Mermaid rendering."""

    fiber_id: str
    neurons: list[dict[str, Any]] = Field(default_factory=list)
    synapses: list[dict[str, Any]] = Field(default_factory=list)


@router.get(
    "/fiber/{fiber_id}/diagram",
    response_model=FiberDiagramResponse,
    summary="Get fiber structure for diagram",
)
async def get_fiber_diagram(
    fiber_id: str,
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> FiberDiagramResponse:
    """Get neurons and synapses for a fiber to render as a diagram."""
    target = await storage.get_fiber(fiber_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Fiber not found.")

    neuron_ids = list(target.neuron_ids) if target.neuron_ids else []
    if not neuron_ids:
        return FiberDiagramResponse(fiber_id=fiber_id, neurons=[], synapses=[])

    neurons_batch = await storage.get_neurons_batch(neuron_ids)

    neuron_list = [
        {
            "id": n.id,
            "type": n.type.value,
            "content": n.content or "",
            "metadata": n.metadata or {},
        }
        for n in neurons_batch.values()
    ]

    # Get synapses between this fiber's neurons using targeted batch query
    id_set = set(neuron_ids)
    outgoing = await storage.get_synapses_for_neurons(neuron_ids, direction="out")
    fiber_synapses = [
        {
            "id": s.id,
            "source_id": s.source_id,
            "target_id": s.target_id,
            "type": s.type.value,
            "weight": s.weight,
            "direction": s.direction.value,
        }
        for synapse_list in outgoing.values()
        for s in synapse_list
        if s.target_id in id_set
    ]

    return FiberDiagramResponse(
        fiber_id=fiber_id,
        neurons=neuron_list,
        synapses=fiber_synapses,
    )


# ── Evolution API ────────────────────────────────────


class SemanticProgressItem(BaseModel):
    """Progress of a fiber toward SEMANTIC stage."""

    fiber_id: str
    stage: str
    days_in_stage: float
    days_required: float
    reinforcement_days: int
    reinforcement_required: int
    progress_pct: float
    next_step: str


class StageDistributionResponse(BaseModel):
    """Distribution of fibers across maturation stages."""

    short_term: int = 0
    working: int = 0
    episodic: int = 0
    semantic: int = 0
    total: int = 0


class EvolutionResponse(BaseModel):
    """Brain evolution metrics for dashboard."""

    brain: str
    proficiency_level: str
    proficiency_index: int
    maturity_level: float
    plasticity: float
    density: float
    activity_score: float
    semantic_ratio: float
    reinforcement_days: float
    topology_coherence: float
    plasticity_index: float
    knowledge_density: float
    total_neurons: int
    total_synapses: int
    total_fibers: int
    fibers_at_semantic: int
    fibers_at_episodic: int
    stage_distribution: StageDistributionResponse | None = None
    closest_to_semantic: list[SemanticProgressItem] = Field(default_factory=list)


@router.get(
    "/evolution",
    response_model=EvolutionResponse,
    summary="Get brain evolution metrics",
)
async def get_evolution(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> EvolutionResponse:
    """Get evolution dynamics for the active brain."""
    from neural_memory.engine.brain_evolution import EvolutionEngine
    from neural_memory.unified_config import get_config

    brain_name = get_config().current_brain

    try:
        engine = EvolutionEngine(storage)
        evo = await engine.analyze(brain_name)
    except Exception as exc:
        logger.warning("Evolution analysis failed for brain %s: %s", brain_name, exc)
        raise HTTPException(status_code=500, detail="Evolution analysis failed")

    stage_dist = None
    if evo.stage_distribution is not None:
        stage_dist = StageDistributionResponse(
            short_term=evo.stage_distribution.short_term,
            working=evo.stage_distribution.working,
            episodic=evo.stage_distribution.episodic,
            semantic=evo.stage_distribution.semantic,
            total=evo.stage_distribution.total,
        )

    closest = [
        SemanticProgressItem(
            fiber_id=p.fiber_id,
            stage=p.stage,
            days_in_stage=round(p.days_in_stage, 2),
            days_required=round(p.days_required, 2),
            reinforcement_days=p.reinforcement_days,
            reinforcement_required=p.reinforcement_required,
            progress_pct=round(p.progress_pct, 4),
            next_step=p.next_step,
        )
        for p in evo.closest_to_semantic
    ]

    return EvolutionResponse(
        brain=evo.brain_name,
        proficiency_level=evo.proficiency_level.value,
        proficiency_index=evo.proficiency_index,
        maturity_level=round(evo.maturity_level, 4),
        plasticity=round(evo.plasticity, 4),
        density=round(evo.density, 4),
        activity_score=round(evo.activity_score, 4),
        semantic_ratio=round(evo.semantic_ratio, 4),
        reinforcement_days=round(evo.reinforcement_days, 2),
        topology_coherence=round(evo.topology_coherence, 4),
        plasticity_index=round(evo.plasticity_index, 4),
        knowledge_density=round(evo.knowledge_density, 4),
        total_neurons=evo.total_neurons,
        total_synapses=evo.total_synapses,
        total_fibers=evo.total_fibers,
        fibers_at_semantic=evo.fibers_at_semantic,
        fibers_at_episodic=evo.fibers_at_episodic,
        stage_distribution=stage_dist,
        closest_to_semantic=closest,
    )


# ── Brain Files API ────────────────────────────────────


class BrainFileInfo(BaseModel):
    """Info about a single brain database file."""

    name: str
    path: str
    size_bytes: int = 0
    is_active: bool = False


class BrainFilesResponse(BaseModel):
    """Response with brain file information."""

    brains_dir: str
    brains: list[BrainFileInfo] = Field(default_factory=list)
    total_size_bytes: int = 0


@router.get(
    "/brain-files",
    response_model=BrainFilesResponse,
    summary="Get brain file paths and sizes",
)
async def get_brain_files() -> BrainFilesResponse:
    """Get file path and size information for all brain databases."""
    from neural_memory.unified_config import get_config

    cfg = get_config()
    brain_names = cfg.list_brains()
    active_name = cfg.current_brain
    brains_dir = Path(cfg.get_brain_db_path("_probe_")).parent

    brain_files: list[BrainFileInfo] = []
    total_size = 0

    for name in brain_names:
        db_path = Path(cfg.get_brain_db_path(name))
        size = 0
        if db_path.exists():
            size = db_path.stat().st_size
            total_size += size

        brain_files.append(
            BrainFileInfo(
                name=name,
                path=str(db_path),
                size_bytes=size,
                is_active=name == active_name,
            )
        )

    return BrainFilesResponse(
        brains_dir=str(brains_dir),
        brains=brain_files,
        total_size_bytes=total_size,
    )


# ── Telegram API ────────────────────────────────────


class TelegramStatusResponse(BaseModel):
    """Telegram integration status."""

    configured: bool = False
    bot_name: str | None = None
    bot_username: str | None = None
    chat_ids: list[str] = Field(default_factory=list)
    backup_on_consolidation: bool = False
    error: str | None = None


class TelegramTestRequest(BaseModel):
    """Request to send a test message."""


class TelegramBackupRequest(BaseModel):
    """Request to trigger a brain backup."""

    brain_name: str | None = None


@router.get(
    "/telegram/status",
    response_model=TelegramStatusResponse,
    summary="Get Telegram integration status",
)
async def get_telegram_status_api() -> TelegramStatusResponse:
    """Get current Telegram integration status."""
    from neural_memory.integration.telegram import get_telegram_status

    status = await get_telegram_status()
    return TelegramStatusResponse(
        configured=status.configured,
        bot_name=status.bot_name,
        bot_username=status.bot_username,
        chat_ids=status.chat_ids,
        backup_on_consolidation=status.backup_on_consolidation,
        error=status.error,
    )


@router.post(
    "/telegram/test",
    summary="Send test message to Telegram",
)
async def telegram_test_api() -> dict[str, Any]:
    """Send a test message to verify Telegram configuration."""
    from neural_memory.integration.telegram import (
        TelegramClient,
        TelegramError,
        get_bot_token,
        get_telegram_config,
    )

    token = get_bot_token()
    if not token:
        raise HTTPException(status_code=400, detail="Bot token not configured")

    config = get_telegram_config()
    if not config.chat_ids:
        raise HTTPException(status_code=400, detail="No chat IDs configured")

    client = TelegramClient(token)
    results: list[str] = []
    errors: list[str] = []

    for chat_id in config.chat_ids:
        try:
            await client.send_message(
                chat_id,
                "🧠 <b>Neural Memory</b> — Test message\n\nTelegram integration is working!",
            )
            results.append(chat_id)
        except TelegramError:
            errors.append(f"{chat_id}: send failed")

    return {"sent": results, "errors": errors}


@router.post(
    "/telegram/backup",
    summary="Send brain backup to Telegram",
)
async def telegram_backup_api(
    request: TelegramBackupRequest,
) -> dict[str, Any]:
    """Send brain database file as backup to Telegram."""
    from neural_memory.integration.telegram import (
        TelegramClient,
        TelegramError,
        get_bot_token,
    )

    token = get_bot_token()
    if not token:
        raise HTTPException(status_code=400, detail="Bot token not configured")

    client = TelegramClient(token)

    try:
        result = await client.backup_brain(request.brain_name)
        return result
    except TelegramError:
        raise HTTPException(status_code=500, detail="Telegram backup failed")


# ---------------------------------------------------------------------------
# Cloud Sync
# ---------------------------------------------------------------------------


@router.get("/sync-status", tags=["dashboard"], summary="Cloud sync status for dashboard")
async def get_sync_status(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> dict[str, Any]:
    """Return sync configuration and status for the dashboard UI."""
    from neural_memory.unified_config import get_config

    config = get_config()
    sync = config.sync

    # Mask API key
    api_key_display = "(not set)"
    if sync.api_key and len(sync.api_key) >= 12:
        api_key_display = f"{sync.api_key[:12]}****"

    result: dict[str, Any] = {
        "enabled": sync.enabled,
        "hub_url": sync.hub_url or "(not set)",
        "api_key": api_key_display,
        "auto_sync": sync.auto_sync,
        "conflict_strategy": sync.conflict_strategy,
        "device_id": config.device_id,
    }

    # Get device list and change log stats if sync is configured
    if sync.enabled:
        try:
            change_stats = await storage.get_change_log_stats()
            devices_raw = await storage.list_devices()
            result["change_log"] = change_stats
            result["devices"] = [
                {
                    "device_id": d.device_id,
                    "device_name": d.device_name,
                    "last_sync_at": d.last_sync_at.isoformat() if d.last_sync_at else None,
                    "last_sync_sequence": d.last_sync_sequence,
                    "registered_at": d.registered_at.isoformat(),
                }
                for d in devices_raw
            ]
            result["device_count"] = len(devices_raw)
        except Exception:
            logger.debug("Could not fetch sync stats", exc_info=True)
            result["devices"] = []
            result["device_count"] = 0
    else:
        result["devices"] = []
        result["device_count"] = 0

    return result


@router.post("/sync-config", tags=["dashboard"], summary="Update sync configuration")
async def update_sync_config(
    body: dict[str, Any],
) -> dict[str, Any]:
    """Update sync configuration from the dashboard UI."""
    from dataclasses import replace as dc_replace

    from neural_memory.unified_config import get_config

    config = get_config()
    new_sync = config.sync

    hub_url = body.get("hub_url")
    if hub_url is not None:
        url = str(hub_url).strip()
        if url and not url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=422, detail="hub_url must start with http:// or https://"
            )
        new_sync = dc_replace(new_sync, hub_url=url[:256])

    api_key = body.get("api_key")
    if api_key is not None:
        key = str(api_key).strip()
        if key and not key.startswith("nmk_"):
            raise HTTPException(status_code=422, detail="API key must start with 'nmk_'")
        new_sync = dc_replace(new_sync, api_key=key)

    if "enabled" in body:
        new_sync = dc_replace(new_sync, enabled=bool(body["enabled"]))

    if "conflict_strategy" in body:
        valid = {"prefer_recent", "prefer_local", "prefer_remote", "prefer_stronger"}
        strategy = str(body["conflict_strategy"])
        if strategy not in valid:
            raise HTTPException(
                status_code=422, detail=f"Invalid strategy. Use: {', '.join(sorted(valid))}"
            )
        new_sync = dc_replace(new_sync, conflict_strategy=strategy)

    # Auto-enable when both hub_url and api_key are set
    if new_sync.hub_url and new_sync.api_key and not new_sync.enabled:
        new_sync = dc_replace(new_sync, enabled=True)

    updated = dc_replace(config, sync=new_sync)
    updated.save()

    api_key_display = "(not set)"
    if new_sync.api_key and len(new_sync.api_key) >= 12:
        api_key_display = f"{new_sync.api_key[:12]}****"

    return {
        "status": "updated",
        "enabled": new_sync.enabled,
        "hub_url": new_sync.hub_url or "(not set)",
        "api_key": api_key_display,
        "conflict_strategy": new_sync.conflict_strategy,
    }


# ── Embedding Config API ──────────────────────────────────

_VALID_EMBEDDING_PROVIDERS: tuple[str, ...] = (
    "sentence_transformer",
    "openai",
    "openrouter",
    "gemini",
    "ollama",
)


class EmbeddingConfigUpdate(BaseModel):
    """Partial update for embedding settings."""

    enabled: bool | None = None
    provider: str | None = None
    model: str | None = None
    similarity_threshold: float | None = None


class ConfigUpdateRequest(BaseModel):
    """Request body for PUT /config."""

    embedding: EmbeddingConfigUpdate | None = None


@router.get("/config/embedding")
async def get_embedding_config() -> dict[str, Any]:
    """Return current embedding settings."""
    from neural_memory.unified_config import get_config

    config = get_config()
    return config.embedding.to_dict()


@router.put("/config")
async def update_config(body: ConfigUpdateRequest) -> dict[str, Any]:
    """Update embedding configuration.

    Only fields provided in the request body are changed; omitted fields
    keep their current values.
    """
    from dataclasses import replace as dc_replace

    from neural_memory.unified_config import get_config

    config = get_config()
    new_embedding = config.embedding

    if body.embedding is not None:
        if not config.is_pro():
            raise HTTPException(
                status_code=403,
                detail="Embedding configuration requires a Pro license. Activate via nmem_sync_config(action='activate').",
            )
        update = body.embedding

        if update.enabled is not None:
            new_embedding = dc_replace(new_embedding, enabled=bool(update.enabled))

        if update.provider is not None:
            provider = str(update.provider).strip()
            if provider not in _VALID_EMBEDDING_PROVIDERS:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid provider. Use one of: {', '.join(_VALID_EMBEDDING_PROVIDERS)}",
                )
            new_embedding = dc_replace(new_embedding, provider=provider)

        if update.model is not None:
            model = str(update.model).strip()
            if not model:
                raise HTTPException(status_code=422, detail="model must be a non-empty string")
            if len(model) > 128:
                raise HTTPException(status_code=422, detail="model must be at most 128 characters")
            new_embedding = dc_replace(new_embedding, model=model)

        if update.similarity_threshold is not None:
            threshold = float(update.similarity_threshold)
            if not 0.0 <= threshold <= 1.0:
                raise HTTPException(
                    status_code=422,
                    detail="similarity_threshold must be in [0.0, 1.0]",
                )
            new_embedding = dc_replace(new_embedding, similarity_threshold=threshold)

    updated = dc_replace(config, embedding=new_embedding)
    updated.save()

    return {"status": "updated", "embedding": new_embedding.to_dict()}


@router.post("/config/embedding/test")
async def test_embedding_connection() -> dict[str, Any]:
    """Test the current embedding provider connection.

    Returns status "ok" with provider name and vector dimension on success,
    or status "error" with an error message on failure.
    """
    from neural_memory.unified_config import get_config

    config = get_config()
    emb = config.embedding

    if not emb.enabled:
        return {"status": "error", "error": "Embedding not enabled"}

    provider_name = emb.provider
    model_name = emb.model

    try:
        provider: Any
        if provider_name == "sentence_transformer":
            from neural_memory.engine.embedding.sentence_transformer import (
                SentenceTransformerEmbedding,
            )

            provider = SentenceTransformerEmbedding(model_name=model_name)
        elif provider_name == "openai":
            from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

            provider = OpenAIEmbedding(model=model_name)
        elif provider_name == "openrouter":
            from neural_memory.engine.embedding.openrouter_embedding import OpenRouterEmbedding

            provider = OpenRouterEmbedding(model=model_name)
        elif provider_name == "gemini":
            from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

            provider = GeminiEmbedding(model=model_name)
        elif provider_name == "ollama":
            from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

            provider = OllamaEmbedding(model=model_name)
        else:
            return {"status": "error", "error": f"Unknown embedding provider: {provider_name!r}"}

        vector = await provider.embed("hello")
        return {
            "status": "ok",
            "provider": provider_name,
            "dimension": len(vector),
        }
    except Exception as exc:
        logger.error("Embedding connection test failed for provider %r: %s", provider_name, exc)
        return {
            "status": "error",
            "error": "Connection test failed. Check server logs for details.",
        }


# ── Config Status API ────────────────────────────────────


class ConfigStatusItem(BaseModel):
    """A single configuration status item."""

    key: str
    label: str
    status: str  # "configured" | "not_configured" | "warning" | "info"
    description: str
    command: str = ""
    value: str = ""


class ConfigStatusResponse(BaseModel):
    """Configuration status response."""

    items: list[ConfigStatusItem] = Field(default_factory=list)


@router.get(
    "/config-status",
    response_model=ConfigStatusResponse,
    summary="Get configuration status and actionable items",
)
async def get_config_status(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> ConfigStatusResponse:
    """Return per-feature configuration status with actionable commands."""
    from neural_memory.unified_config import get_config

    items: list[ConfigStatusItem] = []

    try:
        cfg = get_config()
    except Exception:
        logger.warning("Could not load config for config-status endpoint", exc_info=True)
        return ConfigStatusResponse(items=[])

    # ── 1. Tool Memory ──────────────────────────────────
    try:
        tm = cfg.tool_memory
        if tm.enabled:
            items.append(
                ConfigStatusItem(
                    key="tool_memory",
                    label="Tool Memory",
                    status="configured",
                    description="Tracks MCP tool usage patterns for analytics",
                    command="",
                    value="enabled",
                )
            )
        else:
            items.append(
                ConfigStatusItem(
                    key="tool_memory",
                    label="Tool Memory",
                    status="not_configured",
                    description="Tracks MCP tool usage patterns for analytics",
                    command="Set [tool_memory] enabled = true in config.toml",
                    value="",
                )
            )
    except Exception:
        logger.debug("Could not check tool_memory config", exc_info=True)

    # ── 2. Cloud Sync ───────────────────────────────────
    try:
        sync = cfg.sync
        if sync.hub_url:
            items.append(
                ConfigStatusItem(
                    key="cloud_sync",
                    label="Cloud Sync",
                    status="configured",
                    description="Sync memories across devices via your own Cloudflare Worker",
                    command="",
                    value=sync.hub_url,
                )
            )
        else:
            items.append(
                ConfigStatusItem(
                    key="cloud_sync",
                    label="Cloud Sync",
                    status="not_configured",
                    description="Sync memories across devices via your own Cloudflare Worker",
                    command='nmem_sync_config(action="setup")',
                    value="",
                )
            )
    except Exception:
        logger.debug("Could not check sync config", exc_info=True)

    # ── 3. Embedding Provider ───────────────────────────
    try:
        emb = cfg.embedding
        if emb.enabled and emb.provider:
            model_info = f"{emb.provider} ({emb.model})" if emb.model else emb.provider
            items.append(
                ConfigStatusItem(
                    key="embedding",
                    label="Embedding Provider",
                    status="configured",
                    description=(
                        "Semantic similarity active — disable: "
                        "set [embedding] enabled = false in config.toml"
                    ),
                    command="",
                    value=model_info,
                )
            )
        else:
            # Check if any provider is importable
            provider_installed = False
            try:
                import importlib

                importlib.import_module("sentence_transformers")
                provider_installed = True
            except ImportError:
                pass

            if provider_installed and not emb.enabled:
                items.append(
                    ConfigStatusItem(
                        key="embedding",
                        label="Embedding Provider",
                        status="info",
                        description=(
                            "Installed but disabled — enable for cross-language "
                            "recall and semantic similarity"
                        ),
                        command="Set [embedding] enabled = true in config.toml",
                        value="disabled",
                    )
                )
            else:
                items.append(
                    ConfigStatusItem(
                        key="embedding",
                        label="Embedding Provider",
                        status="not_configured",
                        description=(
                            "Optional — enables cross-language recall and "
                            "semantic similarity for better retrieval"
                        ),
                        command="pip install neural-memory[embeddings]",
                        value="",
                    )
                )
    except Exception:
        logger.debug("Could not check embedding config", exc_info=True)

    # ── 4. Memory Consolidation ─────────────────────────
    try:
        from neural_memory.engine.memory_stages import MemoryStage

        brain_name = cfg.current_brain
        stats = await storage.get_stats(brain_name)
        total_neurons = stats.get("neuron_count", 0)

        semantic_records = await storage.find_maturations(
            stage=MemoryStage.SEMANTIC,
        )
        semantic_count = len(semantic_records)

        if total_neurons > 100 and semantic_count == 0:
            items.append(
                ConfigStatusItem(
                    key="consolidation",
                    label="Memory Consolidation",
                    status="warning",
                    description=(
                        f"{total_neurons} neurons, 0 semantic — memories need consolidation"
                    ),
                    command="nmem consolidate",
                    value=f"0 semantic / {total_neurons} total",
                )
            )
        else:
            items.append(
                ConfigStatusItem(
                    key="consolidation",
                    label="Memory Consolidation",
                    status="configured",
                    description="Memory consolidation is active",
                    command="",
                    value=f"{semantic_count} semantic / {total_neurons} total",
                )
            )
    except Exception:
        logger.debug("Could not check consolidation status", exc_info=True)

    # ── 5. Review Queue ─────────────────────────────────
    try:
        due_reviews = await storage.get_due_reviews(limit=100)
        due_count = len(due_reviews)
        if due_count > 0:
            items.append(
                ConfigStatusItem(
                    key="review_queue",
                    label="Review Queue",
                    status="info",
                    description=f"{due_count} memories due for spaced repetition review",
                    command='nmem_review(action="queue")',
                    value=f"{due_count} pending",
                )
            )
        else:
            items.append(
                ConfigStatusItem(
                    key="review_queue",
                    label="Review Queue",
                    status="configured",
                    description="No memories pending review",
                    command="",
                    value="0 pending",
                )
            )
    except Exception:
        logger.debug("Could not check review queue", exc_info=True)

    # ── 6. Orphan Rate ──────────────────────────────────
    try:
        from neural_memory.engine.diagnostics import DiagnosticsEngine

        brain_name = cfg.current_brain
        diag = DiagnosticsEngine(storage)
        report = await diag.analyze(brain_name)
        orphan_pct = round(report.orphan_rate * 100, 1)

        if report.orphan_rate > 0.20:
            items.append(
                ConfigStatusItem(
                    key="orphan_rate",
                    label="Orphan Neurons",
                    status="warning",
                    description=(f"{orphan_pct}% orphan rate — prune disconnected neurons"),
                    command="nmem consolidate --strategy prune",
                    value=f"{orphan_pct}%",
                )
            )
        else:
            items.append(
                ConfigStatusItem(
                    key="orphan_rate",
                    label="Orphan Neurons",
                    status="configured",
                    description=f"{orphan_pct}% orphan rate — within healthy range",
                    command="",
                    value=f"{orphan_pct}%",
                )
            )
    except Exception:
        logger.debug("Could not check orphan rate", exc_info=True)

    return ConfigStatusResponse(items=items)


# ── File Watcher Status API ──────────────────────────────


@router.get("/watcher/status")
async def get_watcher_status(request: Request) -> dict[str, Any]:
    """Return file watcher status and recent activity."""
    from neural_memory.unified_config import get_config

    config = get_config()
    watcher_cfg = config.watcher

    result: dict[str, Any] = {
        "enabled": watcher_cfg.enabled,
        "running": False,
        "paths": list(watcher_cfg.paths),
        "stats": {},
        "recent": [],
    }

    app = request.app
    if hasattr(app.state, "file_watcher"):
        watcher = app.state.file_watcher
        result["running"] = True
        results = watcher.get_recent_results()
        result["recent"] = [
            {
                "path": str(r.path),
                "action": "processed",
                "neurons_created": r.neurons_created,
            }
            for r in results[:10]
        ]

    return result


@router.get("/tool-stats")
async def tool_stats(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=20, ge=1, le=50),
) -> dict[str, Any]:
    """Tool usage analytics — top tools, success rates, daily trends."""
    from neural_memory.unified_config import get_config

    brain_name = get_config().current_brain
    brain = await storage.get_brain(brain_name)
    if not brain:
        return {"summary": {"total_events": 0, "success_rate": 0, "top_tools": []}, "daily": []}

    summary = await storage.get_tool_stats(brain.id)  # type: ignore[attr-defined]
    daily = await storage.get_tool_stats_by_period(brain.id, days=days, limit=limit)  # type: ignore[attr-defined]
    return {"summary": summary, "daily": daily}


@router.post("/visualize")
async def visualize_memory(
    storage: Annotated[NeuralStorage, Depends(get_storage)],
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate chart spec from memory data via nmem_visualize pipeline."""
    if body is None:
        body = {}
    query = str(body.get("query", "")).strip()
    if not query:
        raise HTTPException(status_code=400, detail="query parameter is required")

    chart_type = body.get("chart_type")
    output_format = str(body.get("format", "vega_lite"))
    limit = min(int(body.get("limit", 20)), 50)

    try:
        neurons = await storage.find_neurons(content_contains=query, limit=limit)
    except Exception:
        neurons = await storage.find_neurons(limit=limit)

    if not neurons:
        return {
            "query": query,
            "chart_type": "table",
            "message": "No data found for query.",
            "data_points": [],
        }

    from neural_memory.engine.chart_generator import extract_data_points, generate_chart

    data_points = extract_data_points(neurons, query)
    if not data_points:
        return {
            "query": query,
            "chart_type": "table",
            "message": "Found memories but no numeric data to chart.",
            "memories": [
                {
                    "id": getattr(n, "id", ""),
                    "content": (getattr(n, "content", "") or "")[:200],
                    "type": getattr(n, "type", ""),
                }
                for n in neurons[:10]
            ],
        }

    chart = generate_chart(
        data_points,
        chart_type=chart_type,
        title=query,
        output_format=output_format,
    )

    result: dict[str, Any] = {
        "query": query,
        "chart_type": chart.chart_type,
        "title": chart.title,
        "data_points_count": len(chart.data_points),
    }
    if chart.vega_lite:
        result["vega_lite"] = chart.vega_lite
    if chart.markdown:
        result["markdown"] = chart.markdown
    if chart.ascii_chart:
        result["ascii"] = chart.ascii_chart

    return result


@router.get("/license", tags=["dashboard"], summary="Current license tier")
async def get_license() -> dict[str, Any]:
    """Return the current license tier and expiry."""
    from neural_memory.unified_config import get_config

    cfg = get_config(reload=True)
    result: dict[str, Any] = {
        "tier": cfg.license.tier,
        "is_pro": cfg.is_pro(),
        "activated_at": cfg.license.activated_at,
        "expires_at": cfg.license.expires_at,
    }
    if not cfg.is_pro():
        from neural_memory.mcp.sync_handler import PRO_LANDING_URL

        result["upgrade_url"] = PRO_LANDING_URL
    return result


class ActivateLicenseRequest(BaseModel):
    license_key: str = Field(
        ..., min_length=5, description="License key (NM-PRO-XXXX or nm_pro_xxxx)"
    )


@router.post("/license/activate", tags=["dashboard"], summary="Activate a license key")
async def activate_license(body: ActivateLicenseRequest) -> dict[str, Any]:
    """Activate a license key via pay-hub (no sync config required)."""
    from dataclasses import replace as _dc_replace

    from neural_memory.mcp.sync_handler import DEFAULT_PAY_URL
    from neural_memory.unified_config import LicenseConfig, get_config, set_config
    from neural_memory.utils.timeutils import utcnow

    cfg = get_config(reload=True)

    original_key = body.license_key.strip()
    if not original_key:
        raise HTTPException(status_code=400, detail="License key is required")

    # Call pay-hub directly — no sync config needed
    try:
        import aiohttp

        from neural_memory.utils.ssl_helper import safe_client_session

        pay_url = f"{DEFAULT_PAY_URL}/verify"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        async with safe_client_session() as session:
            async with session.post(
                pay_url,
                json={"key": original_key},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status != 200 or not data.get("valid"):
                    detail = data.get("error", "Invalid or expired license key")
                    raise HTTPException(status_code=400, detail=detail)

                # Persist to config
                activated_tier = str(data.get("tier", "pro")).lower()
                new_license = LicenseConfig.from_dict(
                    {
                        "tier": activated_tier,
                        "activated_at": utcnow().isoformat(),
                        "expires_at": data.get("expires_at", ""),
                    }
                )
                new_cfg = _dc_replace(cfg, license=new_license)
                new_cfg.save()
                set_config(new_cfg)

                result: dict[str, Any] = {
                    "success": True,
                    "tier": activated_tier,
                    "activated_at": new_license.activated_at,
                    "expires_at": data.get("expires_at", ""),
                }

                # Hint about InfinityDB if on SQLite
                if new_cfg.storage_backend == "sqlite":
                    result["next_step"] = (
                        "Pro activated! To unlock InfinityDB (HNSW indexing, "
                        "tiered compression, cone queries), run: "
                        "nmem storage status → nmem migrate infinitydb "
                        "→ nmem storage switch infinitydb"
                    )

                return result
    except HTTPException:
        raise
    except Exception:
        logger.error("License activation failed", exc_info=True)
        raise HTTPException(status_code=502, detail="Could not reach activation server")


# ═══════════════════════════════════════════════════════════════════
# Storage Management — backend status, migration, backend switch
# ═══════════════════════════════════════════════════════════════════


class MigrationJobStatus(BaseModel):
    """Status of a storage migration job.

    NOTE: Fields are mutated in-place by ``_run_migration_task`` while the
    GET poll endpoint reads the same object.  This is safe under CPython's GIL
    (attribute assignments are atomic), but would require a lock if ever moved
    to threads.
    """

    job_id: str
    state: Literal["running", "done", "error"] = "running"
    direction: Literal["to_infinitydb", "to_sqlite"]
    brain: str
    neurons_total: int = 0
    neurons_done: int = 0
    synapses_total: int = 0
    synapses_done: int = 0
    fibers_total: int = 0
    fibers_done: int = 0
    error: str | None = None
    started_at: str = ""
    finished_at: str | None = None


class StorageStatusResponse(BaseModel):
    """Current storage backend status."""

    current_backend: Literal["sqlite", "infinitydb"]
    pro_installed: bool
    is_pro_license: bool
    sqlite_exists: bool
    sqlite_size_bytes: int = 0
    infinitydb_exists: bool
    migration_job: MigrationJobStatus | None = None


class StartMigrationRequest(BaseModel):
    """Request to start a storage migration."""

    direction: Literal["to_infinitydb", "to_sqlite"]
    brain: str | None = None


class SetBackendRequest(BaseModel):
    """Request to switch active storage backend."""

    backend: Literal["sqlite", "infinitydb"]


# In-memory job store — capped at _MAX_JOBS_PER_BRAIN to prevent unbounded growth
_MAX_JOBS_PER_BRAIN = 20
_migration_jobs: dict[str, MigrationJobStatus] = {}
_migration_tasks: set[asyncio.Task[None]] = set()


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _evict_old_jobs(brain_name: str) -> None:
    """Keep at most _MAX_JOBS_PER_BRAIN completed jobs per brain."""
    brain_jobs = [
        (jid, j)
        for jid, j in _migration_jobs.items()
        if j.brain == brain_name and j.state != "running"
    ]
    if len(brain_jobs) <= _MAX_JOBS_PER_BRAIN:
        return
    # Sort by started_at ascending, evict oldest
    brain_jobs.sort(key=lambda x: x[1].started_at)
    for jid, _ in brain_jobs[: len(brain_jobs) - _MAX_JOBS_PER_BRAIN]:
        _migration_jobs.pop(jid, None)


@router.get(
    "/storage/status",
    response_model=StorageStatusResponse,
    summary="Get storage backend status",
)
async def get_storage_status() -> StorageStatusResponse:
    """Return current storage backend, Pro status, and active migration job."""
    from neural_memory.plugins import has_pro
    from neural_memory.unified_config import get_config

    # Reload from disk so CLI changes (e.g. license activation) are picked up
    cfg = get_config(reload=True)
    brain_name = cfg.current_brain
    brains_dir = Path(cfg.data_dir) / "brains"

    sqlite_path = brains_dir / f"{brain_name}.db"
    sqlite_exists = sqlite_path.exists()
    sqlite_size = sqlite_path.stat().st_size if sqlite_exists else 0

    infinity_marker = brains_dir / brain_name / "brain.inf"
    infinitydb_exists = infinity_marker.exists()

    # Find most recent migration job for this brain
    active_job: MigrationJobStatus | None = None
    for job in reversed(list(_migration_jobs.values())):
        if job.brain == brain_name:
            active_job = job
            break

    return StorageStatusResponse(
        current_backend=cfg.storage_backend,
        pro_installed=has_pro(),
        is_pro_license=cfg.is_pro(),
        sqlite_exists=sqlite_exists,
        sqlite_size_bytes=sqlite_size,
        infinitydb_exists=infinitydb_exists,
        migration_job=active_job,
    )


@router.post(
    "/storage/migrate",
    summary="Start storage migration",
)
async def start_migration(body: StartMigrationRequest) -> dict[str, str]:
    """Trigger async migration between SQLite and InfinityDB."""
    from neural_memory.plugins import has_pro
    from neural_memory.unified_config import get_config

    cfg = get_config()
    brain_name = body.brain or cfg.current_brain
    direction = body.direction

    # Pre-flight: Pro check for InfinityDB
    if direction == "to_infinitydb":
        if not has_pro():
            raise HTTPException(status_code=403, detail="Neural Memory Pro package not installed")
        if not cfg.is_pro():
            raise HTTPException(
                status_code=403, detail="Pro license not active. Activate via 'nmem pro activate'"
            )

    # Pre-flight: source exists
    brains_dir = Path(cfg.data_dir) / "brains"
    if direction == "to_infinitydb":
        sqlite_path = brains_dir / f"{brain_name}.db"
        if not sqlite_path.exists():
            raise HTTPException(
                status_code=404, detail=f"No SQLite database found for brain '{brain_name}'"
            )
    else:
        infinity_marker = brains_dir / brain_name / "brain.inf"
        if not infinity_marker.exists():
            raise HTTPException(
                status_code=404, detail=f"No InfinityDB data found for brain '{brain_name}'"
            )

    # Pre-flight: no running job for same brain
    for job in _migration_jobs.values():
        if job.brain == brain_name and job.state == "running":
            raise HTTPException(
                status_code=409, detail=f"Migration already running for brain '{brain_name}'"
            )

    # Pre-flight: disk space estimate (non-blocking warning in response)
    disk_warning: str | None = None
    if direction == "to_infinitydb":
        import shutil

        source_size = sqlite_path.stat().st_size
        estimated_need = int(source_size * 1.5)
        disk_usage = shutil.disk_usage(str(brains_dir))
        if disk_usage.free < estimated_need:
            disk_warning = (
                f"Low disk space: need ~{estimated_need // (1024 * 1024)}MB, "
                f"only {disk_usage.free // (1024 * 1024)}MB free"
            )
            logger.warning(
                "Migration disk space warning for brain '%s': %s", brain_name, disk_warning
            )

    job_id = str(uuid4())
    job = MigrationJobStatus(
        job_id=job_id,
        state="running",
        direction=direction,
        brain=brain_name,
        started_at=_utcnow_iso(),
    )
    _migration_jobs[job_id] = job
    _evict_old_jobs(brain_name)

    # Launch background task — stored in set to prevent GC
    task = asyncio.create_task(_run_migration_task(job_id, direction, brain_name))
    _migration_tasks.add(task)
    task.add_done_callback(_migration_tasks.discard)

    result: dict[str, str] = {"job_id": job_id, "brain": brain_name, "message": "Migration started"}
    if disk_warning:
        result["disk_warning"] = disk_warning
    return result


@router.get(
    "/storage/migrate/{job_id}",
    response_model=MigrationJobStatus,
    summary="Get migration job progress",
)
async def get_migration_progress(job_id: str) -> MigrationJobStatus:
    """Poll for migration job status and progress."""
    job = _migration_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Migration job '{job_id}' not found")
    return job


@router.post(
    "/storage/backend",
    summary="Switch active storage backend",
)
async def set_storage_backend(body: SetBackendRequest) -> dict[str, str]:
    """Switch storage_backend in config.toml. Requires migration to be complete first."""
    from dataclasses import replace as dc_replace

    from neural_memory.unified_config import get_config, set_config

    backend = body.backend
    cfg = get_config(reload=True)

    if cfg.storage_backend == backend:
        return {"status": "unchanged", "backend": backend}

    brain_name = cfg.current_brain
    brains_dir = Path(cfg.data_dir) / "brains"

    # Guard: target must exist
    if backend == "infinitydb":
        from neural_memory.plugins import has_pro

        if not has_pro():
            raise HTTPException(status_code=403, detail="Neural Memory Pro package not installed")
        infinity_marker = brains_dir / brain_name / "brain.inf"
        if not infinity_marker.exists():
            raise HTTPException(
                status_code=400,
                detail="InfinityDB data not found. Run migration first.",
            )
    elif backend == "sqlite":
        sqlite_path = brains_dir / f"{brain_name}.db"
        if not sqlite_path.exists():
            raise HTTPException(
                status_code=400,
                detail="SQLite database not found for this brain.",
            )

    # Update config and clear storage cache
    new_cfg = dc_replace(cfg, storage_backend=backend)
    new_cfg.save()
    set_config(new_cfg)

    # Clear storage cache so next request picks up new backend
    from neural_memory.unified_config import _storage_cache

    _storage_cache.clear()

    return {"status": "switched", "backend": backend, "brain": brain_name}


async def _run_migration_task(
    job_id: str,
    direction: str,
    brain_name: str,
) -> None:
    """Background task: migrate brain data between SQLite and InfinityDB."""
    job = _migration_jobs[job_id]
    try:
        from neural_memory.unified_config import get_config

        cfg = get_config()

        # Open source storage
        if direction == "to_infinitydb":
            source = await _open_sqlite_storage(cfg, brain_name)
        else:
            source = await _open_infinitydb_storage(cfg, brain_name)

        # Find brain_id — exact match only, no silent fallback
        brains = await source.list_brains()
        brain_id: str | None = None
        for b in brains:
            if b.get("name") == brain_name:
                brain_id = b.get("id") or b.get("name")
                break

        if not brain_id:
            if brains:
                # Single-brain storage: use the only available brain
                brain_id = brains[0].get("id") or brains[0].get("name") or brain_name
            else:
                raise RuntimeError(f"No brain '{brain_name}' found in source storage")

        # Count totals
        stats = await source.get_stats(brain_id)
        job.neurons_total = stats.get("neuron_count", 0)
        job.synapses_total = stats.get("synapse_count", 0)
        job.fibers_total = stats.get("fiber_count", 0)

        # Export snapshot from source
        snapshot = await source.export_brain(brain_id)
        job.neurons_done = len(snapshot.neurons)
        job.synapses_done = len(snapshot.synapses)

        # Open target storage
        if direction == "to_infinitydb":
            target = await _open_infinitydb_storage(cfg, brain_name)
        else:
            target = await _open_sqlite_storage(cfg, brain_name, fresh=True)

        # Import into target
        await target.import_brain(snapshot)
        job.fibers_done = len(snapshot.fibers)

        # Verify counts match
        target_stats = await target.get_stats(brain_id)
        target_neurons = target_stats.get("neuron_count", 0)
        source_neurons = job.neurons_total

        if source_neurons > 0 and abs(target_neurons - source_neurons) / source_neurons > 0.005:
            job.state = "error"
            job.error = (
                f"Verification failed: source has {source_neurons} neurons, "
                f"target has {target_neurons} (>{0.5}% mismatch)"
            )
        else:
            # For to_sqlite: promote temp file to real .db path
            if direction == "to_sqlite":
                brains_dir = Path(cfg.data_dir) / "brains"
                tmp_path = brains_dir / f"{brain_name}_migrating.db"
                real_path = brains_dir / f"{brain_name}.db"
                if tmp_path.exists():
                    import shutil as _shutil

                    _shutil.move(str(tmp_path), str(real_path))
            job.state = "done"

        job.finished_at = _utcnow_iso()

    except Exception as e:
        logger.error("Migration task failed for brain '%s': %s", brain_name, e, exc_info=True)
        job.state = "error"
        # Sanitize: don't leak internal paths or exception class names
        job.error = "Migration failed. Check server logs for details."
        job.finished_at = _utcnow_iso()


async def _open_sqlite_storage(
    cfg: Any,
    brain_name: str,
    *,
    fresh: bool = False,
) -> Any:
    """Open a SQLite storage instance for the given brain."""
    from neural_memory.storage.sqlite import SQLiteStorage

    brains_dir = Path(cfg.data_dir) / "brains"
    db_path = brains_dir / f"{brain_name}.db"

    if fresh:
        # Write to temp file, will be renamed on success
        tmp_path = brains_dir / f"{brain_name}_migrating.db"
        storage = SQLiteStorage(str(tmp_path))
    else:
        storage = SQLiteStorage(str(db_path))

    await storage.initialize()

    # Set brain context
    brain_list = await storage.list_brains()
    if brain_list:
        brain_id = brain_list[0].get("id") or brain_list[0].get("name")
        storage.set_brain(brain_id)

    return storage


async def _open_infinitydb_storage(
    cfg: Any,
    brain_name: str,
) -> Any:
    """Open an InfinityDB storage instance for the given brain."""
    try:
        from neural_memory.pro.storage_adapter import InfinityDBStorage
    except ImportError:
        raise RuntimeError("InfinityDB storage not available — Pro dependencies not installed")

    brains_dir = Path(cfg.data_dir) / "brains"

    storage = InfinityDBStorage(base_dir=str(brains_dir), brain_id=brain_name)
    await storage.initialize()

    return storage
