"""Tests for brain_evolution: cognitive metrics layer."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.brain_evolution import (
    BrainEvolution,
    EvolutionEngine,
    ProficiencyLevel,
    _compute_activity,
    _compute_decay,
    _compute_plasticity,
    _compute_proficiency,
    _maturity_signal,
)
from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage
from neural_memory.utils.timeutils import utcnow


def _mock_synapse(
    source_id: str = "a",
    target_id: str = "b",
    created_at: datetime | None = None,
    last_activated: datetime | None = None,
    reinforced_count: int = 0,
) -> MagicMock:
    s = MagicMock()
    s.source_id = source_id
    s.target_id = target_id
    s.created_at = created_at or utcnow()
    s.last_activated = last_activated
    s.reinforced_count = reinforced_count
    s.metadata = {}
    return s


def _mock_fiber(
    last_conducted: datetime | None = None,
    frequency: int = 0,
    conductivity: float = 1.0,
    created_at: datetime | None = None,
) -> MagicMock:
    f = MagicMock()
    f.last_conducted = last_conducted
    f.frequency = frequency
    f.conductivity = conductivity
    f.created_at = created_at or utcnow()
    return f


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Storage with empty brain."""
    storage = AsyncMock()
    brain = MagicMock()
    brain.name = "test-brain"
    brain.id = "brain-1"
    storage.get_brain = AsyncMock(return_value=brain)
    storage.get_stats = AsyncMock(
        return_value={"neuron_count": 0, "synapse_count": 0, "fiber_count": 0}
    )
    storage.get_all_synapses = AsyncMock(return_value=[])
    storage.get_neighbors = AsyncMock(return_value=[])
    storage.find_maturations = AsyncMock(return_value=[])
    storage.get_fibers = AsyncMock(return_value=[])
    storage._current_brain_id = "brain-1"
    return storage


class TestBrainEvolutionDataclass:
    """Tests for BrainEvolution immutability."""

    def test_frozen(self) -> None:
        """BrainEvolution is immutable."""
        evo = BrainEvolution(
            brain_id="brain-1",
            brain_name="test",
            computed_at=utcnow(),
            semantic_ratio=0.0,
            reinforcement_days=0.0,
            topology_coherence=0.0,
            plasticity_index=0.0,
            knowledge_density=0.0,
            maturity_level=0.0,
            plasticity=0.0,
            density=0.0,
            proficiency_index=0,
            proficiency_level=ProficiencyLevel.JUNIOR,
            activity_score=0.0,
            total_fibers=0,
            total_synapses=0,
            total_neurons=0,
            fibers_at_semantic=0,
            fibers_at_episodic=0,
        )
        with pytest.raises(AttributeError):
            evo.proficiency_index = 99  # type: ignore[misc]


class TestProficiencyLevel:
    """Tests for proficiency level enum."""

    def test_values(self) -> None:
        assert ProficiencyLevel.JUNIOR.value == "junior"
        assert ProficiencyLevel.SENIOR.value == "senior"
        assert ProficiencyLevel.EXPERT.value == "expert"

    def test_exactly_three_levels(self) -> None:
        """Enum has exactly 3 members."""
        assert len(ProficiencyLevel) == 3


class TestProficiencyComputation:
    """Tests for _compute_proficiency pure function."""

    def test_junior_low_metrics(self) -> None:
        """Low metrics → JUNIOR."""
        index, level = _compute_proficiency(
            semantic_ratio=0.0,
            reinforcement_days=0.0,
            topology_coherence=0.0,
            plasticity_index=0.0,
            decay_factor=1.0,
        )
        assert level == ProficiencyLevel.JUNIOR
        assert index < 25

    def test_senior_medium_metrics(self) -> None:
        """Medium metrics + 5 reinforcement days → SENIOR."""
        index, level = _compute_proficiency(
            semantic_ratio=0.3,
            reinforcement_days=5.0,
            topology_coherence=0.5,
            plasticity_index=0.1,
            decay_factor=1.0,
        )
        assert level == ProficiencyLevel.SENIOR
        assert 25 <= index <= 55

    def test_expert_high_metrics(self) -> None:
        """High metrics + 12 reinforcement days → EXPERT."""
        index, level = _compute_proficiency(
            semantic_ratio=0.8,
            reinforcement_days=12.0,
            topology_coherence=0.8,
            plasticity_index=0.3,
            decay_factor=1.0,
        )
        assert level == ProficiencyLevel.EXPERT
        assert index > 55

    def test_senior_needs_reinforcement_days(self) -> None:
        """High semantic but low reinforcement days → stays JUNIOR (AND condition)."""
        index, level = _compute_proficiency(
            semantic_ratio=0.5,
            reinforcement_days=2.0,  # Below 4
            topology_coherence=0.5,
            plasticity_index=0.2,
            decay_factor=1.0,
        )
        assert level == ProficiencyLevel.JUNIOR

    def test_expert_needs_reinforcement_days(self) -> None:
        """High index but low reinforcement → SENIOR not EXPERT."""
        index, level = _compute_proficiency(
            semantic_ratio=0.9,
            reinforcement_days=6.0,  # Above 4, below 10
            topology_coherence=0.9,
            plasticity_index=0.5,
            decay_factor=1.0,
        )
        assert level == ProficiencyLevel.SENIOR

    def test_decay_reduces_proficiency(self) -> None:
        """Low decay factor reduces proficiency index."""
        index_full, _ = _compute_proficiency(
            semantic_ratio=0.5,
            reinforcement_days=5.0,
            topology_coherence=0.5,
            plasticity_index=0.1,
            decay_factor=1.0,
        )
        index_decayed, _ = _compute_proficiency(
            semantic_ratio=0.5,
            reinforcement_days=5.0,
            topology_coherence=0.5,
            plasticity_index=0.1,
            decay_factor=0.3,
        )
        assert index_decayed < index_full

    def test_index_clamped(self) -> None:
        """Proficiency index stays in 0-100 range."""
        index, _ = _compute_proficiency(
            semantic_ratio=1.0,
            reinforcement_days=20.0,
            topology_coherence=1.0,
            plasticity_index=1.0,
            decay_factor=1.0,
        )
        assert 0 <= index <= 100

    # ── Boundary condition tests ──────────────────────────────

    def test_boundary_expert_at_55(self) -> None:
        """Index exactly 55 with 10+ days → SENIOR (not EXPERT, needs > 55)."""
        # raw = 0.5*0.30 + 1.0*0.25 + 0.4*0.25 + 0.25*0.20 = 0.55 → index=55
        _, level = _compute_proficiency(
            semantic_ratio=0.5,
            reinforcement_days=10.0,
            topology_coherence=0.4,
            plasticity_index=0.05,
            decay_factor=1.0,
        )
        # index = 55, which is NOT > 55
        assert level == ProficiencyLevel.SENIOR

    def test_boundary_expert_at_56(self) -> None:
        """Index 56 with 10+ days → EXPERT."""
        _, level = _compute_proficiency(
            semantic_ratio=0.56,
            reinforcement_days=10.0,
            topology_coherence=0.56,
            plasticity_index=0.112,  # norm = 0.56
            decay_factor=1.0,
        )
        assert level == ProficiencyLevel.EXPERT

    def test_boundary_senior_at_25(self) -> None:
        """Index exactly 25 with 4+ days → SENIOR."""
        _, level = _compute_proficiency(
            semantic_ratio=0.25,
            reinforcement_days=4.0,
            topology_coherence=0.25,
            plasticity_index=0.05,  # norm = 0.25
            decay_factor=1.0,
        )
        assert level == ProficiencyLevel.SENIOR

    def test_boundary_junior_below_25(self) -> None:
        """Index 24 with 4+ days → JUNIOR."""
        # raw = 0.2*0.30 + 0.4*0.25 + 0.2*0.25 + 0.15*0.20 = 0.24 → index=24
        _, level = _compute_proficiency(
            semantic_ratio=0.2,
            reinforcement_days=4.0,
            topology_coherence=0.2,
            plasticity_index=0.03,
            decay_factor=1.0,
        )
        assert level == ProficiencyLevel.JUNIOR

    def test_boundary_senior_needs_4_days(self) -> None:
        """Index 25+ but reinforcement_days < 4 → JUNIOR."""
        _, level = _compute_proficiency(
            semantic_ratio=0.5,
            reinforcement_days=3.99,
            topology_coherence=0.5,
            plasticity_index=0.1,
            decay_factor=1.0,
        )
        assert level == ProficiencyLevel.JUNIOR


class TestMaturitySignal:
    """Tests for _maturity_signal agent-facing function."""

    def test_zero(self) -> None:
        assert _maturity_signal(0.0, 0.0) == 0.0

    def test_full(self) -> None:
        signal = _maturity_signal(0.5, 7.0)
        assert signal == 1.0

    def test_partial(self) -> None:
        signal = _maturity_signal(0.25, 3.5)
        assert 0.0 < signal < 1.0

    def test_clamped_high_ratio(self) -> None:
        """semantic_ratio > 0.5 still caps ratio_signal at 1.0."""
        signal = _maturity_signal(0.8, 7.0)
        assert signal == 1.0

    def test_clamped_high_days(self) -> None:
        """reinforcement_days > 7.0 still caps days_signal at 1.0."""
        signal = _maturity_signal(0.5, 14.0)
        assert signal == 1.0


class TestComputePlasticity:
    """Tests for _compute_plasticity pure function."""

    def test_empty_synapses(self) -> None:
        """No synapses → 0.0."""
        assert _compute_plasticity([], utcnow()) == 0.0

    def test_all_new(self) -> None:
        """All synapses created recently → plasticity 1.0."""
        now = utcnow()
        synapses = [
            _mock_synapse(created_at=now - timedelta(days=1)),
            _mock_synapse(created_at=now - timedelta(days=2)),
            _mock_synapse(created_at=now - timedelta(days=3)),
        ]
        assert _compute_plasticity(synapses, now) == 1.0

    def test_all_old(self) -> None:
        """All synapses old and not reinforced → plasticity 0.0."""
        now = utcnow()
        synapses = [
            _mock_synapse(created_at=now - timedelta(days=30)),
            _mock_synapse(created_at=now - timedelta(days=60)),
        ]
        assert _compute_plasticity(synapses, now) == 0.0

    def test_no_double_count(self) -> None:
        """Synapse that is both new AND reinforced is counted once (not twice)."""
        now = utcnow()
        synapses = [
            _mock_synapse(
                created_at=now - timedelta(days=1),
                last_activated=now - timedelta(hours=2),
            ),
        ]
        result = _compute_plasticity(synapses, now)
        # 1 synapse active / 1 total = 1.0 (not 2.0)
        assert result == 1.0

    def test_bounded_zero_to_one(self) -> None:
        """Plasticity is always in [0.0, 1.0]."""
        now = utcnow()
        # All synapses are both new and recently activated
        synapses = [
            _mock_synapse(
                created_at=now - timedelta(days=i),
                last_activated=now - timedelta(hours=i),
            )
            for i in range(1, 6)
        ]
        result = _compute_plasticity(synapses, now)
        assert 0.0 <= result <= 1.0


class TestComputeActivity:
    """Tests for _compute_activity pure function."""

    def test_zero_fiber_count(self) -> None:
        """Zero fiber_count → 0.0."""
        assert _compute_activity([], utcnow(), fiber_count=0) == 0.0

    def test_all_active(self) -> None:
        """All fibers recently conducted → 1.0."""
        now = utcnow()
        fibers = [
            _mock_fiber(last_conducted=now - timedelta(days=1)),
            _mock_fiber(last_conducted=now - timedelta(days=2)),
        ]
        assert _compute_activity(fibers, now, fiber_count=2) == 1.0

    def test_all_inactive(self) -> None:
        """All fibers old → 0.0."""
        now = utcnow()
        fibers = [
            _mock_fiber(last_conducted=now - timedelta(days=30)),
            _mock_fiber(last_conducted=now - timedelta(days=60)),
        ]
        assert _compute_activity(fibers, now, fiber_count=2) == 0.0

    def test_none_last_conducted(self) -> None:
        """Fibers with last_conducted=None are not counted as active."""
        now = utcnow()
        fibers = [_mock_fiber(last_conducted=None)]
        assert _compute_activity(fibers, now, fiber_count=1) == 0.0


class TestComputeDecay:
    """Tests for _compute_decay pure function."""

    def test_empty_fibers(self) -> None:
        """No fibers → neutral decay 0.5."""
        assert _compute_decay([], utcnow()) == 0.5

    def test_all_none_timestamps(self) -> None:
        """Fibers with both last_conducted and created_at as None → 0.5."""
        f = MagicMock()
        f.last_conducted = None
        f.created_at = None
        assert _compute_decay([f], utcnow()) == 0.5

    def test_recent_usage(self) -> None:
        """Recently used brain → decay close to 1.0."""
        now = utcnow()
        fibers = [_mock_fiber(last_conducted=now - timedelta(hours=1))]
        decay = _compute_decay(fibers, now)
        assert decay > 0.9

    def test_30_day_midpoint(self) -> None:
        """Brain unused for 30 days → decay ≈ 0.5 (sigmoid midpoint)."""
        now = utcnow()
        fibers = [_mock_fiber(last_conducted=now - timedelta(days=30))]
        decay = _compute_decay(fibers, now)
        assert abs(decay - 0.5) < 0.01

    def test_60_day_low(self) -> None:
        """Brain unused for 60 days → decay < 0.1."""
        now = utcnow()
        fibers = [_mock_fiber(last_conducted=now - timedelta(days=60))]
        decay = _compute_decay(fibers, now)
        assert decay < 0.1

    def test_uses_created_at_fallback(self) -> None:
        """When last_conducted is None, uses created_at."""
        now = utcnow()
        fibers = [
            _mock_fiber(last_conducted=None, created_at=now - timedelta(hours=1))
        ]
        decay = _compute_decay(fibers, now)
        assert decay > 0.9


class TestEvolutionEngine:
    """Tests for EvolutionEngine.analyze()."""

    @pytest.mark.asyncio
    async def test_empty_brain(self, mock_storage: AsyncMock) -> None:
        """Empty brain → JUNIOR with zero metrics."""
        engine = EvolutionEngine(mock_storage)
        evo = await engine.analyze("brain-1")

        assert evo.proficiency_level == ProficiencyLevel.JUNIOR
        assert evo.semantic_ratio == 0.0
        assert evo.reinforcement_days == 0.0
        assert evo.plasticity_index == 0.0
        assert evo.fibers_at_semantic == 0
        assert evo.fibers_at_episodic == 0
        assert evo.brain_name == "test-brain"

    @pytest.mark.asyncio
    async def test_with_semantic_maturations(self, mock_storage: AsyncMock) -> None:
        """Fibers at SEMANTIC → semantic_ratio > 0."""
        maturations = [
            MaturationRecord(
                fiber_id=f"f{i}",
                brain_id="brain-1",
                stage=MemoryStage.SEMANTIC if i < 3 else MemoryStage.EPISODIC,
                reinforcement_timestamps=[
                    "2025-01-01T10:00:00",
                    "2025-01-03T10:00:00",
                    "2025-01-05T10:00:00",
                ],
            )
            for i in range(5)
        ]
        mock_storage.find_maturations = AsyncMock(return_value=maturations)

        engine = EvolutionEngine(mock_storage)
        evo = await engine.analyze("brain-1")

        assert evo.semantic_ratio == 0.6  # 3/5
        assert evo.fibers_at_semantic == 3
        assert evo.fibers_at_episodic == 2
        assert evo.reinforcement_days == 3.0  # 3 distinct days

    @pytest.mark.asyncio
    async def test_plasticity_recent_synapses(self, mock_storage: AsyncMock) -> None:
        """Recently created/reinforced synapses → plasticity > 0."""
        now = utcnow()
        synapses = [
            _mock_synapse(created_at=now - timedelta(days=1)),  # new
            _mock_synapse(
                created_at=now - timedelta(days=30),
                last_activated=now - timedelta(days=2),
            ),  # reinforced
            _mock_synapse(created_at=now - timedelta(days=60)),  # old, inactive
        ]
        mock_storage.get_all_synapses = AsyncMock(return_value=synapses)
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 3, "synapse_count": 3, "fiber_count": 1}
        )

        engine = EvolutionEngine(mock_storage)
        evo = await engine.analyze("brain-1")

        # 2 distinct active synapses out of 3 total = 2/3
        assert abs(evo.plasticity_index - 2 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_plasticity_no_double_count(self, mock_storage: AsyncMock) -> None:
        """Synapse both new and reinforced is counted once."""
        now = utcnow()
        synapses = [
            _mock_synapse(
                created_at=now - timedelta(days=1),
                last_activated=now - timedelta(hours=2),
            ),
        ]
        mock_storage.get_all_synapses = AsyncMock(return_value=synapses)
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 2, "synapse_count": 1, "fiber_count": 0}
        )

        engine = EvolutionEngine(mock_storage)
        evo = await engine.analyze("brain-1")

        # 1 active / 1 total = 1.0 (not 2.0)
        assert evo.plasticity_index <= 1.0

    @pytest.mark.asyncio
    async def test_activity_score(self, mock_storage: AsyncMock) -> None:
        """Recently conducted fibers → activity_score > 0."""
        now = utcnow()
        fibers = [
            _mock_fiber(last_conducted=now - timedelta(days=1)),
            _mock_fiber(last_conducted=now - timedelta(days=30)),
            _mock_fiber(last_conducted=now - timedelta(hours=2)),
        ]
        mock_storage.get_fibers = AsyncMock(return_value=fibers)
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 5, "synapse_count": 10, "fiber_count": 3}
        )

        engine = EvolutionEngine(mock_storage)
        evo = await engine.analyze("brain-1")

        # 2 fibers conducted in last 7 days out of 3 total
        assert abs(evo.activity_score - 2 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_agent_signals_bounded(self, mock_storage: AsyncMock) -> None:
        """Agent-facing signals are bounded 0.0-1.0."""
        engine = EvolutionEngine(mock_storage)
        evo = await engine.analyze("brain-1")

        assert 0.0 <= evo.maturity_level <= 1.0
        assert 0.0 <= evo.plasticity <= 1.0
        assert 0.0 <= evo.density <= 1.0

    @pytest.mark.asyncio
    async def test_agent_signals_bounded_extreme(self, mock_storage: AsyncMock) -> None:
        """Agent signals stay bounded with extreme metric values."""
        now = utcnow()
        # High knowledge density (100 synapses / 1 neuron)
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 1, "synapse_count": 100, "fiber_count": 10}
        )
        # All synapses are new → max plasticity
        synapses = [
            _mock_synapse(created_at=now - timedelta(hours=i))
            for i in range(100)
        ]
        mock_storage.get_all_synapses = AsyncMock(return_value=synapses)
        # Recent fibers
        fibers = [
            _mock_fiber(last_conducted=now - timedelta(hours=1))
            for _ in range(10)
        ]
        mock_storage.get_fibers = AsyncMock(return_value=fibers)

        engine = EvolutionEngine(mock_storage)
        evo = await engine.analyze("brain-1")

        assert 0.0 <= evo.maturity_level <= 1.0
        assert 0.0 <= evo.plasticity <= 1.0
        assert 0.0 <= evo.density <= 1.0

    @pytest.mark.asyncio
    async def test_decay_old_brain(self, mock_storage: AsyncMock) -> None:
        """Brain not used for 60 days → decay reduces proficiency."""
        now = utcnow()
        old_fibers = [
            _mock_fiber(
                last_conducted=now - timedelta(days=60),
                created_at=now - timedelta(days=90),
            ),
        ]
        mock_storage.get_fibers = AsyncMock(return_value=old_fibers)
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 5, "synapse_count": 10, "fiber_count": 1}
        )

        # Add some maturations to give non-zero base score
        maturations = [
            MaturationRecord(
                fiber_id="f1",
                brain_id="brain-1",
                stage=MemoryStage.SEMANTIC,
                reinforcement_timestamps=[
                    "2024-01-01T10:00:00",
                    "2024-01-05T10:00:00",
                    "2024-01-10T10:00:00",
                    "2024-01-15T10:00:00",
                    "2024-01-20T10:00:00",
                ],
            ),
        ]
        mock_storage.find_maturations = AsyncMock(return_value=maturations)

        engine = EvolutionEngine(mock_storage)
        evo = await engine.analyze("brain-1")

        # Decay should significantly reduce proficiency
        assert evo.proficiency_index < 50

    @pytest.mark.asyncio
    async def test_unknown_brain_name(self, mock_storage: AsyncMock) -> None:
        """get_brain returns None → brain_name 'unknown'."""
        mock_storage.get_brain = AsyncMock(return_value=None)

        engine = EvolutionEngine(mock_storage)
        evo = await engine.analyze("brain-1")

        assert evo.brain_name == "unknown"

    @pytest.mark.asyncio
    async def test_prefetches_data_once(self, mock_storage: AsyncMock) -> None:
        """analyze() calls get_all_synapses and get_fibers exactly once."""
        engine = EvolutionEngine(mock_storage)
        await engine.analyze("brain-1")

        # get_all_synapses called once (pre-fetched, passed to topology + plasticity)
        assert mock_storage.get_all_synapses.call_count == 1
        # get_fibers called once (pre-fetched, passed to activity + decay)
        assert mock_storage.get_fibers.call_count == 1

    @pytest.mark.asyncio
    async def test_full_integration(self, mock_storage: AsyncMock) -> None:
        """Full integration test with all data sources populated."""
        now = utcnow()

        # Stats
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 10, "synapse_count": 15, "fiber_count": 5}
        )

        # Maturations: 3 SEMANTIC, 2 EPISODIC
        maturations = [
            MaturationRecord(
                fiber_id=f"f{i}",
                brain_id="brain-1",
                stage=MemoryStage.SEMANTIC if i < 3 else MemoryStage.EPISODIC,
                reinforcement_timestamps=[
                    "2025-01-01T10:00:00",
                    "2025-01-03T10:00:00",
                    "2025-01-07T10:00:00",
                    "2025-01-10T10:00:00",
                    "2025-01-15T10:00:00",
                ],
            )
            for i in range(5)
        ]
        mock_storage.find_maturations = AsyncMock(return_value=maturations)

        # Synapses: mix of new, reinforced, and old
        synapses = [
            _mock_synapse("a", "b", created_at=now - timedelta(days=1)),
            _mock_synapse("b", "c", created_at=now - timedelta(days=30),
                          last_activated=now - timedelta(days=2)),
            _mock_synapse("c", "a", created_at=now - timedelta(days=60)),
            _mock_synapse("a", "c", created_at=now - timedelta(days=2)),
        ]
        mock_storage.get_all_synapses = AsyncMock(return_value=synapses)

        # Fibers: mix of recent and old
        fibers = [
            _mock_fiber(last_conducted=now - timedelta(days=1)),
            _mock_fiber(last_conducted=now - timedelta(days=3)),
            _mock_fiber(last_conducted=now - timedelta(days=30)),
            _mock_fiber(last_conducted=now - timedelta(days=60)),
            _mock_fiber(last_conducted=now - timedelta(hours=2)),
        ]
        mock_storage.get_fibers = AsyncMock(return_value=fibers)

        engine = EvolutionEngine(mock_storage)
        evo = await engine.analyze("brain-1")

        # Verify all fields populated with reasonable values
        assert evo.semantic_ratio == 0.6
        assert evo.reinforcement_days == 5.0
        assert evo.fibers_at_semantic == 3
        assert evo.fibers_at_episodic == 2
        assert evo.plasticity_index > 0.0
        assert evo.activity_score > 0.0
        assert evo.topology_coherence >= 0.0
        assert evo.total_fibers == 5
        assert evo.total_synapses == 15
        assert evo.total_neurons == 10
        assert 0.0 <= evo.maturity_level <= 1.0
        assert 0.0 <= evo.plasticity <= 1.0
        assert 0.0 <= evo.density <= 1.0
        assert 0 <= evo.proficiency_index <= 100
        assert evo.proficiency_level in ProficiencyLevel


class TestMCPEvolutionHandler:
    """Tests for nmem_evolution MCP handler."""

    @pytest.mark.asyncio
    async def test_evolution_handler_success(self, mock_storage: AsyncMock) -> None:
        """Normal _evolution handler returns all expected keys."""
        from neural_memory.mcp.server import MCPServer

        server = MCPServer.__new__(MCPServer)
        server.get_storage = AsyncMock(return_value=mock_storage)

        result = await server._evolution({})

        assert "proficiency_level" in result
        assert "proficiency_index" in result
        assert "maturity_level" in result
        assert "plasticity" in result
        assert "density" in result
        assert "activity_score" in result
        assert "brain" in result
        assert result["brain"] == "test-brain"
        assert result["proficiency_level"] == "junior"

    @pytest.mark.asyncio
    async def test_evolution_handler_no_brain(self) -> None:
        """_evolution handler returns error when no brain configured."""
        from neural_memory.mcp.server import MCPServer

        storage = AsyncMock()
        storage.get_brain = AsyncMock(return_value=None)
        storage._current_brain_id = "missing"

        server = MCPServer.__new__(MCPServer)
        server.get_storage = AsyncMock(return_value=storage)

        result = await server._evolution({})
        assert "error" in result
        assert result["error"] == "No brain configured"

    @pytest.mark.asyncio
    async def test_evolution_handler_engine_error(
        self, mock_storage: AsyncMock
    ) -> None:
        """_evolution handler catches engine errors gracefully."""
        from neural_memory.mcp.server import MCPServer

        # Make get_stats raise to simulate engine failure
        mock_storage.get_stats = AsyncMock(side_effect=RuntimeError("DB error"))

        server = MCPServer.__new__(MCPServer)
        server.get_storage = AsyncMock(return_value=mock_storage)

        result = await server._evolution({})
        assert "error" in result
        assert result["error"] == "Evolution analysis failed"
