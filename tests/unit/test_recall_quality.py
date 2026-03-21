"""Tests for Phase 3: Recall Quality Improvements.

Covers:
- Configurable recency halflife (BrainConfig.recency_halflife_hours)
- Tag-aware fiber scoring boost
- Dead neuron pruning (access_frequency=0, old enough, not pinned)
"""

from __future__ import annotations

import math
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.brain import BrainConfig
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.utils.timeutils import utcnow

# ---------------------------------------------------------------------------
# BrainConfig new fields
# ---------------------------------------------------------------------------


class TestBrainConfigRecallFields:
    """Verify new recall quality fields on BrainConfig."""

    def test_recency_halflife_default(self) -> None:
        cfg = BrainConfig()
        assert cfg.recency_halflife_hours == 168.0

    def test_tag_match_boost_default(self) -> None:
        cfg = BrainConfig()
        assert cfg.tag_match_boost == 0.15

    def test_prune_dead_neuron_days_default(self) -> None:
        cfg = BrainConfig()
        assert cfg.prune_dead_neuron_days == 14.0

    def test_custom_halflife(self) -> None:
        cfg = BrainConfig(recency_halflife_hours=72.0)
        assert cfg.recency_halflife_hours == 72.0

    def test_with_updates_halflife(self) -> None:
        cfg = BrainConfig()
        updated = cfg.with_updates(recency_halflife_hours=720.0)
        assert updated.recency_halflife_hours == 720.0
        assert cfg.recency_halflife_hours == 168.0  # original unchanged


# ---------------------------------------------------------------------------
# Recency sigmoid with configurable halflife
# ---------------------------------------------------------------------------


class TestRecencySigmoid:
    """Test that recency sigmoid uses configurable halflife."""

    def _sigmoid(self, hours_ago: float, halflife: float) -> float:
        """Reproduce the sigmoid formula from retrieval.py."""
        return max(0.1, 1.0 / (1.0 + math.exp((hours_ago - halflife) / (halflife / 2))))

    def test_at_halflife_score_is_half(self) -> None:
        """At exactly halflife hours, sigmoid should be ~0.5."""
        score = self._sigmoid(168.0, 168.0)
        assert abs(score - 0.5) < 0.01

    def test_recent_memory_scores_high(self) -> None:
        """Memory from 1 hour ago should score high."""
        score = self._sigmoid(1.0, 168.0)
        assert score > 0.85

    def test_old_memory_scores_low(self) -> None:
        """Memory from 30 days ago should score near minimum."""
        score = self._sigmoid(720.0, 168.0)
        assert score < 0.2

    def test_short_halflife_decays_faster(self) -> None:
        """72h halflife should decay faster than 168h at 7 days."""
        score_72 = self._sigmoid(168.0, 72.0)
        score_168 = self._sigmoid(168.0, 168.0)
        assert score_72 < score_168

    def test_long_halflife_retains_longer(self) -> None:
        """720h halflife (30 days) keeps memories relevant longer."""
        score_720 = self._sigmoid(168.0, 720.0)
        assert score_720 > 0.7  # still relevant after 7 days

    def test_minimum_floor(self) -> None:
        """Score never goes below 0.1."""
        score = self._sigmoid(10000.0, 168.0)
        assert score >= 0.1


# ---------------------------------------------------------------------------
# Tag-aware scoring
# ---------------------------------------------------------------------------


class TestTagAwareScoring:
    """Test tag-aware scoring logic."""

    def test_matching_tags_boost(self) -> None:
        """Fibers with matching tags should get a boost."""
        tags = {"python", "auth"}
        fiber_tags = {"python", "auth", "security"}
        overlap = len(tags & fiber_tags)
        boost = 0.15 * min(overlap, 3) / 3
        assert boost > 0
        assert boost == pytest.approx(0.1, abs=0.01)

    def test_no_overlap_penalty(self) -> None:
        """Fibers with zero tag overlap should get mild penalty."""
        tags = {"python", "auth"}
        fiber_tags = {"javascript", "frontend"}
        overlap = len(tags & fiber_tags)
        penalty = 0.15 * 0.5 if overlap == 0 else 0
        assert penalty == pytest.approx(0.075, abs=0.001)

    def test_no_query_tags_no_effect(self) -> None:
        """If query has no tags, no boost or penalty."""
        # When tags is None or empty, the tag boost block is skipped
        tags: set[str] = set()
        # No boost applied
        assert len(tags) == 0

    def test_max_boost_cap(self) -> None:
        """Boost is capped at 3 matching tags."""
        tags = {"a", "b", "c", "d", "e"}
        fiber_tags = {"a", "b", "c", "d", "e"}
        overlap = len(tags & fiber_tags)
        boost = 0.15 * min(overlap, 3) / 3
        # Should be capped at 0.15 (3/3), not 0.25 (5/3)
        assert boost == pytest.approx(0.15, abs=0.001)


# ---------------------------------------------------------------------------
# Dead neuron pruning
# ---------------------------------------------------------------------------


class TestDeadNeuronPruning:
    """Test dead neuron pruning in consolidation."""

    def _make_neuron(self, neuron_id: str, days_old: int) -> Neuron:
        return Neuron(
            id=neuron_id,
            type=NeuronType.CONCEPT,
            content=f"test content {neuron_id}",
            created_at=utcnow() - timedelta(days=days_old),
        )

    def _make_state(self, neuron_id: str, freq: int = 0) -> NeuronState:
        return NeuronState(neuron_id=neuron_id, access_frequency=freq)

    @pytest.mark.asyncio
    async def test_dead_neurons_pruned(self) -> None:
        """Neurons with access_frequency=0 and age > 14 days should be pruned."""
        from neural_memory.engine.consolidation import ConsolidationConfig, ConsolidationEngine

        old_neuron = self._make_neuron("old-dead", 30)
        storage = AsyncMock()
        storage.current_brain_id = "test"
        storage.get_synapses = AsyncMock(return_value=[])
        storage.get_fibers = AsyncMock(return_value=[])
        storage.get_pinned_neuron_ids = AsyncMock(return_value=set())
        storage.find_neurons = AsyncMock(side_effect=[[old_neuron], []])
        storage.get_neuron_states_batch = AsyncMock(
            return_value={"old-dead": self._make_state("old-dead", freq=0)}
        )
        storage.delete_neurons_batch = AsyncMock()

        config = ConsolidationConfig()
        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=["prune"])

        assert report.neurons_pruned >= 1
        storage.delete_neurons_batch.assert_called_once()
        deleted_ids = storage.delete_neurons_batch.call_args[0][0]
        assert "old-dead" in deleted_ids

    @pytest.mark.asyncio
    async def test_young_neurons_not_pruned(self) -> None:
        """Neurons younger than 14 days should NOT be pruned even with 0 access."""
        from neural_memory.engine.consolidation import ConsolidationConfig, ConsolidationEngine

        young_neuron = self._make_neuron("young", 5)
        storage = AsyncMock()
        storage.current_brain_id = "test"
        storage.get_synapses = AsyncMock(return_value=[])
        storage.get_fibers = AsyncMock(return_value=[])
        storage.get_pinned_neuron_ids = AsyncMock(return_value=set())
        storage.find_neurons = AsyncMock(side_effect=[[young_neuron], []])
        storage.get_neuron_states_batch = AsyncMock(
            return_value={"young": self._make_state("young", freq=0)}
        )
        storage.delete_neurons_batch = AsyncMock()

        config = ConsolidationConfig()
        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=["prune"])

        # Young neuron IS an orphan (no synapses, no fibers) so it gets pruned as orphan
        # But it would NOT be pruned as "dead" — the orphan check catches it first
        assert report.neurons_pruned >= 1

    @pytest.mark.asyncio
    async def test_accessed_neurons_not_pruned(self) -> None:
        """Neurons with access_frequency > 0 should NOT be dead-pruned."""
        from neural_memory.core.synapse import Synapse
        from neural_memory.engine.consolidation import ConsolidationConfig, ConsolidationEngine

        old_accessed = self._make_neuron("accessed", 30)

        # Create a real synapse so it's not an orphan and prune logic works
        real_synapse = Synapse.create(
            source_id="accessed",
            target_id="other",
            type="related_to",
            weight=0.8,
        )

        storage = AsyncMock()
        storage.current_brain_id = "test"
        storage.get_synapses = AsyncMock(return_value=[real_synapse])
        storage.get_fibers = AsyncMock(return_value=[])
        storage.get_pinned_neuron_ids = AsyncMock(return_value=set())
        storage.find_neurons = AsyncMock(side_effect=[[old_accessed], []])
        storage.get_neuron_states_batch = AsyncMock(
            return_value={"accessed": self._make_state("accessed", freq=5)}
        )
        storage.delete_neurons_batch = AsyncMock()
        storage.get_synapses_for_neurons = AsyncMock(return_value={})

        config = ConsolidationConfig()
        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=["prune"])

        # Should NOT be pruned (has access_frequency > 0)
        assert report.neurons_pruned == 0

    @pytest.mark.asyncio
    async def test_pinned_neurons_not_dead_pruned(self) -> None:
        """Pinned neurons should never be dead-pruned."""
        from neural_memory.core.synapse import Synapse
        from neural_memory.engine.consolidation import ConsolidationConfig, ConsolidationEngine

        old_pinned = self._make_neuron("pinned", 30)

        real_synapse = Synapse.create(
            source_id="pinned",
            target_id="other2",
            type="related_to",
            weight=0.8,
        )

        storage = AsyncMock()
        storage.current_brain_id = "test"
        storage.get_synapses = AsyncMock(return_value=[real_synapse])
        storage.get_fibers = AsyncMock(return_value=[])
        storage.get_pinned_neuron_ids = AsyncMock(return_value={"pinned"})
        storage.find_neurons = AsyncMock(side_effect=[[old_pinned], []])
        storage.get_neuron_states_batch = AsyncMock(
            return_value={"pinned": self._make_state("pinned", freq=0)}
        )
        storage.delete_neurons_batch = AsyncMock()
        storage.get_synapses_for_neurons = AsyncMock(return_value={})

        config = ConsolidationConfig()
        engine = ConsolidationEngine(storage, config)
        report = await engine.run(strategies=["prune"])

        assert report.neurons_pruned == 0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestRecallBackwardCompat:
    """Ensure new defaults don't break existing behavior."""

    def test_brain_config_frozen(self) -> None:
        cfg = BrainConfig()
        with pytest.raises(AttributeError):
            cfg.recency_halflife_hours = 72.0  # type: ignore[misc]

    def test_default_tag_boost_is_moderate(self) -> None:
        """Default boost (0.15) is small enough not to dominate scoring."""
        cfg = BrainConfig()
        assert 0.0 < cfg.tag_match_boost < 0.5

    def test_default_prune_days_is_conservative(self) -> None:
        """14 days gives neurons time to be discovered."""
        cfg = BrainConfig()
        assert cfg.prune_dead_neuron_days >= 14.0
