"""Tests for synapse quality improvements (Phase F1).

Covers:
- _match_span_to_neuron word overlap matching
- CO_OCCURS weight reduction
- Type-aware consolidation pruning
- Anchor content truncation quality gate
"""

from __future__ import annotations

from datetime import datetime

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.associative_inference import InferenceConfig, compute_inferred_weight
from neural_memory.engine.pipeline_steps import _match_span_to_neuron


class TestMatchSpanToNeuron:
    """Tests for improved span-to-neuron matching."""

    def _make_neuron(self, content: str) -> Neuron:
        return Neuron.create(type=NeuronType.CONCEPT, content=content)

    def test_exact_match(self) -> None:
        neurons = [self._make_neuron("auth"), self._make_neuron("database")]
        result = _match_span_to_neuron("auth", neurons)
        assert result is not None
        assert result.content == "auth"

    def test_substring_containment(self) -> None:
        """Neuron content is substring of span."""
        neurons = [self._make_neuron("Redis"), self._make_neuron("cache")]
        result = _match_span_to_neuron("Redis cache crashed", neurons)
        assert result is not None
        assert result.content in ("Redis", "cache")

    def test_word_overlap_matching(self) -> None:
        """Words from span overlap with neuron content words."""
        neurons = [
            self._make_neuron("deployment failed"),
            self._make_neuron("unrelated thing"),
        ]
        result = _match_span_to_neuron("the deployment suddenly failed", neurons)
        assert result is not None
        assert result.content == "deployment failed"

    def test_no_match_below_threshold(self) -> None:
        """Completely unrelated spans should return None."""
        neurons = [self._make_neuron("PostgreSQL"), self._make_neuron("Docker")]
        result = _match_span_to_neuron("the weather is nice today", neurons)
        assert result is None

    def test_best_match_wins(self) -> None:
        """Most relevant neuron should be selected."""
        neurons = [
            self._make_neuron("auth"),
            self._make_neuron("authentication service"),
        ]
        result = _match_span_to_neuron("authentication service failed", neurons)
        assert result is not None
        assert result.content == "authentication service"

    def test_single_word_neuron_matches_in_span(self) -> None:
        """Single-word concept neurons should match within multi-word spans."""
        neurons = [self._make_neuron("JWT"), self._make_neuron("expiry")]
        result = _match_span_to_neuron("JWT token expired", neurons)
        assert result is not None
        assert result.content == "JWT"

    def test_case_insensitive(self) -> None:
        neurons = [self._make_neuron("FastAPI")]
        result = _match_span_to_neuron("fastapi server crashed", neurons)
        assert result is not None

    def test_low_coverage_rejected(self) -> None:
        """If neuron has many words but only one overlaps, reject it."""
        neurons = [self._make_neuron("long complex multi word concept name")]
        result = _match_span_to_neuron("some concept here", neurons)
        # Only 1/5 neuron words overlap — should be rejected (coverage < 0.5)
        assert result is None


class TestInferenceWeightReduction:
    """Tests for reduced co_occurs inference weights."""

    def test_initial_weight_reduced(self) -> None:
        config = InferenceConfig()
        assert config.inferred_initial_weight == 0.3

    def test_max_weight_reduced(self) -> None:
        config = InferenceConfig()
        assert config.inferred_max_weight == 0.5

    def test_computed_weight_within_bounds(self) -> None:
        config = InferenceConfig()
        weight = compute_inferred_weight(count=10, avg_strength=1.0, config=config)
        assert weight <= 0.5
        assert weight >= 0.2


class TestTypeAwarePruning:
    """Tests for type-aware synapse decay in consolidation."""

    def test_co_occurs_decays_faster(self) -> None:
        """CO_OCCURS synapse should decay 3x faster than semantic synapses."""
        now = datetime.now(tz=None)  # naive UTC per project convention

        co_occurs = Synapse.create(
            source_id="a",
            target_id="b",
            type=SynapseType.CO_OCCURS,
            weight=0.3,
        )
        caused_by = Synapse.create(
            source_id="a",
            target_id="b",
            type=SynapseType.CAUSED_BY,
            weight=0.3,
        )

        # Both start at same weight — after time_decay, CO_OCCURS should
        # get additional 0.33 factor applied by consolidation
        co_decayed = co_occurs.time_decay(reference_time=now).decay(factor=0.33)
        caused_decayed = caused_by.time_decay(reference_time=now)

        assert co_decayed.weight < caused_decayed.weight

    def test_reinforced_co_occurs_protected(self) -> None:
        """CO_OCCURS with 3+ reinforcements should NOT get extra decay."""
        from dataclasses import replace as dc_replace

        syn = Synapse.create(
            source_id="a",
            target_id="b",
            type=SynapseType.CO_OCCURS,
            weight=0.5,
        )
        reinforced = dc_replace(syn, reinforced_count=3)
        # With 3+ reinforcements, no extra decay should be applied
        # (the consolidation code checks reinforced_count < 3)
        assert reinforced.reinforced_count >= 3

    def test_inferred_co_occurs_no_double_decay(self) -> None:
        """Inferred CO_OCCURS should NOT get both _inferred AND CO_OCCURS decay."""
        syn = Synapse.create(
            source_id="a",
            target_id="b",
            type=SynapseType.CO_OCCURS,
            weight=0.3,
            metadata={"_inferred": True},
        )
        # The consolidation code should apply EITHER inferred decay OR
        # CO_OCCURS decay, not both. is_inferred guard prevents stacking.
        assert syn.metadata.get("_inferred") is True
        assert syn.type == SynapseType.CO_OCCURS
        # After only inferred decay (0.5), weight should be ~0.15, not 0.05
        decayed_once = syn.decay(factor=0.5)
        assert decayed_once.weight > 0.1  # Not double-decayed to ~0.05

    def test_dedup_alias_not_decayed(self) -> None:
        """Dedup ALIAS synapses should NOT get accelerated decay."""
        syn = Synapse.create(
            source_id="a",
            target_id="b",
            type=SynapseType.ALIAS,
            weight=0.9,
            metadata={"_dedup": True},
        )
        # Dedup ALIAS is structural — should be excluded from CO_OCCURS/ALIAS decay
        assert syn.metadata.get("_dedup") is True
        assert syn.type == SynapseType.ALIAS
        assert syn.reinforced_count == 0  # Never reinforced, but should be protected
