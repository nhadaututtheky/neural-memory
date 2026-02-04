"""Unit tests for Synapse model."""

from __future__ import annotations

from neural_memory.core.synapse import (
    BIDIRECTIONAL_TYPES,
    INVERSE_TYPES,
    Direction,
    Synapse,
    SynapseType,
)


class TestSynapse:
    """Tests for Synapse class."""

    def test_create_synapse(self) -> None:
        """Test creating a synapse with factory method."""
        synapse = Synapse.create(
            source_id="neuron-1",
            target_id="neuron-2",
            type=SynapseType.RELATED_TO,
            weight=0.7,
        )

        assert synapse.source_id == "neuron-1"
        assert synapse.target_id == "neuron-2"
        assert synapse.type == SynapseType.RELATED_TO
        assert synapse.weight == 0.7
        assert synapse.id  # Should have generated ID

    def test_create_with_explicit_id(self) -> None:
        """Test creating synapse with explicit ID."""
        synapse = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.LEADS_TO,
            synapse_id="custom-syn",
        )

        assert synapse.id == "custom-syn"

    def test_weight_clamped(self) -> None:
        """Test that weight is clamped to 0.0-1.0."""
        high = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.RELATED_TO,
            weight=1.5,
        )
        assert high.weight == 1.0

        low = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.RELATED_TO,
            weight=-0.5,
        )
        assert low.weight == 0.0

    def test_auto_detect_bidirectional(self) -> None:
        """Test that bidirectional types are auto-detected."""
        for syn_type in BIDIRECTIONAL_TYPES:
            synapse = Synapse.create(
                source_id="n1",
                target_id="n2",
                type=syn_type,
            )
            assert synapse.direction == Direction.BIDIRECTIONAL

    def test_unidirectional_by_default(self) -> None:
        """Test that non-bidirectional types are unidirectional."""
        synapse = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.CAUSED_BY,
        )
        assert synapse.direction == Direction.UNIDIRECTIONAL

    def test_explicit_direction_overrides(self) -> None:
        """Test that explicit direction overrides auto-detection."""
        synapse = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.CO_OCCURS,  # Normally bidirectional
            direction=Direction.UNIDIRECTIONAL,
        )
        assert synapse.direction == Direction.UNIDIRECTIONAL

    def test_reinforce(self) -> None:
        """Test reinforcing a synapse."""
        synapse = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.RELATED_TO,
            weight=0.5,
        )

        reinforced = synapse.reinforce(delta=0.1)

        # Original unchanged
        assert synapse.weight == 0.5
        assert synapse.reinforced_count == 0

        # New synapse updated
        assert reinforced.weight == 0.6
        assert reinforced.reinforced_count == 1
        assert reinforced.last_activated is not None

    def test_reinforce_capped_at_one(self) -> None:
        """Test that reinforcement is capped at 1.0."""
        synapse = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.RELATED_TO,
            weight=0.95,
        )

        reinforced = synapse.reinforce(delta=0.1)
        assert reinforced.weight == 1.0

    def test_decay(self) -> None:
        """Test decaying a synapse weight."""
        synapse = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.RELATED_TO,
            weight=1.0,
        )

        decayed = synapse.decay(factor=0.9)

        assert decayed.weight == 0.9
        assert synapse.weight == 1.0  # Original unchanged

    def test_get_inverse_type(self) -> None:
        """Test getting inverse synapse type."""
        synapse = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.BEFORE,
        )

        assert synapse.get_inverse_type() == SynapseType.AFTER

    def test_connects(self) -> None:
        """Test connects method."""
        synapse = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.RELATED_TO,
        )

        assert synapse.connects("n1")
        assert synapse.connects("n2")
        assert not synapse.connects("n3")

    def test_other_end(self) -> None:
        """Test other_end method."""
        synapse = Synapse.create(
            source_id="n1",
            target_id="n2",
            type=SynapseType.RELATED_TO,
        )

        assert synapse.other_end("n1") == "n2"
        assert synapse.other_end("n2") == "n1"
        assert synapse.other_end("n3") is None

    def test_all_synapse_types(self) -> None:
        """Test all synapse types can be created."""
        for syn_type in SynapseType:
            synapse = Synapse.create(
                source_id="n1",
                target_id="n2",
                type=syn_type,
            )
            assert synapse.type == syn_type

    def test_inverse_types_are_symmetric(self) -> None:
        """Test that inverse types are symmetric."""
        for type_a, type_b in INVERSE_TYPES.items():
            assert INVERSE_TYPES.get(type_b) == type_a
