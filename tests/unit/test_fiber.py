"""Unit tests for Fiber data structure."""

from __future__ import annotations

from datetime import datetime

import pytest

from neural_memory.core.fiber import Fiber


class TestFiber:
    """Tests for Fiber class."""

    def test_create_fiber(self) -> None:
        """Test creating a new fiber."""
        fiber = Fiber.create(
            neuron_ids={"n1", "n2", "n3"},
            synapse_ids={"s1", "s2"},
            anchor_neuron_id="n1",
        )

        assert fiber.anchor_neuron_id == "n1"
        assert "n1" in fiber.neuron_ids
        assert "n2" in fiber.neuron_ids
        assert "n3" in fiber.neuron_ids
        assert fiber.conductivity == 1.0
        assert fiber.pathway == ["n1"]  # Default pathway starts with anchor

    def test_create_fiber_with_pathway(self) -> None:
        """Test creating a fiber with explicit pathway."""
        fiber = Fiber.create(
            neuron_ids={"n1", "n2", "n3"},
            synapse_ids={"s1", "s2"},
            anchor_neuron_id="n1",
            pathway=["n1", "n2", "n3"],
        )

        assert fiber.pathway == ["n1", "n2", "n3"]
        assert fiber.pathway_length == 3

    def test_create_fiber_anchor_not_in_neurons_raises(self) -> None:
        """Test that creating a fiber with invalid anchor raises error."""
        with pytest.raises(ValueError, match="must be in neuron_ids"):
            Fiber.create(
                neuron_ids={"n1", "n2"},
                synapse_ids={"s1"},
                anchor_neuron_id="n3",  # Not in neuron_ids
            )


class TestFiberPathway:
    """Tests for Fiber pathway functionality."""

    @pytest.fixture
    def fiber(self) -> Fiber:
        """Create a test fiber with pathway."""
        return Fiber.create(
            neuron_ids={"n1", "n2", "n3", "n4"},
            synapse_ids={"s1", "s2", "s3"},
            anchor_neuron_id="n1",
            pathway=["n1", "n2", "n3", "n4"],
        )

    def test_pathway_length(self, fiber: Fiber) -> None:
        """Test pathway_length property."""
        assert fiber.pathway_length == 4

    def test_pathway_position_found(self, fiber: Fiber) -> None:
        """Test finding position of neuron in pathway."""
        assert fiber.pathway_position("n1") == 0
        assert fiber.pathway_position("n2") == 1
        assert fiber.pathway_position("n3") == 2
        assert fiber.pathway_position("n4") == 3

    def test_pathway_position_not_found(self, fiber: Fiber) -> None:
        """Test pathway_position returns None for missing neuron."""
        assert fiber.pathway_position("n5") is None

    def test_is_in_pathway(self, fiber: Fiber) -> None:
        """Test is_in_pathway method."""
        assert fiber.is_in_pathway("n1") is True
        assert fiber.is_in_pathway("n2") is True
        assert fiber.is_in_pathway("n5") is False


class TestFiberConductivity:
    """Tests for Fiber conductivity functionality."""

    @pytest.fixture
    def fiber(self) -> Fiber:
        """Create a test fiber."""
        return Fiber.create(
            neuron_ids={"n1", "n2"},
            synapse_ids={"s1"},
            anchor_neuron_id="n1",
            pathway=["n1", "n2"],
        )

    def test_default_conductivity(self, fiber: Fiber) -> None:
        """Test default conductivity is 1.0."""
        assert fiber.conductivity == 1.0

    def test_with_conductivity(self, fiber: Fiber) -> None:
        """Test with_conductivity creates new fiber with updated conductivity."""
        new_fiber = fiber.with_conductivity(0.5)

        # Original unchanged
        assert fiber.conductivity == 1.0

        # New fiber has updated conductivity
        assert new_fiber.conductivity == 0.5
        assert new_fiber.id == fiber.id

    def test_with_conductivity_clamped(self, fiber: Fiber) -> None:
        """Test conductivity is clamped to 0.0-1.0."""
        high = fiber.with_conductivity(1.5)
        assert high.conductivity == 1.0

        low = fiber.with_conductivity(-0.5)
        assert low.conductivity == 0.0


class TestFiberConduct:
    """Tests for Fiber conduct method."""

    @pytest.fixture
    def fiber(self) -> Fiber:
        """Create a test fiber."""
        return Fiber.create(
            neuron_ids={"n1", "n2"},
            synapse_ids={"s1"},
            anchor_neuron_id="n1",
        )

    def test_conduct_updates_last_conducted(self, fiber: Fiber) -> None:
        """Test conduct updates last_conducted timestamp."""
        assert fiber.last_conducted is None

        now = datetime.utcnow()
        conducted = fiber.conduct(conducted_at=now)

        assert conducted.last_conducted == now
        assert fiber.last_conducted is None  # Original unchanged

    def test_conduct_increments_frequency(self, fiber: Fiber) -> None:
        """Test conduct increments frequency."""
        assert fiber.frequency == 0

        conducted = fiber.conduct()
        assert conducted.frequency == 1

        conducted_again = conducted.conduct()
        assert conducted_again.frequency == 2

    def test_conduct_reinforces_conductivity(self, fiber: Fiber) -> None:
        """Test conduct with reinforce=True increases conductivity."""
        # Start with lower conductivity
        low_fiber = fiber.with_conductivity(0.5)

        conducted = low_fiber.conduct(reinforce=True)

        assert conducted.conductivity > 0.5
        assert conducted.conductivity == 0.52  # +0.02 reinforcement

    def test_conduct_without_reinforce(self, fiber: Fiber) -> None:
        """Test conduct without reinforcement."""
        low_fiber = fiber.with_conductivity(0.5)

        conducted = low_fiber.conduct(reinforce=False)

        assert conducted.conductivity == 0.5  # Unchanged

    def test_conduct_conductivity_capped_at_one(self, fiber: Fiber) -> None:
        """Test conductivity doesn't exceed 1.0 with reinforcement."""
        high_fiber = fiber.with_conductivity(0.99)

        conducted = high_fiber.conduct(reinforce=True)

        assert conducted.conductivity == 1.0  # Capped at 1.0


class TestFiberImmutability:
    """Tests verifying Fiber immutability patterns."""

    @pytest.fixture
    def fiber(self) -> Fiber:
        """Create a test fiber."""
        return Fiber.create(
            neuron_ids={"n1", "n2"},
            synapse_ids={"s1"},
            anchor_neuron_id="n1",
            pathway=["n1", "n2"],
        )

    def test_access_returns_new_fiber(self, fiber: Fiber) -> None:
        """Test access returns new fiber without modifying original."""
        original_freq = fiber.frequency

        accessed = fiber.access()

        assert fiber.frequency == original_freq
        assert accessed.frequency == original_freq + 1
        assert accessed.id == fiber.id

    def test_with_salience_returns_new_fiber(self, fiber: Fiber) -> None:
        """Test with_salience returns new fiber."""
        original_salience = fiber.salience

        updated = fiber.with_salience(0.8)

        assert fiber.salience == original_salience
        assert updated.salience == 0.8

    def test_add_tags_returns_new_fiber(self, fiber: Fiber) -> None:
        """Test add_tags returns new fiber."""
        original_tags = fiber.tags.copy()

        updated = fiber.add_tags("important", "reviewed")

        assert fiber.tags == original_tags
        assert "important" in updated.tags
        assert "reviewed" in updated.tags
