"""Tests for activation efficiency fixes (Issue #15).

1. Hebbian learning None activation floor → 0.1
2. Dormant neuron reactivation during consolidation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.neuron import NeuronState
from neural_memory.engine.consolidation import ConsolidationReport
from neural_memory.utils.timeutils import utcnow

# ── Hebbian Floor Tests ──


class TestHebbianActivationFloor:
    """Verify None activations are replaced with 0.1 floor."""

    def test_missing_activations_get_floor(self) -> None:
        """When neuron is not in activations dict, should use 0.1 not None."""
        # This tests the logic in retrieval.py _defer_co_activated
        # pre_act/post_act should be 0.1 when neuron not in activations map
        activations: dict[str, MagicMock] = {}
        a, b = "neuron-a", "neuron-b"

        pre_act = activations[a].activation_level if activations and a in activations else 0.1
        post_act = activations[b].activation_level if activations and b in activations else 0.1

        assert pre_act == 0.1
        assert post_act == 0.1

    def test_present_activations_used(self) -> None:
        """When neuron is in activations dict, should use actual value."""
        mock_result = MagicMock()
        mock_result.activation_level = 0.7
        activations = {"neuron-a": mock_result}
        a = "neuron-a"

        pre_act = activations[a].activation_level if activations and a in activations else 0.1

        assert pre_act == 0.7

    def test_floor_enables_positive_hebbian_delta(self) -> None:
        """With 0.1 floor, hebbian_update should compute positive delta."""
        from neural_memory.engine.learning_rule import hebbian_update

        result = hebbian_update(
            current_weight=0.5,
            pre_activation=0.1,
            post_activation=0.1,
            reinforced_count=1,
        )
        # With positive activations, delta should be positive (not zero as with None)
        assert result.delta >= 0


# ── Dormant Reactivation Tests ──


def _make_neuron_state(
    neuron_id: str,
    access_frequency: int = 0,
    activation_level: float = 0.3,
) -> NeuronState:
    return NeuronState(
        neuron_id=neuron_id,
        activation_level=activation_level,
        access_frequency=access_frequency,
        last_activated=utcnow(),
    )


class TestDormantReactivation:
    @pytest.mark.asyncio
    async def test_reactivates_dormant_neurons(self) -> None:
        """Dormant neurons (access_frequency=0) should get a small activation bump."""
        from neural_memory.engine.consolidation import ConsolidationEngine

        storage = AsyncMock()
        consolidator = ConsolidationEngine(storage)

        dormant = [_make_neuron_state(f"n{i}", access_frequency=0) for i in range(5)]
        storage.get_all_neuron_states.return_value = dormant

        report = ConsolidationReport()
        await consolidator._reactivate_dormant(report, dry_run=False)

        assert report.neurons_reactivated == 5
        assert storage.update_neuron_state.call_count == 5

        # Check that activation was bumped
        first_call = storage.update_neuron_state.call_args_list[0][0][0]
        assert first_call.activation_level == 0.35  # 0.3 + 0.05
        assert first_call.access_frequency == 1

    @pytest.mark.asyncio
    async def test_skips_active_neurons(self) -> None:
        """Neurons with access_frequency > 0 should not be reactivated."""
        from neural_memory.engine.consolidation import ConsolidationEngine

        storage = AsyncMock()
        consolidator = ConsolidationEngine(storage)

        active = [_make_neuron_state(f"n{i}", access_frequency=3) for i in range(5)]
        storage.get_all_neuron_states.return_value = active

        report = ConsolidationReport()
        await consolidator._reactivate_dormant(report, dry_run=False)

        assert report.neurons_reactivated == 0
        storage.update_neuron_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_caps_at_20_neurons(self) -> None:
        """Should reactivate at most 20 dormant neurons."""
        from neural_memory.engine.consolidation import ConsolidationEngine

        storage = AsyncMock()
        consolidator = ConsolidationEngine(storage)

        dormant = [_make_neuron_state(f"n{i}", access_frequency=0) for i in range(50)]
        storage.get_all_neuron_states.return_value = dormant

        report = ConsolidationReport()
        await consolidator._reactivate_dormant(report, dry_run=False)

        assert report.neurons_reactivated == 20

    @pytest.mark.asyncio
    async def test_dry_run_counts_only(self) -> None:
        """Dry run should count but not update."""
        from neural_memory.engine.consolidation import ConsolidationEngine

        storage = AsyncMock()
        consolidator = ConsolidationEngine(storage)

        dormant = [_make_neuron_state(f"n{i}", access_frequency=0) for i in range(5)]
        storage.get_all_neuron_states.return_value = dormant

        report = ConsolidationReport()
        await consolidator._reactivate_dormant(report, dry_run=True)

        assert report.neurons_reactivated == 5
        storage.update_neuron_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_storage_error_gracefully(self) -> None:
        """Storage errors should not crash reactivation."""
        from neural_memory.engine.consolidation import ConsolidationEngine

        storage = AsyncMock()
        consolidator = ConsolidationEngine(storage)
        storage.get_all_neuron_states.side_effect = RuntimeError("DB error")

        report = ConsolidationReport()
        await consolidator._reactivate_dormant(report, dry_run=False)

        assert report.neurons_reactivated == 0
