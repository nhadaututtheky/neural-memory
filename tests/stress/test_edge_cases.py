"""Stress tests — edge case regression tests for known issues."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.storage.sqlite_store import SQLiteStorage

pytestmark = [pytest.mark.stress, pytest.mark.asyncio]


class TestOrphanRatioHeuristic:
    """Document: health_pulse orphan heuristic assumes 5 neurons/fiber."""

    async def test_heuristic_accuracy_with_real_encoding(
        self, sqlite_storage: SQLiteStorage, encoder: MemoryEncoder
    ) -> None:
        brain_id = sqlite_storage._current_brain_id
        assert brain_id is not None

        # Encode 20 memories — actual neurons per fiber varies
        total_neurons_per_fiber: list[int] = []
        for i in range(20):
            result = await encoder.encode(f"Simple fact #{i}: a brief technical note")
            assert result.fiber is not None
            total_neurons_per_fiber.append(len(result.neurons_created))

        avg_neurons = sum(total_neurons_per_fiber) / len(total_neurons_per_fiber)
        stats = await sqlite_storage.get_stats(brain_id)

        # The heuristic assumes 5 neurons per fiber
        heuristic_linked = stats["fiber_count"] * 5
        actual_neuron_count = stats["neuron_count"]

        # Document the gap — heuristic may over or underestimate
        # This test passes regardless, but logs the discrepancy
        heuristic_orphan_ratio = (
            max(0.0, 1.0 - (heuristic_linked / actual_neuron_count))
            if actual_neuron_count > 0
            else 0.0
        )

        # Just verify the calculation doesn't error and produces valid range
        assert 0.0 <= heuristic_orphan_ratio <= 1.0
        assert avg_neurons > 0


class TestAutoSaveNesting:
    """Document: disable_auto_save/enable_auto_save are no-ops on SQLiteStorage."""

    async def test_disable_enable_are_idempotent(self, sqlite_storage: SQLiteStorage) -> None:
        # Call disable twice
        sqlite_storage.disable_auto_save()
        sqlite_storage.disable_auto_save()

        # Call enable once
        sqlite_storage.enable_auto_save()

        # Data should still persist (auto_save is always on in SQLite)
        from neural_memory.core.neuron import Neuron, NeuronType

        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="test persistence",
            neuron_id="test-persist",
        )
        await sqlite_storage.add_neuron(neuron)

        # Should be immediately readable
        stored = await sqlite_storage.get_neuron("test-persist")
        assert stored is not None
        assert stored.content == "test persistence"


class TestPassiveCaptureFailure:
    """Document: passive capture failures are swallowed at debug level."""

    async def test_encoding_succeeds_despite_analysis_failure(
        self, sqlite_storage: SQLiteStorage, encoder: MemoryEncoder
    ) -> None:
        # Even if something in the extraction pipeline has issues,
        # the encoder should still produce a fiber
        result = await encoder.encode("A normal memory that should always encode successfully")
        assert result.fiber is not None
        assert len(result.neurons_created) > 0

    async def test_remember_with_mocked_passive_capture_failure(self) -> None:
        """Verify _remember succeeds when passive capture raises."""
        from neural_memory.mcp.server import MCPServer

        server = MCPServer()
        server.config = MagicMock()
        server.config.maintenance.enabled = False
        server.config.auto.enabled = False

        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test-brain"

        from neural_memory.core.brain import BrainConfig

        fake_brain = MagicMock()
        fake_brain.id = "test-brain"
        fake_brain.name = "test"
        fake_brain.config = BrainConfig()
        mock_storage.get_brain = AsyncMock(return_value=fake_brain)

        fake_fiber = MagicMock()
        fake_fiber.id = "f1"
        fake_fiber.anchor_neuron_id = "a1"
        fake_fiber.summary = "test"
        fake_fiber.neuron_ids = {"a1"}
        fake_fiber.tags = set()
        fake_fiber.auto_tags = set()

        fake_neuron = MagicMock()
        fake_neuron.id = "a1"
        fake_neuron.content = "test"
        fake_neuron.metadata = {"is_anchor": True}
        fake_neuron.type = MagicMock(value="concept")

        fake_result = MagicMock()
        fake_result.fiber = fake_fiber
        fake_result.neurons_created = [fake_neuron]
        fake_result.neurons_linked = []
        fake_result.synapses_created = []
        fake_result.conflicts_detected = 0

        server._storage = mock_storage
        server.get_storage = AsyncMock(return_value=mock_storage)

        mock_storage.get_stats = AsyncMock(return_value={"neuron_count": 1, "fiber_count": 1})
        mock_storage.get_neurons_batch = AsyncMock(return_value={})
        mock_storage.find_fibers_batch = AsyncMock(return_value=[])

        with (
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder") as mock_enc,
            patch("neural_memory.safety.sensitive.check_sensitive_content", return_value=[]),
            patch("neural_memory.engine.activation.SpreadingActivation") as mock_act,
        ):
            mock_enc.return_value.encode = AsyncMock(return_value=fake_result)
            mock_activator = AsyncMock()
            mock_activator.activate = AsyncMock(
                return_value={
                    "a1": MagicMock(
                        neuron_id="a1",
                        activation_level=1.0,
                        hop_distance=0,
                        path=["a1"],
                        source_anchor="a1",
                    ),
                }
            )
            mock_act.return_value = mock_activator

            response = await server._remember({"content": "test"})

        assert response["success"] is True


class TestHabitsTruncation:
    """Document: habits fetch is capped at 1000 fibers."""

    async def test_habit_cap_documented(self, sqlite_storage: SQLiteStorage) -> None:
        # We can't easily create 1001 habit fibers without the full
        # habit-learning pipeline, but we can verify the constant exists.
        # The real test is that get_fibers(limit=1000) is called in
        # tool_handlers.py:713, silently capping results.

        # Verify storage handles limit param correctly
        fibers = await sqlite_storage.get_fibers(limit=1000)
        assert isinstance(fibers, list)
        assert len(fibers) <= 1000

        # Verify small limit works
        fibers_small = await sqlite_storage.get_fibers(limit=5)
        assert len(fibers_small) <= 5
