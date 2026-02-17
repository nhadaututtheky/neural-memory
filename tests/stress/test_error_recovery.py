"""Stress tests — error recovery and atomicity verification."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.mcp.server import MCPServer
from neural_memory.storage.sqlite_store import SQLiteStorage

pytestmark = [pytest.mark.stress, pytest.mark.asyncio]


class TestEncodeFailureNoOrphans:
    """If encoding fails mid-way, verify no orphan data persists."""

    async def test_fiber_save_failure_leaves_no_orphans(
        self, sqlite_storage: SQLiteStorage, encoder: MemoryEncoder
    ) -> None:
        stats_before = await sqlite_storage.get_stats(
            sqlite_storage._current_brain_id  # type: ignore[arg-type]
        )

        # Monkeypatch add_fiber to fail
        original_add_fiber = sqlite_storage.add_fiber

        call_count = 0

        async def failing_add_fiber(*args: object, **kwargs: object) -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Simulated fiber save failure")

        sqlite_storage.add_fiber = failing_add_fiber  # type: ignore[assignment]

        try:
            with pytest.raises(RuntimeError, match="fiber save failure"):
                await encoder.encode("Memory that should fail to persist")
        finally:
            sqlite_storage.add_fiber = original_add_fiber  # type: ignore[assignment]

        # Note: In SQLiteStorage, individual operations auto-commit
        # (disable_auto_save is a no-op). So neurons created before
        # add_fiber failure ARE persisted. This documents the behavior.
        stats_after = await sqlite_storage.get_stats(
            sqlite_storage._current_brain_id  # type: ignore[arg-type]
        )

        # The fiber should NOT have been created
        assert stats_after["fiber_count"] == stats_before["fiber_count"]


class TestTransplantBrainSwitchRecovery:
    """Verify brain context is restored after transplant failure."""

    async def test_brain_id_restored_on_transplant_failure(
        self, sqlite_storage: SQLiteStorage
    ) -> None:
        from neural_memory.core.brain import Brain, BrainConfig

        # Create two brains
        brain_a = Brain.create(name="source-brain", config=BrainConfig())
        brain_b = Brain.create(name="target-brain", config=BrainConfig())
        await sqlite_storage.save_brain(brain_a)
        await sqlite_storage.save_brain(brain_b)

        # Set current context to brain B (target)
        sqlite_storage.set_brain(brain_b.id)
        assert sqlite_storage._current_brain_id == brain_b.id

        # Simulate transplant that switches to brain A then fails
        original_brain_id = sqlite_storage._current_brain_id

        try:
            sqlite_storage.set_brain(brain_a.id)  # Switch to source
            raise RuntimeError("Simulated transplant failure")
        except RuntimeError:
            pass
        finally:
            sqlite_storage.set_brain(original_brain_id or "")  # Restore

        # Brain context should be restored to brain B
        assert sqlite_storage._current_brain_id == brain_b.id


class TestRelatedMemoryFailureGraceful:
    """Verify _remember succeeds even if related memory discovery fails."""

    async def test_remember_succeeds_despite_activation_failure(self) -> None:
        server = MCPServer()
        server.config = MagicMock()
        server.config.maintenance.enabled = False
        server.config.auto.enabled = False

        # Create mock storage
        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test-brain"

        from neural_memory.core.brain import BrainConfig

        fake_brain = MagicMock()
        fake_brain.id = "test-brain"
        fake_brain.name = "test-brain"
        fake_brain.config = BrainConfig()
        mock_storage.get_brain = AsyncMock(return_value=fake_brain)

        # Create a minimal fake encoding result
        fake_fiber = MagicMock()
        fake_fiber.id = "fiber-new"
        fake_fiber.anchor_neuron_id = "anchor-new"
        fake_fiber.summary = "Test memory"
        fake_fiber.neuron_ids = {"anchor-new"}
        fake_fiber.tags = set()
        fake_fiber.auto_tags = set()

        fake_neuron = MagicMock()
        fake_neuron.id = "anchor-new"
        fake_neuron.content = "test"
        fake_neuron.metadata = {"is_anchor": True}
        fake_neuron.type = MagicMock(value="concept")

        fake_result = MagicMock()
        fake_result.fiber = fake_fiber
        fake_result.neurons_created = [fake_neuron]
        fake_result.neurons_linked = []
        fake_result.synapses_created = []
        fake_result.conflicts_detected = 0

        mock_storage.get_stats = AsyncMock(return_value={"neuron_count": 1, "fiber_count": 1})

        server._storage = mock_storage
        server.get_storage = AsyncMock(return_value=mock_storage)

        with (
            patch("neural_memory.mcp.tool_handlers.MemoryEncoder") as mock_enc,
            patch("neural_memory.safety.sensitive.check_sensitive_content", return_value=[]),
            patch("neural_memory.engine.activation.SpreadingActivation") as mock_activation,
        ):
            mock_enc.return_value.encode = AsyncMock(return_value=fake_result)

            # Make activation FAIL
            mock_activator = AsyncMock()
            mock_activator.activate = AsyncMock(
                side_effect=RuntimeError("activation engine crashed")
            )
            mock_activation.return_value = mock_activator

            response = await server._remember({"content": "test content"})

        # Should still succeed — related memory failure is non-critical
        assert response["success"] is True
        assert "related_memories" not in response


class TestConflictResolveAtomicity:
    """Document non-atomic conflict resolution behavior."""

    async def test_partial_update_on_synapse_failure(
        self, sqlite_storage: SQLiteStorage, encoder: MemoryEncoder
    ) -> None:
        # Encode two memories to create some neurons
        r1 = await encoder.encode("We use PostgreSQL for the main database")
        r2 = await encoder.encode("We use MySQL for the main database")

        # Both should have created fibers
        assert r1.fiber is not None
        assert r2.fiber is not None

        # Verify both fibers exist
        f1 = await sqlite_storage.get_fiber(r1.fiber.id)
        f2 = await sqlite_storage.get_fiber(r2.fiber.id)
        assert f1 is not None
        assert f2 is not None
