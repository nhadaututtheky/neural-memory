"""Tests for storage/factory.py — storage creation and HybridStorage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.core.brain_mode import BrainMode, BrainModeConfig

# ─────────── create_storage ───────────


class TestCreateStorage:
    """Tests for create_storage() factory function."""

    @pytest.mark.asyncio
    async def test_local_mode_with_path(self, tmp_path: object) -> None:
        from neural_memory.storage.factory import create_storage

        db_path = str(tmp_path) + "/test.db"  # type: ignore[operator]
        config = BrainModeConfig(mode=BrainMode.LOCAL)
        storage = await create_storage(config, "brain-1", local_path=db_path)

        from neural_memory.storage.sqlite_store import SQLiteStorage

        assert isinstance(storage, SQLiteStorage)
        await storage.close()

    @pytest.mark.asyncio
    async def test_local_mode_without_path_returns_in_memory(self) -> None:
        from neural_memory.storage.factory import create_storage
        from neural_memory.storage.memory_store import InMemoryStorage

        config = BrainModeConfig(mode=BrainMode.LOCAL)
        storage = await create_storage(config, "brain-1")
        assert isinstance(storage, InMemoryStorage)

    @pytest.mark.asyncio
    async def test_shared_mode_without_config_raises(self) -> None:
        from neural_memory.storage.factory import create_storage

        config = BrainModeConfig(mode=BrainMode.SHARED)
        with pytest.raises(ValueError, match="SharedConfig required"):
            await create_storage(config, "brain-1")

    @pytest.mark.asyncio
    async def test_shared_mode_creates_shared_storage(self) -> None:
        from neural_memory.core.brain_mode import SharedConfig
        from neural_memory.storage.factory import create_storage

        config = BrainModeConfig(
            mode=BrainMode.SHARED,
            shared=SharedConfig(server_url="http://localhost:8000"),
        )

        with patch("neural_memory.storage.factory.SharedStorage") as mock_cls:
            mock_instance = AsyncMock()
            mock_cls.return_value = mock_instance
            storage = await create_storage(config, "brain-1")

        mock_instance.connect.assert_awaited_once()
        assert storage is mock_instance

    @pytest.mark.asyncio
    async def test_hybrid_mode_without_config_raises(self) -> None:
        from neural_memory.storage.factory import create_storage

        config = BrainModeConfig(mode=BrainMode.HYBRID)
        with pytest.raises(ValueError, match="HybridConfig required"):
            await create_storage(config, "brain-1")

    @pytest.mark.asyncio
    async def test_hybrid_mode_creates_hybrid_storage(self) -> None:
        from neural_memory.core.brain_mode import HybridConfig
        from neural_memory.storage.factory import create_storage

        config = BrainModeConfig(
            mode=BrainMode.HYBRID,
            hybrid=HybridConfig(
                local_path="./test.db",
                server_url="http://localhost:8000",
            ),
        )

        mock_hybrid = AsyncMock()
        with patch(
            "neural_memory.storage.factory.HybridStorage.create",
            new_callable=AsyncMock,
            return_value=mock_hybrid,
        ):
            storage = await create_storage(config, "brain-1")

        assert storage is mock_hybrid

    @pytest.mark.asyncio
    async def test_unknown_mode_raises(self) -> None:
        from neural_memory.storage.factory import create_storage

        config = MagicMock()
        config.mode = "UNKNOWN_MODE"
        with pytest.raises(ValueError, match="Unknown brain mode"):
            await create_storage(config, "brain-1")


# ─────────── HybridStorage ───────────


class TestHybridStorage:
    """Tests for HybridStorage wrapper."""

    def _make_hybrid(self, *, auto_sync: bool = True) -> tuple[object, AsyncMock, AsyncMock]:
        """Create a HybridStorage with mocked local/remote."""
        from neural_memory.storage.factory import HybridStorage

        local = AsyncMock()
        remote = AsyncMock()
        remote.is_connected = False
        storage = HybridStorage(local=local, remote=remote, auto_sync_on_encode=auto_sync)
        storage._brain_id = "brain-1"
        return storage, local, remote

    def test_set_brain(self) -> None:
        storage, local, remote = self._make_hybrid()
        storage.set_brain("new-brain")  # type: ignore[union-attr]
        local.set_brain.assert_called_with("new-brain")
        remote.set_brain.assert_called_with("new-brain")

    @pytest.mark.asyncio
    async def test_add_neuron_local_and_remote(self) -> None:
        storage, local, remote = self._make_hybrid(auto_sync=True)
        local.add_neuron = AsyncMock(return_value="n-1")
        remote.is_connected = True

        neuron = MagicMock()
        result = await storage.add_neuron(neuron)  # type: ignore[union-attr]

        assert result == "n-1"
        local.add_neuron.assert_awaited_once_with(neuron)
        remote.add_neuron.assert_awaited_once_with(neuron)

    @pytest.mark.asyncio
    async def test_add_neuron_no_sync_when_disabled(self) -> None:
        storage, local, remote = self._make_hybrid(auto_sync=False)
        local.add_neuron = AsyncMock(return_value="n-1")

        neuron = MagicMock()
        await storage.add_neuron(neuron)  # type: ignore[union-attr]

        local.add_neuron.assert_awaited_once()
        remote.add_neuron.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_add_neuron_remote_failure_is_graceful(self) -> None:
        storage, local, remote = self._make_hybrid(auto_sync=True)
        local.add_neuron = AsyncMock(return_value="n-1")
        remote.is_connected = True
        remote.add_neuron = AsyncMock(side_effect=ConnectionError("down"))

        neuron = MagicMock()
        result = await storage.add_neuron(neuron)  # type: ignore[union-attr]

        assert result == "n-1"  # Local still succeeds

    @pytest.mark.asyncio
    async def test_get_neuron_delegates_to_local(self) -> None:
        storage, local, remote = self._make_hybrid()
        local.get_neuron = AsyncMock(return_value=MagicMock())

        result = await storage.get_neuron("n-1")  # type: ignore[union-attr]

        local.get_neuron.assert_awaited_once_with("n-1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_find_neurons_delegates_to_local(self) -> None:
        storage, local, remote = self._make_hybrid()
        local.find_neurons = AsyncMock(return_value=[])

        result = await storage.find_neurons(type="concept")  # type: ignore[union-attr]

        local.find_neurons.assert_awaited_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_update_neuron_syncs(self) -> None:
        storage, local, remote = self._make_hybrid(auto_sync=True)
        remote.is_connected = True
        neuron = MagicMock()

        await storage.update_neuron(neuron)  # type: ignore[union-attr]

        local.update_neuron.assert_awaited_once_with(neuron)
        remote.update_neuron.assert_awaited_once_with(neuron)

    @pytest.mark.asyncio
    async def test_delete_neuron_syncs(self) -> None:
        storage, local, remote = self._make_hybrid(auto_sync=True)
        local.delete_neuron = AsyncMock(return_value=True)
        remote.is_connected = True

        result = await storage.delete_neuron("n-1")  # type: ignore[union-attr]

        assert result is True
        remote.delete_neuron.assert_awaited_once_with("n-1")

    @pytest.mark.asyncio
    async def test_add_synapse_syncs(self) -> None:
        storage, local, remote = self._make_hybrid(auto_sync=True)
        local.add_synapse = AsyncMock(return_value="s-1")
        remote.is_connected = True
        synapse = MagicMock()

        result = await storage.add_synapse(synapse)  # type: ignore[union-attr]

        assert result == "s-1"
        remote.add_synapse.assert_awaited_once_with(synapse)

    @pytest.mark.asyncio
    async def test_delete_synapse_syncs(self) -> None:
        storage, local, remote = self._make_hybrid(auto_sync=True)
        local.delete_synapse = AsyncMock(return_value=True)
        remote.is_connected = True

        result = await storage.delete_synapse("s-1")  # type: ignore[union-attr]

        assert result is True

    @pytest.mark.asyncio
    async def test_add_fiber_syncs(self) -> None:
        storage, local, remote = self._make_hybrid(auto_sync=True)
        local.add_fiber = AsyncMock(return_value="f-1")
        remote.is_connected = True
        fiber = MagicMock()

        result = await storage.add_fiber(fiber)  # type: ignore[union-attr]

        assert result == "f-1"
        remote.add_fiber.assert_awaited_once_with(fiber)

    @pytest.mark.asyncio
    async def test_update_fiber_syncs(self) -> None:
        storage, local, remote = self._make_hybrid(auto_sync=True)
        remote.is_connected = True
        fiber = MagicMock()

        await storage.update_fiber(fiber)  # type: ignore[union-attr]

        local.update_fiber.assert_awaited_once_with(fiber)
        remote.update_fiber.assert_awaited_once_with(fiber)

    @pytest.mark.asyncio
    async def test_delete_fiber_syncs(self) -> None:
        storage, local, remote = self._make_hybrid(auto_sync=True)
        local.delete_fiber = AsyncMock(return_value=True)
        remote.is_connected = True

        result = await storage.delete_fiber("f-1")  # type: ignore[union-attr]

        assert result is True

    @pytest.mark.asyncio
    async def test_read_operations_delegate_to_local(self) -> None:
        """Test that all read-only operations delegate to local."""
        storage, local, remote = self._make_hybrid()

        # get_synapse
        await storage.get_synapse("s-1")  # type: ignore[union-attr]
        local.get_synapse.assert_awaited_once_with("s-1")

        # get_synapses
        await storage.get_synapses(source_id="n-1")  # type: ignore[union-attr]
        local.get_synapses.assert_awaited_once()

        # get_neighbors
        await storage.get_neighbors("n-1")  # type: ignore[union-attr]
        local.get_neighbors.assert_awaited_once()

        # get_path
        await storage.get_path("n-1", "n-2")  # type: ignore[union-attr]
        local.get_path.assert_awaited_once()

        # get_fiber
        await storage.get_fiber("f-1")  # type: ignore[union-attr]
        local.get_fiber.assert_awaited_once()

        # find_fibers
        await storage.find_fibers(tags=["test"])  # type: ignore[union-attr]
        local.find_fibers.assert_awaited_once()

        # get_fibers
        await storage.get_fibers()  # type: ignore[union-attr]
        local.get_fibers.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_brain_operations_delegate_to_local(self) -> None:
        storage, local, remote = self._make_hybrid()

        brain = MagicMock()
        await storage.save_brain(brain)  # type: ignore[union-attr]
        local.save_brain.assert_awaited_once_with(brain)

        await storage.get_brain("brain-1")  # type: ignore[union-attr]
        local.get_brain.assert_awaited_once_with("brain-1")

        await storage.export_brain("brain-1")  # type: ignore[union-attr]
        local.export_brain.assert_awaited_once_with("brain-1")

        snapshot = MagicMock()
        await storage.import_brain(snapshot, "brain-1")  # type: ignore[union-attr]
        local.import_brain.assert_awaited_once_with(snapshot, "brain-1")

        await storage.get_stats("brain-1")  # type: ignore[union-attr]
        local.get_stats.assert_awaited_once_with("brain-1")

        await storage.get_enhanced_stats("brain-1")  # type: ignore[union-attr]
        local.get_enhanced_stats.assert_awaited_once_with("brain-1")

        await storage.clear("brain-1")  # type: ignore[union-attr]
        local.clear.assert_awaited_once_with("brain-1")

    @pytest.mark.asyncio
    async def test_suggest_neurons_delegates(self) -> None:
        storage, local, remote = self._make_hybrid()
        local.suggest_neurons = AsyncMock(return_value=[])

        result = await storage.suggest_neurons("test")  # type: ignore[union-attr]

        local.suggest_neurons.assert_awaited_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_neuron_state_delegates(self) -> None:
        storage, local, remote = self._make_hybrid()

        await storage.get_neuron_state("n-1")  # type: ignore[union-attr]
        local.get_neuron_state.assert_awaited_once_with("n-1")

        state = MagicMock()
        await storage.update_neuron_state(state)  # type: ignore[union-attr]
        local.update_neuron_state.assert_awaited_once_with(state)

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        storage, local, remote = self._make_hybrid()

        await storage.close()  # type: ignore[union-attr]

        local.close.assert_awaited_once()
        remote.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ensure_connected(self) -> None:
        storage, local, remote = self._make_hybrid()
        remote.is_connected = False

        await storage._ensure_connected()  # type: ignore[union-attr]

        remote.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ensure_connected_noop_if_connected(self) -> None:
        storage, local, remote = self._make_hybrid()
        remote.is_connected = True

        await storage._ensure_connected()  # type: ignore[union-attr]

        remote.connect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sync_pushes_when_no_remote_brain(self) -> None:
        storage, local, remote = self._make_hybrid()
        remote.is_connected = True

        local_snapshot = MagicMock()
        local.export_brain = AsyncMock(return_value=local_snapshot)
        remote.export_brain = AsyncMock(side_effect=Exception("Not found"))

        result = await storage.sync()  # type: ignore[union-attr]

        assert result["pushed"] is True
        assert result["pulled"] is False
        remote.import_brain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sync_no_brain_raises(self) -> None:
        storage, local, remote = self._make_hybrid()
        remote.is_connected = True
        storage._brain_id = None  # type: ignore[union-attr]

        with pytest.raises(ValueError, match="No brain set"):
            await storage.sync()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_sync_merge(self) -> None:
        storage, local, remote = self._make_hybrid()
        remote.is_connected = True

        local_snapshot = MagicMock()
        remote_snapshot = MagicMock()
        local.export_brain = AsyncMock(return_value=local_snapshot)
        remote.export_brain = AsyncMock(return_value=remote_snapshot)

        mock_report = MagicMock()
        mock_report.neurons_added = 1
        mock_report.neurons_updated = 0
        mock_report.neurons_skipped = 0
        mock_report.synapses_added = 2
        mock_report.synapses_updated = 0
        mock_report.fibers_added = 1
        mock_report.conflicts = []

        merged = MagicMock()

        with patch(
            "neural_memory.engine.merge.merge_snapshots",
            return_value=(merged, mock_report),
        ):
            result = await storage.sync()  # type: ignore[union-attr]

        assert result["pushed"] is True
        assert result["pulled"] is True
        assert result["merge_report"]["neurons_added"] == 1
        assert result["merge_report"]["synapses_added"] == 2


class TestHybridStorageCreate:
    """Tests for HybridStorage.create() classmethod."""

    @pytest.mark.asyncio
    async def test_create_initializes_local(self, tmp_path: object) -> None:
        from neural_memory.storage.factory import HybridStorage

        db_path = str(tmp_path) + "/hybrid.db"  # type: ignore[operator]

        with patch("neural_memory.storage.factory.SharedStorage"):
            storage = await HybridStorage.create(
                local_path=db_path,
                server_url="http://localhost:8000",
                brain_id="brain-1",
            )

        assert storage._brain_id == "brain-1"
        assert storage._local is not None
        await storage._local.close()
