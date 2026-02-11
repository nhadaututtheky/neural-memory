"""Storage factory for creating storage based on configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.core.brain_mode import BrainMode, BrainModeConfig
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.storage.shared_store import SharedStorage
from neural_memory.storage.sqlite_store import SQLiteStorage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neural_memory.core.brain import Brain, BrainSnapshot
    from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
    from neural_memory.core.synapse import Synapse
    from neural_memory.storage.base import NeuralStorage


async def create_storage(
    config: BrainModeConfig,
    brain_id: str,
    *,
    local_path: str | None = None,
) -> NeuralStorage:
    """
    Create a storage instance based on configuration.

    Args:
        config: Brain mode configuration
        brain_id: ID of the brain to connect to
        local_path: Path for local SQLite storage (used in LOCAL mode)

    Returns:
        Configured storage instance

    Examples:
        # Local mode with SQLite
        config = BrainModeConfig.local()
        storage = await create_storage(config, "brain-1", local_path="./brain.db")

        # Shared mode
        config = BrainModeConfig.shared_mode("http://localhost:8000")
        storage = await create_storage(config, "brain-1")

        # Hybrid mode
        config = BrainModeConfig.hybrid_mode("./local.db", "http://localhost:8000")
        storage = await create_storage(config, "brain-1")
    """
    if config.mode == BrainMode.LOCAL:
        if local_path:
            local_storage = SQLiteStorage(local_path)
            await local_storage.initialize()
            local_storage.set_brain(brain_id)
            return local_storage
        else:
            mem_storage = InMemoryStorage()
            mem_storage.set_brain(brain_id)
            return mem_storage

    elif config.mode == BrainMode.SHARED:
        if not config.shared:
            raise ValueError("SharedConfig required for SHARED mode")

        shared_storage = SharedStorage(
            server_url=config.shared.server_url,
            brain_id=brain_id,
            timeout=config.shared.timeout,
            api_key=config.shared.api_key,
        )
        await shared_storage.connect()
        return shared_storage

    elif config.mode == BrainMode.HYBRID:
        if not config.hybrid:
            raise ValueError("HybridConfig required for HYBRID mode")

        # For hybrid mode, return a HybridStorage that wraps both local and remote
        hybrid_storage = await HybridStorage.create(
            local_path=config.hybrid.local_path,
            server_url=config.hybrid.server_url,
            brain_id=brain_id,
            api_key=config.hybrid.api_key,
            sync_strategy=config.hybrid.sync_strategy,
            auto_sync_on_encode=config.hybrid.auto_sync_on_encode,
        )
        return hybrid_storage  # type: ignore[return-value]

    else:
        raise ValueError(f"Unknown brain mode: {config.mode}")


class HybridStorage:
    """
    Hybrid storage that combines local SQLite with remote sync.

    Provides offline-first capability with optional sync to server.
    """

    def __init__(
        self,
        local: SQLiteStorage,
        remote: SharedStorage,
        *,
        auto_sync_on_encode: bool = True,
    ) -> None:
        self._local = local
        self._remote = remote
        self._auto_sync = auto_sync_on_encode
        self._brain_id: str | None = None

    @classmethod
    async def create(
        cls,
        local_path: str,
        server_url: str,
        brain_id: str,
        *,
        api_key: str | None = None,
        sync_strategy: str = "bidirectional",
        auto_sync_on_encode: bool = True,
    ) -> HybridStorage:
        """Create and initialize hybrid storage."""
        local = SQLiteStorage(local_path)
        await local.initialize()
        local.set_brain(brain_id)

        remote = SharedStorage(
            server_url=server_url,
            brain_id=brain_id,
            api_key=api_key,
        )
        # Don't connect remote immediately - connect on demand

        storage = cls(
            local=local,
            remote=remote,
            auto_sync_on_encode=auto_sync_on_encode,
        )
        storage._brain_id = brain_id
        return storage

    def set_brain(self, brain_id: str) -> None:
        """Set the current brain context."""
        self._brain_id = brain_id
        self._local.set_brain(brain_id)
        self._remote.set_brain(brain_id)

    # Delegate all NeuralStorage methods to local storage
    # Sync to remote when appropriate

    async def add_neuron(self, neuron: Neuron) -> str:
        """Add neuron locally, optionally sync."""
        result = await self._local.add_neuron(neuron)
        if self._auto_sync:
            try:
                await self._ensure_connected()
                await self._remote.add_neuron(neuron)
            except (ConnectionError, OSError) as e:
                logger.debug("Remote sync failed for add_neuron: %s", e)
        return result

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        """Get neuron from local storage."""
        return await self._local.get_neuron(neuron_id)

    async def find_neurons(self, **kwargs: Any) -> list[Neuron]:
        """Find neurons in local storage."""
        return await self._local.find_neurons(**kwargs)

    async def update_neuron(self, neuron: Neuron) -> None:
        """Update neuron locally, optionally sync."""
        await self._local.update_neuron(neuron)
        if self._auto_sync:
            try:
                await self._ensure_connected()
                await self._remote.update_neuron(neuron)
            except (ConnectionError, OSError) as e:
                logger.debug("Remote sync failed for update_neuron: %s", e)

    async def delete_neuron(self, neuron_id: str) -> bool:
        """Delete neuron locally, optionally sync."""
        result = await self._local.delete_neuron(neuron_id)
        if self._auto_sync:
            try:
                await self._ensure_connected()
                await self._remote.delete_neuron(neuron_id)
            except (ConnectionError, OSError) as e:
                logger.debug("Remote sync failed for delete_neuron: %s", e)
        return result

    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: NeuronType | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Suggest neurons from local storage."""
        return await self._local.suggest_neurons(prefix, type_filter=type_filter, limit=limit)

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        return await self._local.get_neuron_state(neuron_id)

    async def update_neuron_state(self, state: NeuronState) -> None:
        await self._local.update_neuron_state(state)

    async def add_synapse(self, synapse: Synapse) -> str:
        result = await self._local.add_synapse(synapse)
        if self._auto_sync:
            try:
                await self._ensure_connected()
                await self._remote.add_synapse(synapse)
            except (ConnectionError, OSError) as e:
                logger.debug("Remote sync failed for add_synapse: %s", e)
        return result

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        return await self._local.get_synapse(synapse_id)

    async def get_synapses(self, **kwargs: Any) -> list[Synapse]:
        return await self._local.get_synapses(**kwargs)

    async def update_synapse(self, synapse: Synapse) -> None:
        await self._local.update_synapse(synapse)
        if self._auto_sync:
            try:
                await self._ensure_connected()
                await self._remote.update_synapse(synapse)
            except (ConnectionError, OSError) as e:
                logger.debug("Remote sync failed for update_synapse: %s", e)

    async def delete_synapse(self, synapse_id: str) -> bool:
        result = await self._local.delete_synapse(synapse_id)
        if self._auto_sync:
            try:
                await self._ensure_connected()
                await self._remote.delete_synapse(synapse_id)
            except (ConnectionError, OSError) as e:
                logger.debug("Remote sync failed for delete_synapse: %s", e)
        return result

    async def get_neighbors(self, neuron_id: str, **kwargs: Any) -> Any:
        return await self._local.get_neighbors(neuron_id, **kwargs)

    async def get_path(self, source_id: str, target_id: str, max_hops: int = 4) -> Any:
        return await self._local.get_path(source_id, target_id, max_hops)

    async def add_fiber(self, fiber: Any) -> str:
        result = await self._local.add_fiber(fiber)
        if self._auto_sync:
            try:
                await self._ensure_connected()
                await self._remote.add_fiber(fiber)
            except (ConnectionError, OSError) as e:
                logger.debug("Remote sync failed for add_fiber: %s", e)
        return result

    async def get_fiber(self, fiber_id: str) -> Any:
        return await self._local.get_fiber(fiber_id)

    async def find_fibers(self, **kwargs: Any) -> Any:
        return await self._local.find_fibers(**kwargs)

    async def update_fiber(self, fiber: Any) -> None:
        await self._local.update_fiber(fiber)
        if self._auto_sync:
            try:
                await self._ensure_connected()
                await self._remote.update_fiber(fiber)
            except (ConnectionError, OSError) as e:
                logger.debug("Remote sync failed for update_fiber: %s", e)

    async def delete_fiber(self, fiber_id: str) -> bool:
        result = await self._local.delete_fiber(fiber_id)
        if self._auto_sync:
            try:
                await self._ensure_connected()
                await self._remote.delete_fiber(fiber_id)
            except (ConnectionError, OSError) as e:
                logger.debug("Remote sync failed for delete_fiber: %s", e)
        return result

    async def get_fibers(self, **kwargs: Any) -> Any:
        return await self._local.get_fibers(**kwargs)

    async def save_brain(self, brain: Brain) -> None:
        await self._local.save_brain(brain)

    async def get_brain(self, brain_id: str) -> Brain | None:
        return await self._local.get_brain(brain_id)

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        return await self._local.export_brain(brain_id)

    async def import_brain(
        self, snapshot: BrainSnapshot, target_brain_id: str | None = None
    ) -> str:
        return await self._local.import_brain(snapshot, target_brain_id)

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        return await self._local.get_stats(brain_id)

    async def get_enhanced_stats(self, brain_id: str) -> dict[str, Any]:
        return await self._local.get_enhanced_stats(brain_id)

    async def clear(self, brain_id: str) -> None:
        await self._local.clear(brain_id)

    # Sync operations

    async def sync(
        self,
        strategy: str = "prefer_local",
    ) -> dict[str, Any]:
        """
        Manually trigger a full sync with remote server.

        Args:
            strategy: Conflict resolution strategy
                (prefer_local, prefer_remote, prefer_recent, prefer_stronger)

        Returns:
            Sync statistics including merge report
        """
        from neural_memory.engine.merge import ConflictStrategy, merge_snapshots

        await self._ensure_connected()

        if not self._brain_id:
            raise ValueError("No brain set")

        local_snapshot = await self._local.export_brain(self._brain_id)

        # Get remote snapshot
        try:
            remote_snapshot = await self._remote.export_brain(self._brain_id)
        except Exception:
            logger.debug("Remote brain not found, pushing local version", exc_info=True)
            await self._remote.import_brain(local_snapshot, self._brain_id)
            return {"pushed": True, "pulled": False, "merge_report": None}

        # Merge snapshots
        conflict_strategy = ConflictStrategy(strategy)
        merged_snapshot, merge_report = merge_snapshots(
            local=local_snapshot,
            incoming=remote_snapshot,
            strategy=conflict_strategy,
        )

        # Clear local and reimport merged
        await self._local.clear(self._brain_id)
        await self._local.import_brain(merged_snapshot, self._brain_id)
        self._local.set_brain(self._brain_id)

        # Push merged to remote
        await self._remote.import_brain(merged_snapshot, self._brain_id)

        return {
            "pushed": True,
            "pulled": True,
            "merge_report": {
                "neurons_added": merge_report.neurons_added,
                "neurons_updated": merge_report.neurons_updated,
                "neurons_skipped": merge_report.neurons_skipped,
                "synapses_added": merge_report.synapses_added,
                "synapses_updated": merge_report.synapses_updated,
                "fibers_added": merge_report.fibers_added,
                "conflicts": len(merge_report.conflicts),
            },
        }

    async def _ensure_connected(self) -> None:
        """Ensure remote storage is connected."""
        if not self._remote.is_connected:
            await self._remote.connect()

    async def close(self) -> None:
        """Close all connections."""
        await self._local.close()
        await self._remote.disconnect()
