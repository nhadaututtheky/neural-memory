"""Storage factory for creating storage based on configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.core.brain_mode import BrainMode, BrainModeConfig
from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.storage.shared_store import SharedStorage

# Unified SQL adapter (new)
from neural_memory.storage.sql import SQLStorage
from neural_memory.storage.sql.postgres_dialect import PostgresDialect
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.storage.sqlite_store import SQLiteStorage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neural_memory.core.brain import Brain, BrainSnapshot
    from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
    from neural_memory.core.synapse import Synapse


async def create_storage(
    config: BrainModeConfig,
    brain_id: str,
    *,
    local_path: str | None = None,
    backend: str | None = None,
) -> NeuralStorage:
    """
    Create a storage instance based on configuration.

    Args:
        config: Brain mode configuration
        brain_id: ID of the brain to connect to
        local_path: Path for local SQLite storage (used in LOCAL mode)
        backend: Storage backend to use. ``"unified"`` selects the new
            :class:`SQLStorage` adapter; ``"postgres"`` creates a unified
            ``SQLStorage(PostgresDialect(...))``. ``None`` (default) keeps
            the original ``SQLiteStorage`` / ``PostgreSQLStorage`` for full
            backward compatibility.

    Returns:
        Configured storage instance

    Examples:
        # Local mode with SQLite (legacy default)
        config = BrainModeConfig.local()
        storage = await create_storage(config, "brain-1", local_path="./brain.db")

        # Local mode with unified SQLStorage adapter
        storage = await create_storage(
            config, "brain-1", local_path="./brain.db", backend="unified"
        )

        # Shared mode
        config = BrainModeConfig.shared_mode("http://localhost:8000")
        storage = await create_storage(config, "brain-1")

        # Hybrid mode
        config = BrainModeConfig.hybrid_mode("./local.db", "http://localhost:8000")
        storage = await create_storage(config, "brain-1")
    """
    if config.mode == BrainMode.LOCAL:
        if local_path:
            # --- Unified SQL adapter paths ---
            if backend == "unified":
                dialect = SQLiteDialect(db_path=local_path)
                unified = SQLStorage(dialect)
                await unified.initialize()
                unified.set_brain(brain_id)
                logger.info("Using unified SQLStorage with SQLiteDialect")
                return unified

            if backend == "postgres":
                # Expects local_path to be unused; connection params come
                # from environment or a future config extension.  For now
                # provide sensible defaults that can be overridden.
                pg_dialect = PostgresDialect()
                unified = SQLStorage(pg_dialect)
                await unified.initialize()
                unified.set_brain(brain_id)
                logger.info("Using unified SQLStorage with PostgresDialect")
                return unified

            # --- Legacy path (default) ---
            # Check if Pro plugin provides an alternative storage engine
            pro_storage = await _try_pro_storage(local_path, brain_id)
            if pro_storage is not None:
                return pro_storage

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
        return hybrid_storage

    else:
        raise ValueError(f"Unknown brain mode: {config.mode}")


async def _try_pro_storage(local_path: str, brain_id: str) -> NeuralStorage | None:
    """Try to create Pro storage if plugin provides one.

    Returns None if Pro is not installed or storage creation fails.
    Falls back silently to let the caller use SQLite.
    """
    try:
        from neural_memory.plugins import get_storage_class

        storage_cls = get_storage_class()
        if storage_cls is None:
            return None

        from pathlib import Path

        base_dir = Path(local_path).parent
        storage = storage_cls(base_dir, brain_id=brain_id)
        await storage.open()
        logger.info("Using Pro storage engine: %s", storage_cls.__name__)
        return storage  # type: ignore[no-any-return]
    except Exception:
        logger.debug("Pro storage not available, falling back to SQLite", exc_info=True)
        return None


class HybridStorage(NeuralStorage):
    """
    Hybrid storage that combines local SQLite with remote sync.

    Provides offline-first capability with optional sync to server.

    Subclasses :class:`NeuralStorage` so every abstract core method is
    accounted for at instantiation (the explicit overrides below delegate to
    the local SQLite backend). Batch / extended methods that are *not*
    overridden here are forwarded to ``self._local`` via :meth:`__getattr__`,
    so the previously-missing surface (``get_neurons_batch``, typed memory,
    change log, devices, merkle, alerts, sources, …) no longer raises
    ``AttributeError`` (closes #30).
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

    @property
    def brain_id(self) -> str | None:
        """Current brain context (HybridStorage tracks it on ``_brain_id``)."""
        return self._brain_id

    @property
    def current_brain_id(self) -> str | None:
        """Alias for :attr:`brain_id` (backward compatibility)."""
        return self._brain_id

    def __getattr__(self, name: str) -> Any:
        """Delegate any non-overridden storage method to the local backend.

        ``__getattr__`` is only consulted for attributes not found through the
        normal lookup, so the explicit overrides above always take precedence.
        Dunder / private names are excluded to avoid masking real
        AttributeErrors (e.g. during pickling or copy).
        """
        if name.startswith("_"):
            raise AttributeError(name)
        # ``self._local`` is set in __init__; if it isn't yet, fall through to
        # a normal AttributeError rather than recursing.
        local = self.__dict__.get("_local")
        if local is None:
            raise AttributeError(name)
        return getattr(local, name)

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

    async def get_path(
        self, source_id: str, target_id: str, max_hops: int = 4, bidirectional: bool = False
    ) -> Any:
        return await self._local.get_path(
            source_id, target_id, max_hops, bidirectional=bidirectional
        )

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
        except (ConnectionError, OSError, TimeoutError, ValueError, KeyError) as exc:
            logger.warning(
                "Remote brain not found or unreachable (%s), pushing local version",
                type(exc).__name__,
            )
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


def _install_local_delegators() -> None:
    """Generate read-through delegators to ``self._local`` for the extended /
    batch storage surface HybridStorage does not override explicitly.

    HybridStorage subclasses :class:`NeuralStorage`, so it *inherits*
    ``ExtendedStorage``'s concrete ``raise NotImplementedError`` stubs (typed
    memory, change log, devices, merkle, alerts, sources, …) plus the optional
    ``CoreStorage`` batch helpers. Because those are real inherited methods,
    ``__getattr__`` never fires for them — they would raise NotImplementedError
    instead of delegating. We therefore install explicit delegators for every
    such method that HybridStorage does not already define (closes #30).
    """
    import inspect

    from neural_memory.storage.base import CoreStorage, ExtendedStorage

    def _make_delegator(method_name: str, *, is_async: bool) -> Any:
        if is_async:

            async def _adelegator(self: HybridStorage, *args: Any, **kwargs: Any) -> Any:
                return await getattr(self._local, method_name)(*args, **kwargs)

            deleg: Any = _adelegator
        else:

            def _sdelegator(self: HybridStorage, *args: Any, **kwargs: Any) -> Any:
                return getattr(self._local, method_name)(*args, **kwargs)

            deleg = _sdelegator

        deleg.__name__ = method_name
        deleg.__qualname__ = f"HybridStorage.{method_name}"
        deleg.__doc__ = f"Delegate {method_name}() to the local backend."
        return deleg

    # Optional CoreStorage helpers (non-abstract, default to a base impl that
    # assumes a single SQL backend) that should hit the local store directly.
    optional_core = {
        "get_neurons_batch",
        "get_neuron_states_batch",
        "update_neuron_states_batch",
        "update_synapses_batch",
        "find_neurons_exact_batch",
        "find_reflex_neurons",
        "search_fiber_summaries",
        "update_fiber_metadata",
        "find_fibers_batch",
        "get_synapses_for_neurons",
        "get_neuron_hashes",
        "has_neuron_by_content_hash",
        "batch_save",
        "disable_auto_save",
        "enable_auto_save",
    }

    names: set[str] = set(optional_core)
    for name in vars(ExtendedStorage):
        if name.startswith("_"):
            continue
        if callable(getattr(ExtendedStorage, name, None)):
            names.add(name)

    for name in names:
        # Never shadow a method HybridStorage defines explicitly.
        if name in HybridStorage.__dict__:
            continue
        inherited = getattr(HybridStorage, name, None)
        # Only install if the name is a real (inherited) callable on the class.
        if not callable(inherited):
            continue
        # ``batch_save``/auto-save toggles live on CoreStorage; ensure they exist.
        if name in optional_core and not (hasattr(CoreStorage, name) or hasattr(ExtendedStorage, name)):
            continue
        is_async = inspect.iscoroutinefunction(inherited)
        setattr(HybridStorage, name, _make_delegator(name, is_async=is_async))


_install_local_delegators()
