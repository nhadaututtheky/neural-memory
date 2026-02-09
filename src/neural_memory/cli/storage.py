"""JSON-based persistent storage for CLI."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from neural_memory.core.brain import Brain, BrainConfig, BrainSnapshot
from neural_memory.storage.memory_store import InMemoryStorage


class PersistentStorage(InMemoryStorage):
    """InMemoryStorage with JSON file persistence.

    Wraps InMemoryStorage and adds save/load functionality for CLI use.
    Data is stored in ~/.neural-memory/brains/<brain_name>.json
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize persistent storage.

        Args:
            file_path: Path to the JSON file for this brain.
        """
        super().__init__()
        self._file_path = file_path
        self._auto_save = True

    @classmethod
    async def load(cls, file_path: Path) -> PersistentStorage:
        """Load storage from file, or create new if doesn't exist."""
        storage = cls(file_path)

        if file_path.exists():
            await storage._load_from_file()
        else:
            # Create default brain
            file_path.parent.mkdir(parents=True, exist_ok=True)
            brain_name = file_path.stem
            brain = Brain.create(name=brain_name)
            await storage.save_brain(brain)
            storage.set_brain(brain.id)
            await storage._save_to_file()

        return storage

    async def _load_from_file(self) -> None:
        """Load data from JSON file."""
        with open(self._file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct brain
        brain_data = data.get("brain", {})
        if brain_data:
            config = BrainConfig(**brain_data.get("config", {}))
            brain = Brain(
                id=brain_data["id"],
                name=brain_data["name"],
                config=config,
                owner_id=brain_data.get("owner_id"),
                is_public=brain_data.get("is_public", False),
                created_at=datetime.fromisoformat(brain_data["created_at"]),
                updated_at=datetime.fromisoformat(brain_data["updated_at"]),
            )
            await super().save_brain(brain)
            self.set_brain(brain.id)

        # Import snapshot data if exists
        if "snapshot" in data:
            snapshot_data = data["snapshot"]
            snapshot = BrainSnapshot(
                brain_id=snapshot_data["brain_id"],
                brain_name=snapshot_data["brain_name"],
                exported_at=datetime.fromisoformat(snapshot_data["exported_at"]),
                version=snapshot_data["version"],
                neurons=snapshot_data["neurons"],
                synapses=snapshot_data["synapses"],
                fibers=snapshot_data["fibers"],
                config=snapshot_data.get("config", {}),
                metadata=snapshot_data.get("metadata", {}),
            )
            await self.import_brain(snapshot, brain.id if brain_data else None)

    async def _save_to_file(self) -> None:
        """Save data to JSON file."""
        if not self._current_brain_id:
            return

        brain = await self.get_brain(self._current_brain_id)
        if not brain:
            return

        # Export as snapshot
        snapshot = await self.export_brain(self._current_brain_id)

        data = {
            "brain": {
                "id": brain.id,
                "name": brain.name,
                "config": {
                    "decay_rate": brain.config.decay_rate,
                    "reinforcement_delta": brain.config.reinforcement_delta,
                    "activation_threshold": brain.config.activation_threshold,
                    "max_spread_hops": brain.config.max_spread_hops,
                    "max_context_tokens": brain.config.max_context_tokens,
                },
                "owner_id": brain.owner_id,
                "is_public": brain.is_public,
                "created_at": brain.created_at.isoformat(),
                "updated_at": brain.updated_at.isoformat(),
            },
            "snapshot": {
                "brain_id": snapshot.brain_id,
                "brain_name": snapshot.brain_name,
                "exported_at": snapshot.exported_at.isoformat(),
                "version": snapshot.version,
                "neurons": snapshot.neurons,
                "synapses": snapshot.synapses,
                "fibers": snapshot.fibers,
                "config": snapshot.config,
                "metadata": snapshot.metadata,
            },
            "saved_at": datetime.now().isoformat(),
        }

        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    async def save(self) -> None:
        """Explicitly save to file."""
        await self._save_to_file()

    # Override methods to auto-save

    async def add_neuron(self, neuron: Any) -> str:
        """Add neuron and save."""
        result = await super().add_neuron(neuron)
        if self._auto_save:
            await self._save_to_file()
        return result

    async def add_synapse(self, synapse: Any) -> str:
        """Add synapse and save."""
        result = await super().add_synapse(synapse)
        if self._auto_save:
            await self._save_to_file()
        return result

    async def add_fiber(self, fiber: Any) -> str:
        """Add fiber and save."""
        result = await super().add_fiber(fiber)
        if self._auto_save:
            await self._save_to_file()
        return result

    async def batch_save(self) -> None:
        """Save after batch operations (call manually when auto_save is off)."""
        await self._save_to_file()

    def disable_auto_save(self) -> None:
        """Disable auto-save for batch operations."""
        self._auto_save = False

    def enable_auto_save(self) -> None:
        """Enable auto-save."""
        self._auto_save = True

    async def close(self) -> None:
        """Flush pending data and release the storage instance.

        Safe to call multiple times. Ensures any unsaved state is persisted
        before the storage reference is discarded.
        """
        await self._save_to_file()
