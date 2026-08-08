"""Public-surface parity matrix for legacy and unified SQLite adapters."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.brain_mode import BrainModeConfig, StorageAdapter
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.factory import create_storage
from neural_memory.utils.timeutils import utcnow


async def _open_storage(
    tmp_path: Path,
    adapter: StorageAdapter,
    suffix: str,
) -> NeuralStorage:
    storage = await create_storage(
        BrainModeConfig.local(storage_adapter=adapter),
        "pilot",
        local_path=str(tmp_path / f"{adapter}-{suffix}.db"),
    )
    await storage.save_brain(Brain.create(name="pilot", brain_id="pilot"))
    storage.set_brain("pilot")
    return storage


async def _run_public_scenario(
    tmp_path: Path,
    adapter: StorageAdapter,
) -> dict[str, object]:
    storage = await _open_storage(tmp_path, adapter, "source")
    target = await _open_storage(tmp_path, adapter, "target")
    try:
        now = utcnow()
        old_neuron = Neuron(
            id="n-old",
            type=NeuronType.CONCEPT,
            content="old concept",
            content_hash=101,
            created_at=now - timedelta(days=2),
        )
        new_neuron = Neuron(
            id="n-new",
            type=NeuronType.CONCEPT,
            content="new concept",
            content_hash=202,
            created_at=now,
        )
        await storage.add_neuron(old_neuron)
        await storage.add_neuron(new_neuron)

        tagged = replace(
            Fiber.create(
                neuron_ids={old_neuron.id, new_neuron.id},
                synapse_ids=set(),
                anchor_neuron_id=old_neuron.id,
                tags={"alpha", "beta"},
                fiber_id="f-tagged",
            ),
            created_at=now - timedelta(days=1),
            salience=0.8,
        )
        untyped = replace(
            Fiber.create(
                neuron_ids={old_neuron.id},
                synapse_ids=set(),
                anchor_neuron_id=old_neuron.id,
                tags={"alpha"},
                fiber_id="f-untyped",
            ),
            created_at=now - timedelta(hours=12),
            salience=0.6,
        )
        expired = replace(
            Fiber.create(
                neuron_ids={old_neuron.id},
                synapse_ids=set(),
                anchor_neuron_id=old_neuron.id,
                tags={"beta"},
                fiber_id="f-expired",
            ),
            created_at=now - timedelta(hours=6),
            salience=0.4,
        )
        for fiber in (tagged, untyped, expired):
            await storage.add_fiber(fiber)

        expired_memory = replace(
            TypedMemory.create(
                fiber_id=expired.id,
                memory_type=MemoryType.TODO,
                priority=Priority.NORMAL,
                source="parity",
            ),
            expires_at=now - timedelta(seconds=1),
        )
        await storage.add_typed_memory(expired_memory)

        and_ids = {
            fiber.id
            for fiber in await storage.find_fibers(
                tags={"alpha", "beta"},
                tag_mode="and",
            )
        }
        or_ids = {
            fiber.id
            for fiber in await storage.find_fibers(
                tags={"alpha", "beta"},
                tag_mode="or",
            )
        }
        cutoff_neurons = {
            neuron.id
            for neuron in await storage.find_neurons(
                created_before=now - timedelta(days=1),
            )
        }
        visible_fibers = {
            fiber.id
            for fiber in await storage.find_fibers_batch(
                [old_neuron.id],
                created_before=now,
            )
        }

        await storage.save_brain(Brain.create(name="other", brain_id="other"))
        storage.set_brain("other")
        await storage.add_neuron(
            Neuron.create(
                type=NeuronType.CONCEPT,
                content="other brain concept",
                neuron_id="n-other",
            )
        )
        isolated_ids = {
            neuron.id for neuron in await storage.find_neurons(content_contains="concept")
        }
        storage.set_brain("pilot")

        snapshot = await storage.export_brain("pilot")
        imported_id = await target.import_brain(snapshot, target_brain_id="pilot-copy")
        target.set_brain(imported_id)
        imported_stats = await target.get_stats(imported_id)

        fetched = await storage.get_neuron(old_neuron.id)
        updated = old_neuron.with_metadata(parity=True)
        await storage.update_neuron(updated)
        updated_row = await storage.get_neuron(old_neuron.id)
        deleted = await storage.delete_neuron(new_neuron.id)

        return {
            "fetched": fetched.content if fetched else None,
            "updated": updated_row.metadata if updated_row else None,
            "deleted": deleted,
            "remaining_neurons": sorted(
                neuron.id for neuron in await storage.find_neurons(limit=100)
            ),
            "and_tags": sorted(and_ids),
            "or_tags": sorted(or_ids),
            "cutoff_neurons": sorted(cutoff_neurons),
            "visible_fibers": sorted(visible_fibers),
            "isolated_ids": sorted(isolated_ids),
            "snapshot_counts": {
                "neurons": len(snapshot.neurons),
                "fibers": len(snapshot.fibers),
            },
            "imported_stats": imported_stats,
        }
    finally:
        await storage.close()
        await target.close()


@pytest.mark.asyncio
async def test_public_storage_adapter_parity(tmp_path: Path) -> None:
    legacy = await _run_public_scenario(tmp_path, "legacy")
    unified = await _run_public_scenario(tmp_path, "unified")

    assert legacy == unified
    assert legacy["and_tags"] == ["f-tagged"]
    assert legacy["or_tags"] == ["f-expired", "f-tagged", "f-untyped"]
    assert legacy["cutoff_neurons"] == ["n-old"]
    assert legacy["visible_fibers"] == ["f-tagged", "f-untyped"]
    assert legacy["isolated_ids"] == ["n-other"]
