"""Regression tests for the CRITICAL Merkle pull-sync data-loss bug (finding #1).

The Merkle sync path is pull/reconcile only: it never pushes the requesting
device's new local entities to the hub. The original client-side
``process_merkle_response`` computed deletes as ``local_ids - remote_ids`` and
deleted every entity the hub lacked — destroying memories created locally since
the last sync that were never pushed.

These tests use REAL SQLite storage (not mocks) to prove that new local
memories survive a full/pull Merkle reconcile, and that genuine remote
deletions (already-synced entities the hub no longer has) still apply.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.sync.protocol import (
    ConflictStrategy,
    MerkleBucketDiff,
    MerkleSyncResponse,
)
from neural_memory.sync.sync_engine import SyncEngine


@pytest.fixture
async def storage() -> SQLiteStorage:
    """Real SQLite storage with a brain context set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "device.db"
        store = SQLiteStorage(db_path)
        await store.initialize()

        brain = Brain.create(name="regression_brain")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        yield store

        await store.close()


def _neuron(neuron_id: str, content: str) -> Neuron:
    return Neuron.create(
        type=NeuronType.CONCEPT,
        content=content,
        neuron_id=neuron_id,
        content_hash=abs(hash(content)) % (2**31),
    )


async def _bucket_prefix_for(storage: SQLiteStorage, neuron_id: str) -> str:
    """The Merkle bucket prefix that a neuron id falls into."""
    key = neuron_id[:2].lower() if len(neuron_id) >= 2 else neuron_id.lower().ljust(2, "0")
    return f"neurons/{key}"


class TestMerklePullSyncDataLoss:
    """The core CRITICAL #1 regression: new local memories must survive."""

    @pytest.mark.asyncio
    async def test_new_local_memory_survives_merkle_pull(self, storage: SQLiteStorage) -> None:
        """A locally-created, un-pushed neuron must NOT be deleted by a pull.

        Scenario: device adds neuron "aa-new" locally and records the insert in
        the change_log with synced=0 (never pushed). The hub knows nothing about
        it, so a Merkle diff for that bucket returns remote_ids without it.
        Before the fix, the client deleted it (local - remote). After the fix,
        it is preserved because it is still pending-push.
        """
        # Local-only neuron, sharing a bucket prefix so it lands in the diff.
        local_new = _neuron("aanew-0001", "fresh local memory")
        await storage.add_neuron(local_new)
        # Record as an unsynced (pending-push) change.
        await storage.record_change(
            entity_type="neuron",
            entity_id=local_new.id,
            operation="insert",
            device_id="dev-local",
            payload={"id": local_new.id, "content": local_new.content},
        )

        # The hub also has a neuron in the same bucket that we DON'T have.
        hub_neuron_payload = {
            "id": "aahub-0002",
            "type": "concept",
            "content": "hub-only memory",
            "content_hash": 999,
            "created_at": "2026-01-10T00:00:00",
            "updated_at": "2026-01-10T00:00:00",
            "metadata": {},
        }

        prefix = await _bucket_prefix_for(storage, "aanew-0001")
        # Hub's diff for this bucket: it lists only its own entity, not ours.
        response = MerkleSyncResponse(
            status="diff",
            hub_root_hash="x",
            changed_prefixes=[prefix],
            diffs=[
                MerkleBucketDiff(
                    entity_type="neuron",
                    prefix=prefix,
                    entity_ids=["aahub-0002"],
                    entities=[hub_neuron_payload],
                )
            ],
            hub_sequence=5000,
        )

        engine = SyncEngine(storage, device_id="dev-local", strategy=ConflictStrategy.PREFER_RECENT)
        result = await engine.process_merkle_response(response, local_buckets={})

        # Our new local neuron must STILL exist (not deleted).
        survived = await storage.get_neuron(local_new.id)
        assert survived is not None, "CRITICAL: new local memory was destroyed by Merkle pull"
        assert survived.content == "fresh local memory"

        # The hub-only neuron should have been inserted locally.
        pulled = await storage.get_neuron("aahub-0002")
        assert pulled is not None
        assert pulled.content == "hub-only memory"

        # And nothing was deleted.
        assert result["deleted"] == 0

    @pytest.mark.asyncio
    async def test_genuine_remote_deletion_still_applies(self, storage: SQLiteStorage) -> None:
        """An already-synced local entity the hub no longer has IS deleted.

        This confirms the delete-gate only spares *un-pushed* entities and does
        not regress legitimate tombstone propagation: an entity that was pushed
        (synced=1) and then deleted on another device must be removed locally.
        """
        synced_neuron = _neuron("bbsync-0001", "previously synced memory")
        await storage.add_neuron(synced_neuron)
        # Record + mark synced so it is NOT pending-push.
        seq = await storage.record_change(
            entity_type="neuron",
            entity_id=synced_neuron.id,
            operation="insert",
            device_id="dev-local",
            payload={"id": synced_neuron.id},
        )
        await storage.mark_synced(seq)

        prefix = await _bucket_prefix_for(storage, "bbsync-0001")
        # Hub bucket differs but does NOT include our synced neuron → real delete.
        response = MerkleSyncResponse(
            status="diff",
            hub_root_hash="x",
            changed_prefixes=[prefix],
            diffs=[
                MerkleBucketDiff(
                    entity_type="neuron",
                    prefix=prefix,
                    entity_ids=[],  # hub has nothing in this bucket anymore
                    entities=[],
                )
            ],
            hub_sequence=6000,
        )

        engine = SyncEngine(storage, device_id="dev-local")
        result = await engine.process_merkle_response(response, local_buckets={})

        # The synced-then-remotely-deleted neuron must be gone.
        assert await storage.get_neuron(synced_neuron.id) is None
        assert result["deleted"] == 1

    @pytest.mark.asyncio
    async def test_merkle_pull_does_not_mark_local_synced(self, storage: SQLiteStorage) -> None:
        """The Merkle pull must not mark local changes synced (findings #1/#12).

        Marking against hub_sequence would flag un-pushed local changes synced,
        so they would never be pushed. After a pull the pending change must
        remain unsynced and ready for the change-log push round.
        """
        local_new = _neuron("ccnew-0001", "pending push memory")
        await storage.add_neuron(local_new)
        await storage.record_change(
            entity_type="neuron",
            entity_id=local_new.id,
            operation="insert",
            device_id="dev-local",
            payload={"id": local_new.id},
        )

        response = MerkleSyncResponse(
            status="diff",
            hub_root_hash="x",
            changed_prefixes=[],
            diffs=[],
            hub_sequence=9999,
        )

        engine = SyncEngine(storage, device_id="dev-local")
        await engine.process_merkle_response(response, local_buckets={})

        # The pending change must still be unsynced.
        unsynced = await storage.get_unsynced_changes(limit=100)
        unsynced_ids = {c.entity_id for c in unsynced}
        assert local_new.id in unsynced_ids, "Merkle pull wrongly marked local change synced"


class TestMerkleBucketFetchScale:
    """Finding #44: per-bucket fetch must not be capped at a 10000-row slab."""

    @pytest.mark.asyncio
    async def test_bucket_fetch_uses_prefix_not_slab(self, storage: SQLiteStorage) -> None:
        """_fetch_bucket_entities returns exactly the bucket's entities by prefix."""
        # Two neurons in bucket "dd", one in bucket "ee".
        await storage.add_neuron(_neuron("dd000001", "bucket dd one"))
        await storage.add_neuron(_neuron("dd000002", "bucket dd two"))
        await storage.add_neuron(_neuron("ee000003", "bucket ee"))

        engine = SyncEngine(storage, device_id="dev-local")
        dd = await engine._fetch_bucket_entities("neuron", "neurons/dd")

        ids = {e["id"] for e in dd}
        assert ids == {"dd000001", "dd000002"}


class TestFiberMerkleFingerprint:
    """Finding #43: non-summary fiber edits must change the Merkle leaf hash."""

    @pytest.mark.asyncio
    async def test_salience_change_alters_bucket_hash(self, storage: SQLiteStorage) -> None:
        """A salience/tag edit (summary unchanged) must change the fiber bucket hash."""
        anchor = _neuron("ff000001", "anchor neuron")
        await storage.add_neuron(anchor)

        fiber = Fiber.create(
            neuron_ids={anchor.id},
            synapse_ids=set(),
            anchor_neuron_id=anchor.id,
            summary="stable summary",
            fiber_id="ff000099",
        )
        await storage.add_fiber(fiber)

        await storage.compute_merkle_root("fiber", is_pro=True)
        tree_before = await storage.get_merkle_tree("fiber", is_pro=True)
        bucket_prefix = "fibers/ff"
        hash_before = tree_before.get(bucket_prefix, "")
        assert hash_before, "expected a fiber bucket hash to exist"

        # Mutate ONLY non-summary fields: salience + auto_tags + frequency.
        mutated = dc_replace(
            fiber,
            salience=0.95,
            frequency=fiber.frequency + 7,
            auto_tags=fiber.auto_tags | {"newtag"},
        )
        await storage.update_fiber(mutated)

        await storage.compute_merkle_root("fiber", is_pro=True)
        tree_after = await storage.get_merkle_tree("fiber", is_pro=True)
        hash_after = tree_after.get(bucket_prefix, "")

        assert hash_after != hash_before, (
            "non-summary fiber edit did not change the Merkle leaf — finding #43 not fixed"
        )

    @pytest.mark.asyncio
    async def test_dialect_path_salience_change_alters_bucket_hash(self, tmp_path: Path) -> None:
        """Same #43 fix on the unified SQL dialect backend (non-deprecated path).

        Exercises ``storage/sql/mixins/merkle.py`` so the multi-column fiber
        fingerprint SQL is valid against SQLiteDialect.
        """
        from neural_memory.storage.sql import SQLStorage
        from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect

        store = SQLStorage(SQLiteDialect(db_path=str(tmp_path / "dialect.db")))
        await store.initialize()
        brain = Brain.create(name="dialect_brain")
        await store.save_brain(brain)
        store.set_brain(brain.id)
        try:
            anchor = _neuron("gg000001", "dialect anchor")
            await store.add_neuron(anchor)
            fiber = Fiber.create(
                neuron_ids={anchor.id},
                synapse_ids=set(),
                anchor_neuron_id=anchor.id,
                summary="stable",
                fiber_id="gg000099",
            )
            await store.add_fiber(fiber)

            await store.compute_merkle_root("fiber", is_pro=True)
            before = (await store.get_merkle_tree("fiber", is_pro=True)).get("fibers/gg", "")
            assert before

            mutated = dc_replace(fiber, salience=0.9, auto_tags=fiber.auto_tags | {"x"})
            await store.update_fiber(mutated)

            await store.compute_merkle_root("fiber", is_pro=True)
            after = (await store.get_merkle_tree("fiber", is_pro=True)).get("fibers/gg", "")

            assert after != before, "dialect path: non-summary fiber edit did not change leaf (#43)"
        finally:
            await store.close()
