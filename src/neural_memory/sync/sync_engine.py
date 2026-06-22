"""Sync engine orchestrator for multi-device incremental sync."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.sync.incremental_merge import merge_change_lists
from neural_memory.sync.protocol import (
    ConflictStrategy,
    MerkleBucketDiff,
    MerkleSyncRequest,
    MerkleSyncResponse,
    SyncChange,
    SyncRequest,
    SyncResponse,
    SyncStatus,
)
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


class SyncEngine:
    """Top-level orchestrator for multi-device incremental sync.

    Manages the sync lifecycle:
    1. Read local pending changes
    2. Send to hub
    3. Apply remote changes
    4. Mark synced
    5. Update watermark
    """

    def __init__(
        self,
        storage: NeuralStorage,
        device_id: str,
        strategy: ConflictStrategy = ConflictStrategy.PREFER_RECENT,
    ) -> None:
        self._storage = storage
        self._device_id = device_id
        self._strategy = strategy

    async def prepare_sync_request(self, brain_id: str) -> SyncRequest:
        """Prepare a sync request with local pending changes."""
        # Get the last known sync sequence
        device = await self._storage.get_device(self._device_id)
        last_sequence = device.last_sync_sequence if device else 0

        # Get unsynced local changes
        local_changes = await self._storage.get_unsynced_changes(limit=1000)

        sync_changes = [
            SyncChange(
                sequence=change.id,
                entity_type=change.entity_type,
                entity_id=change.entity_id,
                operation=change.operation,
                device_id=change.device_id,
                changed_at=change.changed_at.isoformat(),
                payload=change.payload,
            )
            for change in local_changes
        ]

        return SyncRequest(
            device_id=self._device_id,
            brain_id=brain_id,
            last_sequence=last_sequence,
            changes=sync_changes,
            strategy=self._strategy,
        )

    async def process_sync_response(
        self,
        response: SyncResponse,
        request: SyncRequest | None = None,
    ) -> dict[str, Any]:
        """Process a sync response from the hub — apply remote changes locally.

        Args:
            response: The hub's response (remote changes + hub watermark).
            request: The request we sent. Its changes determine which LOCAL
                change-log rows are now durably on the hub and may be marked
                synced. Required to mark local changes synced correctly.
        """
        applied = 0
        skipped = 0

        for change in response.changes:
            # Skip changes we originated
            if change.device_id == self._device_id:
                skipped += 1
                continue

            try:
                await self._apply_remote_change(change)
                applied += 1
            except Exception:
                logger.warning(
                    "Failed to apply remote change: %s %s %s",
                    change.operation,
                    change.entity_type,
                    change.entity_id,
                    exc_info=True,
                )
                skipped += 1

        # Mark LOCAL changes synced using the LOCAL change-log id space, NOT the
        # hub's sequence (finding #12). `mark_synced` runs against local
        # change_log.id; the hub_sequence aggregates many devices and is far
        # larger than any local id, so passing it marks ALL pending local
        # changes synced — including ones created after the snapshot and never
        # sent — silently losing them. Only mark up to the max local change id
        # that was actually included in this request.
        if request is not None and request.changes:
            max_local_synced = max(c.sequence for c in request.changes)
            if max_local_synced > 0:
                await self._storage.mark_synced(max_local_synced)

        # Track the hub watermark separately (remote "pulled up to").
        if response.hub_sequence > 0:
            await self._storage.update_device_sync(self._device_id, response.hub_sequence)

        return {
            "applied": applied,
            "skipped": skipped,
            "conflicts": len(response.conflicts),
            "hub_sequence": response.hub_sequence,
        }

    async def handle_hub_sync(self, request: SyncRequest) -> SyncResponse:
        """Handle an incoming sync request as the hub.

        This is called on the hub side to process incoming changes
        and return changes the requesting device hasn't seen.
        """
        # Get changes the requesting device hasn't seen
        remote_changes_raw = await self._storage.get_changes_since(
            request.last_sequence, limit=1000
        )

        remote_changes = [
            SyncChange(
                sequence=c.id,
                entity_type=c.entity_type,
                entity_id=c.entity_id,
                operation=c.operation,
                device_id=c.device_id,
                changed_at=c.changed_at.isoformat(),
                payload=c.payload,
            )
            for c in remote_changes_raw
            if c.device_id != request.device_id  # Don't send back their own changes
        ]

        # Resolve conflicts between incoming device changes and hub's existing remote changes
        # using the device's preferred strategy. `merged` holds the neural-aware
        # winner per entity (weight=max, frequency=sum, tags=union, delete-wins);
        # finding #13: previously discarded, applying raw request.changes instead,
        # which clobbered cross-device edits with last-applied-wins.
        merged_changes, conflicts_list = merge_change_lists(
            list(request.changes), remote_changes, request.strategy
        )

        # Record every incoming device change in the hub's change log (so other
        # devices pull them). Recording is over the raw incoming stream — the log
        # is the authoritative per-device event history.
        for change in request.changes:
            await self._storage.record_change(
                entity_type=change.entity_type,
                entity_id=change.entity_id,
                operation=change.operation,
                device_id=change.device_id,
                payload=change.payload,
            )

        # Apply the MERGED winners to hub storage so the neural-aware resolution
        # (not raw last-applied-wins) is what actually persists. Only apply the
        # winners that involve an incoming device change for this entity; remote-
        # only entities are already present in hub storage.
        incoming_keys = {(c.entity_type, c.entity_id) for c in request.changes}
        for change in merged_changes:
            if (change.entity_type, change.entity_id) not in incoming_keys:
                continue
            try:
                await self._apply_remote_change(change)
            except Exception:
                logger.warning(
                    "Hub failed to apply merged change: %s %s",
                    change.operation,
                    change.entity_id,
                    exc_info=True,
                )

        # Get current hub sequence
        stats = await self._storage.get_change_log_stats()
        hub_sequence = stats.get("last_sequence", 0)

        # Update device's last sync
        await self._storage.update_device_sync(request.device_id, hub_sequence)

        return SyncResponse(
            hub_sequence=hub_sequence,
            changes=remote_changes,
            conflicts=conflicts_list,
            status=SyncStatus.SUCCESS,
        )

    # ── Merkle delta sync ─────────────────────────────────────────────

    async def prepare_merkle_request(
        self,
        brain_id: str,
        *,
        is_pro: bool = False,
    ) -> MerkleSyncRequest | None:
        """Build a Merkle sync request with all local bucket hashes.

        Returns ``None`` if not Pro or if hash cache is incomplete.
        """
        if not is_pro:
            return None

        from neural_memory.sync.merkle import ENTITY_TYPES

        buckets: dict[str, dict[str, str]] = {}
        for entity_type in ENTITY_TYPES:
            # Ensure hashes are fresh
            await self._storage.compute_merkle_root(entity_type, is_pro=True)
            tree = await self._storage.get_merkle_tree(entity_type, is_pro=True)
            if tree:
                buckets[entity_type] = tree

        root_hash = await self._storage.get_merkle_root(is_pro=True)
        if root_hash is None:
            return None

        return MerkleSyncRequest(
            device_id=self._device_id,
            brain_id=brain_id,
            root_hash=root_hash,
            buckets=buckets,
            strategy=self._strategy,
        )

    async def handle_merkle_sync(
        self,
        request: MerkleSyncRequest,
        *,
        is_pro: bool = False,
    ) -> MerkleSyncResponse:
        """Hub-side handler: compare bucket hashes and return diffs.

        Computes local Merkle hashes, compares with device's buckets,
        and returns entity payloads for differing buckets.
        """
        if not is_pro:
            return MerkleSyncResponse(status="error", message="Merkle sync requires Pro")

        from neural_memory.sync.merkle import ENTITY_TYPES

        # Compute local hashes
        for entity_type in ENTITY_TYPES:
            await self._storage.compute_merkle_root(entity_type, is_pro=True)

        hub_root = await self._storage.get_merkle_root(is_pro=True)
        hub_root = hub_root or ""

        # Fast path: roots match = in sync
        if hub_root == request.root_hash:
            stats = await self._storage.get_change_log_stats()
            return MerkleSyncResponse(
                status="in_sync",
                hub_root_hash=hub_root,
                hub_sequence=stats.get("last_sequence", 0),
            )

        # Compare bucket-by-bucket
        changed_prefixes: list[str] = []
        diffs: list[MerkleBucketDiff] = []

        for entity_type in ENTITY_TYPES:
            local_tree = await self._storage.get_merkle_tree(entity_type, is_pro=True)
            remote_buckets = request.buckets.get(entity_type, {})

            # Find differing bucket prefixes
            all_prefixes = set(local_tree) | set(remote_buckets)
            type_prefix = f"{entity_type}s"

            for prefix in sorted(all_prefixes):
                # Skip the type-root entry (e.g. "neurons") — compare buckets only
                if prefix == type_prefix:
                    continue

                local_hash = local_tree.get(prefix, "")
                remote_hash = remote_buckets.get(prefix, "")

                if local_hash != remote_hash:
                    changed_prefixes.append(prefix)

                    # Fetch entities for this bucket via a direct prefix query
                    # (finding #44) — not a 10000-row slab filtered in Python.
                    entities = await self._fetch_bucket_entities(entity_type, prefix)
                    entity_ids = [e["id"] for e in entities]

                    diffs.append(
                        MerkleBucketDiff(
                            entity_type=entity_type,
                            prefix=prefix,
                            entity_ids=entity_ids,
                            entities=entities,
                        )
                    )

        stats = await self._storage.get_change_log_stats()
        return MerkleSyncResponse(
            status="diff",
            hub_root_hash=hub_root,
            changed_prefixes=changed_prefixes,
            diffs=diffs,
            hub_sequence=stats.get("last_sequence", 0),
        )

    async def process_merkle_response(
        self,
        response: MerkleSyncResponse,
        local_buckets: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        """Client-side: apply diffs from Merkle sync response.

        For each differing bucket, computes insert/update/delete sets
        by comparing entity ID lists.
        """
        if response.status == "in_sync":
            return {"applied": 0, "deleted": 0, "status": "in_sync"}

        applied = 0
        deleted = 0

        # Finding #1 (CRITICAL): the client must NOT delete a local entity just
        # because the hub doesn't have it. Any entity created locally since the
        # last sync (still pending-push in the change_log with synced=0) is
        # absent from the hub but must be preserved and pushed, not destroyed.
        # Gather the set of unsynced (pending-push) local entity IDs so the
        # delete pass can skip them.
        pending_push_ids = await self._get_unsynced_entity_ids()

        for diff in response.diffs:
            remote_ids = set(diff.entity_ids)
            remote_entities = {e["id"]: e for e in diff.entities}

            # Get local IDs for this bucket via a direct prefix query (#44).
            local_entities = await self._fetch_bucket_entities(diff.entity_type, diff.prefix)
            local_ids = {e["id"] for e in local_entities}

            # Inserts: remote has, local doesn't
            for eid in remote_ids - local_ids:
                payload = remote_entities.get(eid, {})
                if payload:
                    change = SyncChange(
                        sequence=0,
                        entity_type=diff.entity_type,
                        entity_id=eid,
                        operation="insert",
                        device_id="hub",
                        changed_at=payload.get("updated_at", utcnow().isoformat()),
                        payload=payload,
                    )
                    try:
                        await self._apply_remote_change(change)
                        applied += 1
                    except Exception:
                        logger.warning("Merkle insert failed: %s %s", diff.entity_type, eid)

            # Updates: both have, hash differs (hub sends full payload)
            for eid in remote_ids & local_ids:
                payload = remote_entities.get(eid, {})
                if payload:
                    change = SyncChange(
                        sequence=0,
                        entity_type=diff.entity_type,
                        entity_id=eid,
                        operation="update",
                        device_id="hub",
                        changed_at=payload.get("updated_at", utcnow().isoformat()),
                        payload=payload,
                    )
                    try:
                        await self._apply_remote_change(change)
                        applied += 1
                    except Exception:
                        logger.warning("Merkle update failed: %s %s", diff.entity_type, eid)

            # Deletes: local has, remote doesn't — BUT only delete entities the
            # hub has actually seen. A local-only id that is still pending-push
            # (synced=0 in the change_log) is a NEW local entity, not a remote
            # deletion; deleting it would silently destroy un-pushed data
            # (finding #1). Skip those — they are pushed on the next change-log
            # sync round instead.
            for eid in local_ids - remote_ids:
                if eid in pending_push_ids:
                    logger.debug(
                        "Merkle: preserving un-pushed local %s %s (pending push)",
                        diff.entity_type,
                        eid,
                    )
                    continue
                change = SyncChange(
                    sequence=0,
                    entity_type=diff.entity_type,
                    entity_id=eid,
                    operation="delete",
                    device_id="hub",
                    changed_at=utcnow().isoformat(),
                )
                try:
                    await self._apply_remote_change(change)
                    deleted += 1
                except Exception:
                    logger.warning("Merkle delete failed: %s %s", diff.entity_type, eid)

        # Update only the hub watermark (remote "pulled up to"). Do NOT call
        # mark_synced(hub_sequence) here: the Merkle path is a pull/reconcile,
        # it does not push local change-log rows, and hub_sequence is in the
        # hub's id space — marking local changes synced against it would lose
        # un-pushed local data (findings #1 and #12). Local changes are flushed
        # by the change-log push path, which marks them synced correctly.
        if response.hub_sequence > 0:
            await self._storage.update_device_sync(self._device_id, response.hub_sequence)

        return {
            "applied": applied,
            "deleted": deleted,
            "changed_prefixes": len(response.changed_prefixes),
            "hub_sequence": response.hub_sequence,
            "status": "diff",
        }

    async def _fetch_bucket_entities(
        self,
        entity_type: str,
        prefix: str,
    ) -> list[dict[str, Any]]:
        """Fetch all entities in a Merkle bucket as sync-payload dicts.

        Finding #44: the old implementation pulled up to 10000 rows
        (``find_neurons(limit=10000)``) then filtered by ``id[:2]`` in Python,
        so on a brain with more than 10000 entities everything past the slab
        was invisible — entities looked "missing" and skewed insert/delete
        detection. Instead, resolve the exact IDs in this bucket via a SQL
        prefix query (``get_bucket_entity_ids``) and fetch each entity by ID,
        which is bounded by the bucket size, not the whole brain.

        ``prefix`` is the full bucket prefix, e.g. ``"neurons/0a"``.
        """
        # Resolve the exact set of IDs in this bucket via a direct prefix query.
        bucket_key = prefix.split("/")[-1] if "/" in prefix else ""
        try:
            ids = await self._storage.get_bucket_entity_ids(entity_type, prefix, is_pro=True)
        except (NotImplementedError, AttributeError):
            ids = []

        if not ids:
            # Fallback for backends/mocks without get_bucket_entity_ids: derive
            # IDs from the (bounded) collection. We still avoid re-fetching full
            # objects below by reusing the ones we already loaded here.
            return await self._fetch_bucket_entities_fallback(entity_type, bucket_key)

        entities: list[dict[str, Any]] = []

        if entity_type == "neuron":
            for nid in ids:
                n = await self._storage.get_neuron(nid)
                if n is not None:
                    entities.append(self._neuron_to_payload(n))

        elif entity_type == "synapse":
            for sid in ids:
                s = await self._storage.get_synapse(sid)
                if s is not None:
                    entities.append(self._synapse_to_payload(s))

        elif entity_type == "fiber":
            for fid in ids:
                f = await self._storage.get_fiber(fid)
                if f is not None:
                    entities.append(self._fiber_to_payload(f))

        return entities

    async def _fetch_bucket_entities_fallback(
        self,
        entity_type: str,
        bucket_key: str,
    ) -> list[dict[str, Any]]:
        """Collection-scan fallback when get_bucket_entity_ids is unavailable.

        Paginates through the collection in pages rather than a single 10000-row
        slab so large brains are still covered (finding #44). Used only by
        backends/test doubles that do not implement get_bucket_entity_ids.
        """
        entities: list[dict[str, Any]] = []
        page = 5000

        def _bucket_of(eid: str) -> str:
            return eid[:2].lower() if len(eid) >= 2 else eid.lower().ljust(2, "0")

        if entity_type == "neuron":
            offset = 0
            while True:
                neurons = await self._storage.find_neurons(limit=page, offset=offset)
                if not neurons:
                    break
                for n in neurons:
                    if _bucket_of(n.id) == bucket_key:
                        entities.append(self._neuron_to_payload(n))
                if len(neurons) < page:
                    break
                offset += page

        elif entity_type == "synapse":
            synapses = await self._storage.get_synapses()
            for s in synapses:
                if _bucket_of(s.id) == bucket_key:
                    entities.append(self._synapse_to_payload(s))

        elif entity_type == "fiber":
            fibers = await self._storage.find_fibers(limit=page)
            for f in fibers:
                if _bucket_of(f.id) == bucket_key:
                    entities.append(self._fiber_to_payload(f))

        return entities

    @staticmethod
    def _neuron_to_payload(n: Neuron) -> dict[str, Any]:
        """Serialize a Neuron into a sync-payload dict."""
        updated_at = getattr(n, "updated_at", None) or n.created_at
        return {
            "id": n.id,
            "type": n.type.value if hasattr(n.type, "value") else str(n.type),
            "content": n.content,
            "content_hash": n.content_hash,
            "created_at": n.created_at.isoformat() if n.created_at else "",
            "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else "",
            "metadata": n.metadata or {},
        }

    @staticmethod
    def _synapse_to_payload(s: Synapse) -> dict[str, Any]:
        """Serialize a Synapse into a sync-payload dict."""
        return {
            "id": s.id,
            "source_id": s.source_id,
            "target_id": s.target_id,
            "type": s.type.value if hasattr(s.type, "value") else str(s.type),
            "weight": s.weight,
            "direction": s.direction.value if hasattr(s.direction, "value") else str(s.direction),
            "content_hash": getattr(s, "content_hash", 0),
            "reinforced_count": s.reinforced_count,
            "created_at": s.created_at.isoformat() if s.created_at else "",
            "metadata": s.metadata or {},
        }

    @staticmethod
    def _fiber_to_payload(f: Fiber) -> dict[str, Any]:
        """Serialize a Fiber into a sync-payload dict."""
        return {
            "id": f.id,
            "anchor_neuron_id": f.anchor_neuron_id,
            "summary": f.summary or "",
            "conductivity": f.conductivity,
            "salience": f.salience,
            "frequency": f.frequency,
            "neuron_ids": list(f.neuron_ids),
            "synapse_ids": list(f.synapse_ids),
            "pathway": list(f.pathway),
            "auto_tags": list(f.auto_tags),
            "agent_tags": list(f.agent_tags),
            "created_at": f.created_at.isoformat() if f.created_at else "",
            "metadata": f.metadata or {},
        }

    async def _get_unsynced_entity_ids(self) -> set[str]:
        """Return the set of local entity IDs with un-pushed change-log rows.

        These are entities created/modified locally since the last successful
        push (``synced=0`` in the change_log). They are absent from the hub not
        because the hub deleted them, but because they were never sent — so the
        Merkle delete pass must preserve them (finding #1).
        """
        try:
            unsynced = await self._storage.get_unsynced_changes(limit=10000)
        except (NotImplementedError, AttributeError):
            return set()
        return {c.entity_id for c in unsynced}

    async def _apply_remote_change(self, change: SyncChange) -> None:
        """Apply a single remote change to local storage.

        This is a best-effort application — entities may not exist locally
        for update/delete, and that's OK (eventual consistency).
        """
        entity_type = change.entity_type
        operation = change.operation
        payload = change.payload

        # Delete operations don't need a payload — just remove by ID
        if operation == "delete":
            if entity_type == "neuron":
                await self._storage.delete_neuron(change.entity_id)
            elif entity_type == "synapse":
                await self._storage.delete_synapse(change.entity_id)
            elif entity_type == "fiber":
                await self._storage.delete_fiber(change.entity_id)
            else:
                logger.warning("Unknown entity_type in delete: %s", entity_type)
            return

        # Insert/update require a payload to reconstruct the entity
        if not payload:
            logger.warning(
                "Empty payload for %s %s %s — skipping",
                operation,
                entity_type,
                change.entity_id,
            )
            return

        if entity_type == "neuron":
            neuron = self._neuron_from_payload(payload)
            if operation == "insert":
                # Dedup check: skip if content_hash matches existing neuron
                if neuron.content_hash and await self._has_neuron_content_hash(neuron.content_hash):
                    logger.info("Sync dedup: skipping neuron %s (content hash match)", neuron.id)
                    return
                try:
                    await self._storage.add_neuron(neuron)
                except ValueError:
                    await self._storage.update_neuron(neuron)
            else:  # update
                try:
                    await self._storage.update_neuron(neuron)
                except ValueError:
                    await self._storage.add_neuron(neuron)

        elif entity_type == "synapse":
            synapse = self._synapse_from_payload(payload)
            if operation == "insert":
                try:
                    await self._storage.add_synapse(synapse)
                except ValueError:
                    await self._storage.update_synapse(synapse)
            else:  # update
                try:
                    await self._storage.update_synapse(synapse)
                except ValueError:
                    await self._storage.add_synapse(synapse)

        elif entity_type == "fiber":
            fiber = self._fiber_from_payload(payload)
            if operation == "insert":
                # Dedup check: if a fiber with same anchor neuron exists, merge tags
                merged = await self._try_merge_fiber(fiber)
                if merged:
                    logger.info("Sync dedup: merged fiber %s into existing", fiber.id)
                    return
                try:
                    await self._storage.add_fiber(fiber)
                except ValueError:
                    await self._storage.update_fiber(fiber)
            else:  # update
                try:
                    await self._storage.update_fiber(fiber)
                except ValueError:
                    await self._storage.add_fiber(fiber)

        else:
            logger.warning("Unknown entity_type: %s", entity_type)
            return

        logger.debug(
            "Applied remote change: %s %s %s from device %s",
            operation,
            entity_type,
            change.entity_id,
            change.device_id,
        )

    # ── Sync dedup helpers ──────────────────────────────────────────────

    async def _has_neuron_content_hash(self, content_hash: int) -> bool:
        """Check if a neuron with this content hash already exists locally."""
        try:
            return await self._storage.has_neuron_by_content_hash(content_hash)
        except Exception:
            return False

    async def _try_merge_fiber(self, incoming: Fiber) -> bool:
        """Try to merge incoming fiber into an existing one with same anchor.

        Returns True if merged (caller should skip insert), False if no match.
        """
        if not incoming.anchor_neuron_id:
            return False

        try:
            from dataclasses import replace as dc_replace

            existing_fibers = await self._storage.find_fibers_batch(
                [incoming.anchor_neuron_id], limit_per_neuron=1
            )
            if not existing_fibers:
                return False

            existing = existing_fibers[0]
            if existing.id == incoming.id:
                return False  # Same fiber ID — let normal update handle it

            # Merge tags from incoming into existing
            merged_auto_tags = existing.auto_tags | incoming.auto_tags
            merged_agent_tags = existing.agent_tags | incoming.agent_tags
            merged_meta = {**(existing.metadata or {}), **(incoming.metadata or {})}

            updated = dc_replace(
                existing,
                auto_tags=merged_auto_tags,
                agent_tags=merged_agent_tags,
                metadata=merged_meta,
                frequency=existing.frequency + 1,
            )
            await self._storage.update_fiber(updated)
            return True
        except Exception:
            logger.debug("Fiber merge check failed (non-critical)", exc_info=True)
            return False

    # ── Payload-to-entity reconstruction ─────────────────────────────────

    @staticmethod
    def _neuron_from_payload(payload: dict[str, Any]) -> Neuron:
        """Reconstruct a Neuron from sync payload dict."""
        created_at_raw = payload.get("created_at")
        created_at = datetime.fromisoformat(created_at_raw) if created_at_raw else utcnow()

        metadata = payload.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Neuron(
            id=payload["id"],
            type=NeuronType(payload.get("type", "concept")),
            content=payload.get("content", ""),
            metadata=metadata,
            content_hash=payload.get("content_hash", 0),
            created_at=created_at,
        )

    @staticmethod
    def _synapse_from_payload(payload: dict[str, Any]) -> Synapse:
        """Reconstruct a Synapse from sync payload dict."""
        created_at_raw = payload.get("created_at")
        created_at = datetime.fromisoformat(created_at_raw) if created_at_raw else utcnow()

        last_activated_raw = payload.get("last_activated")
        last_activated = datetime.fromisoformat(last_activated_raw) if last_activated_raw else None

        metadata = payload.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Synapse(
            id=payload["id"],
            source_id=payload.get("source_id", ""),
            target_id=payload.get("target_id", ""),
            type=SynapseType(payload.get("type", "related_to")),
            weight=payload.get("weight", 0.5),
            direction=Direction(payload.get("direction", "uni")),
            metadata=metadata,
            reinforced_count=payload.get("reinforced_count", 0),
            last_activated=last_activated,
            created_at=created_at,
        )

    @staticmethod
    def _fiber_from_payload(payload: dict[str, Any]) -> Fiber:
        """Reconstruct a Fiber from sync payload dict."""
        created_at_raw = payload.get("created_at")
        created_at = datetime.fromisoformat(created_at_raw) if created_at_raw else utcnow()

        time_start_raw = payload.get("time_start")
        time_start = datetime.fromisoformat(time_start_raw) if time_start_raw else None

        time_end_raw = payload.get("time_end")
        time_end = datetime.fromisoformat(time_end_raw) if time_end_raw else None

        last_conducted_raw = payload.get("last_conducted")
        last_conducted = datetime.fromisoformat(last_conducted_raw) if last_conducted_raw else None

        metadata = payload.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        # Parse set/list fields with safe defaults
        neuron_ids_raw = payload.get("neuron_ids", [])
        if isinstance(neuron_ids_raw, str):
            neuron_ids_raw = json.loads(neuron_ids_raw)
        neuron_ids = set(neuron_ids_raw)

        synapse_ids_raw = payload.get("synapse_ids", [])
        if isinstance(synapse_ids_raw, str):
            synapse_ids_raw = json.loads(synapse_ids_raw)
        synapse_ids = set(synapse_ids_raw)

        pathway_raw = payload.get("pathway", [])
        if isinstance(pathway_raw, str):
            pathway_raw = json.loads(pathway_raw)
        pathway: list[str] = list(pathway_raw)

        auto_tags_raw = payload.get("auto_tags", [])
        if isinstance(auto_tags_raw, str):
            auto_tags_raw = json.loads(auto_tags_raw)
        auto_tags = set(auto_tags_raw)

        agent_tags_raw = payload.get("agent_tags", [])
        if isinstance(agent_tags_raw, str):
            agent_tags_raw = json.loads(agent_tags_raw)
        agent_tags = set(agent_tags_raw)

        return Fiber(
            id=payload["id"],
            neuron_ids=neuron_ids,
            synapse_ids=synapse_ids,
            anchor_neuron_id=payload.get("anchor_neuron_id", ""),
            pathway=pathway,
            conductivity=payload.get("conductivity", 1.0),
            last_conducted=last_conducted,
            time_start=time_start,
            time_end=time_end,
            coherence=payload.get("coherence", 0.0),
            salience=payload.get("salience", 0.0),
            frequency=payload.get("frequency", 0),
            summary=payload.get("summary"),
            auto_tags=auto_tags,
            agent_tags=agent_tags,
            metadata=metadata,
            compression_tier=payload.get("compression_tier", 0),
            created_at=created_at,
        )
