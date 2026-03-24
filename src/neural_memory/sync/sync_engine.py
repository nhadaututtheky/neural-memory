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

    async def process_sync_response(self, response: SyncResponse) -> dict[str, Any]:
        """Process a sync response from the hub — apply remote changes locally."""
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

        # Mark local changes as synced
        if response.hub_sequence > 0:
            await self._storage.mark_synced(response.hub_sequence)
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
        # using the device's preferred strategy
        _, conflicts_list = merge_change_lists(
            list(request.changes), remote_changes, request.strategy
        )

        # Record and apply incoming changes from the device
        for change in request.changes:
            # Always record in hub's change log first
            await self._storage.record_change(
                entity_type=change.entity_type,
                entity_id=change.entity_id,
                operation=change.operation,
                device_id=change.device_id,
                payload=change.payload,
            )
            try:
                await self._apply_remote_change(change)
            except Exception:
                logger.warning(
                    "Hub failed to apply change: %s %s",
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

                    # Fetch entities for this bucket
                    bucket_key = prefix.split("/")[-1] if "/" in prefix else ""
                    entities = await self._fetch_bucket_entities(entity_type, bucket_key)
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

        for diff in response.diffs:
            remote_ids = set(diff.entity_ids)
            remote_entities = {e["id"]: e for e in diff.entities}

            # Get local IDs for this bucket
            bucket_key = diff.prefix.split("/")[-1] if "/" in diff.prefix else ""
            local_entities = await self._fetch_bucket_entities(diff.entity_type, bucket_key)
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

            # Deletes: local has, remote doesn't
            for eid in local_ids - remote_ids:
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

        # Update sync watermark
        if response.hub_sequence > 0:
            await self._storage.mark_synced(response.hub_sequence)
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
        bucket_key: str,
    ) -> list[dict[str, Any]]:
        """Fetch all entities whose ID starts with ``bucket_key`` prefix.

        Returns dicts suitable for sync payloads.
        """
        entities: list[dict[str, Any]] = []

        if entity_type == "neuron":
            neurons = await self._storage.find_neurons(limit=10000)
            for n in neurons:
                nid = n.id[:2].lower() if len(n.id) >= 2 else n.id.lower().ljust(2, "0")
                if nid == bucket_key:
                    entities.append(
                        {
                            "id": n.id,
                            "type": n.type.value if hasattr(n.type, "value") else str(n.type),
                            "content": n.content,
                            "content_hash": n.content_hash,
                            "created_at": n.created_at.isoformat() if n.created_at else "",
                            "updated_at": getattr(n, "updated_at", n.created_at),
                            "metadata": n.metadata or {},
                        }
                    )
                    if hasattr(entities[-1]["updated_at"], "isoformat"):
                        entities[-1]["updated_at"] = entities[-1]["updated_at"].isoformat()

        elif entity_type == "synapse":
            synapses = await self._storage.get_synapses()
            for s in synapses:
                sid = s.id[:2].lower() if len(s.id) >= 2 else s.id.lower().ljust(2, "0")
                if sid == bucket_key:
                    entities.append(
                        {
                            "id": s.id,
                            "source_id": s.source_id,
                            "target_id": s.target_id,
                            "type": s.type.value if hasattr(s.type, "value") else str(s.type),
                            "weight": s.weight,
                            "direction": s.direction.value
                            if hasattr(s.direction, "value")
                            else str(s.direction),
                            "content_hash": getattr(s, "content_hash", 0),
                            "reinforced_count": s.reinforced_count,
                            "created_at": s.created_at.isoformat() if s.created_at else "",
                            "metadata": s.metadata or {},
                        }
                    )

        elif entity_type == "fiber":
            fibers = await self._storage.find_fibers(limit=10000)
            for f in fibers:
                fid = f.id[:2].lower() if len(f.id) >= 2 else f.id.lower().ljust(2, "0")
                if fid == bucket_key:
                    entities.append(
                        {
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
                    )

        return entities

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
