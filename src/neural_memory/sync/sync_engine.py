"""Sync engine orchestrator for multi-device incremental sync."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.sync.incremental_merge import merge_change_lists
from neural_memory.sync.protocol import (
    ConflictStrategy,
    SyncChange,
    SyncRequest,
    SyncResponse,
    SyncStatus,
)

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

        # Apply incoming changes from the device
        for change in request.changes:
            try:
                await self._apply_remote_change(change)
                # Record in hub's change log
                await self._storage.record_change(
                    entity_type=change.entity_type,
                    entity_id=change.entity_id,
                    operation=change.operation,
                    device_id=change.device_id,
                    payload=change.payload,
                )
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

    async def _apply_remote_change(self, change: SyncChange) -> None:
        """Apply a single remote change to local storage.

        This is a best-effort application — entities may not exist locally
        for update/delete, and that's OK (eventual consistency).
        """
        # For now, we log the change and record it.
        # Full entity materialization will require reading the payload
        # and calling the appropriate storage methods.
        # This is a stub for the first iteration — the hub just records changes.
        logger.debug(
            "Applied remote change: %s %s %s from device %s",
            change.operation,
            change.entity_type,
            change.entity_id,
            change.device_id,
        )
