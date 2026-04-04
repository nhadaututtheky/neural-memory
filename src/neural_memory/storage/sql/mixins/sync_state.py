"""Sync state persistence mixin — dialect-agnostic."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from neural_memory.integration.models import SyncState
from neural_memory.storage.sql.dialect import Dialect

logger = logging.getLogger(__name__)


class SyncStateMixin:
    """Mixin: persist and retrieve sync state for external source integrations."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def get_sync_state(
        self, source: str, collection: str, brain_id: str | None = None
    ) -> SyncState | None:
        """Load persisted sync state for a source/collection pair.

        Args:
            source: Source system name (e.g. "mem0")
            collection: Source collection name
            brain_id: Brain ID (uses current brain if None)

        Returns:
            SyncState if found, None otherwise
        """
        d = self._dialect
        bid = brain_id or self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT source_system, source_collection, last_sync_at,
                      records_imported, last_record_id, metadata
               FROM sync_states
               WHERE brain_id = {d.ph(1)} AND source_system = {d.ph(2)} AND source_collection = {d.ph(3)}""",
            [bid, source, collection],
        )

        if row is None:
            return None

        last_sync_at = None
        if row["last_sync_at"]:
            try:
                val = row["last_sync_at"]
                last_sync_at = (
                    val if isinstance(val, datetime) else datetime.fromisoformat(str(val))
                )
            except (ValueError, TypeError):
                logger.warning(
                    "Corrupt last_sync_at in sync_states for %s/%s: %r",
                    source,
                    collection,
                    row["last_sync_at"],
                )

        metadata: dict[str, Any] = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Corrupt metadata JSON in sync_states for %s/%s",
                    source,
                    collection,
                )

        return SyncState(
            source_system=row["source_system"],
            source_collection=row["source_collection"],
            last_sync_at=last_sync_at,
            records_imported=row["records_imported"] or 0,
            last_record_id=row["last_record_id"],
            metadata=metadata,
        )

    async def save_sync_state(self, state: SyncState, brain_id: str | None = None) -> None:
        """Persist sync state for a source/collection pair.

        Uses INSERT OR REPLACE for upsert semantics.

        Args:
            state: The SyncState to persist
            brain_id: Brain ID (uses current brain if None)
        """
        d = self._dialect
        bid = brain_id or self._get_brain_id()

        last_sync_iso = d.serialize_dt(state.last_sync_at)
        metadata_json = json.dumps(state.metadata) if state.metadata else "{}"

        upsert_sql = d.upsert_sql(
            "sync_states",
            [
                "brain_id",
                "source_system",
                "source_collection",
                "last_sync_at",
                "records_imported",
                "last_record_id",
                "metadata",
            ],
            ["brain_id", "source_system", "source_collection"],
            ["last_sync_at", "records_imported", "last_record_id", "metadata"],
        )
        await d.execute(
            upsert_sql,
            [
                bid,
                state.source_system,
                state.source_collection,
                last_sync_iso,
                state.records_imported,
                state.last_record_id,
                metadata_json,
            ],
        )
