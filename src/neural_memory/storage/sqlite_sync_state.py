"""SQLite mixin for sync state persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.integration.models import SyncState

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class SQLiteSyncStateMixin:
    """Mixin: persist and retrieve sync state for external source integrations."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

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
        conn = self._ensure_conn()
        bid = brain_id or self._get_brain_id()

        async with conn.execute(
            """SELECT source_system, source_collection, last_sync_at,
                      records_imported, last_record_id, metadata
               FROM sync_states
               WHERE brain_id = ? AND source_system = ? AND source_collection = ?""",
            (bid, source, collection),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        last_sync_at = None
        if row["last_sync_at"]:
            try:
                last_sync_at = datetime.fromisoformat(row["last_sync_at"])
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
        conn = self._ensure_conn()
        bid = brain_id or self._get_brain_id()

        last_sync_iso = state.last_sync_at.isoformat() if state.last_sync_at else None
        metadata_json = json.dumps(state.metadata) if state.metadata else "{}"

        await conn.execute(
            """INSERT OR REPLACE INTO sync_states
               (brain_id, source_system, source_collection, last_sync_at,
                records_imported, last_record_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                bid,
                state.source_system,
                state.source_collection,
                last_sync_iso,
                state.records_imported,
                state.last_record_id,
                metadata_json,
            ),
        )
        await conn.commit()
