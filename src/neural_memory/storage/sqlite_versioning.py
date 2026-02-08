"""SQLite versioning mixin — version storage operations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from neural_memory.engine.brain_versioning import BrainVersion

if TYPE_CHECKING:
    import aiosqlite


class SQLiteVersioningMixin:
    """Mixin providing brain version persistence for SQLiteStorage."""

    _conn: aiosqlite.Connection | None

    def _ensure_conn(self) -> aiosqlite.Connection: ...

    async def save_version(
        self,
        brain_id: str,
        version: BrainVersion,
        snapshot_json: str,
    ) -> None:
        """Persist a brain version with its snapshot data."""
        conn = self._ensure_conn()
        await conn.execute(
            """INSERT INTO brain_versions
               (id, brain_id, version_name, version_number, description,
                neuron_count, synapse_count, fiber_count, snapshot_hash,
                snapshot_data, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version.id,
                brain_id,
                version.version_name,
                version.version_number,
                version.description,
                version.neuron_count,
                version.synapse_count,
                version.fiber_count,
                version.snapshot_hash,
                snapshot_json,
                version.created_at.isoformat(),
                json.dumps(version.metadata),
            ),
        )
        await conn.commit()

    async def get_version(
        self,
        brain_id: str,
        version_id: str,
    ) -> tuple[BrainVersion, str] | None:
        """Get a version and its snapshot JSON by ID."""
        conn = self._ensure_conn()
        async with conn.execute(
            "SELECT * FROM brain_versions WHERE brain_id = ? AND id = ?",
            (brain_id, version_id),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        version = _row_to_version(row)
        return version, row["snapshot_data"]

    async def list_versions(
        self,
        brain_id: str,
        limit: int = 20,
    ) -> list[BrainVersion]:
        """List versions for a brain, most recent first."""
        conn = self._ensure_conn()
        async with conn.execute(
            """SELECT * FROM brain_versions
               WHERE brain_id = ?
               ORDER BY version_number DESC
               LIMIT ?""",
            (brain_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        return [_row_to_version(row) for row in rows]

    async def get_next_version_number(self, brain_id: str) -> int:
        """Get the next auto-incrementing version number for a brain."""
        conn = self._ensure_conn()
        async with conn.execute(
            "SELECT MAX(version_number) as max_num FROM brain_versions WHERE brain_id = ?",
            (brain_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None or row["max_num"] is None:
            return 1
        return row["max_num"] + 1

    async def delete_version(self, brain_id: str, version_id: str) -> bool:
        """Delete a specific version."""
        conn = self._ensure_conn()
        cursor = await conn.execute(
            "DELETE FROM brain_versions WHERE brain_id = ? AND id = ?",
            (brain_id, version_id),
        )
        await conn.commit()
        return cursor.rowcount > 0


def _row_to_version(row: aiosqlite.Row) -> BrainVersion:
    """Convert a database row to a BrainVersion."""
    metadata_raw = row["metadata"]
    metadata = json.loads(metadata_raw) if metadata_raw else {}

    return BrainVersion(
        id=row["id"],
        brain_id=row["brain_id"],
        version_name=row["version_name"],
        version_number=row["version_number"],
        description=row["description"] or "",
        neuron_count=row["neuron_count"],
        synapse_count=row["synapse_count"],
        fiber_count=row["fiber_count"],
        snapshot_hash=row["snapshot_hash"],
        created_at=datetime.fromisoformat(row["created_at"]),
        metadata=metadata,
    )
