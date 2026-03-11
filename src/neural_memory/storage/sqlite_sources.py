"""SQLite mixin for source registry operations."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.source import Source, SourceStatus, SourceType
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class SQLiteSourcesMixin:
    """Mixin providing source registry CRUD for SQLiteStorage."""

    # ------------------------------------------------------------------
    # Protocol stubs — satisfied by SQLiteStorage at runtime.
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_source(self, source: Source) -> str:
        """Insert a source record. Returns the source ID."""
        conn = self._ensure_conn()

        await conn.execute(
            """INSERT INTO sources
               (id, brain_id, name, source_type, version, effective_date,
                expires_at, status, file_hash, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source.id,
                source.brain_id,
                source.name,
                source.source_type.value,
                source.version,
                source.effective_date.isoformat() if source.effective_date else None,
                source.expires_at.isoformat() if source.expires_at else None,
                source.status.value,
                source.file_hash,
                json.dumps(source.metadata),
                source.created_at.isoformat(),
                source.updated_at.isoformat(),
            ),
        )
        await conn.commit()
        return source.id

    async def get_source(self, source_id: str) -> Source | None:
        """Get a source by ID within the current brain."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT * FROM sources WHERE brain_id = ? AND id = ?",
            (brain_id, source_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        col_names = [d[0] for d in (cursor.description or [])]
        return _row_to_source(dict(zip(col_names, row, strict=False)))

    async def list_sources(
        self,
        source_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Source]:
        """List sources for the current brain, with optional filters."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)

        conditions = ["brain_id = ?"]
        params: list[Any] = [brain_id]

        if source_type is not None:
            conditions.append("source_type = ?")
            params.append(source_type)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        where = " AND ".join(conditions)
        params.append(limit)

        async with conn.execute(
            f"SELECT * FROM sources WHERE {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ) as cursor:
            rows = await cursor.fetchall()
            col_names = [d[0] for d in (cursor.description or [])]
            return [_row_to_source(dict(zip(col_names, r, strict=False))) for r in rows]

    async def update_source(
        self,
        source_id: str,
        status: str | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a source. Returns True if the row was modified."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        sets: list[str] = ["updated_at = ?"]
        params: list[Any] = [utcnow().isoformat()]

        if status is not None:
            try:
                SourceStatus(status)
            except ValueError:
                raise ValueError(
                    f"Invalid status: {status!r}. Must be one of {[s.value for s in SourceStatus]}"
                )
            sets.append("status = ?")
            params.append(status)
        if version is not None:
            sets.append("version = ?")
            params.append(version)
        if metadata is not None:
            sets.append("metadata = ?")
            params.append(json.dumps(metadata))

        set_clause = ", ".join(sets)
        params.extend([brain_id, source_id])

        cursor = await conn.execute(
            f"UPDATE sources SET {set_clause} WHERE brain_id = ? AND id = ?",
            params,
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def delete_source(self, source_id: str) -> bool:
        """Delete a source. Returns True if deleted."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM sources WHERE brain_id = ? AND id = ?",
            (brain_id, source_id),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def count_neurons_for_source(self, source_id: str) -> int:
        """Count neurons linked to a source via SOURCE_OF synapses."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """SELECT COUNT(DISTINCT target_id) FROM synapses
               WHERE brain_id = ? AND source_id = ? AND type = 'source_of'""",
            (brain_id, source_id),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def find_source_by_name(self, name: str) -> Source | None:
        """Find a source by exact name within the current brain."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT * FROM sources WHERE brain_id = ? AND name = ?",
            (brain_id, name),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        col_names = [d[0] for d in (cursor.description or [])]
        return _row_to_source(dict(zip(col_names, row, strict=False)))


def _row_to_source(row: dict[str, Any]) -> Source:
    """Convert a database row dict to a Source dataclass."""

    def _parse_dt(val: object) -> datetime | None:
        if val is None:
            return None
        return datetime.fromisoformat(str(val))

    raw_metadata = row.get("metadata", "{}")
    try:
        metadata = json.loads(str(raw_metadata)) if raw_metadata else {}
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "_row_to_source: invalid JSON in metadata for source %s, using {}", row.get("id")
        )
        metadata = {}

    raw_source_type = str(row["source_type"])
    try:
        source_type = SourceType(raw_source_type)
    except ValueError:
        logger.warning(
            "_row_to_source: unknown source_type %r for source %s, falling back to DOCUMENT",
            raw_source_type,
            row.get("id"),
        )
        source_type = SourceType.DOCUMENT

    raw_status = str(row["status"])
    try:
        status = SourceStatus(raw_status)
    except ValueError:
        logger.warning(
            "_row_to_source: unknown status %r for source %s, falling back to ACTIVE",
            raw_status,
            row.get("id"),
        )
        status = SourceStatus.ACTIVE

    return Source(
        id=str(row["id"]),
        brain_id=str(row["brain_id"]),
        name=str(row["name"]),
        source_type=source_type,
        version=str(row.get("version") or ""),
        effective_date=_parse_dt(row.get("effective_date")),
        expires_at=_parse_dt(row.get("expires_at")),
        status=status,
        file_hash=str(row.get("file_hash") or ""),
        metadata=metadata,
        created_at=_parse_dt(row.get("created_at")) or utcnow(),
        updated_at=_parse_dt(row.get("updated_at")) or utcnow(),
    )
