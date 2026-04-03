"""Source registry operations mixin — dialect-agnostic."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from neural_memory.core.source import Source, SourceStatus, SourceType
from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _row_to_source(row: dict[str, Any]) -> Source:
    """Convert a database row dict to a Source dataclass."""

    def _parse_dt(val: object) -> datetime | None:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
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


class SourcesMixin:
    """Mixin providing source registry CRUD for SQLStorage."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_source(self, source: Source) -> str:
        """Insert a source record. Returns the source ID."""
        d = self._dialect

        await d.execute(
            f"""INSERT INTO sources
                (id, brain_id, name, source_type, version, effective_date,
                 expires_at, status, file_hash, metadata, created_at, updated_at)
                VALUES ({d.phs(12)})""",
            [
                source.id,
                source.brain_id,
                source.name,
                source.source_type.value,
                source.version,
                d.serialize_dt(source.effective_date),
                d.serialize_dt(source.expires_at),
                source.status.value,
                source.file_hash,
                json.dumps(source.metadata),
                d.serialize_dt(source.created_at),
                d.serialize_dt(source.updated_at),
            ],
        )
        return source.id

    async def get_source(self, source_id: str) -> Source | None:
        """Get a source by ID within the current brain."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT * FROM sources WHERE brain_id = {d.ph(1)} AND id = {d.ph(2)}",
            [brain_id, source_id],
        )
        if row is None:
            return None
        return _row_to_source(row)

    async def list_sources(
        self,
        source_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Source]:
        """List sources for the current brain, with optional filters."""
        d = self._dialect
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)

        conditions = [f"brain_id = {d.ph(1)}"]
        params: list[Any] = [brain_id]

        if source_type is not None:
            conditions.append(f"source_type = {d.ph(len(params) + 1)}")
            params.append(source_type)
        if status is not None:
            conditions.append(f"status = {d.ph(len(params) + 1)}")
            params.append(status)

        where = " AND ".join(conditions)
        params.append(limit)

        rows = await d.fetch_all(
            f"SELECT * FROM sources WHERE {where} ORDER BY created_at DESC LIMIT {d.ph(len(params))}",
            params,
        )
        return [_row_to_source(r) for r in rows]

    async def update_source(
        self,
        source_id: str,
        status: str | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a source. Returns True if the row was modified."""
        d = self._dialect
        brain_id = self._get_brain_id()

        sets: list[str] = [f"updated_at = {d.ph(1)}"]
        params: list[Any] = [d.serialize_dt(utcnow())]

        if status is not None:
            try:
                SourceStatus(status)
            except ValueError:
                raise ValueError(
                    f"Invalid status: {status!r}. Must be one of {[s.value for s in SourceStatus]}"
                )
            sets.append(f"status = {d.ph(len(params) + 1)}")
            params.append(status)
        if version is not None:
            sets.append(f"version = {d.ph(len(params) + 1)}")
            params.append(version)
        if metadata is not None:
            sets.append(f"metadata = {d.ph(len(params) + 1)}")
            params.append(json.dumps(metadata))

        set_clause = ", ".join(sets)
        params.extend([brain_id, source_id])

        count = await d.execute_count(
            f"UPDATE sources SET {set_clause} WHERE brain_id = {d.ph(len(params) - 1)} AND id = {d.ph(len(params))}",
            params,
        )
        return count > 0

    async def delete_source(self, source_id: str) -> bool:
        """Delete a source. Returns True if deleted."""
        d = self._dialect
        brain_id = self._get_brain_id()

        count = await d.execute_count(
            f"DELETE FROM sources WHERE brain_id = {d.ph(1)} AND id = {d.ph(2)}",
            [brain_id, source_id],
        )
        return count > 0

    async def count_neurons_for_source(self, source_id: str) -> int:
        """Count neurons linked to a source via SOURCE_OF synapses."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT COUNT(DISTINCT target_id) as cnt FROM synapses
                WHERE brain_id = {d.ph(1)} AND source_id = {d.ph(2)} AND type = 'source_of'""",
            [brain_id, source_id],
        )
        return int(row.get("cnt", 0)) if row else 0

    async def find_source_by_name(self, name: str) -> Source | None:
        """Find a source by exact name within the current brain."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT * FROM sources WHERE brain_id = {d.ph(1)} AND name = {d.ph(2)}",
            [brain_id, name],
        )
        if row is None:
            return None
        return _row_to_source(row)
