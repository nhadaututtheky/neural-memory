"""Versioning mixin — dialect-agnostic."""

from __future__ import annotations

import base64
import binascii
import json
import zlib
from datetime import datetime

from neural_memory.engine.brain_versioning import BrainVersion
from neural_memory.storage.sql.dialect import Dialect


class VersioningMixin:
    """Mixin providing brain version persistence for SQLStorage."""

    _dialect: Dialect

    async def save_version(
        self,
        brain_id: str,
        version: BrainVersion,
        snapshot_json: str,
    ) -> None:
        """Persist a brain version with its compressed snapshot data."""
        d = self._dialect
        # Compress snapshot for storage efficiency
        compressed = base64.b64encode(zlib.compress(snapshot_json.encode("utf-8"), level=6)).decode(
            "ascii"
        )
        await d.execute(
            f"""INSERT INTO brain_versions
               (id, brain_id, version_name, version_number, description,
                neuron_count, synapse_count, fiber_count, snapshot_hash,
                snapshot_data, created_at, metadata)
               VALUES ({d.phs(12)})""",
            [
                version.id,
                brain_id,
                version.version_name,
                version.version_number,
                version.description,
                version.neuron_count,
                version.synapse_count,
                version.fiber_count,
                version.snapshot_hash,
                compressed,
                d.serialize_dt(version.created_at),
                json.dumps(version.metadata),
            ],
        )

    async def get_version(
        self,
        brain_id: str,
        version_id: str,
    ) -> tuple[BrainVersion, str] | None:
        """Get a version and its snapshot JSON by ID."""
        d = self._dialect

        row = await d.fetch_one(
            f"SELECT * FROM brain_versions WHERE brain_id = {d.ph(1)} AND id = {d.ph(2)}",
            [brain_id, version_id],
        )

        if row is None:
            return None

        version = _row_to_version(row)
        raw_data = row["snapshot_data"]
        # Decompress: try zlib first, fall back to raw JSON for legacy data
        snapshot_json = _decompress_snapshot(raw_data)
        return version, snapshot_json

    async def list_versions(
        self,
        brain_id: str,
        limit: int = 20,
    ) -> list[BrainVersion]:
        """List versions for a brain, most recent first."""
        limit = min(limit, 100)
        d = self._dialect

        rows = await d.fetch_all(
            f"""SELECT * FROM brain_versions
               WHERE brain_id = {d.ph(1)}
               ORDER BY version_number DESC
               LIMIT {d.ph(2)}""",
            [brain_id, limit],
        )

        return [_row_to_version(r) for r in rows]

    async def get_next_version_number(self, brain_id: str) -> int:
        """Get the next auto-incrementing version number for a brain."""
        d = self._dialect

        row = await d.fetch_one(
            f"SELECT MAX(version_number) as max_num FROM brain_versions WHERE brain_id = {d.ph(1)}",
            [brain_id],
        )

        if row is None or row["max_num"] is None:
            return 1
        return int(row["max_num"]) + 1

    async def delete_version(self, brain_id: str, version_id: str) -> bool:
        """Delete a specific version."""
        d = self._dialect

        count = await d.execute_count(
            f"DELETE FROM brain_versions WHERE brain_id = {d.ph(1)} AND id = {d.ph(2)}",
            [brain_id, version_id],
        )
        return count > 0


def _decompress_snapshot(raw_data: object) -> str:
    """Decompress snapshot data, with fallback for uncompressed legacy data."""
    raw_str = str(raw_data)
    try:
        compressed_bytes = base64.b64decode(raw_str)
        return zlib.decompress(compressed_bytes).decode("utf-8")
    except (zlib.error, binascii.Error):
        # Legacy uncompressed data - return as-is
        return raw_str


def _row_to_version(row: dict[str, object]) -> BrainVersion:
    """Convert a database row dict to a BrainVersion."""
    metadata_raw = row.get("metadata")
    metadata = json.loads(str(metadata_raw)) if metadata_raw else {}

    created = row["created_at"]
    created_dt = created if isinstance(created, datetime) else datetime.fromisoformat(str(created))

    return BrainVersion(
        id=str(row["id"]),
        brain_id=str(row["brain_id"]),
        version_name=str(row["version_name"]),
        version_number=int(str(row["version_number"])),
        description=str(row["description"] or ""),
        neuron_count=int(str(row["neuron_count"])),
        synapse_count=int(str(row["synapse_count"])),
        fiber_count=int(str(row["fiber_count"])),
        snapshot_hash=str(row["snapshot_hash"]),
        created_at=created_dt,
        metadata=metadata,
    )
