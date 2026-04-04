"""Training files operations mixin — dialect-agnostic."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

# Maximum file size to hash (2GB — streaming hash, safe for large files)
_MAX_HASH_SIZE = 2 * 1024 * 1024 * 1024


class TrainingFilesMixin:
    """Mixin providing training file tracking CRUD operations."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def get_training_file_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """Look up a training file record by content hash.

        Returns:
            Dict with file record or None if not found.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT * FROM training_files WHERE brain_id = {d.ph(1)} AND file_hash = {d.ph(2)}",
            [brain_id, file_hash],
        )
        return dict(row) if row else None

    async def upsert_training_file(
        self,
        *,
        file_hash: str,
        file_path: str,
        file_size: int,
        chunks_total: int = 0,
        chunks_completed: int = 0,
        status: str = "pending",
        domain_tag: str = "",
    ) -> str:
        """Create or update a training file record.

        Returns:
            The record ID.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        # Check if record exists
        existing = await self.get_training_file_by_hash(file_hash)
        if existing:
            record_id: str = existing["id"]
            await d.execute(
                f"""UPDATE training_files
                   SET chunks_total = {d.ph(1)}, chunks_completed = {d.ph(2)}, status = {d.ph(3)},
                       trained_at = {d.ph(4)}
                   WHERE id = {d.ph(5)} AND brain_id = {d.ph(6)}""",
                [
                    chunks_total,
                    chunks_completed,
                    status,
                    d.serialize_dt(utcnow()) if status == "completed" else None,
                    record_id,
                    brain_id,
                ],
            )
            return record_id

        record_id = str(uuid4())
        await d.execute(
            f"""INSERT INTO training_files
               (id, brain_id, file_hash, file_path, file_size,
                chunks_total, chunks_completed, status, domain_tag, created_at)
               VALUES ({d.phs(10)})""",
            [
                record_id,
                brain_id,
                file_hash,
                file_path,
                file_size,
                chunks_total,
                chunks_completed,
                status,
                domain_tag,
                d.serialize_dt(utcnow()),
            ],
        )
        return record_id

    async def update_training_file_progress(
        self, record_id: str, chunks_completed: int, status: str = "in_progress"
    ) -> None:
        """Update training file progress (for resume support)."""
        d = self._dialect
        brain_id = self._get_brain_id()

        trained_at = d.serialize_dt(utcnow()) if status == "completed" else None
        await d.execute(
            f"""UPDATE training_files
               SET chunks_completed = {d.ph(1)}, status = {d.ph(2)}, trained_at = COALESCE({d.ph(3)}, trained_at)
               WHERE id = {d.ph(4)} AND brain_id = {d.ph(5)}""",
            [chunks_completed, status, trained_at, record_id, brain_id],
        )

    async def get_training_stats(self) -> dict[str, Any]:
        """Get training file statistics for current brain."""
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT
                 COUNT(*) as total_files,
                 SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                 SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                 SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                 SUM(chunks_completed) as total_chunks
               FROM training_files WHERE brain_id = {d.ph(1)}""",
            [brain_id],
        )
        if row is None:
            return {
                "total_files": 0,
                "completed": 0,
                "in_progress": 0,
                "failed": 0,
                "total_chunks": 0,
            }
        return {
            "total_files": row.get("total_files") or 0,
            "completed": row.get("completed") or 0,
            "in_progress": row.get("in_progress") or 0,
            "failed": row.get("failed") or 0,
            "total_chunks": row.get("total_chunks") or 0,
        }


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file's content.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest of the SHA-256 hash.

    Raises:
        ValueError: If file is too large.
    """
    file_size = file_path.stat().st_size
    if file_size > _MAX_HASH_SIZE:
        raise ValueError(f"File too large to hash: {file_size} bytes (max {_MAX_HASH_SIZE})")

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
