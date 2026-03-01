"""SQLite training files operations mixin — tracks trained files for dedup and resume."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

# Maximum file size to hash (50MB safety limit)
_MAX_HASH_SIZE = 50 * 1024 * 1024


class SQLiteTrainingFilesMixin:
    """Mixin providing training file tracking CRUD operations."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def get_training_file_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """Look up a training file record by content hash.

        Returns:
            Dict with file record or None if not found.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM training_files WHERE brain_id = ? AND file_hash = ?",
            (brain_id, file_hash),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)

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
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        # Check if record exists
        existing = await self.get_training_file_by_hash(file_hash)
        if existing:
            record_id: str = existing["id"]
            await conn.execute(
                """UPDATE training_files
                   SET chunks_total = ?, chunks_completed = ?, status = ?,
                       trained_at = ?
                   WHERE id = ? AND brain_id = ?""",
                (
                    chunks_total,
                    chunks_completed,
                    status,
                    utcnow().isoformat() if status == "completed" else None,
                    record_id,
                    brain_id,
                ),
            )
            await conn.commit()
            return record_id

        record_id = str(uuid4())
        await conn.execute(
            """INSERT INTO training_files
               (id, brain_id, file_hash, file_path, file_size,
                chunks_total, chunks_completed, status, domain_tag, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record_id,
                brain_id,
                file_hash,
                file_path,
                file_size,
                chunks_total,
                chunks_completed,
                status,
                domain_tag,
                utcnow().isoformat(),
            ),
        )
        await conn.commit()
        return record_id

    async def update_training_file_progress(
        self, record_id: str, chunks_completed: int, status: str = "in_progress"
    ) -> None:
        """Update training file progress (for resume support)."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        trained_at = utcnow().isoformat() if status == "completed" else None
        await conn.execute(
            """UPDATE training_files
               SET chunks_completed = ?, status = ?, trained_at = COALESCE(?, trained_at)
               WHERE id = ? AND brain_id = ?""",
            (chunks_completed, status, trained_at, record_id, brain_id),
        )
        await conn.commit()

    async def get_training_stats(self) -> dict[str, Any]:
        """Get training file statistics for current brain."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            """SELECT
                 COUNT(*) as total_files,
                 SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                 SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                 SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                 SUM(chunks_completed) as total_chunks
               FROM training_files WHERE brain_id = ?""",
            (brain_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return {"total_files": 0, "completed": 0, "in_progress": 0, "failed": 0, "total_chunks": 0}
            return {
                "total_files": row[0] or 0,
                "completed": row[1] or 0,
                "in_progress": row[2] or 0,
                "failed": row[3] or 0,
                "total_chunks": row[4] or 0,
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
