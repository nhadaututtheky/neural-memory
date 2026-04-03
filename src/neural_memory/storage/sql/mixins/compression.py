"""Compression backup storage mixin — dialect-agnostic."""

from __future__ import annotations

import logging
from typing import Any

from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class CompressionMixin:
    """Mixin providing compression backup CRUD for SQLStorage.

    Saves and retrieves pre-compression snapshots of neuron content so that
    tier-1 and tier-2 compressions can be reversed.  Tier-3 and tier-4
    compressions are irreversible and therefore do not create backups.
    """

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_compression_backup(
        self,
        fiber_id: str,
        original_content: str,
        compression_tier: int,
        original_token_count: int,
        compressed_token_count: int,
    ) -> None:
        """Upsert a compression backup for a fiber.

        If a backup already exists for *fiber_id* it is replaced so that the
        most recent pre-compression snapshot is always available.

        Args:
            fiber_id: The fiber whose content is being backed up.
            original_content: The full original text before compression.
            compression_tier: The tier to which the fiber was compressed.
            original_token_count: Approximate token count before compression.
            compressed_token_count: Approximate token count after compression.
        """
        d = self._dialect
        brain_id = self._get_brain_id()
        now = d.serialize_dt(utcnow())

        await d.execute(
            f"""INSERT INTO compression_backups
                (fiber_id, brain_id, original_content, compression_tier,
                 compressed_at, original_token_count, compressed_token_count)
                VALUES ({d.phs(7)})
                ON CONFLICT (brain_id, fiber_id) DO UPDATE SET
                  original_content       = excluded.original_content,
                  compression_tier       = excluded.compression_tier,
                  compressed_at          = excluded.compressed_at,
                  original_token_count   = excluded.original_token_count,
                  compressed_token_count = excluded.compressed_token_count""",
            [
                fiber_id,
                brain_id,
                original_content,
                compression_tier,
                now,
                original_token_count,
                compressed_token_count,
            ],
        )

    async def get_compression_backup(self, fiber_id: str) -> dict[str, Any] | None:
        """Retrieve the compression backup for *fiber_id*, if any.

        Args:
            fiber_id: The fiber ID to look up.

        Returns:
            A dict with keys ``fiber_id``, ``brain_id``, ``original_content``,
            ``compression_tier``, ``compressed_at``, ``original_token_count``,
            ``compressed_token_count``; or ``None`` if no backup exists.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"""SELECT fiber_id, brain_id, original_content, compression_tier,
                      compressed_at, original_token_count, compressed_token_count
               FROM compression_backups
               WHERE brain_id = {d.ph(1)} AND fiber_id = {d.ph(2)}""",
            [brain_id, fiber_id],
        )
        return row if row else None

    async def delete_compression_backup(self, fiber_id: str) -> bool:
        """Delete the compression backup for *fiber_id*.

        Called after a successful decompression to remove the now-stale
        backup row.

        Args:
            fiber_id: The fiber ID whose backup should be removed.

        Returns:
            True if a row was deleted, False if no backup existed.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        count = await d.execute_count(
            f"DELETE FROM compression_backups WHERE brain_id = {d.ph(1)} AND fiber_id = {d.ph(2)}",
            [brain_id, fiber_id],
        )
        return count > 0

    async def get_compression_stats(self) -> dict[str, Any]:
        """Return aggregate compression statistics for the current brain.

        The result includes:
        - ``total_backups``: total number of saved backups.
        - ``by_tier``: dict mapping tier integer to backup count.
        - ``total_tokens_saved``: sum of (original - compressed) token counts.

        Returns:
            Dict with compression statistics.
        """
        d = self._dialect
        brain_id = self._get_brain_id()

        # Per-tier counts and token aggregates.
        rows = await d.fetch_all(
            f"""SELECT
                   compression_tier,
                   COUNT(*) AS backup_count,
                   SUM(original_token_count)   AS total_original,
                   SUM(compressed_token_count) AS total_compressed
               FROM compression_backups
               WHERE brain_id = {d.ph(1)}
               GROUP BY compression_tier
               ORDER BY compression_tier""",
            [brain_id],
        )

        total_backups = 0
        total_tokens_saved = 0
        by_tier: dict[int, int] = {}

        for row in rows:
            tier: int = int(row["compression_tier"])
            count: int = int(row["backup_count"])
            original: int = int(row["total_original"] or 0)
            compressed: int = int(row["total_compressed"] or 0)

            by_tier[tier] = count
            total_backups += count
            total_tokens_saved += max(0, original - compressed)

        return {
            "total_backups": total_backups,
            "by_tier": by_tier,
            "total_tokens_saved": total_tokens_saved,
        }
