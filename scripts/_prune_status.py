#!/usr/bin/env python3
"""Shared exclusion-marker logic for the `prune-*.py` maintenance scripts.

WHY THIS MODULE EXISTS
----------------------
Hiding a neuron from recall means setting ``metadata._status``. It does NOT
mean setting the ``lifecycle_state`` column, for two independent reasons:

1. No retrieval path reads that column. ``row_to_neuron`` ignores it entirely;
   visibility is gated by ``Neuron.status`` (from ``metadata._status``) via
   ``_filter_by_status`` in ``engine/retrieval.py``, which default-allows only
   ``'active'``.
2. The column is derived, not user-settable. The ``lifecycle`` consolidation
   strategy recomputes it from heat and age for EVERY neuron on every run
   (``engine/consolidation.py``), so any value written here is overwritten by
   the next consolidation pass.

Both prune scripts originally wrote only ``lifecycle_state``, so everything
they "pruned" stayed fully recallable — and on a real brain the marker had
already been erased by consolidation. The logic lived in two copies, which is
why the same bug existed twice; it lives here now so a third script cannot
repeat it.

See issue #195.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

# Statuses that already hide a neuron from default recall. A row in one of
# these is left alone, which also makes re-running a prune idempotent.
ALREADY_HIDDEN = ("expired", "superseded")

# Predicate for "this row is still visible to recall". Callers AND their own
# match condition onto it. Deliberately keyed on `_status`, not on
# `lifecycle_state` — re-running therefore also repairs rows that an older
# version of a script marked `stale` without setting `_status`.
STILL_VISIBLE_SQL = (
    "COALESCE(json_extract(metadata, '$._status'), 'active') NOT IN ('expired', 'superseded')"
)

# Sets the marker recall actually reads, stashing the prior value so a restore
# is faithful rather than assuming everything was 'active'. `lifecycle_state`
# is still written for continuity with earlier backups; it carries no meaning.
MARK_PRUNED_SET_SQL = (
    "lifecycle_state = 'stale', "
    "metadata = json_set(COALESCE(metadata, '{}'), "
    "    '$.pruned_reason', ?, "
    "    '$._prev_status', COALESCE(json_extract(metadata, '$._status'), 'active'), "
    "    '$._status', 'expired')"
)

# Restores `_status` from the stash, then drops both bookkeeping keys.
RESTORE_SET_SQL = (
    "lifecycle_state = NULL, "
    "metadata = json_remove("
    "    json_set(COALESCE(metadata, '{}'), "
    "        '$._status', COALESCE(json_extract(metadata, '$._prev_status'), 'active')), "
    "    '$.pruned_reason', '$._prev_status')"
)


def make_stdout_safe() -> None:
    """Stop unencodable characters in previewed content from killing the run.

    These scripts echo neuron content to stdout. On a Windows console defaulting
    to cp1252 that raises UnicodeEncodeError on the first Vietnamese character —
    which crashed a real run mid-brain, after committing some brains but not
    others. Replacing the offending characters keeps the report readable and the
    run atomic-ish; it does not touch what is written to the database.
    """
    stream = sys.stdout
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")  # type: ignore[union-attr]


def get_brain_base() -> Path:
    """Resolve the brain storage directory from BRAIN_PATH or the default."""
    env_path = os.environ.get("BRAIN_PATH", "")
    if env_path:
        return Path(env_path)
    return Path.home() / ".neuralmemory"


def find_brain_dbs() -> list[Path]:
    """Find every brain database under the storage directory."""
    candidates: list[Path] = []
    base = get_brain_base()

    main = base / "brain.db"
    if main.exists():
        candidates.append(main)

    brains_dir = base / "brains"
    if brains_dir.exists():
        candidates.extend(sorted(brains_dir.glob("*.db")))

    for f in sorted(base.glob("*.db")):
        if f not in candidates:
            candidates.append(f)

    return candidates


def has_neuron_columns(cur: sqlite3.Cursor, *columns: str) -> bool:
    """True if a `neurons` table exists with all the given columns.

    Older or alternate-schema brains are skipped rather than crashing the run.
    """
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='neurons'")
    if cur.fetchone() is None:
        return False
    present = {r[1] for r in cur.execute("PRAGMA table_info(neurons)")}
    return set(columns).issubset(present)


def restore_by_reason(
    cur: sqlite3.Cursor,
    neuron_ids: list[str],
    pruned_reason: str,
) -> int:
    """Un-hide neurons this script pruned. Returns the number restored.

    Only rows still carrying ``pruned_reason`` are touched, so a row that some
    other process has since modified is left alone.
    """
    if not neuron_ids:
        return 0

    restorable = []
    for nid in neuron_ids:
        cur.execute(
            "SELECT json_extract(metadata, '$.pruned_reason') FROM neurons WHERE id = ?",
            (nid,),
        )
        row = cur.fetchone()
        if row and row[0] == pruned_reason:
            restorable.append(nid)

    if not restorable:
        return 0

    placeholders = ",".join("?" for _ in restorable)
    cur.execute(
        f"UPDATE neurons SET {RESTORE_SET_SQL} WHERE id IN ({placeholders})",  # noqa: S608 — static SET fragment, '?' params
        restorable,
    )
    return cur.rowcount
