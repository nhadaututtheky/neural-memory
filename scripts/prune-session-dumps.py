#!/usr/bin/env python3
"""Prune legacy `[SESSION] …` transcript-dump neurons.

Sibling to `prune-noisy-concepts.py`. That script prunes low-signal *concept/
entity* neurons (single-word code identifiers, generic nouns). This one targets a
different, higher-mass noise source that it does NOT cover: whole-session
**transcript dumps** saved as neurons in the `[SESSION] Project … Turns … Cost …
Duration … Key exchanges …` format (produced by older auto-capture / session-flush
paths before the current pattern-based capture).

These dumps are walls of conversational text — they violate the "≤3 sentences, no
transcripts" memory-quality rule, dominate recall's "Related Information" surface,
inflate the neuron count, and drag down the brain's freshness/purity grade. A real
audit (my-brain.v2) found 428 such neurons active in a single brain.

Same safe, reversible mechanism as the concept pruner:
  1. Match neurons whose content starts with the literal ``[SESSION]`` marker.
     (In SQLite ``LIKE`` the brackets are literal — only ``%``/``_`` are wildcards —
     so ``content LIKE '[SESSION]%'`` matches exactly that prefix.)
  2. Mark matches with BOTH exclusion markers:
       - ``metadata._status = 'expired'`` — the marker retrieval actually gates on:
         ``_filter_by_status`` default-allows only ``active`` (NeuronStatus contract,
         see ``core/neuron.py``), so this is what removes dumps from recall.
       - ``lifecycle_state = 'stale'`` — bookkeeping column used by this script's
         own idempotence check, consistent with ``prune-noisy-concepts.py``.
     The original ``_status`` is preserved in ``metadata._prev_status`` so
     ``--unprune`` restores it faithfully. Nothing is deleted.

Usage:
  python scripts/prune-session-dumps.py            # dry-run (default)
  python scripts/prune-session-dumps.py --execute  # apply pruning (writes backup)
  python scripts/prune-session-dumps.py --dump      # save affected to JSON only
  python scripts/prune-session-dumps.py --unprune prune-session-dump.json  # restore

Supports BRAIN_PATH env var to override the ~/.neuralmemory base directory.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

# ── Match pattern ───────────────────────────────────────────
# Neurons whose content begins with this literal marker are session-transcript
# dumps. LIKE treats the brackets literally (only % and _ are wildcards), so this
# is an exact prefix match, not a character class.
SESSION_PREFIX = "[SESSION]%"
BACKUP_FILE = "prune-session-dump.json"
PRUNED_REASON = "session_transcript_dump"

# A dump still surfaces in recall unless metadata._status hides it — that is the
# only marker retrieval reads. Match on that (NOT on lifecycle_state) so re-running
# the script also repairs rows an older version marked 'stale' without setting
# _status (they were still fully recallable).
NEEDS_PRUNE_WHERE = (
    "content LIKE ? "
    "AND COALESCE(json_extract(metadata, '$._status'), 'active') NOT IN ('expired', 'superseded')"
)

# ── Helpers ─────────────────────────────────────────────────


def _get_brain_base() -> Path:
    """Resolve brain storage directory from env var or default."""
    env_path = os.environ.get("BRAIN_PATH", "")
    if env_path:
        return Path(env_path)
    return Path.home() / ".neuralmemory"


def find_brain_dbs() -> list[Path]:
    """Find all brain databases in the neural-memory directory."""
    candidates: list[Path] = []
    base = _get_brain_base()

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


def _prunable(cur: sqlite3.Cursor) -> bool:
    """True only if this DB has a `neurons` table with the columns we touch —
    older/alternate-schema brains may lack `content`/`lifecycle_state`; skip them
    rather than crash."""
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='neurons'")
    if cur.fetchone() is None:
        return False
    cols = {r[1] for r in cur.execute("PRAGMA table_info(neurons)")}
    return {"content", "lifecycle_state"}.issubset(cols)


def count_session_dumps(cur: sqlite3.Cursor) -> int:
    """Count session-dump neurons still visible to recall."""
    cur.execute(
        f"SELECT COUNT(*) FROM neurons WHERE {NEEDS_PRUNE_WHERE}",  # noqa: S608 — static WHERE fragment, '?' params
        (SESSION_PREFIX,),
    )
    result = cur.fetchone()
    return result[0] if result else 0


def report_sample(cur: sqlite3.Cursor, limit: int = 8) -> list[tuple]:
    """Show a sample of session-dump neurons (type + one-line content preview)."""
    cur.execute(
        "SELECT type, substr(replace(content, char(10), ' '), 1, 70) AS preview "  # noqa: S608 — static WHERE fragment; LIMIT is an internal int
        f"FROM neurons WHERE {NEEDS_PRUNE_WHERE} LIMIT {int(limit)}",
        (SESSION_PREFIX,),
    )
    return cur.fetchall()


def dump_affected(dbs: list[Path]) -> list[dict]:
    """Dump affected (brain, neuron_id, content) rows for rollback."""
    affected: list[dict] = []
    for db_path in dbs:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        if not _prunable(cur):
            conn.close()
            continue
        cur.execute(
            f"SELECT id, type, content, lifecycle_state FROM neurons WHERE {NEEDS_PRUNE_WHERE}",  # noqa: S608 — static WHERE fragment, '?' params
            (SESSION_PREFIX,),
        )
        for row in cur.fetchall():
            affected.append(
                {
                    "brain": str(db_path),
                    "neuron_id": row[0],
                    "type": row[1],
                    "content": row[2],
                    "lifecycle_state": row[3],
                }
            )
        conn.close()
    return affected


def prune(cur: sqlite3.Cursor) -> int:
    """Mark active session-dump neurons excluded from recall. Returns affected row count.

    Sets ``metadata._status = 'expired'`` (the value retrieval's
    ``_filter_by_status`` actually gates on) plus the ``lifecycle_state``
    bookkeeping column. The prior ``_status`` is stashed in
    ``metadata._prev_status`` so unprune can restore it exactly.
    """
    cur.execute(
        "UPDATE neurons "  # noqa: S608 — static WHERE fragment, '?' params
        "SET lifecycle_state = 'stale', "
        "metadata = json_set(COALESCE(metadata, '{}'), "
        "    '$.pruned_reason', ?, "
        "    '$._prev_status', COALESCE(json_extract(metadata, '$._status'), 'active'), "
        "    '$._status', 'expired') "
        f"WHERE {NEEDS_PRUNE_WHERE}",
        (PRUNED_REASON, SESSION_PREFIX),
    )
    return cur.rowcount


def unprune(dump_path: str) -> int:
    """Restore neurons from a previously saved dump file."""
    with open(dump_path) as f:
        records = json.load(f)
    if not records:
        print("No records to restore.")
        return 0

    brains: dict[str, list[dict]] = {}
    for r in records:
        brains.setdefault(r["brain"], []).append(r)

    total = 0
    for brain_path, brain_records in brains.items():
        conn = sqlite3.connect(brain_path)
        cur = conn.cursor()
        restorable = []
        for r in brain_records:
            # Restore only rows this script pruned (identified by its own marker),
            # not rows some other process happened to mark stale.
            cur.execute(
                "SELECT json_extract(metadata, '$.pruned_reason') FROM neurons WHERE id = ?",
                (r["neuron_id"],),
            )
            row = cur.fetchone()
            if row and row[0] == PRUNED_REASON:
                restorable.append(r["neuron_id"])
        if restorable:
            placeholders = ",".join("?" for _ in restorable)
            # Restore _status from the stashed _prev_status (default 'active'),
            # then drop the bookkeeping keys this script added.
            cur.execute(
                "UPDATE neurons SET lifecycle_state = NULL, "  # noqa: S608 — interpolation is '?' placeholders only
                "metadata = json_remove("
                "    json_set(COALESCE(metadata, '{}'), "
                "        '$._status', COALESCE(json_extract(metadata, '$._prev_status'), 'active')), "
                "    '$.pruned_reason', '$._prev_status') "
                f"WHERE id IN ({placeholders})",
                restorable,
            )
            conn.commit()
            total += cur.rowcount
            print(f"  Restored {cur.rowcount} neurons in {Path(brain_path).name}")
        else:
            print(f"  No stale neurons to restore in {Path(brain_path).name}")
        conn.close()
    return total


# ── Main ────────────────────────────────────────────────────


def main() -> None:
    args = set(sys.argv[1:])
    dry_run = "--execute" not in args
    do_dump = "--dump" in args
    do_unprune = "--unprune" in args

    dump_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--unprune" and i + 1 < len(sys.argv):
            dump_path = sys.argv[i + 1]

    if do_unprune:
        if not dump_path:
            print("Error: --unprune requires a dump file path")
            print(f"  Usage: python {sys.argv[0]} --unprune {BACKUP_FILE}")
            sys.exit(1)
        print("=" * 60)
        print("  NeuralMemory -- Session-Dump Unprune")
        print(f"  Dump file: {dump_path}")
        print("=" * 60)
        print(f"\nRestored {unprune(dump_path)} neuron(s) total")
        return

    print("=" * 60)
    print("  NeuralMemory -- Session-Transcript-Dump Prune")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)

    dbs = find_brain_dbs()
    print(f"\nFound {len(dbs)} brain database(s).")

    if do_dump:
        affected = dump_affected(dbs)
        if not affected:
            # Don't clobber an existing rollback backup with an empty list
            # (e.g. running --dump after --execute finds nothing active).
            print(f"\nNo active session-dump neurons found; {BACKUP_FILE} not written")
            return
        with open(BACKUP_FILE, "w") as f:
            json.dump(affected, f, indent=2)
        print(f"\nDumped {len(affected)} affected neuron(s) to {BACKUP_FILE}")
        print(f"  Use --unprune {BACKUP_FILE} to restore")
        return

    total_found = 0
    total_pruned = 0
    all_affected: list[dict] = []

    for db_path in dbs:
        name = db_path.name
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        if not _prunable(cur):
            conn.close()
            continue

        found = count_session_dumps(cur)
        total_found += found
        if found == 0:
            conn.close()
            continue

        print(f"\n  [{name}] {found} session-dump neuron(s)")
        for typ, preview in report_sample(cur):
            print(f'      {typ:8s} "{preview}"')

        if not dry_run:
            all_affected.extend(dump_affected([db_path]))
            pruned = prune(cur)
            conn.commit()
            total_pruned += pruned
            print(f"    -> Pruned {pruned}")
        else:
            print(f"    -> Would prune {found} (use --execute)")
        conn.close()

    if not dry_run and all_affected:
        with open(BACKUP_FILE, "w") as f:
            json.dump(all_affected, f, indent=2)
        print(f"\n  Backup saved to {BACKUP_FILE} (use --unprune to restore)")

    print(f"\n{'=' * 60}")
    if dry_run:
        print(f"  Total: {total_found} session-dump neuron(s) across {len(dbs)} DB(s)")
        print("  Run with --execute to apply (writes a rollback backup)")
    else:
        print(f"  Pruned: {total_pruned} session-dump neuron(s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
