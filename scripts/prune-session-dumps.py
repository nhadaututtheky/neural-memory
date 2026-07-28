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

Dump text reaches recall through THREE independent surfaces, and all three
have to be cleared — clearing fewer leaves the transcript readable:

  1. ``neurons.content`` — matched on the session-flush signatures in
     ``SESSION_PATTERNS``, anywhere in the text rather than only as a prefix.
     Matches are marked with ``metadata._status = 'expired'``, which is the
     marker retrieval actually gates on (``_filter_by_status`` default-allows
     only ``active``). ``lifecycle_state`` is written too but means nothing —
     see ``scripts/_prune_status.py``. The prior ``_status`` is stashed in
     ``metadata._prev_status`` so ``--unprune`` restores it faithfully.
  2. ``fibers.summary`` — read directly by the fiber-summary retrieval tier,
     which consults only the fiber's *anchor* neuron for status. A dump fiber
     anchored on an ordinary concept therefore leaks regardless of step 1.
  3. ``fibers.essence`` — a second, separate text field on the same row.

Nothing is deleted: neurons are hidden, fiber text is blanked, and both are
saved to the JSON backup for ``--unprune``.

Usage:
  python scripts/prune-session-dumps.py            # dry-run (default)
  python scripts/prune-session-dumps.py --execute  # apply pruning (writes backup)
  python scripts/prune-session-dumps.py --dump      # save affected to JSON only
  python scripts/prune-session-dumps.py --unprune prune-session-dump.json  # restore

Supports BRAIN_PATH env var to override the ~/.neuralmemory base directory.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from _prune_status import (
    MARK_PRUNED_SET_SQL,
    STILL_VISIBLE_SQL,
    find_brain_dbs,
    has_neuron_columns,
    make_stdout_safe,
    restore_by_reason,
)

BACKUP_FILE = "prune-session-dump.json"
PRUNED_REASON = "session_transcript_dump"

# ── Match patterns ──────────────────────────────────────────
# Signatures of the session-flush template. LIKE treats the brackets literally
# (only % and _ are wildcards), so "[SESSION]" is matched as text.
#
# Matching the marker ANYWHERE, not just as a prefix, is deliberate: concept
# extraction re-emitted fragments of these transcripts as their own neurons,
# and those start with things like "Duration = 74s" or "User", so a prefix rule
# missed them entirely.
#
# Length is NOT part of the signature, and must not be: the longest neurons in
# a real brain turned out to be imported Vietnamese legal texts and company tax
# records. A size threshold would have destroyed them. These three markers hit
# 50 dump fragments and 0 of those documents.
SESSION_PATTERNS = (
    "%[SESSION]%",
    "%Key exchanges:%",
    "%SESSION END%",
)

_CONTENT_MATCH = " OR ".join("content LIKE ?" for _ in SESSION_PATTERNS)

# A dump still surfaces in recall unless metadata._status hides it — that is the
# only marker retrieval reads. Match on that (NOT on lifecycle_state) so re-running
# the script also repairs rows an older version marked 'stale' without setting
# _status (they were still fully recallable).
NEEDS_PRUNE_WHERE = f"({_CONTENT_MATCH}) AND {STILL_VISIBLE_SQL}"

# ── Helpers ─────────────────────────────────────────────────


def _fibers_prunable(cur: sqlite3.Cursor) -> bool:
    """True if this DB has a `fibers` table with both text columns we clear."""
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fibers'")
    if cur.fetchone() is None:
        return False
    cols = {r[1] for r in cur.execute("PRAGMA table_info(fibers)")}
    return {"summary", "essence"}.issubset(cols)


# Fibers carry the dump text in TWO columns and both reach recall independently.
_FIBER_MATCH = {
    col: " OR ".join(f"{col} LIKE ?" for _ in SESSION_PATTERNS) for col in ("summary", "essence")
}


def count_dump_fibers(cur: sqlite3.Cursor) -> int:
    """Count fibers carrying session-dump text in `summary` or `essence`.

    Neuron-level pruning does not cover these. The fiber-summary retrieval tier
    reads ``fibers.summary`` directly, and ``_filter_fibers_by_status`` only
    consults the *anchor* neuron, so a dump fiber anchored on an ordinary
    concept neuron keeps leaking the transcript. ``essence`` is a second,
    separate surface — blanking only the summary leaves the dump readable
    through the essence field, which is exactly what happened the first time.
    """
    cur.execute(
        f"SELECT COUNT(*) FROM fibers WHERE ({_FIBER_MATCH['summary']}) "  # noqa: S608 — static fragments, '?' params
        f"OR ({_FIBER_MATCH['essence']})",
        (*SESSION_PATTERNS, *SESSION_PATTERNS),
    )
    result = cur.fetchone()
    return result[0] if result else 0


def prune_fibers(cur: sqlite3.Cursor) -> tuple[int, int]:
    """Blank dump text from fiber `summary` and `essence`.

    Returns ``(summaries_blanked, essences_blanked)``.

    The fiber itself is kept — it still links real neurons; only the polluted
    text goes. Retrieval skips fibers with an empty summary
    (``_try_fiber_summary_tier``: ``if not summary: continue``), and the
    ``fibers_au`` trigger keeps the ``fibers_fts`` index in sync automatically.
    Originals go to the JSON backup for ``--unprune``.
    """
    cur.execute(
        f"UPDATE fibers SET summary = NULL WHERE {_FIBER_MATCH['summary']}",  # noqa: S608 — static fragment, '?' params
        SESSION_PATTERNS,
    )
    summaries = cur.rowcount
    cur.execute(
        f"UPDATE fibers SET essence = NULL WHERE {_FIBER_MATCH['essence']}",  # noqa: S608 — static fragment, '?' params
        SESSION_PATTERNS,
    )
    return summaries, cur.rowcount


def count_session_dumps(cur: sqlite3.Cursor) -> int:
    """Count session-dump neurons still visible to recall."""
    cur.execute(
        f"SELECT COUNT(*) FROM neurons WHERE {NEEDS_PRUNE_WHERE}",  # noqa: S608 — static WHERE fragment, '?' params
        SESSION_PATTERNS,
    )
    result = cur.fetchone()
    return result[0] if result else 0


def report_sample(cur: sqlite3.Cursor, limit: int = 8) -> list[tuple]:
    """Show a sample of session-dump neurons (type + one-line content preview)."""
    cur.execute(
        "SELECT type, substr(replace(content, char(10), ' '), 1, 70) AS preview "  # noqa: S608 — static WHERE fragment; LIMIT is an internal int
        f"FROM neurons WHERE {NEEDS_PRUNE_WHERE} LIMIT {int(limit)}",
        SESSION_PATTERNS,
    )
    return cur.fetchall()


def dump_affected(dbs: list[Path]) -> list[dict]:
    """Dump affected neuron and fiber rows for rollback.

    Records carry a ``kind`` discriminator (``"neuron"`` / ``"fiber"``). Backups
    written by earlier versions have no ``kind`` key; ``unprune`` treats those as
    neurons for backward compatibility.
    """
    affected: list[dict] = []
    for db_path in dbs:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        if has_neuron_columns(cur, "content", "metadata"):
            cur.execute(
                f"SELECT id, type, content, lifecycle_state FROM neurons WHERE {NEEDS_PRUNE_WHERE}",  # noqa: S608 — static WHERE fragment, '?' params
                SESSION_PATTERNS,
            )
            for row in cur.fetchall():
                affected.append(
                    {
                        "kind": "neuron",
                        "brain": str(db_path),
                        "neuron_id": row[0],
                        "type": row[1],
                        "content": row[2],
                        "lifecycle_state": row[3],
                    }
                )
        if _fibers_prunable(cur):
            cur.execute(
                f"SELECT id, summary, essence FROM fibers "  # noqa: S608 — static fragments, '?' params
                f"WHERE ({_FIBER_MATCH['summary']}) OR ({_FIBER_MATCH['essence']})",
                (*SESSION_PATTERNS, *SESSION_PATTERNS),
            )
            for row in cur.fetchall():
                affected.append(
                    {
                        "kind": "fiber",
                        "brain": str(db_path),
                        "fiber_id": row[0],
                        "summary": row[1],
                        "essence": row[2],
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
        f"UPDATE neurons SET {MARK_PRUNED_SET_SQL} WHERE {NEEDS_PRUNE_WHERE}",  # noqa: S608 — static fragments, '?' params
        (PRUNED_REASON, *SESSION_PATTERNS),
    )
    return cur.rowcount


def unprune(dump_path: str) -> int:
    """Restore neurons and fiber summaries from a previously saved dump file."""
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

        # Fiber summaries: restore only where the summary is still blank, so a
        # summary rewritten since the backup is never silently clobbered.
        fiber_restored = 0
        for r in brain_records:
            if r.get("kind") != "fiber":
                continue
            # Restore each column only where it is still blank, so text
            # rewritten since the backup is never clobbered. Backups predating
            # essence support simply have no "essence" key.
            cur.execute(
                "UPDATE fibers SET summary = ? WHERE id = ? AND summary IS NULL",
                (r["summary"], r["fiber_id"]),
            )
            fiber_restored += cur.rowcount
            if r.get("essence") is not None:
                cur.execute(
                    "UPDATE fibers SET essence = ? WHERE id = ? AND essence IS NULL",
                    (r["essence"], r["fiber_id"]),
                )
                fiber_restored += cur.rowcount
        if fiber_restored:
            conn.commit()
            total += fiber_restored
            print(f"  Restored {fiber_restored} fiber summaries in {Path(brain_path).name}")

        # Records without `kind` come from a pre-fiber-support backup — neurons.
        neuron_ids = [r["neuron_id"] for r in brain_records if r.get("kind", "neuron") == "neuron"]
        restored = restore_by_reason(cur, neuron_ids, PRUNED_REASON)
        if restored:
            conn.commit()
            total += restored
            print(f"  Restored {restored} neurons in {Path(brain_path).name}")
        else:
            print(f"  Nothing pruned by this script to restore in {Path(brain_path).name}")
        conn.close()
    return total


# ── Main ────────────────────────────────────────────────────


def main() -> None:
    make_stdout_safe()
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
    total_fibers = 0
    total_pruned = 0
    total_fibers_pruned = 0
    all_affected: list[dict] = []

    for db_path in dbs:
        name = db_path.name
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        has_neurons = has_neuron_columns(cur, "content", "metadata")
        has_fibers = _fibers_prunable(cur)
        if not has_neurons and not has_fibers:
            conn.close()
            continue

        found = count_session_dumps(cur) if has_neurons else 0
        found_fibers = count_dump_fibers(cur) if has_fibers else 0
        total_found += found
        total_fibers += found_fibers
        if found == 0 and found_fibers == 0:
            conn.close()
            continue

        print(f"\n  [{name}] {found} session-dump neuron(s), {found_fibers} dump fiber(s)")
        if found:
            for typ, preview in report_sample(cur):
                print(f'      {typ:8s} "{preview}"')

        if not dry_run:
            all_affected.extend(dump_affected([db_path]))
            pruned = prune(cur) if has_neurons else 0
            blanked_sum, blanked_ess = prune_fibers(cur) if has_fibers else (0, 0)
            conn.commit()
            total_pruned += pruned
            total_fibers_pruned += blanked_sum + blanked_ess
            print(
                f"    -> Pruned {pruned} neuron(s), blanked "
                f"{blanked_sum} summaries + {blanked_ess} essences"
            )
        else:
            print(f"    -> Would prune {found} neuron(s) + {found_fibers} fiber(s) (use --execute)")
        conn.close()

    if not dry_run and all_affected:
        with open(BACKUP_FILE, "w") as f:
            json.dump(all_affected, f, indent=2)
        print(f"\n  Backup saved to {BACKUP_FILE} (use --unprune to restore)")

    print(f"\n{'=' * 60}")
    if dry_run:
        print(
            f"  Total: {total_found} session-dump neuron(s) + "
            f"{total_fibers} dump fiber(s) across {len(dbs)} DB(s)"
        )
        print("  Run with --execute to apply (writes a rollback backup)")
    else:
        print(
            f"  Pruned: {total_pruned} session-dump neuron(s), "
            f"blanked {total_fibers_pruned} fiber text field(s)"
        )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
