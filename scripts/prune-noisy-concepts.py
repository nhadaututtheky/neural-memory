#!/usr/bin/env python3
"""One-time prune of noisy concept/entity neurons stored before the #156 filter.

The #156 fix (ExtractConceptNeuronsStep) filters low-signal concepts at encoding
time, but existing noisy neurons remain in the database and surface during recall.
This script:
  1. Expands the noise set to cover common single-word code identifiers and
     generic nouns that carry zero topical signal.
  2. Marks matching neurons as stale (lifecycle_state = 'stale') so the recall
     pipeline naturally skips them.
  3. Reports what was cleaned so we can assess if more filtering is needed.

Usage:
  python scripts/prune-noisy-concepts.py           # dry-run (default)
  python scripts/prune-noisy-concepts.py --execute  # apply pruning
  python scripts/prune-noisy-concepts.py --dump     # save affected to JSON
  python scripts/prune-noisy-concepts.py --unprune prune-dump.json  # restore

Supports BRAIN_PATH env var to override ~/.neuralmemory base directory.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

# ── Noise patterns ──────────────────────────────────────────
# Short generic words that leak through entity extraction and
# produce noise in the "Related Information" recall surface.

SHORT_NOISE: set[str] = {
    # Single-letter / very short (len <= 2) — these are almost never
    # conversation-relevant entities
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "id",
    "Id",
    "ID",
    "IDK",
    "ok",
    "OK",
    "Ok",
    "no",
    "No",
    "NO",
    "go",
    "Go",
    "GO",
    "do",
    "Do",
    "DO",
    "is",
    "Is",
    "IS",
    "it",
    "It",
    "IT",
    "to",
    "To",
    "in",
    "In",
    "IN",
    "on",
    "On",
    "ON",
    "at",
    "At",
    "by",
    "By",
    "or",
    "Or",
    "OR",
    "as",
    "As",
    "AS",
    "if",
    "If",
    "IF",
    "be",
    "Be",
    "up",
    "Up",
    "UP",
    "my",
    "My",
    "re",
    "Re",
    "vs",
    "np",  # numpy alias — code noise
    "os",  # operating system — too generic
    "ws",  # websocket — code noise
    "ai",
    "Ai",
    "AI",
    "ui",
    "Ui",
    "UI",
    "db",
    "Db",
    "DB",
    "io",
    "Io",
    "IO",
    "px",  # pixel unit
    "ms",  # millisecond / Microsoft
    "vm",
    "Vm",
    "VM",
    "pi",
    "Pi",
    "PI",
    "3d",
    "2d",
    "4k",
    "et",  # latin "and" — false positive
    "ad",
    "Ad",
    "AD",  # advertisement / anno domini
    "ex",
    "Ex",  # example
    "pm",
    "Pm",
    "PM",  # project management / time
    "am",
    "Am",
    "AM",
    "us",
    "Us",
    "US",  # pronoun / country
    "we",
    "We",
    "he",
    "He",
    "she",
    "She",
    "hi",
    "Hi",
    "oh",
    "Oh",
    "ah",
    "Ah",
    "ha",
}

# Short code identifiers (len 3-5) that appear frequently as entities
# but carry no conversational signal.
CODE_NOISE: set[str] = {
    # Common programming terms that aren't topics
    "any",
    "Any",
    "all",
    "All",
    "new",
    "New",
    "old",
    "Old",
    "set",
    "Set",
    "get",
    "Get",
    "put",
    "Put",
    "add",
    "Add",
    "del",
    "Del",
    "pop",
    "Pop",
    "has",
    "Has",
    "len",
    "Len",
    "str",
    "Str",
    "int",
    "Int",
    "bool",
    "Bool",
    "dict",
    "Dict",
    "list",
    "List",
    "map",
    "Map",
    "arr",
    "Arr",
    "obj",
    "Obj",
    "key",
    "Key",
    "val",
    "Val",
    "arg",
    "Arg",
    "var",
    "Var",
    "let",
    "Let",
    "def",
    "Def",
    "class",
    "Class",
    "type",
    "Type",
    "enum",
    "Enum",
    "impl",
    "Impl",
    "trait",
    "Trait",
    "path",
    "Path",
    "dir",
    "Dir",
    "file",
    "File",
    "line",
    "Line",
    "rows",
    "cols",
    "col",
    "Col",
    "row",
    "Row",
    "tab",
    "Tab",
    "fmt",
    "Fmt",
    "json",
    "JSON",
    "Json",
    "xml",
    "XML",
    "Xml",
    "yaml",
    "Yaml",
    "YAML",
    "toml",
    "Toml",
    "TOML",
    "csv",
    "CSV",
    "url",
    "URL",
    "Url",
    "uri",
    "URI",
    "Uri",
    "ip",
    "Ip",
    "IP",
    "port",
    "host",
    "node",
    "Node",
    "edge",
    "Edge",
    "tree",
    "Tree",
    "hash",
    "Hash",
    "heap",
    "Heap",
    "stack",
    "Stack",
    "queue",
    "Queue",
    "link",
    "Link",
    "core",
    "Core",
    "main",
    "Main",
    "init",
    "Init",
    "exit",
    "Exit",
    "open",
    "Open",
    "close",
    "Close",
    "read",
    "Read",
    "write",
    "Write",
    "load",
    "Load",
    "save",
    "Save",
    "move",
    "Move",
    "find",
    "Find",
    "sort",
    "Sort",
    "join",
    "Join",
    "split",
    "Split",
    "trim",
    "Trim",
    "fill",
    "Fill",
    "bind",
    "Bind",
    "call",
    "Call",
    "next",
    "Next",
    "last",
    "Last",
    "prev",
    "Prev",
    "math",
    "Math",
    "uuid",
    "Uuid",
    "UUID",
    "uuid4",
    "date",
    "Date",
    "time",
    "Time",
    "utc",
    "UTC",
    "now",
    "Now",
    "sync",
    "Sync",
    "async",
    "Async",
    "await",
    "Await",
    "todo",
    "Todo",
    "TODO",
    "fixme",
    "Fixme",
    "FIXME",
    "hack",
    "Hack",
    "docs",
    "Docs",
    "test",
    "Test",
    "spec",
    "Spec",
    "unit",
    "Unit",
    "bench",
    "Bench",
    "prod",
    "Prod",
    "dev",
    "Dev",
    "staging",
    "build",
    "Build",
    "deploy",
    "Deploy",
    "ship",
    "Ship",
    "log",
    "Log",
    "debug",
    "Debug",
    "trace",
    "Trace",
    "error",
    "Error",
    "warn",
    "fail",
    "pass",
}

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

    # Main brain
    main = base / "brain.db"
    if main.exists():
        candidates.append(main)

    # Named brains
    brains_dir = base / "brains"
    if brains_dir.exists():
        candidates.extend(brains_dir.glob("*.db"))

    # Also check for any .db files directly in the base dir (alternate layouts)
    for f in base.glob("*.db"):
        if f not in candidates:
            candidates.append(f)

    return candidates


def _noise_placeholders() -> tuple[set[str], str]:
    """Build placeholder string for SQL IN clause from noise sets."""
    noise_lower = {w.lower() for w in SHORT_NOISE | CODE_NOISE}
    placeholders = ",".join("?" for _ in noise_lower)
    return noise_lower, placeholders


def count_noisy_neurons(cur: sqlite3.Cursor) -> int:
    """Count neurons matching noise patterns."""
    noise_lower, placeholders = _noise_placeholders()
    sql = (
        "SELECT COUNT(*) FROM neurons "
        "WHERE type IN ('concept', 'entity') "
        f"AND LOWER(content) IN ({placeholders}) "
        "AND (lifecycle_state IS NULL OR lifecycle_state != 'stale')"
    )
    cur.execute(sql, list(noise_lower))
    result = cur.fetchone()
    return result[0] if result else 0


def prune_noisy_neurons(cur: sqlite3.Cursor, dry_run: bool = True) -> int:
    """Mark noisy neurons as stale. Returns count of affected rows."""
    noise_lower, placeholders = _noise_placeholders()

    if dry_run:
        dry_sql = (
            "SELECT COUNT(*) FROM neurons "
            "WHERE type IN ('concept', 'entity') "
            f"AND LOWER(content) IN ({placeholders}) "
            "AND (lifecycle_state IS NULL OR lifecycle_state != 'stale')"
        )
        cur.execute(dry_sql, list(noise_lower))
        result = cur.fetchone()
        return result[0] if result else 0

    empty_obj = "'{}'"
    json_val = "'" + '"noisy_low_signal"' + "'"
    update_sql = (
        "UPDATE neurons "
        "SET lifecycle_state = 'stale', "
        f"metadata = json_set(COALESCE(metadata, {empty_obj}), "
        f"'$.pruned_reason', {json_val}) "
        "WHERE type IN ('concept', 'entity') "
        f"AND LOWER(content) IN ({placeholders}) "
        "AND (lifecycle_state IS NULL OR lifecycle_state != 'stale')"
    )
    cur.execute(update_sql, list(noise_lower))
    return cur.rowcount


def report_noise_sample(cur: sqlite3.Cursor, limit: int = 15) -> list[tuple]:
    """Show a sample of noisy neurons that would be / were pruned."""
    noise_lower, placeholders = _noise_placeholders()
    sample_sql = (
        "SELECT type, content, COUNT(*) as cnt "
        "FROM neurons "
        "WHERE type IN ('concept', 'entity') "
        f"AND LOWER(content) IN ({placeholders}) "
        "AND (lifecycle_state IS NULL OR lifecycle_state != 'stale') "
        "GROUP BY type, content "
        "ORDER BY cnt DESC "
        f"LIMIT {limit}"
    )
    cur.execute(sample_sql, list(noise_lower))
    return cur.fetchall()


def dump_affected_neurons(dbs: list[Path]) -> list[dict]:
    """Dump all affected (brain, neuron_id, content) to a list for rollback."""
    noise_lower = {w.lower() for w in SHORT_NOISE | CODE_NOISE}
    placeholders = ",".join("?" for _ in noise_lower)
    affected: list[dict] = []

    for db_path in dbs:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='neurons'")
        if not cur.fetchone():
            conn.close()
            continue

        cur.execute(
            f"""
            SELECT rowid, id, type, content, lifecycle_state
            FROM neurons
            WHERE type IN ('concept', 'entity')
            AND LOWER(content) IN ({placeholders})
            AND (lifecycle_state IS NULL OR lifecycle_state != 'stale')
            """,
            list(noise_lower),
        )
        for row in cur.fetchall():
            affected.append(
                {
                    "brain": str(db_path),
                    "rowid": row[0],
                    "neuron_id": row[1],
                    "type": row[2],
                    "content": row[3],
                    "lifecycle_state": row[4],
                }
            )
        conn.close()

    return affected


def unprune_neurons(dump_path: str) -> int:
    """Restore neurons from a previously saved dump file."""
    with open(dump_path) as f:
        records = json.load(f)

    if not records:
        print("No records to restore.")
        return 0

    # Group by brain file
    brains: dict[str, list[dict]] = {}
    for r in records:
        brains.setdefault(r["brain"], []).append(r)

    total_restored = 0
    for brain_path, brain_records in brains.items():
        conn = sqlite3.connect(brain_path)
        cur = conn.cursor()
        restorable = []
        for r in brain_records:
            nid = r["neuron_id"]
            cur.execute("SELECT lifecycle_state FROM neurons WHERE id = ?", (nid,))
            row = cur.fetchone()
            if row and row[0] == "stale":
                restorable.append(nid)

        if restorable:
            placeholders = ",".join("?" for _ in restorable)
            cur.execute(
                f"""
                UPDATE neurons
                SET lifecycle_state = NULL,
                    metadata = json_remove(COALESCE(metadata, '{{}}'), '$.pruned_reason')
                WHERE id IN ({placeholders})
                """,
                restorable,
            )
            conn.commit()
            total_restored += cur.rowcount
            print(f"  Restored {cur.rowcount} neurons in {Path(brain_path).name}")
        else:
            print(f"  No stale neurons to restore in {Path(brain_path).name}")
        conn.close()

    return total_restored


# ── Main ────────────────────────────────────────────────────


def main() -> None:
    """Run the prune, showing report with optional --execute to apply."""
    args = set(sys.argv[1:])
    dry_run = "--execute" not in args
    do_dump = "--dump" in args
    do_unprune = "--unprune" in args

    # Extract dump path
    dump_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--unprune" and i + 1 < len(sys.argv):
            dump_path = sys.argv[i + 1]

    # Unprune mode
    if do_unprune and dump_path:
        print("=" * 60)
        print("  NeuralMemory -- Unprune Mode")
        print(f"  Dump file: {dump_path}")
        print("=" * 60)
        restored = unprune_neurons(dump_path)
        print(f"\nRestored {restored} neuron(s) total")
        return

    print("=" * 60)
    print("  NeuralMemory -- Noisy Concept/Entity Prune")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)

    dbs = find_brain_dbs()
    print(f"\nFound {len(dbs)} brain database(s):")

    # Dump-all mode: save affected neurons to JSON for rollback
    if do_dump:
        affected = dump_affected_neurons(dbs)
        dump_file = Path("prune-dump.json")
        with open(dump_file, "w") as f:
            json.dump(affected, f, indent=2)
        print(f"\nDumped {len(affected)} affected neuron(s) to {dump_file}")
        print("  Use --unprune prune-dump.json to restore")
        if not dry_run:
            print("  Note: --dump + --execute saves the backup BEFORE pruning")
        return

    total_noisy = 0
    total_affected: list[dict] = []
    total_pruned = 0

    for db_path in dbs:
        name = db_path.name
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        # Check if table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='neurons'")
        if not cur.fetchone():
            print(f"\n  [{name}] No neurons table -- skipping")
            conn.close()
            continue

        # Count
        noisy = count_noisy_neurons(cur)
        total_noisy += noisy

        if noisy == 0:
            print(f"\n  [{name}] No noisy neurons found +")
            conn.close()
            continue

        # Sample
        sample = report_noise_sample(cur)
        print(f"\n  [{name}] {noisy} noisy neurons found")
        print("    Top offenders:")
        for row in sample[:8]:
            print(f'      {row[0]:10s} "{row[1]:20s}" x {row[2]}')

        if not dry_run:
            # Save backup before pruning
            affected = dump_affected_neurons([db_path])
            total_affected.extend(affected)
            pruned = prune_noisy_neurons(cur, dry_run=False)
            total_pruned += pruned
            conn.commit()
            print(f"    -> Pruned {pruned} neurons")
        else:
            print(f"    -> Would prune {noisy} (use --execute to apply)")

        conn.close()

    # Save backup dump if anything was pruned
    if not dry_run and total_affected:
        dump_file = Path("prune-dump.json")
        with open(dump_file, "w") as f:
            json.dump(total_affected, f, indent=2)
        print(f"\n  Backup saved to {dump_file} (use --unprune to restore)")

    print(f"\n{'=' * 60}")
    if dry_run:
        print(f"  Total: {total_noisy} noisy neurons found across {len(dbs)} DB(s)")
        print("  Run with --execute to apply")
        print("  Run with --dump to save affected list (for rollback)")
    else:
        print(f"  Pruned: {total_pruned} noisy neurons")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
