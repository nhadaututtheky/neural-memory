#!/usr/bin/env python3
"""Sync scattered references (tool count, test count, schema version) across docs.

Derives single source of truth from code, then scans all .md files for stale
references and optionally fixes them.

Usage:
    python scripts/sync_refs.py          # Check mode — report stale references
    python scripts/sync_refs.py --fix    # Fix mode — update stale references in-place
    python scripts/sync_refs.py --json   # Machine-readable output

Designed to be called from pre_ship.py as an automated check.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Files to NEVER touch (historical records, snapshots, migration guides)
SKIP_FILES = {
    "CHANGELOG.md",
    "docs/changelog.md",
    "AUDIT-REPORT.md",
    "docs/guides/schema-v21-migration.md",  # Historical migration — references old schema versions
}

# Paths to skip entirely (not shipped docs)
SKIP_DIRS = {
    ".rune",
    "node_modules",
    ".git",
    "coverage",
    "__pycache__",
}

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
WARN = "\033[93m WARN \033[0m"


@dataclass
class RefFinding:
    """A stale reference found in a file."""

    file: str
    line_num: int
    old_text: str
    new_text: str
    ref_type: str  # "tool_count", "test_count", "schema_version"


@dataclass
class TruthValues:
    """Single source of truth derived from code."""

    mcp_tool_count: int = 0
    test_count_approx: int = 0  # Rounded to nearest 100
    schema_version: int = 0
    version: str = ""
    errors: list[str] = field(default_factory=list)


def derive_truth() -> TruthValues:
    """Derive truth values from code — no guessing, no config files."""
    truth = TruthValues()

    # 1. MCP tool count — count schema definitions
    schema_file = ROOT / "src" / "neural_memory" / "mcp" / "tool_schemas.py"
    if schema_file.exists():
        text = schema_file.read_text(encoding="utf-8")
        # Each tool schema has "name": "nmem_*" — count these (not references in tiers/descriptions)
        truth.mcp_tool_count = len(re.findall(r'"name":\s*"nmem_\w+"', text))
    else:
        truth.errors.append("tool_schemas.py not found")

    # 2. Test count — run pytest --co -q (collect only)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "--co", "-q", "--timeout=10"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        m = re.search(r"(\d+) tests? collected", result.stdout)
        if m:
            raw_count = int(m.group(1))
            # Round down to nearest 100 for "X00+" style
            truth.test_count_approx = (raw_count // 100) * 100
    except (subprocess.TimeoutExpired, FileNotFoundError):
        truth.errors.append("pytest collection failed or timed out")

    # 3. Schema version — read from sqlite_schema.py
    schema_py = ROOT / "src" / "neural_memory" / "storage" / "sqlite_schema.py"
    if schema_py.exists():
        text = schema_py.read_text(encoding="utf-8")
        m = re.search(r"SCHEMA_VERSION\s*=\s*(\d+)", text)
        if m:
            truth.schema_version = int(m.group(1))
    else:
        truth.errors.append("sqlite_schema.py not found")

    # 4. Version — from pyproject.toml
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if m:
            truth.version = m.group(1)
    else:
        truth.errors.append("pyproject.toml not found")

    return truth


def _should_skip(path: Path) -> bool:
    """Check if file should be skipped."""
    rel = path.relative_to(ROOT).as_posix()

    # Skip historical files
    if rel in SKIP_FILES:
        return True

    # Skip non-shipped directories
    for skip_dir in SKIP_DIRS:
        if rel.startswith(skip_dir + "/") or rel == skip_dir:
            return True

    return False


def scan_stale_refs(truth: TruthValues) -> list[RefFinding]:
    """Scan all .md files for stale references."""
    findings: list[RefFinding] = []

    # Build patterns to search for
    patterns: list[tuple[re.Pattern[str], str, str]] = []

    # Tool count patterns — match "N tools" or "N MCP tools" where N != truth
    if truth.mcp_tool_count > 0:
        # Match any number followed by " tools" or " MCP tools" (but not in code blocks)
        patterns.append(
            (
                re.compile(r"\b(\d+)(\s+(?:MCP\s+)?tools)\b"),
                "tool_count",
                str(truth.mcp_tool_count),
            )
        )
        # Also match "other N automatically" pattern (quickstart)
        patterns.append(
            (
                re.compile(r"\b(other\s+)(\d+)(\s+automatically)\b"),
                "tool_count_other",
                str(truth.mcp_tool_count - 3),  # "other X automatically" = total - 3 core
            )
        )

    # Test count patterns — match "N+ tests" or "N tests"
    if truth.test_count_approx > 0:
        patterns.append(
            (
                re.compile(r"\b(\d{4})\+?\s+tests\b"),
                "test_count",
                f"{truth.test_count_approx}+",
            )
        )

    # Schema version in non-historical context — "schema v{N}"
    # Only match in ROADMAP-style "current state" lines, not historical refs
    if truth.schema_version > 0:
        patterns.append(
            (
                re.compile(r"(schema\s+v)(\d+)"),
                "schema_version",
                str(truth.schema_version),
            )
        )

    md_files = sorted(ROOT.rglob("*.md"))

    for md_file in md_files:
        if _should_skip(md_file):
            continue

        try:
            lines = md_file.read_text(encoding="utf-8").split("\n")
        except (OSError, UnicodeDecodeError):
            continue

        rel_path = md_file.relative_to(ROOT).as_posix()

        for line_num, line in enumerate(lines, 1):
            # Skip code blocks
            if line.strip().startswith("```") or line.strip().startswith("`"):
                continue

            for pattern, ref_type, truth_val in patterns:
                for match in pattern.finditer(line):
                    if ref_type == "tool_count":
                        old_num = match.group(1)
                        if old_num != truth_val and int(old_num) != truth.mcp_tool_count:
                            # Only flag if the number looks like a tool count (20-200 range)
                            if 20 <= int(old_num) <= 200:
                                old_text = match.group(0)
                                new_text = old_text.replace(old_num, truth_val, 1)
                                findings.append(
                                    RefFinding(
                                        file=rel_path,
                                        line_num=line_num,
                                        old_text=old_text,
                                        new_text=new_text,
                                        ref_type=ref_type,
                                    )
                                )

                    elif ref_type == "tool_count_other":
                        old_num = match.group(2)
                        if old_num != truth_val:
                            old_text = match.group(0)
                            new_text = f"{match.group(1)}{truth_val}{match.group(3)}"
                            findings.append(
                                RefFinding(
                                    file=rel_path,
                                    line_num=line_num,
                                    old_text=old_text,
                                    new_text=new_text,
                                    ref_type=ref_type,
                                )
                            )

                    elif ref_type == "test_count":
                        old_num = match.group(1)
                        if int(old_num) < truth.test_count_approx:
                            old_text = match.group(0)
                            new_text = f"{truth.test_count_approx}+ tests"
                            findings.append(
                                RefFinding(
                                    file=rel_path,
                                    line_num=line_num,
                                    old_text=old_text,
                                    new_text=new_text,
                                    ref_type=ref_type,
                                )
                            )

                    elif ref_type == "schema_version":
                        old_ver = match.group(2)
                        # Only flag if it looks like a "current state" reference
                        # Skip historical migration references (e.g., "schema v22 → v23")
                        if "→" in line or "->" in line or "migration" in line.lower():
                            continue
                        if int(old_ver) < truth.schema_version:
                            # Only flag significant staleness (not every historical mention)
                            context = line.lower()
                            if any(
                                kw in context
                                for kw in [
                                    "current",
                                    "state",
                                    "version",
                                    "backend",
                                    "persist",
                                ]
                            ):
                                old_text = match.group(0)
                                new_text = f"{match.group(1)}{truth.schema_version}"
                                findings.append(
                                    RefFinding(
                                        file=rel_path,
                                        line_num=line_num,
                                        old_text=old_text,
                                        new_text=new_text,
                                        ref_type=ref_type,
                                    )
                                )

    return findings


def apply_fixes(findings: list[RefFinding]) -> int:
    """Apply fixes to files. Returns number of files modified."""
    # Group by file
    by_file: dict[str, list[RefFinding]] = {}
    for f in findings:
        by_file.setdefault(f.file, []).append(f)

    modified = 0
    for rel_path, file_findings in by_file.items():
        path = ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        original = text

        for finding in file_findings:
            text = text.replace(finding.old_text, finding.new_text, 1)

        if text != original:
            path.write_text(text, encoding="utf-8")
            modified += 1

    return modified


def main() -> int:
    fix_mode = "--fix" in sys.argv
    json_mode = "--json" in sys.argv

    if not json_mode:
        print("Deriving truth from code...")

    truth = derive_truth()

    if truth.errors:
        for err in truth.errors:
            print(f"  [{WARN}] {err}")

    if not json_mode:
        print(f"  MCP tools:      {truth.mcp_tool_count}")
        print(f"  Tests:          {truth.test_count_approx}+")
        print(f"  Schema:         v{truth.schema_version}")
        print(f"  Version:        {truth.version}")
        print()
        print("Scanning docs for stale references...")

    findings = scan_stale_refs(truth)

    if json_mode:
        result = {
            "truth": {
                "mcp_tool_count": truth.mcp_tool_count,
                "test_count_approx": truth.test_count_approx,
                "schema_version": truth.schema_version,
                "version": truth.version,
            },
            "findings": [
                {
                    "file": f.file,
                    "line": f.line_num,
                    "old": f.old_text,
                    "new": f.new_text,
                    "type": f.ref_type,
                }
                for f in findings
            ],
            "stale_count": len(findings),
        }
        print(json.dumps(result, indent=2))
        return 1 if findings else 0

    if not findings:
        print(f"  [{PASS}] All references up-to-date")
        return 0

    # Group by type for display
    by_type: dict[str, list[RefFinding]] = {}
    for f in findings:
        by_type.setdefault(f.ref_type, []).append(f)

    for ref_type, type_findings in by_type.items():
        print(f"\n  {ref_type} ({len(type_findings)} stale):")
        for f in type_findings:
            print(f"    {f.file}:{f.line_num}  {f.old_text!r} -> {f.new_text!r}")

    if fix_mode:
        modified = apply_fixes(findings)
        print(f"\n  [{PASS}] Fixed {len(findings)} references in {modified} files")
        return 0

    print(f"\n  [{FAIL}] {len(findings)} stale references found")
    print("  Run: python scripts/sync_refs.py --fix")
    return 1


if __name__ == "__main__":
    sys.exit(main())
