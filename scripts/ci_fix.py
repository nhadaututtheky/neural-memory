#!/usr/bin/env python3
"""Fix all CI-checked issues in one shot.

Runs the same checks as .github/workflows/ci.yml (lint job) and auto-fixes:
1. ruff check --fix (unused imports, import sorting, etc.)
2. ruff format (code formatting)
3. Verify clean (no remaining errors)

Usage:
    python scripts/ci_fix.py          # Fix + verify
    python scripts/ci_fix.py --check  # Verify only (no fixes)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"

TARGETS = ["src/", "tests/"]


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    check_only = "--check" in sys.argv
    errors = 0

    if not check_only:
        # Step 1: Auto-fix lint issues
        print("Fixing lint issues...", end=" ", flush=True)
        result = run(["python", "-m", "ruff", "check", "--fix", *TARGETS])
        if "fixed" in result.stdout or result.returncode == 0:
            print(f"{PASS} ruff check --fix")
        else:
            print(f"{FAIL} ruff check --fix")
            print(result.stdout)
            print(result.stderr)

        # Step 2: Auto-format
        print("Formatting code...", end=" ", flush=True)
        result = run(["python", "-m", "ruff", "format", *TARGETS])
        print(f"{PASS} ruff format")

    # Step 3: Verify lint clean
    print("Verifying lint...", end=" ", flush=True)
    result = run(["python", "-m", "ruff", "check", *TARGETS])
    if result.returncode == 0:
        print(f"{PASS} ruff check")
    else:
        print(f"{FAIL} ruff check")
        print(result.stdout)
        errors += 1

    # Step 4: Verify format clean
    print("Verifying format...", end=" ", flush=True)
    result = run(["python", "-m", "ruff", "format", "--check", *TARGETS])
    if result.returncode == 0:
        print(f"{PASS} ruff format --check")
    else:
        print(f"{FAIL} ruff format --check")
        print(result.stdout)
        errors += 1

    # Step 5: Security scan (same as CI)
    print("Security scan...", end=" ", flush=True)
    result = run(
        [
            "python",
            "-m",
            "ruff",
            "check",
            "src/",
            "--select",
            "S",
            "--ignore",
            "S101,S110,S112,S311,S324",
        ]
    )
    if result.returncode == 0:
        print(f"{PASS} security rules")
    else:
        print(f"{FAIL} security rules")
        print(result.stdout)
        errors += 1

    if errors:
        print(f"\n{FAIL} {errors} check(s) failed")
        return 1

    print(f"\n{PASS} All CI checks pass - safe to push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
