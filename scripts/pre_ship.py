#!/usr/bin/env python3
"""Pre-ship verification for Neural Memory.

Runs all automated checks before a release to catch common issues:
- Version consistency across 6 files
- Ruff lint + format
- Mypy type check
- Fast unit tests
- Import smoke test
- CHANGELOG has current version entry
- Auto-type classifier smoke test
- Cognitive layer integration test
- Documentation freshness (auto-generated docs up-to-date)
- Scattered reference sync (tool count, test count, schema version)

Usage:
    python scripts/pre_ship.py          # Run all checks
    python scripts/pre_ship.py --fix    # Auto-fix what's possible (ruff)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
WARN = "\033[93m WARN \033[0m"
SKIP = "\033[90m SKIP \033[0m"

failures: list[str] = []
warnings: list[str] = []


def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), **kwargs)


def check(name: str, passed: bool, detail: str = "") -> None:
    if passed:
        print(f"  [{PASS}] {name}")
    else:
        failures.append(name)
        msg = f"  [{FAIL}] {name}"
        if detail:
            msg += f"\n         {detail}"
        print(msg)


def warn(name: str, detail: str = "") -> None:
    warnings.append(name)
    msg = f"  [{WARN}] {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)


# ── 1. Version Consistency ──────────────────────────────────────


def check_versions() -> None:
    print("\n1. Version Consistency")

    version_files: dict[str, str | None] = {
        "pyproject.toml": None,
        "src/neural_memory/__init__.py": None,
        ".claude-plugin/plugin.json": None,
        ".claude-plugin/marketplace.json (metadata)": None,
        ".claude-plugin/marketplace.json (plugins)": None,
        "tests/unit/test_health_fixes.py": None,
        "tests/unit/test_markdown_export.py": None,
    }

    # pyproject.toml
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    canonical = m.group(1) if m else "NOT_FOUND"
    version_files["pyproject.toml"] = canonical

    # __init__.py
    text = (ROOT / "src/neural_memory/__init__.py").read_text()
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    version_files["src/neural_memory/__init__.py"] = m.group(1) if m else "NOT_FOUND"

    # plugin.json
    text = (ROOT / ".claude-plugin/plugin.json").read_text()
    m = re.search(r'"version"\s*:\s*"([^"]+)"', text)
    version_files[".claude-plugin/plugin.json"] = m.group(1) if m else "NOT_FOUND"

    # marketplace.json (2 occurrences)
    text = (ROOT / ".claude-plugin/marketplace.json").read_text()
    versions = re.findall(r'"version"\s*:\s*"([^"]+)"', text)
    version_files[".claude-plugin/marketplace.json (metadata)"] = (
        versions[0] if len(versions) > 0 else "NOT_FOUND"
    )
    version_files[".claude-plugin/marketplace.json (plugins)"] = (
        versions[1] if len(versions) > 1 else "NOT_FOUND"
    )

    # test_health_fixes.py
    text = (ROOT / "tests/unit/test_health_fixes.py").read_text()
    m = re.search(r'__version__\s*==\s*"([^"]+)"', text)
    version_files["tests/unit/test_health_fixes.py"] = m.group(1) if m else "NOT_FOUND"

    # test_markdown_export.py
    text = (ROOT / "tests/unit/test_markdown_export.py").read_text()
    m = re.search(r'"version"\s*:\s*"([^"]+)"', text)
    version_files["tests/unit/test_markdown_export.py"] = m.group(1) if m else "NOT_FOUND"

    all_match = all(v == canonical for v in version_files.values())
    check("All 6 files match", all_match, f"Expected {canonical}")

    if not all_match:
        for path, ver in version_files.items():
            status = "ok" if ver == canonical else f"MISMATCH: {ver}"
            print(f"           {path}: {status}")

    # CHANGELOG has entry for this version
    changelog = (ROOT / "CHANGELOG.md").read_text()
    has_entry = f"[{canonical}]" in changelog
    check(f"CHANGELOG has [{canonical}] entry", has_entry)


# ── 2. Ruff ─────────────────────────────────────────────────────


def check_ruff(fix: bool = False) -> None:
    print("\n2. Ruff Lint & Format")

    if fix:
        run(["ruff", "check", "--fix", "src/", "tests/"])
        run(["ruff", "format", "src/", "tests/"])

    result = run(["ruff", "check", "src/", "tests/"])
    check(
        "ruff check",
        result.returncode == 0,
        result.stdout.strip()[:200] if result.returncode != 0 else "",
    )

    result = run(["ruff", "format", "--check", "src/", "tests/"])
    check(
        "ruff format",
        result.returncode == 0,
        "Run: ruff format src/ tests/" if result.returncode != 0 else "",
    )


# ── 3. Mypy ─────────────────────────────────────────────────────


def check_mypy() -> None:
    print("\n3. Type Check (mypy)")

    result = run(["mypy", "src/", "--ignore-missing-imports"])
    passed = result.returncode == 0
    detail = ""
    if not passed:
        lines = result.stdout.strip().split("\n")
        error_lines = [l for l in lines if "error:" in l]
        detail = (
            f"{len(error_lines)} errors. First: {error_lines[0]}"
            if error_lines
            else result.stdout[:200]
        )
    check("mypy src/", passed, detail)


# ── 4. Import Smoke Test ────────────────────────────────────────


def check_imports() -> None:
    print("\n4. Import Smoke Test")

    result = run(
        [sys.executable, "-c", "import neural_memory; print(f'v{neural_memory.__version__}')"]
    )
    check(
        "import neural_memory",
        result.returncode == 0,
        result.stderr.strip()[:200] if result.returncode != 0 else "",
    )


# ── 5. Fast Tests ───────────────────────────────────────────────


def check_tests() -> None:
    print("\n5. Fast Unit Tests")

    import tempfile

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/",
        "-x",
        "-q",
        "--timeout=30",
        "--ignore=tests/unit/test_consolidation.py",
        "-m",
        "not stress",
    ]
    # Use temp file to avoid Windows subprocess PIPE buffering deadlock
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as tmp:
        try:
            result = subprocess.run(
                cmd, stdout=tmp, stderr=subprocess.STDOUT,
                cwd=str(ROOT), timeout=600,
            )
        except subprocess.TimeoutExpired:
            check("pytest tests/unit/", False, "timed out after 600s")
            return
        tmp.seek(0)
        output = tmp.read()
    passed = result.returncode == 0
    lines = output.strip().split("\n")
    summary = lines[-1] if lines else ""
    check("pytest tests/unit/", passed, summary if not passed else f"({summary})")


# ── 6. Auto-Type Classifier Smoke Test ─────────────────────────


def check_classifier() -> None:
    print("\n6. Auto-Type Classifier Smoke Test")

    # These test cases cover the bias bug fixed in v2.27.1
    # (DECISION was classified as INSIGHT because "because" matched first)
    test_cases = [
        # (content, expected_type)
        ("Chose PostgreSQL over MongoDB because we need ACID for payments", "decision"),
        ("Decided to use Redis instead of Memcached for caching", "decision"),
        ("Rejected GraphQL due to team inexperience", "decision"),
        ("Bug: the auth middleware crashes when cookie is expired", "error"),
        ("Error: connection refused when connecting to Redis on port 6379", "error"),
        ("Learned that asyncio.gather swallows exceptions by default", "insight"),
        ("Turns out the bottleneck was JSON serialization, not DB queries", "insight"),
        ("TODO: add rate limiting to the /api/upload endpoint", "todo"),
        ("Need to migrate the database before next release", "todo"),
        ("User prefers dark mode for all dashboards", "preference"),
        ("Always use parameterized queries for SQL", "instruction"),
        ("Deploy process: build, test, push to registry, update k8s", "workflow"),
        ("API endpoint is /v2/users", "fact"),
    ]

    result = run(
        [
            sys.executable,
            "-c",
            "from neural_memory.core.memory_types import suggest_memory_type; "
            "import json, sys; "
            "cases = json.loads(sys.argv[1]); "
            "results = [(c, e, suggest_memory_type(c).value) for c, e in cases]; "
            "failures = [(c, e, a) for c, e, a in results if e != a]; "
            "print(json.dumps(failures))",
            json.dumps(test_cases),
        ]
    )

    if result.returncode != 0:
        check("classifier import", False, result.stderr.strip()[:200])
        return

    classifier_failures = json.loads(result.stdout.strip())

    if classifier_failures:
        details = "; ".join(
            f"'{c[:40]}...' expected={e} got={a}" for c, e, a in classifier_failures[:3]
        )
        check(f"classifier ({len(test_cases)} cases)", False, details)
    else:
        check(f"classifier ({len(test_cases)} cases)", True)


# ── 7. Cognitive Layer Integration ─────────────────────────────


def check_cognitive() -> None:
    print("\n7. Cognitive Layer Integration")

    # Verify cognitive engine functions are importable and produce sane outputs
    result = run(
        [
            sys.executable,
            "-c",
            "from neural_memory.engine.cognitive import update_confidence, detect_auto_resolution; "
            "c = update_confidence(0.5, 'for', 0.7, 0, 0); "
            "assert 0.5 < c < 0.8, f'confidence {c} out of range'; "
            "c2 = update_confidence(0.5, 'against', 0.7, 0, 0); "
            "assert 0.2 < c2 < 0.5, f'confidence {c2} out of range'; "
            "assert detect_auto_resolution(0.95, 3, 0) == 'confirmed'; "
            "assert detect_auto_resolution(0.05, 0, 3) == 'refuted'; "
            "assert detect_auto_resolution(0.5, 1, 1) is None; "
            "print('ok')",
        ]
    )
    check(
        "cognitive engine (confidence + auto-resolution)",
        result.returncode == 0 and "ok" in result.stdout,
        result.stderr.strip()[:200] if result.returncode != 0 else "",
    )


# ── 8. OpenClaw Plugin ──────────────────────────────────────────


def check_plugin() -> None:
    print("\n8. OpenClaw Plugin")

    plugin_dir = ROOT / "integrations" / "neuralmemory"
    pkg = plugin_dir / "package.json"
    manifest = plugin_dir / "openclaw.plugin.json"

    if not pkg.exists():
        warn("Plugin package.json not found")
        return

    # Check name matches manifest id
    pkg_text = pkg.read_text()
    manifest_text = manifest.read_text()

    pkg_name = re.search(r'"name"\s*:\s*"([^"]+)"', pkg_text)
    manifest_id = re.search(r'"id"\s*:\s*"([^"]+)"', manifest_text)

    if pkg_name and manifest_id:
        check(
            "Plugin name matches manifest id",
            pkg_name.group(1) == manifest_id.group(1),
            f"package.json name={pkg_name.group(1)}, manifest id={manifest_id.group(1)}",
        )

    # Check versions match between package.json and manifest
    pkg_ver = re.search(r'"version"\s*:\s*"([^"]+)"', pkg_text)
    manifest_ver = re.search(r'"version"\s*:\s*"([^"]+)"', manifest_text)
    if pkg_ver and manifest_ver:
        check(
            "Plugin versions consistent",
            pkg_ver.group(1) == manifest_ver.group(1),
            f"package.json={pkg_ver.group(1)}, manifest={manifest_ver.group(1)}",
        )


# ── 9. Documentation Freshness ─────────────────────────────────


def check_docs() -> None:
    print("\n9. Documentation Freshness")

    for script, label in [
        ("scripts/gen_mcp_docs.py", "MCP tools reference"),
        ("scripts/gen_cli_docs.py", "CLI reference"),
    ]:
        script_path = ROOT / script
        if not script_path.exists():
            warn(f"{script} not found — skipping {label} check")
            continue

        result = subprocess.run(
            [sys.executable, str(script_path), "--check"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        check(
            f"{label} up-to-date",
            result.returncode == 0,
            result.stdout.strip()[:200] if result.returncode != 0 else "",
        )


# ── 10. Scattered Reference Sync ──────────────────────────────


def check_refs(fix: bool = False) -> None:
    print("\n10. Scattered Reference Sync")

    script = ROOT / "scripts" / "sync_refs.py"
    if not script.exists():
        warn("scripts/sync_refs.py not found")
        return

    cmd = [sys.executable, str(script)]
    if fix:
        cmd.append("--fix")
    else:
        cmd.append("--json")

    result = run(cmd)

    if fix:
        check(
            "sync_refs --fix",
            result.returncode == 0,
            result.stdout.strip()[-200:] if result.returncode != 0 else "",
        )
        return

    if result.returncode != 0 and result.stdout.strip():
        try:
            data = json.loads(result.stdout.strip())
            stale = data.get("stale_count", 0)
            findings = data.get("findings", [])
            if stale > 0:
                samples = "; ".join(f"{f['file']}:{f['line']} {f['old']!r}" for f in findings[:3])
                check(
                    f"references up-to-date ({stale} stale)",
                    False,
                    f"{samples}... Run: python scripts/sync_refs.py --fix",
                )
            else:
                check("references up-to-date", True)
        except json.JSONDecodeError:
            check("sync_refs", False, result.stderr.strip()[:200])
    else:
        check("references up-to-date", result.returncode == 0)


# ── Main ────────────────────────────────────────────────────────


def main() -> int:
    fix = "--fix" in sys.argv

    print("=" * 60)
    print("  Neural Memory — Pre-Ship Verification")
    print("=" * 60)

    check_versions()
    check_ruff(fix=fix)
    check_mypy()
    check_imports()
    check_tests()
    check_classifier()
    check_cognitive()
    check_plugin()
    check_docs()
    check_refs(fix=fix)

    print("\n" + "=" * 60)
    if failures:
        print(f"  {FAIL} {len(failures)} check(s) failed:")
        for f in failures:
            print(f"    - {f}")
        print("=" * 60)
        return 1
    elif warnings:
        print(f"  {WARN} All checks passed with {len(warnings)} warning(s)")
        print("=" * 60)
        return 0
    else:
        print(f"  {PASS} All checks passed!")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
