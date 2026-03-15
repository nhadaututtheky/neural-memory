#!/usr/bin/env python3
"""Check all distribution channels for Neural Memory and report version mismatches.

Usage:
    python scripts/check_distribution.py           # Check all channels
    python scripts/check_distribution.py --fix     # Print commands to sync out-of-date channels
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Constants ───────────────────────────────────────────────────────────────

PACKAGE_NAME = "neural-memory"
NPM_PACKAGE = "neuralmemory"
VSCE_ID = "neuralmem.neuralmemory"
CLAWHUB_SLUG = "neural-memory"
GITHUB_REPO = "nhadaututtheky/neural-memory"

ROOT = Path(__file__).resolve().parent.parent

# All files that must be updated on every version bump
ALL_VERSION_FILES: list[str] = [
    "pyproject.toml",
    "src/neural_memory/__init__.py",
    ".claude-plugin/plugin.json",
    ".claude-plugin/marketplace.json",  # 2 occurrences
    "tests/unit/test_health_fixes.py",
    "tests/unit/test_markdown_export.py",
]

# ANSI colors — disable if stdout is not a TTY or on Windows without VT support
_USE_COLOR = sys.stdout.isatty() and (sys.platform != "win32" or os.environ.get("TERM"))
_GREEN = "\033[92m" if _USE_COLOR else ""
_RED = "\033[91m" if _USE_COLOR else ""
_YELLOW = "\033[93m" if _USE_COLOR else ""
_RESET = "\033[0m" if _USE_COLOR else ""
_BOLD = "\033[1m" if _USE_COLOR else ""

# Status icons — use ASCII on Windows to avoid cp1252 encoding errors
_ICON_OK = "[OK]" if sys.platform == "win32" else "✅"
_ICON_FAIL = "[!!]" if sys.platform == "win32" else "❌"
_ICON_WARN = "[??]" if sys.platform == "win32" else "⚠"

# ── Result model ────────────────────────────────────────────────────────────

STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_WARN = "warn"


@dataclass
class ChannelResult:
    name: str
    version: Optional[str]
    status: str  # STATUS_OK | STATUS_MISMATCH | STATUS_WARN
    detail: str = ""

    def status_icon(self) -> str:
        if self.status == STATUS_OK:
            return f"{_GREEN}{_ICON_OK}{_RESET}"
        elif self.status == STATUS_MISMATCH:
            return f"{_RED}{_ICON_FAIL}{_RESET}"
        else:
            return f"{_YELLOW}{_ICON_WARN}{_RESET}"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr).

    Uses shell=True on Windows so that npm/clawdhub/.cmd executables are found.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
            shell=(sys.platform == "win32"),
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", f"Timed out after {timeout}s"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except OSError as exc:
        return -1, "", str(exc)


def _fetch_json(url: str, timeout: int = 10) -> Optional[dict]:
    """Fetch a URL and parse as JSON. Returns None on any error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "check_distribution/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


# ── Channel checkers ─────────────────────────────────────────────────────────


def check_local_init() -> ChannelResult:
    """Read __version__ from src/neural_memory/__init__.py."""
    path = ROOT / "src" / "neural_memory" / "__init__.py"
    try:
        text = path.read_text(encoding="utf-8")
        match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
        if match:
            return ChannelResult("Local __init__.py", match.group(1), STATUS_OK)
        return ChannelResult("Local __init__.py", None, STATUS_WARN, "Pattern not found")
    except OSError as exc:
        return ChannelResult("Local __init__.py", None, STATUS_WARN, str(exc))


def check_pyproject() -> ChannelResult:
    """Read version from pyproject.toml."""
    path = ROOT / "pyproject.toml"
    try:
        text = path.read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if match:
            return ChannelResult("pyproject.toml", match.group(1), STATUS_OK)
        return ChannelResult("pyproject.toml", None, STATUS_WARN, "Pattern not found")
    except OSError as exc:
        return ChannelResult("pyproject.toml", None, STATUS_WARN, str(exc))


def check_plugin_json() -> ChannelResult:
    """Read version from .claude-plugin/plugin.json."""
    path = ROOT / ".claude-plugin" / "plugin.json"
    try:
        text = path.read_text(encoding="utf-8")
        match = re.search(r'"version"\s*:\s*"([^"]+)"', text)
        if match:
            return ChannelResult(".claude-plugin/plugin.json", match.group(1), STATUS_OK)
        return ChannelResult(".claude-plugin/plugin.json", None, STATUS_WARN, "Pattern not found")
    except OSError as exc:
        return ChannelResult(".claude-plugin/plugin.json", None, STATUS_WARN, str(exc))


def check_marketplace_json() -> list[ChannelResult]:
    """Read both version occurrences from .claude-plugin/marketplace.json."""
    path = ROOT / ".claude-plugin" / "marketplace.json"
    try:
        text = path.read_text(encoding="utf-8")
        versions = re.findall(r'"version"\s*:\s*"([^"]+)"', text)
        results: list[ChannelResult] = []
        results.append(ChannelResult(
            "marketplace.json (metadata)",
            versions[0] if len(versions) > 0 else None,
            STATUS_OK if versions else STATUS_WARN,
            "" if versions else "Pattern not found",
        ))
        results.append(ChannelResult(
            "marketplace.json (plugins)",
            versions[1] if len(versions) > 1 else None,
            STATUS_OK if len(versions) > 1 else STATUS_WARN,
            "" if len(versions) > 1 else "Second occurrence not found",
        ))
        return results
    except OSError as exc:
        return [
            ChannelResult("marketplace.json (metadata)", None, STATUS_WARN, str(exc)),
            ChannelResult("marketplace.json (plugins)", None, STATUS_WARN, str(exc)),
        ]


def check_test_health_fixes() -> ChannelResult:
    """Read hardcoded version from tests/unit/test_health_fixes.py."""
    path = ROOT / "tests" / "unit" / "test_health_fixes.py"
    try:
        text = path.read_text(encoding="utf-8")
        match = re.search(r'__version__\s*==\s*"([^"]+)"', text)
        if match:
            return ChannelResult("test_health_fixes.py", match.group(1), STATUS_OK)
        return ChannelResult("test_health_fixes.py", None, STATUS_WARN, "Pattern not found")
    except OSError as exc:
        return ChannelResult("test_health_fixes.py", None, STATUS_WARN, str(exc))


def check_test_markdown_export() -> ChannelResult:
    """Read hardcoded version from tests/unit/test_markdown_export.py."""
    path = ROOT / "tests" / "unit" / "test_markdown_export.py"
    try:
        text = path.read_text(encoding="utf-8")
        match = re.search(r'"version"\s*:\s*"([^"]+)"', text)
        if match:
            return ChannelResult("test_markdown_export.py", match.group(1), STATUS_OK)
        return ChannelResult("test_markdown_export.py", None, STATUS_WARN, "Pattern not found")
    except OSError as exc:
        return ChannelResult("test_markdown_export.py", None, STATUS_WARN, str(exc))


def check_pypi() -> ChannelResult:
    """Fetch latest version from PyPI JSON API."""
    data = _fetch_json(f"https://pypi.org/pypi/{PACKAGE_NAME}/json")
    if data is None:
        return ChannelResult("PyPI", None, STATUS_WARN, "Network error or package not found")
    try:
        version = data["info"]["version"]
        return ChannelResult("PyPI", version, STATUS_OK)
    except (KeyError, TypeError):
        return ChannelResult("PyPI", None, STATUS_WARN, "Unexpected response format")


def check_npm() -> ChannelResult:
    """Fetch latest version via `npm view`."""
    code, stdout, stderr = _run(["npm", "view", NPM_PACKAGE, "version"], timeout=20)
    if code == 0 and stdout:
        return ChannelResult("npm", stdout.strip(), STATUS_OK)
    # npm exits non-zero for packages that don't exist; detect E404
    if "E404" in stderr or "npm error 404" in stderr.lower():
        return ChannelResult("npm", None, STATUS_WARN, "Package not found on npm")
    if code == -1:
        return ChannelResult("npm", None, STATUS_WARN, stderr or "npm CLI not available")
    return ChannelResult("npm", None, STATUS_WARN, stderr[:120] or "Unknown npm error")


def check_vscode_marketplace() -> ChannelResult:
    """Fetch extension version from VS Code Marketplace REST API."""
    url = "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery"
    publisher, ext_name = VSCE_ID.split(".", 1)
    payload = json.dumps({
        "filters": [{"criteria": [{"filterType": 7, "value": VSCE_ID}]}],
        "flags": 0x200,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json;api-version=7.1-preview.1",
        "User-Agent": "check_distribution/1.0",
    }
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        extensions = data.get("results", [{}])[0].get("extensions", [])
        if not extensions:
            return ChannelResult("VS Code Marketplace", None, STATUS_WARN, "Extension not published yet")
        version = extensions[0]["versions"][0]["version"]
        return ChannelResult("VS Code Marketplace", version, STATUS_OK)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError, KeyError, IndexError) as exc:
        return ChannelResult("VS Code Marketplace", None, STATUS_WARN, str(exc)[:120])


def check_openvsx() -> ChannelResult:
    """Fetch extension version from Open VSX Registry."""
    publisher, ext_name = VSCE_ID.split(".", 1)
    data = _fetch_json(f"https://open-vsx.org/api/{publisher}/{ext_name}")
    if data is None:
        return ChannelResult("Open VSX", None, STATUS_WARN, "Network error or extension not found")
    if "error" in data:
        return ChannelResult("Open VSX", None, STATUS_WARN, str(data["error"])[:120])
    try:
        version = data["version"]
        return ChannelResult("Open VSX", version, STATUS_OK)
    except (KeyError, TypeError):
        return ChannelResult("Open VSX", None, STATUS_WARN, "Unexpected response format")


def check_clawhub() -> ChannelResult:
    """Fetch version via `clawdhub inspect` CLI."""
    code, stdout, stderr = _run(["clawdhub", "inspect", CLAWHUB_SLUG, "--json"], timeout=20)
    if code == 0 and stdout:
        try:
            data = json.loads(stdout)
            version = (
                data.get("version")
                or data.get("latestVersion", {}).get("version")
                or data.get("skill", {}).get("tags", {}).get("latest")
                or data.get("info", {}).get("version")
            )
            if version:
                return ChannelResult("ClawHub", str(version), STATUS_OK)
            return ChannelResult("ClawHub", None, STATUS_WARN, "version key missing in response")
        except json.JSONDecodeError:
            return ChannelResult("ClawHub", None, STATUS_WARN, "Non-JSON response from clawdhub")
    if code == -1:
        return ChannelResult("ClawHub", None, STATUS_WARN, stderr or "clawdhub CLI not available")
    return ChannelResult("ClawHub", None, STATUS_WARN, (stderr or stdout)[:120] or "Unknown error")


def check_github_release() -> ChannelResult:
    """Fetch latest release tag via `gh release view`."""
    code, stdout, stderr = _run(
        ["gh", "release", "view", "--repo", GITHUB_REPO, "--json", "tagName"],
        timeout=20,
    )
    if code == 0 and stdout:
        try:
            data = json.loads(stdout)
            tag = data.get("tagName", "")
            version = tag.lstrip("v")
            return ChannelResult("GitHub Release", version, STATUS_OK)
        except json.JSONDecodeError:
            return ChannelResult("GitHub Release", None, STATUS_WARN, "Non-JSON response from gh")
    if code == -1:
        return ChannelResult("GitHub Release", None, STATUS_WARN, stderr or "gh CLI not available")
    # No release yet
    if "release not found" in stderr.lower() or "no releases" in stderr.lower():
        return ChannelResult("GitHub Release", None, STATUS_WARN, "No releases published yet")
    return ChannelResult("GitHub Release", None, STATUS_WARN, (stderr or stdout)[:120] or "Unknown error")


# ── Fix hints ────────────────────────────────────────────────────────────────


def _fix_hints(channel: ChannelResult, local_version: str) -> list[str]:
    """Return shell commands needed to bring this channel in sync."""
    name = channel.name
    if name == "PyPI":
        return [
            "# Publish to PyPI:",
            f"python -m build",
            f"twine upload dist/neural_memory-{local_version}*",
        ]
    if name == "npm":
        return [
            "# Publish to npm:",
            f"cd integrations/neuralmemory",
            f"npm version {local_version} --no-git-tag-version",
            f"npm publish",
        ]
    if name == "VS Code Marketplace":
        return [
            "# Publish to VS Code Marketplace:",
            f"cd vscode-extension",
            f"npx @vscode/vsce publish {local_version}",
        ]
    if name == "Open VSX":
        return [
            "# Publish to Open VSX:",
            f"cd vscode-extension",
            f"npx ovsx publish -p $OVSX_TOKEN",
        ]
    if name == "ClawHub":
        return [
            "# Publish to ClawHub:",
            f"clawdhub publish --slug {CLAWHUB_SLUG} --version {local_version}",
        ]
    if name == "GitHub Release":
        return [
            "# Create GitHub Release:",
            f'gh release create v{local_version} --repo {GITHUB_REPO} --title "v{local_version}" --notes "Release v{local_version}"',
        ]
    if "pyproject" in name.lower():
        return [f'# Update pyproject.toml: version = "{local_version}"']
    if "__init__" in name.lower():
        return [f'# Update src/neural_memory/__init__.py: __version__ = "{local_version}"']
    if "plugin.json" in name:
        return [f'# Update .claude-plugin/plugin.json: "version": "{local_version}"']
    if "marketplace.json" in name:
        return [f'# Update .claude-plugin/marketplace.json (both occurrences): "version": "{local_version}"']
    if "test_health_fixes" in name:
        return [f'# Update tests/unit/test_health_fixes.py: __version__ == "{local_version}"']
    if "test_markdown_export" in name:
        return [f'# Update tests/unit/test_markdown_export.py: "version": "{local_version}"']
    return [f"# Manually sync {name} to version {local_version}"]


# ── Table rendering ──────────────────────────────────────────────────────────


def _pad(text: str, width: int) -> str:
    """Pad text to width, ignoring ANSI escape codes in length calculation."""
    visible_len = len(re.sub(r"\033\[[0-9;]*m", "", text))
    return text + " " * max(0, width - visible_len)


def print_table(results: list[ChannelResult]) -> None:
    col_channel = 34
    col_version = 14
    col_status = 8

    header = (
        f"  {'Channel':<{col_channel}} {'Version':<{col_version}} Status"
    )
    divider = "  " + "-" * (col_channel + col_version + col_status + 4)

    print(header)
    print(divider)

    for r in results:
        version_str = r.version or "N/A"
        icon = r.status_icon()
        detail = f"  ({r.detail})" if r.detail and r.status == STATUS_WARN else ""
        line = f"  {r.name:<{col_channel}} {version_str:<{col_version}} {icon}{detail}"
        print(line)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    fix_mode = "--fix" in sys.argv

    print()
    print(f"{_BOLD}{'=' * 62}{_RESET}")
    print(f"{_BOLD}  Neural Memory - Distribution Channel Check{_RESET}")
    print(f"{_BOLD}{'=' * 62}{_RESET}")

    # Step 1: Get the authoritative local version
    local_result = check_local_init()
    local_version = local_result.version

    if not local_version:
        print(f"\n{_RED}ERROR: Could not read local __version__ from __init__.py{_RESET}")
        return 1

    print(f"\n  Canonical version: {_BOLD}{local_version}{_RESET}  (from src/neural_memory/__init__.py)\n")

    # Step 2: Collect all channel results
    print("  Checking channels (network calls may take a moment)...")
    print()

    all_results: list[ChannelResult] = []

    # --- Local / file checks (fast, no network) ---
    all_results.append(local_result)  # already fetched
    all_results.append(check_pyproject())
    all_results.append(check_plugin_json())
    all_results.extend(check_marketplace_json())
    all_results.append(check_test_health_fixes())
    all_results.append(check_test_markdown_export())

    # --- Remote / CLI checks (may involve network) ---
    all_results.append(check_pypi())
    all_results.append(check_npm())
    all_results.append(check_vscode_marketplace())
    all_results.append(check_openvsx())
    all_results.append(check_clawhub())
    all_results.append(check_github_release())

    # Channels with independent versioning — don't compare against canonical
    _INDEPENDENT_VERSION_CHANNELS = {"npm", "VS Code Marketplace", "Open VSX"}

    # Step 3: Classify each result against canonical version
    for r in all_results:
        if r.status == STATUS_WARN:
            continue  # already marked as warn (check failed)
        if r.name in _INDEPENDENT_VERSION_CHANNELS:
            r.status = STATUS_OK  # own versioning, not compared to Python package
            r.detail = "independent version"
        elif r.version == local_version:
            r.status = STATUS_OK
        else:
            r.status = STATUS_MISMATCH

    # Step 4: Print table
    print_table(all_results)

    # Step 5: Summary
    ok_count = sum(1 for r in all_results if r.status == STATUS_OK)
    mismatch_results = [r for r in all_results if r.status == STATUS_MISMATCH]
    warn_results = [r for r in all_results if r.status == STATUS_WARN]
    total = len(all_results)

    print()
    print(f"{_BOLD}{'=' * 62}{_RESET}")

    if not mismatch_results:
        sync_label = f"{_GREEN}{ok_count}/{total} channels in sync{_RESET}"
        if warn_results:
            warn_names = ", ".join(r.name for r in warn_results)
            print(f"  {sync_label}  |  {_YELLOW}{_ICON_WARN} {len(warn_results)} could not be checked: {warn_names}{_RESET}")
        else:
            print(f"  {sync_label} {_ICON_OK}")
    else:
        mismatch_names = ", ".join(r.name for r in mismatch_results)
        print(f"  {_RED}MISMATCH: {mismatch_names} need updating{_RESET}")
        print(f"  {ok_count}/{total} channels in sync")

    print(f"{_BOLD}{'=' * 62}{_RESET}")

    # Step 6: --fix output
    if fix_mode:
        channels_to_fix = [r for r in all_results if r.status == STATUS_MISMATCH]
        if not channels_to_fix:
            print(f"\n  {_GREEN}Nothing to fix — all channels match v{local_version}.{_RESET}\n")
        else:
            print(f"\n{_BOLD}  Commands to bring channels in sync with v{local_version}:{_RESET}\n")
            for r in channels_to_fix:
                print(f"  {_BOLD}[{r.name}]{_RESET}")
                for hint in _fix_hints(r, local_version):
                    print(f"    {hint}")
                print()

    return 1 if mismatch_results else 0


if __name__ == "__main__":
    sys.exit(main())
