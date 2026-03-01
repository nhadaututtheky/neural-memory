"""Zero-config setup for NeuralMemory.

Handles first-time initialization: config, brain, and MCP auto-configuration.
Called by `nmem init` to set up everything in one command.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import typer


def find_nmem_command() -> dict[str, Any]:
    """Find the best command to run the MCP server.

    Priority:
    1. nmem-mcp entry point (cleanest)
    2. nmem CLI with mcp subcommand
    3. python -m fallback (uses absolute path, normalized for Windows)
    """
    nmem_mcp = shutil.which("nmem-mcp")
    if nmem_mcp:
        return {"command": "nmem-mcp"}

    nmem = shutil.which("nmem")
    if nmem:
        return {"command": "nmem", "args": ["mcp"]}

    return {"command": _normalize_path(sys.executable), "args": ["-m", "neural_memory.mcp"]}


def setup_config(data_dir: Path, *, force: bool = False) -> bool:
    """Create ~/.neuralmemory/ with config.toml and brains/ directory.

    Returns True if config was created/updated, False if skipped.
    """
    from neural_memory.unified_config import UnifiedConfig

    config_path = data_dir / "config.toml"

    if config_path.exists() and not force:
        return False

    config = UnifiedConfig(data_dir=data_dir)
    config.save()

    brains_dir = data_dir / "brains"
    brains_dir.mkdir(parents=True, exist_ok=True)

    return True


def setup_brain(data_dir: Path) -> str:
    """Ensure default brain SQLite DB exists.

    Returns the brain name.
    """
    brains_dir = data_dir / "brains"
    brains_dir.mkdir(parents=True, exist_ok=True)

    db_path = brains_dir / "default.db"
    if not db_path.exists():
        db_path.touch()

    return "default"


def setup_mcp_claude() -> str:
    """Auto-configure MCP in Claude Code (~/.claude/mcp_servers.json).

    Returns status string: "added", "exists", "failed", or "not_found".
    """
    claude_dir = Path.home() / ".claude"
    if not claude_dir.exists():
        return "not_found"

    config_path = claude_dir / "mcp_servers.json"
    mcp_entry = find_nmem_command()

    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = config_path.read_text(encoding="utf-8").strip()
            if raw:
                existing = json.loads(raw)
        except (json.JSONDecodeError, OSError):
            existing = {}

    if "neural-memory" in existing:
        return "exists"

    try:
        existing["neural-memory"] = mcp_entry
        config_path.write_text(
            json.dumps(existing, indent=2) + "\n",
            encoding="utf-8",
        )
        return "added"
    except OSError:
        return "failed"


def setup_mcp_cursor() -> str:
    """Auto-configure MCP in Cursor (~/.cursor/mcp.json).

    Returns status string: "added", "exists", "failed", or "not_found".
    """
    cursor_dir = Path.home() / ".cursor"
    if not cursor_dir.exists():
        return "not_found"

    config_path = cursor_dir / "mcp.json"
    mcp_entry = find_nmem_command()

    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = config_path.read_text(encoding="utf-8").strip()
            if raw:
                existing = json.loads(raw)
        except (json.JSONDecodeError, OSError):
            existing = {}

    servers = existing.get("mcpServers", {})
    if "neural-memory" in servers:
        return "exists"

    try:
        servers["neural-memory"] = mcp_entry
        existing["mcpServers"] = servers
        config_path.write_text(
            json.dumps(existing, indent=2) + "\n",
            encoding="utf-8",
        )
        return "added"
    except OSError:
        return "failed"


def _normalize_path(path: str) -> str:
    """Normalize a filesystem path for use in shell commands.

    On Windows, convert backslashes to forward slashes and quote if
    the path contains spaces. This ensures commands work in both
    CMD/PowerShell and Git Bash/MSYS2 environments.
    """
    normalized = path.replace("\\", "/")
    if " " in normalized:
        normalized = f'"{normalized}"'
    return normalized


def _find_hook_command(entry_point: str, cli_subcommand: str, module: str) -> str:
    """Find the best shell command for a NeuralMemory hook.

    Priority:
    1. Dedicated pip entry point  — cleanest, cross-platform
    2. nmem <subcommand>          — works if nmem CLI is on PATH
    3. python -m <module>         — always-available fallback (uses absolute path)
    """
    if shutil.which(entry_point):
        return entry_point
    if shutil.which("nmem"):
        return f"nmem {cli_subcommand}"
    python = _normalize_path(sys.executable)
    return f"{python} -m {module}"


def _find_pre_compact_command() -> str:
    return _find_hook_command("nmem-hook-pre-compact", "flush", "neural_memory.hooks.pre_compact")


def _find_stop_command() -> str:
    return _find_hook_command("nmem-hook-stop", "stop-hook", "neural_memory.hooks.stop")


def _find_post_tool_use_command() -> str:
    return _find_hook_command(
        "nmem-hook-post-tool-use", "post-tool-use-hook", "neural_memory.hooks.post_tool_use"
    )


def _is_nmem_hook_present(entries: list[dict[str, Any]]) -> bool:
    """Return True if any NeuralMemory hook is already registered in the entry list."""
    for entry in entries:
        for hook in entry.get("hooks", []):
            cmd = hook.get("command", "")
            if "neural_memory" in cmd or "nmem" in cmd:
                return True
    return False


def _load_settings(settings_path: Path) -> dict[str, Any]:
    if not settings_path.exists():
        return {}
    try:
        raw = settings_path.read_text(encoding="utf-8").strip()
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, OSError):
        return {}


def setup_hooks_claude() -> str:
    """Auto-configure PreCompact and Stop hooks in Claude Code (~/.claude/settings.json).

    Injects hook entries so NeuralMemory captures memories both before context
    compaction (PreCompact) and at normal session end (Stop).
    Safe to call repeatedly — skips hooks that are already present.

    Returns status string: "added", "exists", "failed", or "not_found".
    """
    claude_dir = Path.home() / ".claude"
    if not claude_dir.exists():
        return "not_found"

    settings_path = claude_dir / "settings.json"
    existing = _load_settings(settings_path)
    hooks_section: dict[str, Any] = existing.setdefault("hooks", {})

    added = 0
    hook_specs = [
        ("PreCompact", _find_pre_compact_command(), 30),
        ("Stop", _find_stop_command(), 30),
        ("PostToolUse", _find_post_tool_use_command(), 5),
    ]

    # Matcher for PostToolUse: skip internal/noisy tools
    post_tool_matcher = 'tool != "TodoRead" && tool != "TodoWrite" && tool != "TaskList"'

    for event, cmd_str, timeout in hook_specs:
        entries: list[dict[str, Any]] = hooks_section.setdefault(event, [])
        if _is_nmem_hook_present(entries):
            continue
        entry: dict[str, Any] = {
            "hooks": [{"type": "command", "command": cmd_str, "timeout": timeout}],
        }
        if event == "PostToolUse":
            entry["matcher"] = post_tool_matcher
        entries.append(entry)
        added += 1

    if added == 0:
        return "exists"

    try:
        settings_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        return "added"
    except OSError:
        return "failed"


def _discover_bundled_skills() -> dict[str, Path]:
    """Find all bundled skills (dirs containing SKILL.md).

    Returns mapping of skill name to SKILL.md path.
    """
    from neural_memory.skills import SKILLS_DIR

    skills: dict[str, Path] = {}
    for child in sorted(SKILLS_DIR.iterdir()):
        skill_file = child / "SKILL.md"
        if child.is_dir() and skill_file.is_file():
            skills[child.name] = skill_file
    return skills


def setup_skills(*, force: bool = False) -> dict[str, str]:
    """Install bundled skills to ~/.claude/skills/.

    Args:
        force: Overwrite existing files even if different.

    Returns:
        Mapping of skill name to status string:
        "installed", "exists", "updated", "update available", or "failed".
        Returns {"Skills": "not_found ..."} if ~/.claude/ doesn't exist.
    """
    claude_dir = Path.home() / ".claude"
    if not claude_dir.exists():
        return {"Skills": "not_found (~/.claude/ not found)"}

    bundled = _discover_bundled_skills()
    if not bundled:
        return {"Skills": "no bundled skills found"}

    target_dir = claude_dir / "skills"
    results: dict[str, str] = {}

    for name, source_path in bundled.items():
        dest_dir = target_dir / name
        dest_file = dest_dir / "SKILL.md"

        try:
            source_content = source_path.read_text(encoding="utf-8")
            if dest_file.exists():
                existing_content = dest_file.read_text(encoding="utf-8")
                if existing_content == source_content:
                    results[name] = "exists"
                elif force:
                    dest_file.write_text(source_content, encoding="utf-8")
                    results[name] = "updated"
                else:
                    results[name] = "update available"
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file.write_text(source_content, encoding="utf-8")
                results[name] = "installed"
        except OSError:
            results[name] = "failed"

    return results


def print_summary(results: dict[str, str]) -> None:
    """Print formatted setup summary."""
    typer.echo()
    typer.secho("  NeuralMemory Setup", bold=True)
    typer.echo()

    status_icons = {
        "ok": typer.style("[OK]", fg=typer.colors.GREEN),
        "skip": typer.style("[--]", fg=typer.colors.YELLOW),
        "fail": typer.style("[!!]", fg=typer.colors.RED),
    }

    for label, detail in results.items():
        icon = status_icons.get(_classify_status(detail), status_icons["skip"])
        typer.echo(f"  {icon} {label:<16}{detail}")

    typer.echo()


def _classify_status(detail: str) -> str:
    """Classify a result detail string into ok/skip/fail."""
    lower = detail.lower()
    if any(word in lower for word in ("created", "added", "ready", "installed", "updated")):
        return "ok"
    if any(word in lower for word in ("exists", "already")):
        return "ok"
    if any(word in lower for word in ("not detected", "skipped", "not found")):
        return "skip"
    return "fail"
