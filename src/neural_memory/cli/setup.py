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


def _claude_json_has_server(claude_json_path: Path, server_name: str) -> bool:
    """Check if a server is already registered in ~/.claude.json."""
    if not claude_json_path.exists():
        return False
    try:
        raw = claude_json_path.read_text(encoding="utf-8").strip()
        if not raw:
            return False
        data = json.loads(raw)
        servers = data.get("mcpServers", {})
        return server_name in servers
    except (json.JSONDecodeError, OSError):
        return False


def _add_via_claude_cli(scope: str, command_args: list[str]) -> bool:
    """Try to add MCP server via `claude mcp add` CLI.

    Returns True on success, False if claude CLI is not available or fails.
    """
    import subprocess

    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False

    cmd = [claude_bin, "mcp", "add", "-s", scope, "neural-memory", "--"]
    cmd.extend(command_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _add_via_claude_json(claude_json_path: Path, mcp_entry: dict[str, Any]) -> bool:
    """Fallback: write MCP config directly to ~/.claude.json."""
    try:
        existing: dict[str, Any] = {}
        if claude_json_path.exists():
            raw = claude_json_path.read_text(encoding="utf-8").strip()
            if raw:
                existing = json.loads(raw)

        servers: dict[str, Any] = existing.setdefault("mcpServers", {})
        servers["neural-memory"] = mcp_entry
        claude_json_path.write_text(
            json.dumps(existing, indent=2) + "\n",
            encoding="utf-8",
        )
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _cleanup_stale_mcp_servers_json() -> None:
    """Remove the deprecated ~/.claude/mcp_servers.json if it exists.

    Claude Code does NOT read this file — it was a pre-release path.
    Entries here cause user confusion since they appear configured but don't work.
    """
    stale = Path.home() / ".claude" / "mcp_servers.json"
    if stale.exists():
        try:
            stale.unlink()
        except OSError:
            pass


def setup_mcp_claude() -> str:
    """Auto-configure MCP in Claude Code.

    Strategy:
    1. Use ``claude mcp add --scope user`` CLI if available (official method).
    2. Fallback: write directly to ``~/.claude.json`` > ``mcpServers``.
    3. Clean up deprecated ``~/.claude/mcp_servers.json`` (Claude Code ignores it).

    Returns status string: "added", "exists", "failed", or "not_found".
    """
    claude_dir = Path.home() / ".claude"
    if not claude_dir.exists():
        return "not_found"

    claude_json = claude_dir.parent / ".claude.json"

    # Already registered?
    if _claude_json_has_server(claude_json, "neural-memory"):
        _cleanup_stale_mcp_servers_json()
        return "exists"

    # Build the command to register
    mcp_entry = find_nmem_command()
    command_args: list[str] = [mcp_entry["command"]]
    command_args.extend(mcp_entry.get("args", []))

    # Try official CLI first
    if _add_via_claude_cli("user", command_args):
        _cleanup_stale_mcp_servers_json()
        return "added"

    # Fallback: direct JSON write
    if _add_via_claude_json(claude_json, mcp_entry):
        _cleanup_stale_mcp_servers_json()
        return "added"

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


def _find_session_start_command() -> str:
    return _find_hook_command(
        "nmem-hook-session-start", "session-start-hook", "neural_memory.hooks.session_start"
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
    """Auto-configure NeuralMemory hooks in Claude Code (``~/.claude/settings.json``).

    Registers four lifecycle handlers so memory capture + context injection
    happen with zero manual effort:

    * ``SessionStart`` — inject project surface as system message
    * ``PreCompact`` — flush memory before context loss
    * ``PostToolUse`` — append tool events for deferred consolidation
    * ``Stop`` — drain buffer + regenerate surface at session end

    Safe to call repeatedly — skips hooks that are already present.

    Returns ``"added"`` (new entries written), ``"exists"`` (no-op), ``"failed"``
    (I/O error), or ``"not_found"`` (no ``~/.claude`` directory).
    """
    claude_dir = Path.home() / ".claude"
    if not claude_dir.exists():
        return "not_found"

    settings_path = claude_dir / "settings.json"
    existing = _load_settings(settings_path)
    hooks_section: dict[str, Any] = existing.setdefault("hooks", {})

    added = 0
    hook_specs = [
        ("SessionStart", _find_session_start_command(), 10),
        ("PreCompact", _find_pre_compact_command(), 30),
        ("Stop", _find_stop_command(), 30),
        ("PostToolUse", _find_post_tool_use_command(), 5),
    ]

    # Matcher for PostToolUse: skip internal/noisy tools.
    # The hook script also filters these defensively (see _NOISE_TOOLS).
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


# ─────────────────────────────────────────────────────────────────────────────
# Codex CLI hook installer
#
# Codex uses TOML config at ``~/.codex/config.toml`` with array-of-tables under
# ``[[hooks.<Event>]]``. Matcher field is a **regex** (Codex), not the boolean
# expression used by Claude. Events supported: SessionStart, PreToolUse,
# PermissionRequest, PostToolUse, UserPromptSubmit, Stop — Codex has no
# PreCompact event.
# ─────────────────────────────────────────────────────────────────────────────


# Regex form of the Claude expression ``tool != X && tool != Y && tool != Z``.
# Codex matchers use Python-style regex against ``tool_name``.
_CODEX_POSTTOOL_REGEX = r"^(?!TodoRead$|TodoWrite$|TaskList$).+$"


def _codex_hook_specs() -> list[tuple[str, str | None, str, int]]:
    """Return ``(event, matcher, command, timeout_secs)`` for Codex events.

    Matcher is ``None`` for events where Codex ignores the field
    (UserPromptSubmit, Stop) or where we want to match every fire.
    """
    return [
        ("SessionStart", "startup|resume", _find_session_start_command(), 10),
        ("PostToolUse", _CODEX_POSTTOOL_REGEX, _find_post_tool_use_command(), 5),
        ("Stop", None, _find_stop_command(), 30),
    ]


def _codex_command_is_nmem(cmd: str) -> bool:
    return "neural_memory" in cmd or "nmem-hook-" in cmd or "nmem " in cmd or cmd.startswith("nmem")


def _codex_event_has_nmem_hook(config: dict[str, Any], event: str) -> bool:
    """Check if the Codex ``[[hooks.<event>]]`` array already has a NM entry."""
    hooks_root = config.get("hooks", {})
    if not isinstance(hooks_root, dict):
        return False
    entries = hooks_root.get(event, [])
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if isinstance(entry, dict):
            cmd = str(entry.get("command", ""))
            if _codex_command_is_nmem(cmd):
                return True
    return False


def _codex_load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _toml_escape(value: str) -> str:
    """Escape a string for use inside a TOML basic string."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _render_codex_hook_entry(
    event: str, matcher: str | None, command: str, timeout_secs: int
) -> str:
    """Render one ``[[hooks.<event>]]`` block as TOML text."""
    lines = [f"[[hooks.{event}]]"]
    if matcher is not None:
        lines.append(f'matcher = "{_toml_escape(matcher)}"')
    lines.append(f'command = "{_toml_escape(command)}"')
    lines.append(f"timeout_secs = {timeout_secs}")
    return "\n".join(lines) + "\n"


def setup_hooks_codex() -> str:
    """Auto-configure NeuralMemory hooks in Codex CLI (``~/.codex/config.toml``).

    Appends NM ``[[hooks.<Event>]]`` array-of-tables for SessionStart,
    PostToolUse, and Stop. Skips events that already have an NM entry so
    re-runs are idempotent.

    Returns ``"added"`` (new entries appended), ``"exists"`` (no-op),
    ``"failed"`` (I/O error), or ``"not_found"`` (no ``~/.codex`` directory
    — caller should suggest the user install Codex CLI first).
    """
    codex_dir = Path.home() / ".codex"
    if not codex_dir.exists():
        return "not_found"

    config_path = codex_dir / "config.toml"
    existing = _codex_load_toml(config_path)

    blocks_to_append: list[str] = []
    for event, matcher, command, timeout in _codex_hook_specs():
        if _codex_event_has_nmem_hook(existing, event):
            continue
        blocks_to_append.append(_render_codex_hook_entry(event, matcher, command, timeout))

    if not blocks_to_append:
        return "exists"

    try:
        existing_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
        if existing_text and not existing_text.endswith("\n"):
            existing_text += "\n"
        if existing_text and not existing_text.endswith("\n\n"):
            existing_text += "\n"
        header = (
            "# ── NeuralMemory hooks (auto-added) ──\n"
            "# Lifecycle handlers — safe to re-run `nmem hooks install`.\n"
        )
        config_path.write_text(
            existing_text + header + "\n".join(blocks_to_append), encoding="utf-8"
        )
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
