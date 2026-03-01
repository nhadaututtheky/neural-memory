"""PostToolUse hook: capture MCP tool call metadata for tool memory.

Called by Claude Code after every tool call. Writes lightweight metadata
to a JSONL buffer file for deferred processing. Must complete in < 50ms.

Usage as Claude Code hook:
    Receives JSON on stdin with tool_name, tool_input, tool_output fields.
    Writes one JSONL line to ~/.neuralmemory/tool_events.jsonl.

This hook does NOT access SQLite or perform encoding — all processing
is deferred to the consolidation cycle.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Max characters for args summary (prevent OOM on huge tool inputs)
_MAX_ARGS_CHARS = 200
# Max size for stdout response JSON
_MAX_TOOL_OUTPUT_PREVIEW = 100


def _read_stdin() -> dict[str, Any]:
    """Read Claude Code PostToolUse hook JSON from stdin."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        result: dict[str, Any] = json.loads(raw)
        return result
    except (json.JSONDecodeError, OSError):
        return {}


def _get_buffer_path() -> Path:
    """Get the JSONL buffer file path."""
    data_dir = Path(os.environ.get("NEURALMEMORY_DIR", "")) or (Path.home() / ".neuralmemory")
    return data_dir / "tool_events.jsonl"


def _is_enabled() -> bool:
    """Quick check if tool memory is enabled via config.

    Reads only the [tool_memory] section from config.toml.
    Falls back to False if config is missing or malformed.
    """
    data_dir = Path(os.environ.get("NEURALMEMORY_DIR", "")) or (Path.home() / ".neuralmemory")
    config_path = data_dir / "config.toml"
    if not config_path.exists():
        return False
    try:
        import tomllib

        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return bool(data.get("tool_memory", {}).get("enabled", False))
    except Exception:
        logger.debug("Failed to read tool_memory.enabled from config", exc_info=True)
        return False


def _get_blacklist() -> list[str]:
    """Read blacklist from config.toml."""
    data_dir = Path(os.environ.get("NEURALMEMORY_DIR", "")) or (Path.home() / ".neuralmemory")
    config_path = data_dir / "config.toml"
    if not config_path.exists():
        return []
    try:
        import tomllib

        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        bl = data.get("tool_memory", {}).get("blacklist", [])
        return list(bl) if isinstance(bl, (list, tuple)) else []
    except Exception:
        logger.debug("Failed to read tool_memory.blacklist from config", exc_info=True)
        return []


def _truncate_args(tool_input: Any) -> str:
    """Truncate tool input to a short summary string."""
    if tool_input is None:
        return ""
    try:
        raw = json.dumps(tool_input, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        raw = str(tool_input)
    return raw[:_MAX_ARGS_CHARS]


def _format_event(hook_input: dict[str, Any]) -> dict[str, Any]:
    """Format hook input into a JSONL event dict."""
    from neural_memory.utils.timeutils import utcnow

    tool_name = hook_input.get("tool_name", hook_input.get("tool", ""))
    server_name = hook_input.get("server_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_error = hook_input.get("tool_error")
    duration_ms = hook_input.get("duration_ms", 0)
    session_id = os.environ.get("CLAUDE_SESSION_ID", "")

    return {
        "tool_name": str(tool_name),
        "server_name": str(server_name),
        "args_summary": _truncate_args(tool_input),
        "success": tool_error is None,
        "duration_ms": int(duration_ms) if isinstance(duration_ms, (int, float)) else 0,
        "session_id": session_id,
        "task_context": "",  # Populated by processing engine if session is active
        "created_at": utcnow().isoformat(),
    }


def _append_to_buffer(event: dict[str, Any], buffer_path: Path) -> bool:
    """Append one JSONL line to the buffer file.

    Returns True on success, False on failure.
    """
    try:
        buffer_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=str)
        with open(buffer_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except OSError:
        return False


def _check_buffer_rotation(buffer_path: Path, max_lines: int = 10000) -> None:
    """Truncate buffer if it exceeds max_lines (keep newest half)."""
    if not buffer_path.exists():
        return
    try:
        content = buffer_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        if len(lines) <= max_lines:
            return
        # Keep newest half
        keep = lines[len(lines) // 2 :]
        buffer_path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    except OSError:
        pass


def main() -> None:
    """Entry point for the PostToolUse hook."""
    start = time.monotonic()

    # Fast exit if disabled
    if not _is_enabled():
        # Output empty JSON for hook response
        sys.stdout.write("{}\n")
        return

    hook_input = _read_stdin()
    if not hook_input:
        sys.stdout.write("{}\n")
        return

    tool_name = hook_input.get("tool_name", hook_input.get("tool", ""))
    if not tool_name:
        sys.stdout.write("{}\n")
        return

    # Check blacklist
    blacklist = _get_blacklist()
    for prefix in blacklist:
        if str(tool_name).startswith(prefix):
            sys.stdout.write("{}\n")
            return

    # Format and write event
    event = _format_event(hook_input)
    buffer_path = _get_buffer_path()
    _append_to_buffer(event, buffer_path)

    # Periodic buffer rotation check (every ~100 calls, cheap stat check)
    try:
        if buffer_path.exists() and buffer_path.stat().st_size > 5_000_000:  # > 5MB
            _check_buffer_rotation(buffer_path)
    except OSError:
        pass

    elapsed_ms = (time.monotonic() - start) * 1000
    if elapsed_ms > 50:
        sys.stderr.write(f"[NeuralMemory] PostToolUse hook slow: {elapsed_ms:.0f}ms\n")

    sys.stdout.write("{}\n")


if __name__ == "__main__":
    main()
