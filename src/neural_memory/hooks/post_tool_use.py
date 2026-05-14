"""PostToolUse hook: capture tool call metadata for deferred memory consolidation.

Light hook architecture — uses stdlib only, no ``neural_memory.*`` imports in hot
path. Cold-start measured ~54ms p50, ~65ms p95 on Windows / Python 3.14.

Schema written to ``~/.neuralmemory/tool_events.jsonl`` matches the consumer in
``storage/sql/mixins/tool_events.py`` and ``engine/tool_memory.py``::

    tool_name, server_name, args_summary, success, duration_ms,
    session_id, task_context, created_at  (ISO 8601 UTC)

Concurrent fires are safe — uses ``msvcrt`` (Windows) / ``fcntl`` (POSIX)
advisory locks around the JSONL append.

Disable via env var ``NEURALMEMORY_DISABLE_HOOKS=1`` or ``[tool_memory].enabled
= false`` in ``~/.neuralmemory/config.toml``. Config blacklist extends the
built-in noise filter rather than replacing it.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

_MAX_ARGS_CHARS = 200
_MAX_BUFFER_BYTES = 5_000_000
_NOISE_TOOLS = frozenset(
    {"Read", "Glob", "Grep", "TodoRead", "TodoWrite", "TaskList", "NotebookRead"}
)


def _data_dir() -> str:
    return os.environ.get("NEURALMEMORY_DIR") or os.path.expanduser("~/.neuralmemory")


def _read_stdin() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
    except OSError:
        return {}
    if not raw.strip():
        return {}
    try:
        result = json.loads(raw)
    except ValueError:
        return {}
    return result if isinstance(result, dict) else {}


def _load_config_section(section: str) -> dict[str, Any]:
    """Load one section of ``config.toml``. Returns ``{}`` if missing."""
    config_path = os.path.join(_data_dir(), "config.toml")
    if not os.path.exists(config_path):
        return {}
    try:
        import tomllib

        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, ValueError):
        return {}
    section_data = data.get(section, {})
    return section_data if isinstance(section_data, dict) else {}


def _is_enabled() -> bool:
    """Honor ``[tool_memory].enabled = false`` in config.toml (default: True)."""
    cfg = _load_config_section("tool_memory")
    return bool(cfg.get("enabled", True))


def _get_blacklist() -> list[str]:
    """Extra prefix blacklist from config.toml, layered on _NOISE_TOOLS."""
    cfg = _load_config_section("tool_memory")
    bl = cfg.get("blacklist", [])
    return list(bl) if isinstance(bl, (list, tuple)) else []


def _is_filtered(tool_name: str, extra_prefixes: list[str] | None = None) -> bool:
    if not tool_name:
        return True
    if tool_name in _NOISE_TOOLS:
        return True
    if extra_prefixes:
        for prefix in extra_prefixes:
            if tool_name.startswith(prefix):
                return True
    return False


def _truncate_args(tool_input: Any) -> str:
    if tool_input is None:
        return ""
    try:
        raw = json.dumps(tool_input, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        raw = str(tool_input)
    return raw[:_MAX_ARGS_CHARS]


def _get_buffer_path() -> str:
    return os.path.join(_data_dir(), "tool_events.jsonl")


def _format_event(hook_input: dict[str, Any]) -> dict[str, Any]:
    """Build the JSONL event dict from raw stdin payload."""
    tool_name = str(hook_input.get("tool_name", hook_input.get("tool", "")))
    duration_ms = hook_input.get("duration_ms", 0)
    if not isinstance(duration_ms, (int, float)):
        duration_ms = 0
    session_id = os.environ.get("CLAUDE_SESSION_ID", os.environ.get("CODEX_SESSION_ID", ""))
    return {
        "tool_name": tool_name,
        "server_name": str(hook_input.get("server_name", "")),
        "args_summary": _truncate_args(hook_input.get("tool_input", {})),
        "success": hook_input.get("tool_error") is None,
        "duration_ms": int(duration_ms),
        "session_id": session_id,
        "task_context": "",
        "created_at": datetime.now(UTC).isoformat(),
    }


def _append_to_buffer(event: dict[str, Any], buffer_path: Any) -> bool:
    """Append one JSONL line atomically, safe under concurrent writers.

    Linux/macOS: ``O_APPEND`` makes ``os.write`` atomic at EOF for writes
    under ``PIPE_BUF`` (4KB on Linux, 512B POSIX min) — no explicit lock
    needed.

    Windows: ``O_APPEND`` is NOT atomic when multiple processes race the
    file. We acquire an exclusive ``msvcrt.locking`` byte-0 lock so all
    writers serialize at the OS level. ``LK_LOCK`` retries up to ~10s
    before raising, plenty for our ~50ms hot path.
    """
    path_str = str(buffer_path)
    try:
        parent = os.path.dirname(path_str)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode("utf-8")

        fd = os.open(path_str, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if os.name == "nt":
                import msvcrt

                # Position pointer at byte 0 so all writers contend on the
                # same lock region. O_APPEND still forces writes to EOF.
                os.lseek(fd, 0, 0)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined,unused-ignore]
                try:
                    os.write(fd, line)
                finally:
                    os.lseek(fd, 0, 0)
                    try:
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined,unused-ignore]
                    except OSError:
                        pass
            else:
                os.write(fd, line)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


def _check_buffer_rotation(buffer_path: Any, max_lines: int = 10000) -> None:
    """Truncate buffer to newest half if line count exceeds ``max_lines``."""
    path_str = str(buffer_path)
    if not os.path.exists(path_str):
        return
    try:
        with open(path_str, encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= max_lines:
            return
        keep = lines[len(lines) // 2 :]
        with open(path_str, "w", encoding="utf-8") as f:
            f.writelines(keep)
    except OSError:
        pass


def main() -> None:
    """Entry point: read stdin, filter, append JSONL, exit fast."""
    if os.environ.get("NEURALMEMORY_DISABLE_HOOKS"):
        sys.stdout.write("{}\n")
        return

    if not _is_enabled():
        sys.stdout.write("{}\n")
        return

    hook_input = _read_stdin()
    if not hook_input:
        sys.stdout.write("{}\n")
        return

    tool_name = str(hook_input.get("tool_name", hook_input.get("tool", "")))
    if _is_filtered(tool_name, _get_blacklist()):
        sys.stdout.write("{}\n")
        return

    event = _format_event(hook_input)
    buf_path = _get_buffer_path()
    _append_to_buffer(event, buf_path)

    try:
        if os.path.getsize(buf_path) > _MAX_BUFFER_BYTES:
            _check_buffer_rotation(buf_path)
    except OSError:
        pass

    sys.stdout.write("{}\n")


if __name__ == "__main__":
    main()
