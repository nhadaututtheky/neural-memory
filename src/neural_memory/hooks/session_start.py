"""SessionStart hook: inject Knowledge Surface as context for a new session.

Fires at the beginning of every Claude Code / Codex CLI session. Loads the
project-scoped ``.neuralmemory/surface.nm`` (or the global fallback) and
emits it as ``systemMessage`` so the agent starts with full context.

Stdlib-only — the surface file is pre-rendered by the Stop / consolidation
cycle, so this hook never needs to import ``neural_memory.*``. Fires at most
once per session so even if engine imports were required the cost would be
amortized; keeping it light lets it co-exist with the lazy ``__init__``.

Behaviour:
  * ``source == "resume"`` → skip (context already present)
  * Surface file missing OR empty → skip silently
  * Surface file present → output as systemMessage, truncated to budget

Disable via env ``NEURALMEMORY_DISABLE_HOOKS=1`` or
``[hooks].session_start = false`` in config.toml.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_DEFAULT_BUDGET_CHARS = 6000  # ≈ 1500 tokens at 4 chars/token


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
    cfg = _load_config_section("hooks")
    return bool(cfg.get("session_start", True))


def _emit_empty() -> None:
    sys.stdout.write("{}\n")


def _emit_system_message(text: str, budget_chars: int) -> None:
    """Write a hook response that injects ``text`` as systemMessage."""
    if len(text) > budget_chars:
        text = text[:budget_chars].rstrip() + "\n…[truncated to fit context budget]"
    response = {"systemMessage": text}
    sys.stdout.write(json.dumps(response) + "\n")


def _find_surface_for_cwd(cwd: str) -> str | None:
    """Walk up from ``cwd`` looking for ``.neuralmemory/surface.nm``.

    The user's home directory is intentionally NOT probed — a surface that
    lives at ``~/.neuralmemory/surface.nm`` is the global fallback, not a
    project surface. Stopping at home prevents accidental cross-project
    bleed for sessions started in unrelated subdirectories of home.
    """
    if not cwd:
        return None
    try:
        current = os.path.abspath(cwd)
    except OSError:
        return None
    home = os.path.abspath(os.path.expanduser("~"))
    for _ in range(20):
        if current == home:
            break
        candidate = os.path.join(current, ".neuralmemory", "surface.nm")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def _find_global_surface(brain_name: str = "default") -> str | None:
    candidate = os.path.join(_data_dir(), "surfaces", f"{brain_name}.nm")
    return candidate if os.path.exists(candidate) else None


def _load_surface_text(cwd: str) -> str | None:
    """Project-level surface wins over global. Returns trimmed content or None."""
    path = _find_surface_for_cwd(cwd) or _find_global_surface()
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    return text.strip() or None


def main() -> None:
    if os.environ.get("NEURALMEMORY_DISABLE_HOOKS"):
        _emit_empty()
        return

    if not _is_enabled():
        _emit_empty()
        return

    hook_input = _read_stdin()

    source = str(hook_input.get("source", "startup"))
    if source == "resume":
        # Existing transcript already carries context; don't double-inject.
        _emit_empty()
        return

    cwd = str(hook_input.get("cwd") or os.getcwd())
    surface = _load_surface_text(cwd)
    if not surface:
        _emit_empty()
        return

    cfg = _load_config_section("hooks")
    budget = cfg.get("session_start_budget_chars", _DEFAULT_BUDGET_CHARS)
    if not isinstance(budget, int) or budget <= 0:
        budget = _DEFAULT_BUDGET_CHARS

    header = "## Persistent memory (Neural Memory surface)\n"
    _emit_system_message(header + surface, budget)


if __name__ == "__main__":
    main()
