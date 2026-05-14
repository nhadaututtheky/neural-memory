"""Tests for plugin manifest paths (issues #166, #167).

`.claude-plugin/plugin.json` and `.claude-plugin/hooks/hooks.json` are the
contract with Claude Code's plugin loader. These tests pin the paths so the
manifest doesn't drift back into broken states reported by users.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / ".claude-plugin"


@pytest.mark.skipif(
    not (_PLUGIN_DIR / "plugin.json").exists(),
    reason="plugin manifest only present in repo root, not in installed package",
)
class TestPluginManifest:
    """Pin .claude-plugin/plugin.json invariants."""

    def test_skills_path_resolves_to_real_directory(self) -> None:
        """Issue #166: 'skills' field must point to an existing folder.

        Claude Code's plugin loader resolves the 'skills' string relative to
        the plugin's *root* (the version dir), not relative to plugin.json.
        Using './skills' looks for a top-level skills folder that doesn't
        exist in our layout; the real folder is at .claude-plugin/skills/.
        """
        manifest = json.loads((_PLUGIN_DIR / "plugin.json").read_text())
        skills_rel = manifest.get("skills")
        assert isinstance(skills_rel, str), "plugin.json must declare a skills path"
        # Resolve from plugin root (the parent of .claude-plugin/)
        plugin_root = _PLUGIN_DIR.parent
        resolved = (plugin_root / skills_rel).resolve()
        assert resolved.is_dir(), (
            f"plugin.json skills='{skills_rel}' resolved to {resolved} which "
            "does not exist. Claude Code will refuse to load skills."
        )

    def test_skills_path_contains_expected_skills(self) -> None:
        manifest = json.loads((_PLUGIN_DIR / "plugin.json").read_text())
        skills_rel = manifest["skills"]
        skills_dir = (_PLUGIN_DIR.parent / skills_rel).resolve()
        names = {p.name for p in skills_dir.iterdir() if p.is_dir()}
        # These 3 skills are advertised in plugin metadata; missing any of
        # them means a user-facing capability regression.
        for expected in ("memory-audit", "memory-evolution", "memory-intake"):
            assert expected in names, f"Bundled skill {expected!r} missing from {skills_dir}"


@pytest.mark.skipif(
    not (_PLUGIN_DIR / "hooks" / "hooks.json").exists(),
    reason="plugin hooks manifest only present in repo root",
)
class TestPluginHooksManifest:
    """Pin .claude-plugin/hooks/hooks.json invariants (issue #167)."""

    def test_all_four_lifecycle_hooks_registered(self) -> None:
        manifest = json.loads((_PLUGIN_DIR / "hooks" / "hooks.json").read_text())
        events = {h["event"] for h in manifest.get("hooks", [])}
        # SessionStart was missing in 4.56.0 — issue #167. Pin all four so
        # the plugin install delivers the full lifecycle.
        assert events >= {"SessionStart", "PreCompact", "Stop", "PostToolUse"}, (
            f"Plugin hooks.json missing lifecycle events: {events}"
        )

    def test_session_start_uses_correct_binary(self) -> None:
        manifest = json.loads((_PLUGIN_DIR / "hooks" / "hooks.json").read_text())
        session_start = next(
            (h for h in manifest["hooks"] if h["event"] == "SessionStart"),
            None,
        )
        assert session_start is not None
        assert session_start["command"] == "nmem-hook-session-start", (
            "SessionStart hook must invoke the dedicated binary shipped by "
            "pip install neural-memory; renaming it silently breaks every "
            "plugin install."
        )
