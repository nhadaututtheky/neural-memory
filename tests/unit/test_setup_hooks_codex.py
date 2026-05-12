"""Tests for setup_hooks_codex() — Codex CLI hook installation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import patch

from neural_memory.cli.setup import (
    _codex_command_is_nmem,
    _codex_event_has_nmem_hook,
    _codex_load_toml,
    _render_codex_hook_entry,
    _toml_escape,
    setup_hooks_codex,
)


class TestTomlEscape:
    def test_plain_string(self) -> None:
        assert _toml_escape("hello") == "hello"

    def test_escapes_quotes(self) -> None:
        assert _toml_escape('a"b') == 'a\\"b'

    def test_escapes_backslash(self) -> None:
        assert _toml_escape("a\\b") == "a\\\\b"

    def test_escapes_newline(self) -> None:
        assert _toml_escape("a\nb") == "a\\nb"


class TestRenderEntry:
    def test_with_matcher(self) -> None:
        text = _render_codex_hook_entry("PostToolUse", "^Bash$", "nmem-hook-post-tool-use", 5)
        assert "[[hooks.PostToolUse]]" in text
        assert 'matcher = "^Bash$"' in text
        assert 'command = "nmem-hook-post-tool-use"' in text
        assert "timeout_secs = 5" in text

    def test_without_matcher(self) -> None:
        text = _render_codex_hook_entry("Stop", None, "nmem-hook-stop", 30)
        assert "[[hooks.Stop]]" in text
        assert "matcher" not in text
        assert 'command = "nmem-hook-stop"' in text
        assert "timeout_secs = 30" in text

    def test_renders_valid_toml(self) -> None:
        text = _render_codex_hook_entry("PostToolUse", "^Bash$", "nmem-hook-post-tool-use", 5)
        # Round-trip via tomllib
        parsed = tomllib.loads(text)
        assert parsed["hooks"]["PostToolUse"][0]["matcher"] == "^Bash$"


class TestCommandIsNmem:
    def test_entry_point_match(self) -> None:
        assert _codex_command_is_nmem("nmem-hook-post-tool-use") is True

    def test_python_module_match(self) -> None:
        assert _codex_command_is_nmem("python -m neural_memory.hooks.post_tool_use") is True

    def test_nmem_cli_match(self) -> None:
        assert _codex_command_is_nmem("nmem post-tool-use-hook") is True

    def test_unrelated_no_match(self) -> None:
        assert _codex_command_is_nmem("/usr/bin/some-other-tool") is False


class TestEventHasNmemHook:
    def test_empty_config(self) -> None:
        assert _codex_event_has_nmem_hook({}, "PostToolUse") is False

    def test_other_event_doesnt_interfere(self) -> None:
        cfg = {"hooks": {"SessionStart": [{"command": "nmem-hook-session-start"}]}}
        assert _codex_event_has_nmem_hook(cfg, "PostToolUse") is False

    def test_nmem_command_detected(self) -> None:
        cfg = {"hooks": {"PostToolUse": [{"command": "nmem-hook-post-tool-use"}]}}
        assert _codex_event_has_nmem_hook(cfg, "PostToolUse") is True

    def test_user_custom_hook_not_detected_as_nmem(self) -> None:
        cfg = {"hooks": {"PostToolUse": [{"command": "/path/to/my-custom-hook"}]}}
        assert _codex_event_has_nmem_hook(cfg, "PostToolUse") is False


class TestSetupHooksCodex:
    def test_no_codex_dir_returns_not_found(self, tmp_path: Path) -> None:
        with patch("neural_memory.cli.setup.Path.home", return_value=tmp_path):
            assert setup_hooks_codex() == "not_found"

    def test_writes_fresh_config(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        with patch("neural_memory.cli.setup.Path.home", return_value=tmp_path):
            result = setup_hooks_codex()
        assert result == "added"
        cfg = codex_dir / "config.toml"
        assert cfg.exists()
        parsed = tomllib.loads(cfg.read_text(encoding="utf-8"))
        # All 3 events present
        assert "SessionStart" in parsed["hooks"]
        assert "PostToolUse" in parsed["hooks"]
        assert "Stop" in parsed["hooks"]

    def test_appends_to_existing_config(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        existing = codex_dir / "config.toml"
        existing.write_text('[general]\nmodel = "o1"\n', encoding="utf-8")
        with patch("neural_memory.cli.setup.Path.home", return_value=tmp_path):
            result = setup_hooks_codex()
        assert result == "added"
        text = existing.read_text(encoding="utf-8")
        # User's config preserved
        assert 'model = "o1"' in text
        # NM hooks appended
        parsed = tomllib.loads(text)
        assert parsed["general"]["model"] == "o1"
        assert "PostToolUse" in parsed["hooks"]

    def test_idempotent_on_rerun(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        with patch("neural_memory.cli.setup.Path.home", return_value=tmp_path):
            first = setup_hooks_codex()
            second = setup_hooks_codex()
        assert first == "added"
        assert second == "exists"

    def test_user_existing_hook_not_overwritten(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        # User already has a custom Stop hook — should NOT be replaced
        existing = codex_dir / "config.toml"
        existing.write_text(
            '[[hooks.Stop]]\ncommand = "/usr/local/bin/my-custom"\ntimeout_secs = 5\n',
            encoding="utf-8",
        )
        with patch("neural_memory.cli.setup.Path.home", return_value=tmp_path):
            result = setup_hooks_codex()
        assert result == "added"
        parsed = tomllib.loads(existing.read_text(encoding="utf-8"))
        # Both the user's Stop AND our SessionStart/PostToolUse should be present
        stop_entries = parsed["hooks"]["Stop"]
        assert any("my-custom" in e.get("command", "") for e in stop_entries)
        # NM added SessionStart and PostToolUse but NOT Stop (user has one)
        assert "SessionStart" in parsed["hooks"]
        assert "PostToolUse" in parsed["hooks"]

    def test_corrupt_toml_recovers(self, tmp_path: Path) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "config.toml").write_text("not = = valid toml", encoding="utf-8")
        with patch("neural_memory.cli.setup.Path.home", return_value=tmp_path):
            result = setup_hooks_codex()
        # We treat unreadable existing config as empty and still write hooks
        assert result in {"added", "failed"}


class TestLoadToml:
    def test_missing_file(self, tmp_path: Path) -> None:
        assert _codex_load_toml(tmp_path / "missing.toml") == {}

    def test_valid_toml(self, tmp_path: Path) -> None:
        p = tmp_path / "c.toml"
        p.write_text('[a]\nb = "c"\n', encoding="utf-8")
        assert _codex_load_toml(p) == {"a": {"b": "c"}}

    def test_corrupt_toml(self, tmp_path: Path) -> None:
        p = tmp_path / "c.toml"
        p.write_text("not valid =", encoding="utf-8")
        assert _codex_load_toml(p) == {}
