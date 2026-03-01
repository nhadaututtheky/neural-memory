"""Tests for the PostToolUse hook (v2.17.0)."""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from neural_memory.hooks.post_tool_use import (
    _append_to_buffer,
    _check_buffer_rotation,
    _format_event,
    _get_blacklist,
    _get_buffer_path,
    _is_enabled,
    _read_stdin,
    _truncate_args,
    main,
)


class TestTruncateArgs:
    def test_short_args(self) -> None:
        result = _truncate_args({"query": "test"})
        assert len(result) <= 200
        assert "query" in result

    def test_long_args(self) -> None:
        big_input = {"data": "x" * 500}
        result = _truncate_args(big_input)
        assert len(result) == 200

    def test_none_args(self) -> None:
        assert _truncate_args(None) == ""

    def test_non_serializable(self) -> None:
        """Falls back to str() for non-serializable objects."""
        result = _truncate_args(object())
        assert len(result) > 0


class TestFormatEvent:
    def test_basic_format(self) -> None:
        hook_input = {
            "tool_name": "Read",
            "server_name": "filesystem",
            "tool_input": {"path": "/tmp/test.py"},
            "duration_ms": 50,
        }
        event = _format_event(hook_input)
        assert event["tool_name"] == "Read"
        assert event["server_name"] == "filesystem"
        assert event["success"] is True
        assert event["duration_ms"] == 50
        assert "created_at" in event

    def test_error_event(self) -> None:
        hook_input = {
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
            "tool_error": "Permission denied",
            "duration_ms": 10,
        }
        event = _format_event(hook_input)
        assert event["success"] is False

    def test_missing_fields(self) -> None:
        """Gracefully handles missing optional fields."""
        event = _format_event({"tool_name": "Read"})
        assert event["tool_name"] == "Read"
        assert event["server_name"] == ""
        assert event["duration_ms"] == 0
        assert event["success"] is True

    def test_invalid_duration(self) -> None:
        """Non-numeric duration defaults to 0."""
        event = _format_event({"tool_name": "Read", "duration_ms": "not-a-number"})
        assert event["duration_ms"] == 0

    def test_uses_tool_fallback_key(self) -> None:
        """Falls back from tool_name to tool key."""
        event = _format_event({"tool": "Write"})
        assert event["tool_name"] == "Write"


class TestBufferRotation:
    def test_no_rotation_small_buffer(self, tmp_path: Path) -> None:
        buf = tmp_path / "events.jsonl"
        lines = [json.dumps({"tool_name": f"tool-{i}"}) for i in range(10)]
        buf.write_text("\n".join(lines) + "\n")

        _check_buffer_rotation(buf, max_lines=100)
        assert len(buf.read_text().splitlines()) == 10

    def test_rotation_large_buffer(self, tmp_path: Path) -> None:
        buf = tmp_path / "events.jsonl"
        lines = [json.dumps({"tool_name": f"tool-{i}"}) for i in range(200)]
        buf.write_text("\n".join(lines) + "\n")

        _check_buffer_rotation(buf, max_lines=100)
        remaining = buf.read_text().splitlines()
        assert len(remaining) == 100  # Kept newest half

    def test_rotation_missing_file(self, tmp_path: Path) -> None:
        buf = tmp_path / "nonexistent.jsonl"
        _check_buffer_rotation(buf)  # Should not raise


class TestReadStdin:
    def test_valid_json(self) -> None:
        with patch.object(sys, "stdin", StringIO('{"tool_name": "Read"}')):
            result = _read_stdin()
        assert result == {"tool_name": "Read"}

    def test_empty_stdin(self) -> None:
        with patch.object(sys, "stdin", StringIO("")):
            result = _read_stdin()
        assert result == {}

    def test_invalid_json(self) -> None:
        with patch.object(sys, "stdin", StringIO("not json")):
            result = _read_stdin()
        assert result == {}


class TestIsEnabled:
    def test_no_config_file(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is False

    def test_enabled_true(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[tool_memory]\nenabled = true\n")
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is True

    def test_enabled_false(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[tool_memory]\nenabled = false\n")
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is False

    def test_missing_section(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[general]\nbrain = 'default'\n")
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is False


class TestGetBlacklist:
    def test_no_config(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _get_blacklist() == []

    def test_blacklist_present(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text('[tool_memory]\nblacklist = ["TodoRead", "TaskList"]\n')
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            result = _get_blacklist()
        assert result == ["TodoRead", "TaskList"]


class TestGetBufferPath:
    def test_custom_dir(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            path = _get_buffer_path()
        assert path == tmp_path / "tool_events.jsonl"


class TestAppendToBuffer:
    def test_creates_and_appends(self, tmp_path: Path) -> None:
        buf = tmp_path / "sub" / "events.jsonl"
        event = {"tool_name": "Read", "created_at": "2026-01-01T00:00:00"}
        assert _append_to_buffer(event, buf) is True
        assert buf.exists()
        data = json.loads(buf.read_text().strip())
        assert data["tool_name"] == "Read"

    def test_appends_multiple(self, tmp_path: Path) -> None:
        buf = tmp_path / "events.jsonl"
        _append_to_buffer({"tool_name": "Read"}, buf)
        _append_to_buffer({"tool_name": "Write"}, buf)
        lines = buf.read_text().strip().splitlines()
        assert len(lines) == 2


class TestMain:
    def test_disabled_exits_fast(self, tmp_path: Path) -> None:
        """When tool memory is disabled, main outputs empty JSON."""
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"

    def test_empty_stdin(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[tool_memory]\nenabled = true\n")
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO("")),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"

    def test_no_tool_name(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[tool_memory]\nenabled = true\n")
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO('{"server_name": "test"}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"

    def test_blacklisted_tool(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text('[tool_memory]\nenabled = true\nblacklist = ["Todo"]\n')
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO('{"tool_name": "TodoRead"}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"
        # No event file should be created
        assert not (tmp_path / "tool_events.jsonl").exists()

    def test_successful_capture(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[tool_memory]\nenabled = true\n")
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO('{"tool_name": "Read", "duration_ms": 25}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"
        buf = tmp_path / "tool_events.jsonl"
        assert buf.exists()
        event = json.loads(buf.read_text().strip())
        assert event["tool_name"] == "Read"
        assert event["success"] is True
