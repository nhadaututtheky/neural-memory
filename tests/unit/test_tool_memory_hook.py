"""Tests for the PostToolUse hook (v2.17.0)."""

from __future__ import annotations

import json
from pathlib import Path

from neural_memory.hooks.post_tool_use import (
    _check_buffer_rotation,
    _format_event,
    _truncate_args,
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
