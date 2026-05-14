"""Tests for the PostToolUse light hook.

Covers schema compat with consumers (storage/sql/mixins/tool_events.py),
noise filter, config-toml blacklist overlay, env-var disable, concurrent
fire safety, buffer rotation, and the fast-path early exits in ``main()``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from neural_memory.hooks.post_tool_use import (
    _NOISE_TOOLS,
    _append_to_buffer,
    _check_buffer_rotation,
    _format_event,
    _get_blacklist,
    _get_buffer_path,
    _is_enabled,
    _is_filtered,
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
        result = _truncate_args(object())
        assert len(result) > 0


class TestIsFiltered:
    def test_empty_tool_name(self) -> None:
        assert _is_filtered("") is True

    def test_noise_tool_read(self) -> None:
        assert _is_filtered("Read") is True

    def test_noise_tool_todoread(self) -> None:
        assert _is_filtered("TodoRead") is True

    def test_high_value_tool_edit(self) -> None:
        assert _is_filtered("Edit") is False

    def test_high_value_tool_bash(self) -> None:
        assert _is_filtered("Bash") is False

    def test_mcp_tool_passes(self) -> None:
        assert _is_filtered("mcp__neural-memory__nmem_recall") is False

    def test_extra_prefix_filter(self) -> None:
        assert _is_filtered("CustomTool", ["Custom"]) is True
        assert _is_filtered("OtherTool", ["Custom"]) is False


class TestFormatEvent:
    def test_basic_format(self) -> None:
        hook_input = {
            "tool_name": "Edit",
            "server_name": "filesystem",
            "tool_input": {"path": "/tmp/test.py"},
            "duration_ms": 50,
        }
        event = _format_event(hook_input)
        assert event["tool_name"] == "Edit"
        assert event["server_name"] == "filesystem"
        assert event["success"] is True
        assert event["duration_ms"] == 50
        assert "created_at" in event
        # created_at must be ISO 8601 parseable (consumer uses datetime.fromisoformat)
        from datetime import datetime

        datetime.fromisoformat(event["created_at"])

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
        event = _format_event({"tool_name": "Edit"})
        assert event["tool_name"] == "Edit"
        assert event["server_name"] == ""
        assert event["duration_ms"] == 0
        assert event["success"] is True
        assert event["task_context"] == ""

    def test_invalid_duration(self) -> None:
        event = _format_event({"tool_name": "Edit", "duration_ms": "not-a-number"})
        assert event["duration_ms"] == 0

    def test_uses_tool_fallback_key(self) -> None:
        event = _format_event({"tool": "Write"})
        assert event["tool_name"] == "Write"

    def test_schema_has_required_consumer_fields(self) -> None:
        """storage/sql/mixins/tool_events.py expects these keys for INSERT."""
        event = _format_event({"tool_name": "Edit"})
        for required in (
            "tool_name",
            "server_name",
            "args_summary",
            "success",
            "duration_ms",
            "session_id",
            "task_context",
            "created_at",
        ):
            assert required in event, f"missing consumer-required key: {required}"

    def test_codex_session_id_fallback(self) -> None:
        with patch.dict(os.environ, {"CODEX_SESSION_ID": "cdx-1"}, clear=False):
            os.environ.pop("CLAUDE_SESSION_ID", None)
            event = _format_event({"tool_name": "Edit"})
        assert event["session_id"] == "cdx-1"


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
        assert len(remaining) == 100

    def test_rotation_missing_file(self, tmp_path: Path) -> None:
        buf = tmp_path / "nonexistent.jsonl"
        _check_buffer_rotation(buf)  # must not raise


class TestReadStdin:
    def test_valid_json(self) -> None:
        with patch.object(sys, "stdin", StringIO('{"tool_name": "Edit"}')):
            result = _read_stdin()
        assert result == {"tool_name": "Edit"}

    def test_empty_stdin(self) -> None:
        with patch.object(sys, "stdin", StringIO("")):
            result = _read_stdin()
        assert result == {}

    def test_invalid_json(self) -> None:
        with patch.object(sys, "stdin", StringIO("not json")):
            result = _read_stdin()
        assert result == {}

    def test_non_dict_json(self) -> None:
        with patch.object(sys, "stdin", StringIO('["a", "b"]')):
            result = _read_stdin()
        assert result == {}


class TestIsEnabled:
    def test_no_config_file(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is True

    def test_enabled_true(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[tool_memory]\nenabled = true\n")
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is True

    def test_enabled_false(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[tool_memory]\nenabled = false\n")
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is False

    def test_missing_section(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[general]\nbrain = 'default'\n")
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is True

    def test_corrupt_toml_falls_back(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("not valid = = = toml")
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _is_enabled() is True


class TestGetBlacklist:
    def test_no_config(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _get_blacklist() == []

    def test_blacklist_present(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text(
            '[tool_memory]\nblacklist = ["CustomTool", "OtherPrefix"]\n'
        )
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _get_blacklist() == ["CustomTool", "OtherPrefix"]

    def test_blacklist_invalid_type(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text('[tool_memory]\nblacklist = "notalist"\n')
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            assert _get_blacklist() == []


class TestGetBufferPath:
    def test_custom_dir(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}):
            path = _get_buffer_path()
        assert Path(path) == tmp_path / "tool_events.jsonl"

    def test_default_dir(self) -> None:
        env_without = {k: v for k, v in os.environ.items() if k != "NEURALMEMORY_DIR"}
        with patch.dict(os.environ, env_without, clear=True):
            path = _get_buffer_path()
        assert path.endswith("tool_events.jsonl")
        assert ".neuralmemory" in path.replace("\\", "/")


class TestAppendToBuffer:
    def test_creates_and_appends(self, tmp_path: Path) -> None:
        buf = tmp_path / "sub" / "events.jsonl"
        event = {"tool_name": "Edit", "created_at": "2026-01-01T00:00:00+00:00"}
        assert _append_to_buffer(event, buf) is True
        assert buf.exists()
        data = json.loads(buf.read_text().strip())
        assert data["tool_name"] == "Edit"

    def test_appends_multiple(self, tmp_path: Path) -> None:
        buf = tmp_path / "events.jsonl"
        _append_to_buffer({"tool_name": "Edit"}, buf)
        _append_to_buffer({"tool_name": "Write"}, buf)
        lines = buf.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_concurrent_subprocess_writes_no_corruption(self, tmp_path: Path) -> None:
        """Real scenario: N parallel hook subprocesses append to same file.

        Each fire is its own OS process — this is how Claude Code / Codex run
        hooks. Verifies cross-process locking via O_APPEND or OS file lock.
        """
        n_writers = 6
        events_per_writer = 5
        env = {**os.environ, "NEURALMEMORY_DIR": str(tmp_path)}
        env.pop("NEURALMEMORY_DISABLE_HOOKS", None)

        procs = []
        for tid in range(n_writers):
            payloads = "\n".join(
                json.dumps({"tool_name": "Edit", "tid": tid, "seq": i})
                for i in range(events_per_writer)
            )
            # One subprocess per writer; reads its single stdin payload.
            # Command is fully-controlled (no user input) — ruff S603 false positive.
            for line in payloads.splitlines():
                p = subprocess.Popen(
                    [sys.executable, "-m", "neural_memory.hooks.post_tool_use"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                p.stdin.write(line.encode("utf-8"))  # type: ignore[union-attr]
                p.stdin.close()  # type: ignore[union-attr]
                procs.append(p)

        for p in procs:
            p.wait(timeout=30)
            assert p.returncode == 0, f"hook exited with {p.returncode}"

        buf = tmp_path / "tool_events.jsonl"
        assert buf.exists()
        lines = buf.read_text().splitlines()
        assert len(lines) == n_writers * events_per_writer, (
            f"lost lines: got {len(lines)} expected {n_writers * events_per_writer}"
        )
        for line in lines:
            event = json.loads(line)  # must parse — no torn writes
            assert event["tool_name"] == "Edit"


class TestMain:
    def _stdout(self) -> StringIO:
        return StringIO()

    def test_disabled_via_env_var(self, tmp_path: Path) -> None:
        stdout = self._stdout()
        with (
            patch.dict(
                os.environ,
                {
                    "NEURALMEMORY_DIR": str(tmp_path),
                    "NEURALMEMORY_DISABLE_HOOKS": "1",
                },
            ),
            patch.object(sys, "stdin", StringIO('{"tool_name": "Edit"}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"
        assert not (tmp_path / "tool_events.jsonl").exists()

    def test_disabled_via_config(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[tool_memory]\nenabled = false\n")
        stdout = self._stdout()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}, clear=False),
            patch.object(sys, "stdin", StringIO('{"tool_name": "Edit"}')),
            patch.object(sys, "stdout", stdout),
        ):
            os.environ.pop("NEURALMEMORY_DISABLE_HOOKS", None)
            main()
        assert stdout.getvalue().strip() == "{}"
        assert not (tmp_path / "tool_events.jsonl").exists()

    def test_empty_stdin(self, tmp_path: Path) -> None:
        stdout = self._stdout()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}, clear=False),
            patch.object(sys, "stdin", StringIO("")),
            patch.object(sys, "stdout", stdout),
        ):
            os.environ.pop("NEURALMEMORY_DISABLE_HOOKS", None)
            main()
        assert stdout.getvalue().strip() == "{}"

    def test_no_tool_name(self, tmp_path: Path) -> None:
        stdout = self._stdout()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}, clear=False),
            patch.object(sys, "stdin", StringIO('{"server_name": "test"}')),
            patch.object(sys, "stdout", stdout),
        ):
            os.environ.pop("NEURALMEMORY_DISABLE_HOOKS", None)
            main()
        assert stdout.getvalue().strip() == "{}"
        assert not (tmp_path / "tool_events.jsonl").exists()

    def test_noise_tool_filtered(self, tmp_path: Path) -> None:
        stdout = self._stdout()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}, clear=False),
            patch.object(sys, "stdin", StringIO('{"tool_name": "Read"}')),
            patch.object(sys, "stdout", stdout),
        ):
            os.environ.pop("NEURALMEMORY_DISABLE_HOOKS", None)
            main()
        assert stdout.getvalue().strip() == "{}"
        assert not (tmp_path / "tool_events.jsonl").exists()

    def test_config_blacklist_overlays_noise(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text('[tool_memory]\nblacklist = ["CustomNoise"]\n')
        stdout = self._stdout()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}, clear=False),
            patch.object(sys, "stdin", StringIO('{"tool_name": "CustomNoiseTool"}')),
            patch.object(sys, "stdout", stdout),
        ):
            os.environ.pop("NEURALMEMORY_DISABLE_HOOKS", None)
            main()
        assert stdout.getvalue().strip() == "{}"
        assert not (tmp_path / "tool_events.jsonl").exists()

    def test_successful_capture(self, tmp_path: Path) -> None:
        stdout = self._stdout()
        with (
            patch.dict(os.environ, {"NEURALMEMORY_DIR": str(tmp_path)}, clear=False),
            patch.object(sys, "stdin", StringIO('{"tool_name": "Edit", "duration_ms": 25}')),
            patch.object(sys, "stdout", stdout),
        ):
            os.environ.pop("NEURALMEMORY_DISABLE_HOOKS", None)
            main()
        assert stdout.getvalue().strip() == "{}"
        buf = tmp_path / "tool_events.jsonl"
        assert buf.exists()
        event = json.loads(buf.read_text().strip())
        assert event["tool_name"] == "Edit"
        assert event["duration_ms"] == 25
        assert event["success"] is True


class TestNoiseTools:
    def test_default_noise_set_contains_expected(self) -> None:
        """Lock the default filter to catch accidental shrinkage."""
        for expected in ("Read", "Glob", "Grep", "TodoRead", "TodoWrite", "TaskList"):
            assert expected in _NOISE_TOOLS
