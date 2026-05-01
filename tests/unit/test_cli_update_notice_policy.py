"""Tests for CLI update notice routing and suppression."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from neural_memory.cli.main import _args_request_json, _should_run_update_check, app

runner = CliRunner()


class TestUpdateNoticePolicy:
    """Tests for opportunistic update-check command policy."""

    def test_json_flags_skip_update_check(self) -> None:
        ctx = MagicMock(invoked_subcommand="check")

        assert _should_run_update_check(ctx, ["check", "safe", "--json"]) is False
        assert _should_run_update_check(ctx, ["check", "safe", "-j"]) is False
        assert _should_run_update_check(ctx, ["check", "safe", "--json=true"]) is False

    def test_machine_oriented_commands_skip_update_check(self) -> None:
        for command in ("context", "recall", "stats", "status"):
            ctx = MagicMock(invoked_subcommand=command)
            assert _should_run_update_check(ctx, [command]) is False

    def test_interactive_commands_allow_update_check(self) -> None:
        ctx = MagicMock(invoked_subcommand="check")

        assert _should_run_update_check(ctx, ["check", "safe"]) is True

    def test_json_flag_detection(self) -> None:
        assert _args_request_json(["doctor", "--json"]) is True
        assert _args_request_json(["doctor", "-j"]) is True
        assert _args_request_json(["doctor", "--json=true"]) is True
        assert _args_request_json(["doctor"]) is False


class TestUpdateNoticeOutput:
    """Tests that notices do not contaminate machine-readable output."""

    def test_json_cli_output_stays_clean(self) -> None:
        with (
            patch("sys.argv", ["nmem", "check", "plain text", "--json"]),
            patch("neural_memory.cli.main._warn_if_not_initialized"),
            patch("neural_memory.cli.update_check.run_update_check_background") as mock_start,
        ):
            result = runner.invoke(app, ["check", "plain text", "--json"])

        assert result.exit_code == 0
        assert mock_start.call_count == 0
        payload = json.loads(result.stdout)
        assert payload == {"sensitive": False, "matches": []}

    def test_non_json_cli_can_start_background_update_check(self) -> None:
        with (
            patch("sys.argv", ["nmem", "check", "plain text"]),
            patch("neural_memory.cli.main._warn_if_not_initialized"),
            patch("neural_memory.cli.update_check.run_update_check_background") as mock_start,
        ):
            result = runner.invoke(app, ["check", "plain text"])

        assert result.exit_code == 0
        mock_start.assert_called_once_with()

    def test_update_notice_prints_to_stderr_only(self, capsys) -> None:  # type: ignore[no-untyped-def]
        from neural_memory.cli.update_check import _print_update_notice

        with patch("neural_memory.cli.update_check._is_editable_install", return_value=False):
            _print_update_notice("1.0.0", "1.1.0")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Update available" in captured.err
