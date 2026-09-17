"""Tests that ``nmem forget --hard`` wires the MCP confirmation gate.

Before this fix the CLI never passed ``confirm`` to the facade, so
``facade._forget`` always answered ``pending_confirmation`` — and the CLI
printed that payload as success. Hard deletion was impossible from the
command line, and the failure was invisible.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import typer
from typer.testing import CliRunner

from neural_memory.cli.commands.lifecycle import forget

runner = CliRunner()
app = typer.Typer()
app.command()(forget)


def _invoke(argv: list[str], payload: dict[str, Any]) -> tuple[Any, MagicMock]:
    """Run ``forget`` with a stubbed facade and return (result, facade)."""
    facade = MagicMock()
    facade._forget = AsyncMock(return_value=payload)
    with patch(
        "neural_memory.cli.commands.lifecycle._build_facade",
        new=AsyncMock(return_value=(MagicMock(), facade)),
    ):
        result = runner.invoke(app, argv)
    return result, facade


def _output_of(result: Any) -> str:
    """Combine stdout and stderr across click/typer versions."""
    parts = [result.output or ""]
    stderr = getattr(result, "stderr", "") or ""
    if stderr:
        parts.append(stderr)
    return "\n".join(parts)


class TestForgetConfirmationWiring:
    def test_force_passes_confirm_true(self) -> None:
        _result, facade = _invoke(
            ["mem-1", "--hard", "--force"], {"status": "hard_deleted", "memory_id": "mem-1"}
        )
        assert facade._forget.await_args.args[0]["confirm"] is True

    def test_hard_without_force_passes_confirm_false(self) -> None:
        _result, facade = _invoke(
            ["mem-1", "--hard"], {"status": "hard_deleted", "memory_id": "mem-1"}
        )
        assert facade._forget.await_args.args[0]["confirm"] is False

    def test_force_hard_delete_succeeds(self) -> None:
        result, _facade = _invoke(
            ["mem-1", "--hard", "--force"],
            {
                "status": "hard_deleted",
                "memory_id": "mem-1",
                "message": "Memory permanently deleted",
            },
        )
        assert result.exit_code == 0

    def test_pending_confirmation_is_reported_as_failure(self) -> None:
        """An unmet gate is not a deletion — it must not exit 0."""
        result, _facade = _invoke(
            ["mem-1", "--hard"],
            {"status": "pending_confirmation", "memory_id": "mem-1", "message": "Call again"},
        )
        assert result.exit_code == 1

    def test_pending_confirmation_mentions_force(self) -> None:
        result, _facade = _invoke(
            ["mem-1", "--hard"],
            {"status": "pending_confirmation", "memory_id": "mem-1", "message": "Call again"},
        )
        assert "--force" in _output_of(result)

    def test_soft_delete_is_unaffected(self) -> None:
        result, facade = _invoke(["mem-1"], {"status": "soft_deleted", "memory_id": "mem-1"})
        assert result.exit_code == 0
        payload = facade._forget.await_args.args[0]
        assert payload["hard"] is False
        assert payload["confirm"] is False

    def test_reason_and_id_are_forwarded(self) -> None:
        _result, facade = _invoke(
            ["mem-9", "--hard", "--force", "--reason", "duplicate"],
            {"status": "hard_deleted", "memory_id": "mem-9"},
        )
        payload = facade._forget.await_args.args[0]
        assert payload["memory_id"] == "mem-9"
        assert payload["reason"] == "duplicate"
