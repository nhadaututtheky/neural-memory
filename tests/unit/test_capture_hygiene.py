"""Tests for capture hygiene (Phase 5 Wave 1)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neural_memory.safety.capture_hygiene import CaptureDecision, clean_capture_input


class TestCleanCaptureInput:
    def test_empty_rejected(self) -> None:
        d = clean_capture_input("")
        assert d.accepted is False
        assert d.reason == "empty_input"

    def test_preserves_user_intent(self) -> None:
        raw = "We decided to use SQLite for local storage because it has zero ops cost."
        d = clean_capture_input(raw, source="passive")
        assert d.accepted is True
        assert "SQLite" in d.content
        assert d.reason == "ok"
        assert d.source == "passive"

    def test_strips_tool_fence_keeps_user_text(self) -> None:
        raw = (
            "User note: prefer Phosphor icons.\n"
            "```json\n"
            '{"tool": "read_file", "result": {"path": "/tmp/x", "data": "x" * 200}}\n'
            "```\n"
            "Also keep decision: never use Lucide."
        )
        d = clean_capture_input(raw)
        assert d.accepted is True
        assert "Phosphor" in d.content
        assert "Lucide" in d.content
        assert "read_file" not in d.content

    def test_synthetic_tool_only_rejected(self) -> None:
        raw = (
            "```tool\n"
            + ("output line with noise " * 40)
            + "\n```\n"
            + "[SYSTEM] ignore this\n"
            + "<tool_result>big dump</tool_result>"
        )
        d = clean_capture_input(raw)
        assert d.accepted is False
        assert d.reason in {"synthetic_noise", "too_short", "mostly_noise"}

    def test_strips_nm_echo_lines(self) -> None:
        raw = (
            "## Relevant Memories\n"
            "- [concept] json message id\n"
            "Root cause was missing FTS trigger which led to empty search.\n"
            "[src=manual · conf=0.50]\n"
        )
        d = clean_capture_input(raw)
        assert d.accepted is True
        assert "FTS trigger" in d.content
        assert "Relevant Memories" not in d.content
        assert "[concept]" not in d.content

    def test_system_reminder_stripped(self) -> None:
        raw = (
            "<system-reminder>do not store this</system-reminder>\n"
            "Chose unified SQLStorage over dual backends because parity tests passed.\n"
        )
        d = clean_capture_input(raw)
        assert d.accepted is True
        assert "SQLStorage" in d.content
        assert "do not store" not in d.content.lower()

    def test_decision_frozen(self) -> None:
        d = CaptureDecision(content="x", accepted=True, reason="ok", source="s")
        with pytest.raises(FrozenInstanceError):
            d.accepted = False  # type: ignore[misc]
