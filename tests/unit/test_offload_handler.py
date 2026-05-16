"""Tests for nmem_offload / nmem_inflate MCP tools (Phase 1 agent ergonomics)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.neuron import Neuron


def _make_handler(brain_id: str = "test-brain"):
    """Build a minimal handler with mock storage — mirrors test_show_handler pattern."""
    from neural_memory.mcp.offload_handler import OffloadHandler

    storage = AsyncMock()
    storage.brain_id = brain_id
    storage.current_brain_id = brain_id
    storage._current_brain_id = brain_id

    # In-memory neuron table to round-trip offload → inflate.
    stored: dict[str, Neuron] = {}

    async def _add(neuron: Neuron) -> str:
        stored[neuron.id] = neuron
        return neuron.id

    async def _get(neuron_id: str) -> Neuron | None:
        return stored.get(neuron_id)

    storage.add_neuron = AsyncMock(side_effect=_add)
    storage.get_neuron = AsyncMock(side_effect=_get)

    brain_mock = MagicMock(id=brain_id, config=MagicMock())
    storage.get_brain = AsyncMock(return_value=brain_mock)

    class TestHandler(OffloadHandler):
        config = MagicMock()
        config.safety = MagicMock(auto_redact_min_severity=3)

        async def get_storage(self) -> Any:
            return storage

    return TestHandler(), storage, stored


# ──────────────────── nmem_offload ────────────────────


class TestOffload:
    @pytest.mark.asyncio
    async def test_offload_basic(self) -> None:
        handler, _storage, stored = _make_handler()
        content = "tool result line 1\n" * 100  # ~1.8KB
        result = await handler._offload({"content": content, "tool_name": "ls"})
        assert "ref_id" in result
        assert "summary" in result
        assert "token_saved" in result
        assert "redacted" in result
        # Round-trip stored — sanitizer may normalize trailing whitespace, so
        # compare on content prefix rather than exact equality.
        assert result["ref_id"] in stored
        assert stored[result["ref_id"]].content.startswith("tool result line 1")
        assert stored[result["ref_id"]].content.count("tool result line 1") == 100
        assert stored[result["ref_id"]].ephemeral is True

    @pytest.mark.asyncio
    async def test_offload_summary_truncates(self) -> None:
        handler, _storage, _stored = _make_handler()
        content = "a" * 5000
        result = await handler._offload({"content": content, "tool_name": "read_file"})
        # Summary must be ≤ 250 chars (200 preview + size hint)
        assert len(result["summary"]) <= 250
        # token_saved is positive and proportional to original size
        assert result["token_saved"] > 100

    @pytest.mark.asyncio
    async def test_offload_empty_content(self) -> None:
        handler, _storage, _stored = _make_handler()
        result = await handler._offload({"content": "", "tool_name": "ls"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_offload_explicit_summary(self) -> None:
        """Caller-provided summary wins over auto-generation."""
        handler, _storage, _stored = _make_handler()
        result = await handler._offload(
            {
                "content": "x" * 1000,
                "tool_name": "grep",
                "summary": "12 matches in 3 files",
            }
        )
        assert result["summary"] == "12 matches in 3 files"


# ──────────────────── nmem_inflate ────────────────────


class TestInflate:
    @pytest.mark.asyncio
    async def test_inflate_round_trip(self) -> None:
        handler, _storage, _stored = _make_handler()
        original = "line\n" * 200
        off = await handler._offload({"content": original, "tool_name": "ls"})
        infl = await handler._inflate({"ref_id": off["ref_id"]})
        # Sanitizer may strip trailing whitespace — verify line content survives.
        assert infl["content"].count("line") == 200
        assert infl["tool_name"] == "ls"

    @pytest.mark.asyncio
    async def test_inflate_missing_ref(self) -> None:
        handler, _storage, _stored = _make_handler()
        result = await handler._inflate({"ref_id": "nonexistent-id"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_inflate_missing_arg(self) -> None:
        handler, _storage, _stored = _make_handler()
        result = await handler._inflate({})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_inflate_rejects_non_offload_neuron(self) -> None:
        """_inflate must refuse neurons that weren't created via _offload."""
        from neural_memory.core.neuron import NeuronType

        handler, storage, stored = _make_handler()
        rogue = Neuron.create(
            type=NeuronType.CONCEPT,
            content="not an offload payload",
            metadata={"_source": "manual"},
        )
        stored[rogue.id] = rogue
        result = await handler._inflate({"ref_id": rogue.id})
        assert "error" in result
        assert "not an offloaded payload" in result["error"]


class TestSummaryBounds:
    @pytest.mark.asyncio
    async def test_summary_bounded_with_long_tool_name(self) -> None:
        """Long tool_name must not blow the summary past _MAX_SUMMARY_LEN."""
        handler, _storage, _stored = _make_handler()
        long_tool = "x" * 100  # at schema max
        result = await handler._offload({"content": "a" * 10000, "tool_name": long_tool})
        assert len(result["summary"]) <= 300

    @pytest.mark.asyncio
    async def test_tool_name_truncated_in_storage(self) -> None:
        """Caller-supplied tool_name longer than 100 chars is truncated."""
        handler, _storage, stored = _make_handler()
        too_long = "y" * 5000
        result = await handler._offload({"content": "abc", "tool_name": too_long})
        neuron = stored[result["ref_id"]]
        assert len(neuron.metadata["_tool_name"]) <= 100


class TestSensitiveScan:
    @pytest.mark.asyncio
    async def test_offload_redacts_secrets(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """Tool output containing an API-key pattern is auto-redacted before storage."""
        from neural_memory.safety import sensitive as sensitive_mod

        # Synthetic redactor: replaces the marker with <REDACTED>
        def fake_redact(content: str, min_severity: int = 3):  # type: ignore[no-untyped-def]
            if "SECRET_TOKEN" in content:
                return (
                    content.replace("SECRET_TOKEN", "<REDACTED>"),
                    [MagicMock(type=MagicMock(value="api_key"), severity=4)],
                    0,
                )
            return content, [], 0

        monkeypatch.setattr(sensitive_mod, "auto_redact_content", fake_redact)

        handler, _storage, stored = _make_handler()
        result = await handler._offload(
            {"content": "before SECRET_TOKEN after", "tool_name": "curl"}
        )
        assert result.get("redacted") is True
        neuron = stored[result["ref_id"]]
        assert "SECRET_TOKEN" not in neuron.content
        assert "<REDACTED>" in neuron.content
