"""Tests for stop hook role filtering, memory markers, and embedding dedup."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from neural_memory.hooks.stop import (
    _embedding_dedup,
    _get_entry_role,
    _has_memory_markers,
    read_transcript_tail,
)


class TestGetEntryRole:
    """Tests for _get_entry_role()."""

    def test_direct_user_role(self) -> None:
        assert _get_entry_role({"role": "user", "content": "hello"}) == "user"

    def test_direct_assistant_role(self) -> None:
        assert _get_entry_role({"role": "assistant", "content": "hi"}) == "assistant"

    def test_direct_tool_role(self) -> None:
        assert _get_entry_role({"role": "tool", "content": "result"}) == "tool"

    def test_nested_message_role(self) -> None:
        entry = {"message": {"role": "assistant", "content": "nested"}}
        assert _get_entry_role(entry) == "assistant"

    def test_tool_result_type(self) -> None:
        entry = {"type": "tool_result", "content": "data"}
        assert _get_entry_role(entry) == "tool"

    def test_tool_use_type(self) -> None:
        entry = {"type": "tool_use", "name": "read", "input": {}}
        assert _get_entry_role(entry) == "tool"

    def test_tool_use_id_present(self) -> None:
        entry = {"tool_use_id": "abc123", "content": "result"}
        assert _get_entry_role(entry) == "tool"

    def test_content_list_with_tool_use(self) -> None:
        entry = {
            "content": [
                {"type": "text", "text": "Let me read that"},
                {"type": "tool_use", "name": "read", "input": {}},
            ]
        }
        assert _get_entry_role(entry) == "tool"

    def test_content_list_with_tool_result(self) -> None:
        entry = {"content": [{"type": "tool_result", "tool_use_id": "x"}]}
        assert _get_entry_role(entry) == "tool"

    def test_unknown_entry_defaults_to_user(self) -> None:
        assert _get_entry_role({"text": "some text"}) == "user"

    def test_empty_entry_defaults_to_user(self) -> None:
        assert _get_entry_role({}) == "user"

    def test_content_list_text_only_defaults_to_user(self) -> None:
        entry = {"content": [{"type": "text", "text": "just text"}]}
        assert _get_entry_role(entry) == "user"


class TestHasMemoryMarkers:
    """Tests for _has_memory_markers()."""

    def test_decision_marker(self) -> None:
        assert _has_memory_markers("We decided to use SQLite for storage")

    def test_chose_marker(self) -> None:
        assert _has_memory_markers("I chose React over Vue")

    def test_root_cause_marker(self) -> None:
        assert _has_memory_markers("The root cause was a race condition")

    def test_fixed_marker(self) -> None:
        assert _has_memory_markers("Fixed the import error in server.py")

    def test_insight_marker(self) -> None:
        assert _has_memory_markers("Turns out the config was wrong")

    def test_todo_marker(self) -> None:
        assert _has_memory_markers("TODO: add retry logic for API calls")

    def test_preference_marker(self) -> None:
        assert _has_memory_markers("I prefer using async/await everywhere")

    def test_version_marker(self) -> None:
        assert _has_memory_markers("Released v2.21.0 with cross-language support")

    def test_shipped_marker(self) -> None:
        assert _has_memory_markers("Successfully shipped the new feature")

    def test_committed_marker(self) -> None:
        assert _has_memory_markers("Committed the changes to main branch")

    def test_vietnamese_decision(self) -> None:
        assert _has_memory_markers("Quyết định dùng PostgreSQL thay vì MySQL")

    def test_vietnamese_error(self) -> None:
        assert _has_memory_markers("Lỗi do thiếu import module")

    def test_vietnamese_lesson(self) -> None:
        assert _has_memory_markers("Bài học rút ra từ lần deploy này")

    def test_vietnamese_todo(self) -> None:
        assert _has_memory_markers("Cần phải refactor lại module auth")

    def test_no_markers_generic_text(self) -> None:
        assert not _has_memory_markers("Let me read the file for you")

    def test_no_markers_tool_description(self) -> None:
        assert not _has_memory_markers("I will use the Edit tool to modify this")

    def test_no_markers_short_response(self) -> None:
        assert not _has_memory_markers("Sure, here it is")

    def test_no_markers_code_output(self) -> None:
        assert not _has_memory_markers("The function returns a list of strings")


class TestReadTranscriptTailFiltering:
    """Tests for role-based filtering in read_transcript_tail()."""

    def test_skips_tool_results(self, tmp_path: object) -> None:
        import json
        from pathlib import Path

        p = Path(str(tmp_path)) / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "What is the root cause of the bug?"}),
            json.dumps({"role": "tool", "content": "File contents: lots of code here blah blah"}),
            json.dumps(
                {
                    "role": "assistant",
                    "content": "The root cause was a missing null check in the handler",
                }
            ),
        ]
        p.write_text("\n".join(lines), encoding="utf-8")

        result = read_transcript_tail(str(p))
        assert "root cause" in result.lower()
        assert "File contents:" not in result

    def test_skips_assistant_without_markers(self, tmp_path: object) -> None:
        import json
        from pathlib import Path

        p = Path(str(tmp_path)) / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "Can you help me with this feature?"}),
            json.dumps(
                {"role": "assistant", "content": "Let me read the file and check the code for you"}
            ),
            json.dumps(
                {
                    "role": "assistant",
                    "content": "I decided to use the factory pattern for this module",
                }
            ),
        ]
        p.write_text("\n".join(lines), encoding="utf-8")

        result = read_transcript_tail(str(p))
        # User message included
        assert "help me with this feature" in result
        # Generic assistant response filtered out
        assert "read the file and check" not in result
        # Decision-bearing assistant response included
        assert "factory pattern" in result

    def test_includes_all_user_messages(self, tmp_path: object) -> None:
        import json
        from pathlib import Path

        p = Path(str(tmp_path)) / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "First instruction about the project setup"}),
            json.dumps(
                {"role": "user", "content": "Second instruction about the API design choices"}
            ),
        ]
        p.write_text("\n".join(lines), encoding="utf-8")

        result = read_transcript_tail(str(p))
        assert "project setup" in result
        assert "API design" in result


class TestEmbeddingDedup:
    """Tests for _embedding_dedup()."""

    @pytest.mark.asyncio
    async def test_single_item_passthrough(self) -> None:
        items = [{"content": "Decision: use SQLite", "confidence": 0.9, "type": "decision"}]
        result = await _embedding_dedup(items)
        assert result == items

    @pytest.mark.asyncio
    async def test_empty_list_passthrough(self) -> None:
        result = await _embedding_dedup([])
        assert result == []

    @pytest.mark.asyncio
    async def test_no_provider_returns_original(self) -> None:
        items = [
            {"content": "Item A", "confidence": 0.8, "type": "fact"},
            {"content": "Item B", "confidence": 0.7, "type": "fact"},
        ]
        with patch(
            "neural_memory.engine.semantic_discovery._auto_detect_provider",
            side_effect=RuntimeError("no provider"),
        ):
            result = await _embedding_dedup(items)
        assert result == items

    @pytest.mark.asyncio
    async def test_removes_semantic_duplicates(self) -> None:
        items = [
            {"content": "Decided to use React for frontend", "confidence": 0.9, "type": "decision"},
            {"content": "Chose React for the frontend UI", "confidence": 0.8, "type": "decision"},
            {"content": "Fixed the auth bug in login", "confidence": 0.85, "type": "error"},
        ]

        mock_provider = AsyncMock()
        # Embeddings: items 0 and 1 are near-duplicates, item 2 is different
        mock_provider.embed_batch = AsyncMock(
            return_value=[[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.0, 1.0, 0.0]]
        )

        async def mock_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        mock_provider.similarity = mock_similarity

        with (
            patch(
                "neural_memory.engine.semantic_discovery._auto_detect_provider",
                return_value=("sentence_transformer", "all-MiniLM-L6-v2"),
            ),
            patch(
                "neural_memory.engine.embedding.sentence_transformer.SentenceTransformerEmbedding",
                return_value=mock_provider,
            ),
        ):
            result = await _embedding_dedup(items)

        # Should keep item 0 (higher confidence) and item 2, remove item 1
        assert len(result) == 2
        contents = [r["content"] for r in result]
        assert "Decided to use React for frontend" in contents
        assert "Fixed the auth bug in login" in contents
        assert "Chose React for the frontend UI" not in contents

    @pytest.mark.asyncio
    async def test_keeps_all_when_no_duplicates(self) -> None:
        items = [
            {"content": "Decision about database", "confidence": 0.9, "type": "decision"},
            {"content": "Error in auth module", "confidence": 0.85, "type": "error"},
            {"content": "Insight about caching", "confidence": 0.8, "type": "insight"},
        ]

        mock_provider = AsyncMock()
        mock_provider.embed_batch = AsyncMock(
            return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        async def mock_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        mock_provider.similarity = mock_similarity

        with (
            patch(
                "neural_memory.engine.semantic_discovery._auto_detect_provider",
                return_value=("sentence_transformer", "all-MiniLM-L6-v2"),
            ),
            patch(
                "neural_memory.engine.embedding.sentence_transformer.SentenceTransformerEmbedding",
                return_value=mock_provider,
            ),
        ):
            result = await _embedding_dedup(items)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_skips_api_providers(self) -> None:
        """API-based providers (gemini, openai) are skipped to avoid rate limits."""
        items = [
            {"content": "Item A", "confidence": 0.8, "type": "fact"},
            {"content": "Item B", "confidence": 0.7, "type": "fact"},
        ]
        with patch(
            "neural_memory.engine.semantic_discovery._auto_detect_provider",
            return_value=("gemini", "text-embedding-004"),
        ):
            result = await _embedding_dedup(items)
        assert result == items

    @pytest.mark.asyncio
    async def test_embed_failure_returns_original(self) -> None:
        """If embedding fails, return original list gracefully."""
        items = [
            {"content": "Item A", "confidence": 0.8, "type": "fact"},
            {"content": "Item B", "confidence": 0.7, "type": "fact"},
        ]

        mock_provider = AsyncMock()
        mock_provider.embed_batch = AsyncMock(side_effect=RuntimeError("model not found"))

        with (
            patch(
                "neural_memory.engine.semantic_discovery._auto_detect_provider",
                return_value=("sentence_transformer", "all-MiniLM-L6-v2"),
            ),
            patch(
                "neural_memory.engine.embedding.sentence_transformer.SentenceTransformerEmbedding",
                return_value=mock_provider,
            ),
        ):
            result = await _embedding_dedup(items)
        assert result == items
