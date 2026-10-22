"""Tests for persisting the write-time quality score on fibers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.mcp.server import MCPServer
from neural_memory.safety.encryption import EncryptionResult
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.unified_config import ResponseConfig, ToolTierConfig

RICH = {
    "content": (
        "Decided to use PostgreSQL instead of MongoDB for the billing service "
        "because we need transactions and strong consistency."
    ),
    "type": "decision",
    "tags": ["billing", "database"],
    "context": {"reason": "needs ACID transactions", "alternatives": ["MongoDB", "DynamoDB"]},
    "compact": False,
}
POOR = {"content": "check the thing", "type": "fact", "compact": False}


@pytest.fixture
def server() -> MCPServer:
    with patch("neural_memory.mcp.server.get_config") as mock_get_config:
        mock_get_config.return_value = MagicMock(
            current_brain="test-brain",
            get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            tool_tier=ToolTierConfig(tier="full"),
            response=ResponseConfig(),
        )
        return MCPServer()


@pytest_asyncio.fixture
async def storage() -> AsyncIterator[InMemoryStorage]:
    store = InMemoryStorage()
    brain = Brain.create(name="test-brain", config=BrainConfig(), brain_id="test-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store


async def _remember(server: MCPServer, storage: InMemoryStorage, args: dict[str, Any]) -> Any:
    async def _get_storage() -> InMemoryStorage:
        return storage

    with patch.object(server, "get_storage", _get_storage):
        return await server.call_tool("nmem_remember", args)


async def _stored_metadata(storage: InMemoryStorage, fiber_id: str) -> dict[str, Any]:
    fiber = await storage.get_fiber(fiber_id)
    assert fiber is not None
    return fiber.metadata


class TestPersistQualityScore:
    @pytest.mark.asyncio
    async def test_stored_score_equals_returned_score(
        self, server: MCPServer, storage: InMemoryStorage
    ) -> None:
        result = await _remember(server, storage, dict(RICH))
        assert result["success"] is True
        meta = await _stored_metadata(storage, result["fiber_id"])
        score = meta["_quality_score"]
        assert isinstance(score, int)
        assert 0 <= score <= 10
        assert score == result["score"]
        assert meta["type"] == "decision"

    @pytest.mark.asyncio
    async def test_low_quality_scores_lower(
        self, server: MCPServer, storage: InMemoryStorage
    ) -> None:
        rich = await _remember(server, storage, dict(RICH))
        poor = await _remember(server, storage, dict(POOR))
        rich_score = (await _stored_metadata(storage, rich["fiber_id"]))["_quality_score"]
        poor_score = (await _stored_metadata(storage, poor["fiber_id"]))["_quality_score"]
        assert poor_score < rich_score

    @pytest.mark.asyncio
    async def test_score_memory_called_once(
        self, server: MCPServer, storage: InMemoryStorage
    ) -> None:
        from neural_memory.engine import quality_scorer

        with patch.object(quality_scorer, "score_memory", wraps=quality_scorer.score_memory) as spy:
            result = await _remember(server, storage, dict(RICH))
        assert spy.call_count == 1
        assert result["quality"] in {"low", "medium", "high"}

    @pytest.mark.asyncio
    async def test_encrypted_write_still_carries_score(
        self, server: MCPServer, storage: InMemoryStorage, tmp_path: Path
    ) -> None:
        server.config.encryption = MagicMock(enabled=True, keys_dir=str(tmp_path))
        server.config.data_dir = tmp_path
        fake = EncryptionResult(ciphertext="ciphertext-of-rich-memory", key_id="key-1")
        captured: dict[str, Any] = {}
        real_encode = MemoryEncoder.encode

        async def spy_encode(self: MemoryEncoder, *args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return await real_encode(self, *args, **kwargs)

        with (
            patch("neural_memory.safety.encryption.MemoryEncryptor") as enc_cls,
            patch.object(MemoryEncoder, "encode", spy_encode),
        ):
            enc_cls.return_value.encrypt.return_value = fake
            result = await _remember(server, storage, {**RICH, "encrypted": True})

        assert result["encrypted"] is True
        assert captured["content"] == "ciphertext-of-rich-memory"
        assert captured["metadata"]["_quality_score"] == result["score"]
        assert captured["metadata"]["type"] == "decision"
        meta = await _stored_metadata(storage, result["fiber_id"])
        assert meta["_quality_score"] == result["score"]


class TestRecallUsesStoredScore:
    @pytest.mark.asyncio
    async def test_recall_confidence_reflects_stored_score(
        self, server: MCPServer, storage: InMemoryStorage
    ) -> None:
        rich = await _remember(server, storage, dict(RICH))
        stored = (await _stored_metadata(storage, rich["fiber_id"]))["_quality_score"]

        async def _get_storage() -> InMemoryStorage:
            return storage

        with patch.object(server, "get_storage", _get_storage):
            recalled = await server.call_tool(
                "nmem_recall", {"query": "PostgreSQL billing service decision", "compact": False}
            )
        assert recalled["confidence_score"]["content_quality"] == pytest.approx(stored / 10.0)
        assert stored != 5

    @pytest.mark.asyncio
    async def test_old_fiber_without_score_defaults_to_five(
        self, server: MCPServer, storage: InMemoryStorage
    ) -> None:
        await _remember(server, storage, dict(RICH))
        for fiber in list(storage._fibers["test-brain"].values()):
            legacy_meta = {
                key: value for key, value in fiber.metadata.items() if key != "_quality_score"
            }
            await storage.update_fiber(replace(fiber, metadata=legacy_meta))
        assert all(
            "_quality_score" not in fiber.metadata
            for fiber in storage._fibers["test-brain"].values()
        )

        async def _get_storage() -> InMemoryStorage:
            return storage

        with patch.object(server, "get_storage", _get_storage):
            recalled = await server.call_tool(
                "nmem_recall", {"query": "PostgreSQL billing service decision", "compact": False}
            )
        assert recalled["confidence_score"]["content_quality"] == pytest.approx(0.5)
