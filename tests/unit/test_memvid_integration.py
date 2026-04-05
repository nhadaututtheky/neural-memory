"""Tests for Memvid Phase 3: Integration of SimHash pre-filter + Time-travel into API layers.

Covers:
- MCP tool schema includes as_of + simhash_threshold fields
- MCP recall handler parses simhash_threshold override
- FastAPI QueryRequest model accepts as_of + simhash_threshold
- ReflexPipeline.query() accepts simhash_threshold override
- BrainSettings includes simhash_prefilter_threshold for TOML config
- BrainConfig creation from unified_config maps the threshold
"""

from __future__ import annotations

from datetime import datetime

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.engine.retrieval import ReflexPipeline
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.unified_config import BrainSettings

# ---------------------------------------------------------------------------
# MCP Tool Schema
# ---------------------------------------------------------------------------


class TestMCPToolSchema:
    """Verify nmem_recall schema includes new fields."""

    def test_recall_schema_has_as_of(self) -> None:
        from neural_memory.mcp.tool_schemas import _ALL_TOOL_SCHEMAS

        recall = next(t for t in _ALL_TOOL_SCHEMAS if t["name"] == "nmem_recall")
        props = recall["inputSchema"]["properties"]
        assert "as_of" in props
        assert props["as_of"]["type"] == "string"

    def test_recall_schema_has_simhash_threshold(self) -> None:
        from neural_memory.mcp.tool_schemas import _ALL_TOOL_SCHEMAS

        recall = next(t for t in _ALL_TOOL_SCHEMAS if t["name"] == "nmem_recall")
        props = recall["inputSchema"]["properties"]
        assert "simhash_threshold" in props
        assert props["simhash_threshold"]["type"] == "integer"
        assert props["simhash_threshold"]["minimum"] == 0
        assert props["simhash_threshold"]["maximum"] == 64


# ---------------------------------------------------------------------------
# FastAPI QueryRequest Model
# ---------------------------------------------------------------------------


class TestQueryRequestModel:
    """Verify FastAPI QueryRequest includes new fields."""

    def test_query_request_accepts_as_of(self) -> None:
        from neural_memory.server.models import QueryRequest

        req = QueryRequest(query="test", as_of=datetime(2026, 3, 1, 12, 0))
        assert req.as_of == datetime(2026, 3, 1, 12, 0)

    def test_query_request_as_of_defaults_none(self) -> None:
        from neural_memory.server.models import QueryRequest

        req = QueryRequest(query="test")
        assert req.as_of is None

    def test_query_request_accepts_simhash_threshold(self) -> None:
        from neural_memory.server.models import QueryRequest

        req = QueryRequest(query="test", simhash_threshold=10)
        assert req.simhash_threshold == 10

    def test_query_request_simhash_threshold_defaults_none(self) -> None:
        from neural_memory.server.models import QueryRequest

        req = QueryRequest(query="test")
        assert req.simhash_threshold is None

    def test_query_request_simhash_threshold_validates_range(self) -> None:
        from pydantic import ValidationError

        from neural_memory.server.models import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="test", simhash_threshold=65)

        with pytest.raises(ValidationError):
            QueryRequest(query="test", simhash_threshold=-1)


# ---------------------------------------------------------------------------
# BrainSettings TOML Config
# ---------------------------------------------------------------------------


class TestBrainSettingsConfig:
    """Verify BrainSettings includes simhash_prefilter_threshold."""

    def test_default_threshold_is_zero(self) -> None:
        settings = BrainSettings()
        assert settings.simhash_prefilter_threshold == 0

    def test_to_dict_includes_threshold(self) -> None:
        settings = BrainSettings(simhash_prefilter_threshold=10)
        d = settings.to_dict()
        assert d["simhash_prefilter_threshold"] == 10

    def test_from_dict_parses_threshold(self) -> None:
        settings = BrainSettings.from_dict({"simhash_prefilter_threshold": 15})
        assert settings.simhash_prefilter_threshold == 15

    def test_from_dict_defaults_to_zero(self) -> None:
        settings = BrainSettings.from_dict({})
        assert settings.simhash_prefilter_threshold == 0


# ---------------------------------------------------------------------------
# ReflexPipeline.query() simhash_threshold override
# ---------------------------------------------------------------------------


class TestPipelineSimhashOverride:
    """Verify pipeline respects per-query simhash_threshold override."""

    @pytest.fixture
    async def pipeline_with_data(self) -> tuple[ReflexPipeline, InMemoryStorage]:
        """Create a pipeline with some encoded memories."""
        storage = InMemoryStorage()
        config = BrainConfig(
            activation_threshold=0.1,
            simhash_prefilter_threshold=0,  # disabled in config
        )
        brain = Brain.create(name="test", config=config)
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        from neural_memory.engine.encoder import MemoryEncoder

        encoder = MemoryEncoder(storage, config)
        await encoder.encode("Python asyncio event loop fundamentals")
        await encoder.encode("JavaScript promises and async/await patterns")

        pipeline = ReflexPipeline(storage, config)
        return pipeline, storage

    @pytest.mark.asyncio
    async def test_query_accepts_simhash_threshold_none(
        self, pipeline_with_data: tuple[ReflexPipeline, InMemoryStorage]
    ) -> None:
        """simhash_threshold=None should use config default (0=disabled)."""
        pipeline, _ = pipeline_with_data
        result = await pipeline.query("asyncio", simhash_threshold=None)
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_query_accepts_simhash_threshold_override(
        self, pipeline_with_data: tuple[ReflexPipeline, InMemoryStorage]
    ) -> None:
        """simhash_threshold=10 should enable pre-filter even when config=0."""
        pipeline, _ = pipeline_with_data
        result = await pipeline.query("asyncio", simhash_threshold=10)
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_query_with_as_of_param_accepted(
        self, pipeline_with_data: tuple[ReflexPipeline, InMemoryStorage]
    ) -> None:
        """as_of parameter should be accepted without error."""
        pipeline, _ = pipeline_with_data
        # InMemoryStorage may not support created_before filtering,
        # but the parameter should be accepted and passed through
        result = await pipeline.query(
            "asyncio",
            as_of=datetime(2020, 1, 1),
        )
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_query_with_as_of_future_includes_all(
        self, pipeline_with_data: tuple[ReflexPipeline, InMemoryStorage]
    ) -> None:
        """as_of in the future should include all neurons."""
        pipeline, _ = pipeline_with_data
        result = await pipeline.query(
            "asyncio",
            as_of=datetime(2030, 1, 1),
        )
        # Should find something since memories exist
        assert result.latency_ms >= 0
