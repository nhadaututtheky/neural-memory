"""Tests for PostgreSQL fiber parity methods (Phase 3).

Uses mocked asyncpg pool/connection to test SQL generation and result handling
without requiring a real PostgreSQL instance.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from neural_memory.utils.timeutils import utcnow


def _make_fiber_record(**overrides: Any) -> dict[str, Any]:
    """Create a minimal fiber record dict that row_to_fiber can parse."""
    now = utcnow()
    base: dict[str, Any] = {
        "id": overrides.get("id", "fiber-1"),
        "brain_id": "test-brain",
        "neuron_ids": json.dumps(["n1", "n2"]),
        "synapse_ids": json.dumps([]),
        "anchor_neuron_id": "n1",
        "pathway": json.dumps([]),
        "conductivity": 1.0,
        "last_conducted": now,
        "time_start": None,
        "time_end": None,
        "coherence": 0.5,
        "salience": overrides.get("salience", 0.8),
        "frequency": 1,
        "summary": overrides.get("summary", "test fiber"),
        "tags": json.dumps(list(overrides.get("tags", []))),
        "auto_tags": json.dumps([]),
        "agent_tags": json.dumps([]),
        "metadata": json.dumps(overrides.get("metadata", {})),
        "compression_tier": 0,
        "pinned": 0,
        "created_at": overrides.get("created_at", now),
    }
    return base


class _FakeRecord(dict):
    """Dict that supports both [] and attribute access like asyncpg Record."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def _record(**kw: Any) -> _FakeRecord:
    return _FakeRecord(_make_fiber_record(**kw))


def _make_mixin() -> Any:
    """Create a PostgresFiberMixin instance with mocked pool."""
    from neural_memory.storage.postgres.postgres_fibers import PostgresFiberMixin

    mixin = PostgresFiberMixin.__new__(PostgresFiberMixin)
    mixin._current_brain_id = "test-brain"
    mixin._pool = AsyncMock()
    return mixin


# ---------------------------------------------------------------------------
# find_fibers_batch
# ---------------------------------------------------------------------------


class TestFindFibersBatch:
    @pytest.mark.asyncio
    async def test_empty_neuron_ids(self) -> None:
        mixin = _make_mixin()
        result = await mixin.find_fibers_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_fibers(self) -> None:
        mixin = _make_mixin()
        records = [_record(id="f1"), _record(id="f2")]
        mixin._query_ro = AsyncMock(return_value=records)

        result = await mixin.find_fibers_batch(["n1", "n2"], limit_per_neuron=5)
        assert len(result) == 2
        assert result[0].id == "f1"
        assert result[1].id == "f2"

        # Verify SQL contains key patterns
        sql = mixin._query_ro.call_args[0][0]
        assert "fiber_neurons" in sql
        assert "ROW_NUMBER" in sql
        assert "ANY($2" in sql

    @pytest.mark.asyncio
    async def test_with_tags_and(self) -> None:
        mixin = _make_mixin()
        mixin._query_ro = AsyncMock(return_value=[])

        await mixin.find_fibers_batch(["n1"], tags={"python", "async"}, tag_mode="and")
        sql = mixin._query_ro.call_args[0][0]
        assert "@>" in sql  # JSONB containment for AND mode

    @pytest.mark.asyncio
    async def test_with_tags_or(self) -> None:
        mixin = _make_mixin()
        mixin._query_ro = AsyncMock(return_value=[])

        await mixin.find_fibers_batch(["n1"], tags={"python", "async"}, tag_mode="or")
        sql = mixin._query_ro.call_args[0][0]
        assert "?|" in sql  # ANY key exists for OR mode

    @pytest.mark.asyncio
    async def test_with_created_before(self) -> None:
        mixin = _make_mixin()
        mixin._query_ro = AsyncMock(return_value=[])

        cutoff = utcnow() - timedelta(days=7)
        await mixin.find_fibers_batch(["n1"], created_before=cutoff)
        sql = mixin._query_ro.call_args[0][0]
        assert "created_at" in sql


# ---------------------------------------------------------------------------
# search_fiber_summaries
# ---------------------------------------------------------------------------


class TestSearchFiberSummaries:
    @pytest.mark.asyncio
    async def test_empty_query(self) -> None:
        mixin = _make_mixin()
        result = await mixin.search_fiber_summaries("")
        assert result == []

    @pytest.mark.asyncio
    async def test_fts_query(self) -> None:
        mixin = _make_mixin()
        records = [_record(id="f1", summary="python async patterns")]
        mixin._query_ro = AsyncMock(return_value=records)

        result = await mixin.search_fiber_summaries("python async")
        assert len(result) == 1
        sql = mixin._query_ro.call_args[0][0]
        assert "summary_tsv" in sql
        assert "to_tsquery" in sql

    @pytest.mark.asyncio
    async def test_ilike_fallback(self) -> None:
        mixin = _make_mixin()
        records = [_record(id="f1")]
        call_count = 0

        async def side_effect(*args: Any, **kwargs: Any) -> list[Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("FTS not available")
            return records

        mixin._query_ro = AsyncMock(side_effect=side_effect)

        result = await mixin.search_fiber_summaries("test query")
        assert len(result) == 1
        # Second call should be ILIKE
        sql = mixin._query_ro.call_args[0][0]
        assert "ILIKE" in sql


# ---------------------------------------------------------------------------
# update_fiber_metadata
# ---------------------------------------------------------------------------


class TestUpdateFiberMetadata:
    @pytest.mark.asyncio
    async def test_merges_metadata(self) -> None:
        mixin = _make_mixin()

        from neural_memory.core.fiber import Fiber

        existing = Fiber(
            id="f1",
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            metadata={"existing_key": "value"},
            created_at=utcnow(),
        )
        mixin.get_fiber = AsyncMock(return_value=existing)
        mixin.update_fiber = AsyncMock()

        await mixin.update_fiber_metadata("f1", {"new_key": 42})

        mixin.update_fiber.assert_called_once()
        updated = mixin.update_fiber.call_args[0][0]
        assert updated.metadata["existing_key"] == "value"
        assert updated.metadata["new_key"] == 42

    @pytest.mark.asyncio
    async def test_noop_for_missing_fiber(self) -> None:
        mixin = _make_mixin()
        mixin.get_fiber = AsyncMock(return_value=None)
        mixin.update_fiber = AsyncMock()

        await mixin.update_fiber_metadata("nonexistent", {"key": "val"})
        mixin.update_fiber.assert_not_called()


# ---------------------------------------------------------------------------
# Stats methods
# ---------------------------------------------------------------------------


class TestStatsMethodsPg:
    @pytest.mark.asyncio
    async def test_get_stale_fiber_count(self) -> None:
        mixin = _make_mixin()
        mixin._query_one = AsyncMock(return_value={"cnt": 42})

        result = await mixin.get_stale_fiber_count("brain-test", stale_days=30)
        assert result == 42
        sql = mixin._query_one.call_args[0][0]
        assert "COUNT(*)" in sql
        assert "last_conducted" in sql

    @pytest.mark.asyncio
    async def test_get_fiber_stage_counts(self) -> None:
        mixin = _make_mixin()
        mixin._query_ro = AsyncMock(
            return_value=[
                {"stage": "episodic", "cnt": 10},
                {"stage": "semantic", "cnt": 5},
            ]
        )

        result = await mixin.get_fiber_stage_counts("brain-test")
        assert result == {"episodic": 10, "semantic": 5}
        sql = mixin._query_ro.call_args[0][0]
        assert "_stage" in sql  # Uses metadata JSONB extraction

    @pytest.mark.asyncio
    async def test_get_total_fiber_count(self) -> None:
        mixin = _make_mixin()
        mixin._query_one = AsyncMock(return_value={"cnt": 100})

        result = await mixin.get_total_fiber_count()
        assert result == 100

    @pytest.mark.asyncio
    async def test_batch_update_ghost_shown(self) -> None:
        mixin = _make_mixin()
        mixin._query = AsyncMock(return_value="UPDATE 3")

        from neural_memory.utils.timeutils import utcnow

        ts = utcnow()
        result = await mixin.batch_update_ghost_shown(["f1", "f2", "f3"], ts)
        mixin._query.assert_called_once()
        sql = mixin._query.call_args[0][0]
        assert "last_ghost_shown_at" in sql
        assert "ANY($3" in sql
        assert result == 3

    @pytest.mark.asyncio
    async def test_batch_update_ghost_shown_empty(self) -> None:
        mixin = _make_mixin()
        mixin._query = AsyncMock()

        from neural_memory.utils.timeutils import utcnow

        result = await mixin.batch_update_ghost_shown([], utcnow())
        mixin._query.assert_not_called()
        assert result == 0


# ---------------------------------------------------------------------------
# Keyword document frequency
# ---------------------------------------------------------------------------


class TestKeywordDfPg:
    @pytest.mark.asyncio
    async def test_get_keyword_df_batch(self) -> None:
        mixin = _make_mixin()
        mixin._query_ro = AsyncMock(
            return_value=[
                {"keyword": "python", "fiber_count": 15},
                {"keyword": "async", "fiber_count": 8},
            ]
        )

        result = await mixin.get_keyword_df_batch(["python", "async", "missing"])
        assert result == {"python": 15, "async": 8}

    @pytest.mark.asyncio
    async def test_get_keyword_df_empty(self) -> None:
        mixin = _make_mixin()
        result = await mixin.get_keyword_df_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_increment_keyword_df(self) -> None:
        mixin = _make_mixin()
        mixin._executemany = AsyncMock()

        await mixin.increment_keyword_df(["python", "async", "python"])  # dedup
        mixin._executemany.assert_called_once()
        sql = mixin._executemany.call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "fiber_count" in sql
        # Should be 2 unique keywords, not 3
        args_list = mixin._executemany.call_args[0][1]
        assert len(args_list) == 2

    @pytest.mark.asyncio
    async def test_increment_keyword_df_empty(self) -> None:
        mixin = _make_mixin()
        mixin._executemany = AsyncMock()

        await mixin.increment_keyword_df([])
        mixin._executemany.assert_not_called()


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestPgSchemaMigration:
    def test_migrations_include_new_columns(self) -> None:
        from neural_memory.storage.postgres.postgres_schema import _MIGRATIONS

        migration_text = " ".join(_MIGRATIONS)
        assert "last_ghost_shown_at" in migration_text
        assert "summary_tsv" in migration_text
        assert "keyword_document_frequency" in migration_text
        assert "fibers_summary_tsv_trigger" in migration_text
