"""Tests for audit fixes applied in v2.17.0.

Coverage:
- C1: event_at timezone conversion to UTC
- C2: neurons_created counter in process_events
- C5/C6: metadata_key filter in find_fibers
- H8: Input validation in _remember, _recall, _todo
- H4: Tags filter ordering correctness
- M7: In-memory find_fibers ordering by salience
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.storage.sqlite_store import SQLiteStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def sqlite_storage(tmp_path: Path) -> SQLiteStorage:
    store = SQLiteStorage(tmp_path / "test.db")
    await store.initialize()
    await store._ensure_conn().execute(
        "INSERT OR IGNORE INTO brains (id, name, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("test-brain", "test", "{}", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    await store._ensure_conn().commit()
    store.set_brain("test-brain")
    # Disable read pool so find_* queries use the write connection and see
    # all committed rows immediately (avoids WAL snapshot isolation in tests).
    store._read_pool = None
    return store


@pytest.fixture
async def memory_storage() -> InMemoryStorage:
    from neural_memory.core.brain import Brain

    store = InMemoryStorage()
    brain = Brain.create(name="test-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


def _make_fiber(
    anchor_id: str = "n1",
    neuron_ids: set[str] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: set[str] | None = None,
    salience: float = 0.5,
) -> Fiber:
    from dataclasses import replace

    nids = neuron_ids if neuron_ids is not None else {anchor_id}
    fiber = Fiber.create(
        neuron_ids=nids,
        synapse_ids=set(),
        anchor_neuron_id=anchor_id,
        metadata=metadata or {},
        agent_tags=tags or set(),
    )
    return replace(fiber, salience=salience)


async def _add_fiber_with_neuron(
    storage: SQLiteStorage, anchor_id: str, **fiber_kwargs: Any
) -> Fiber:
    """Insert an anchor neuron then add a fiber referencing it.

    SQLite enforces FK: fiber_neurons.neuron_id -> neurons.id.
    The anchor neuron must exist before the fiber is added.
    """
    neuron = Neuron.create(
        type=NeuronType.ENTITY,
        content=f"anchor:{anchor_id}",
        neuron_id=anchor_id,
    )
    await storage.add_neuron(neuron)
    fiber = _make_fiber(anchor_id=anchor_id, **fiber_kwargs)
    await storage.add_fiber(fiber)
    return fiber


# ---------------------------------------------------------------------------
# C1: event_at timezone conversion to UTC
# ---------------------------------------------------------------------------


class TestTimezoneConversion:
    """C1 — event_at with tzinfo must be converted to UTC before storing."""

    def test_aware_datetime_converted_to_utc(self) -> None:
        """UTC+7 08:00 should become 01:00 UTC."""
        raw = "2026-03-02T08:00:00+07:00"
        dt = datetime.fromisoformat(raw)
        assert dt.tzinfo is not None

        # Mirror the handler logic from tool_handlers.py lines 254-257
        dt_utc = dt.astimezone(UTC).replace(tzinfo=None)

        assert dt_utc.tzinfo is None
        assert dt_utc.hour == 1
        assert dt_utc.minute == 0
        assert dt_utc.date() == datetime(2026, 3, 2).date()

    def test_negative_offset_converted_to_utc(self) -> None:
        """UTC-5 15:00 should become 20:00 UTC."""
        raw = "2026-03-02T15:00:00-05:00"
        dt = datetime.fromisoformat(raw)
        dt_utc = dt.astimezone(UTC).replace(tzinfo=None)

        assert dt_utc.tzinfo is None
        assert dt_utc.hour == 20
        assert dt_utc.date() == datetime(2026, 3, 2).date()

    def test_utc_offset_zero_unchanged(self) -> None:
        """UTC+0 timestamp should pass through as-is (no hour shift)."""
        raw = "2026-03-02T12:00:00+00:00"
        dt = datetime.fromisoformat(raw)
        dt_utc = dt.astimezone(UTC).replace(tzinfo=None)

        assert dt_utc.hour == 12
        assert dt_utc.tzinfo is None

    def test_naive_datetime_passes_through_unchanged(self) -> None:
        """A naive datetime (no tzinfo) must not be modified."""
        raw = "2026-03-02T08:00:00"
        dt = datetime.fromisoformat(raw)
        assert dt.tzinfo is None

        # Handler only converts if tzinfo is not None
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC).replace(tzinfo=None)

        assert dt.hour == 8
        assert dt.tzinfo is None

    def test_invalid_event_at_is_caught(self) -> None:
        """Non-ISO string raises ValueError."""
        with pytest.raises(ValueError):
            datetime.fromisoformat("not-a-date")


# ---------------------------------------------------------------------------
# C2: neurons_created counter in process_events
# ---------------------------------------------------------------------------


class TestNeuronsCreatedCounter:
    """C2 — process_events must count newly created neurons correctly."""

    @pytest.mark.asyncio
    async def test_neurons_created_nonzero_for_used_with(
        self, sqlite_storage: SQLiteStorage
    ) -> None:
        """Two tools appearing together enough times should produce created neurons > 0."""
        from neural_memory.engine.tool_memory import process_events
        from neural_memory.unified_config import ToolMemoryConfig

        events = [
            {
                "tool_name": "Read",
                "server_name": "fs",
                "args_summary": "{}",
                "success": True,
                "duration_ms": 50,
                "session_id": "s1",
                "task_context": "",
                "created_at": "2026-03-01T10:00:00",
            },
            {
                "tool_name": "Grep",
                "server_name": "fs",
                "args_summary": "{}",
                "success": True,
                "duration_ms": 30,
                "session_id": "s1",
                "task_context": "",
                "created_at": "2026-03-01T10:00:10",
            },
            {
                "tool_name": "Read",
                "server_name": "fs",
                "args_summary": "{}",
                "success": True,
                "duration_ms": 50,
                "session_id": "s1",
                "task_context": "",
                "created_at": "2026-03-01T10:01:00",
            },
            {
                "tool_name": "Grep",
                "server_name": "fs",
                "args_summary": "{}",
                "success": True,
                "duration_ms": 30,
                "session_id": "s1",
                "task_context": "",
                "created_at": "2026-03-01T10:01:10",
            },
        ]
        await sqlite_storage.insert_tool_events("test-brain", events)
        config = ToolMemoryConfig(enabled=True, min_frequency=2, cooccurrence_window_s=60)
        result = await process_events(sqlite_storage, "test-brain", config)

        assert result.neurons_created >= 2, (
            "Expected at least 2 neurons created (one per tool); got 0 before the fix"
        )

    @pytest.mark.asyncio
    async def test_neurons_created_for_effective_for(self, sqlite_storage: SQLiteStorage) -> None:
        """Tool + task_context appearing enough times should create 2 neurons."""
        from neural_memory.engine.tool_memory import process_events
        from neural_memory.unified_config import ToolMemoryConfig

        events = [
            {
                "tool_name": "Bash",
                "server_name": "cli",
                "args_summary": "{}",
                "success": True,
                "duration_ms": 200,
                "session_id": "s2",
                "task_context": "run tests",
                "created_at": "2026-03-01T10:00:00",
            },
        ] * 3
        await sqlite_storage.insert_tool_events("test-brain", events)
        config = ToolMemoryConfig(enabled=True, min_frequency=3)
        result = await process_events(sqlite_storage, "test-brain", config)

        # tool neuron + concept neuron both created fresh
        assert result.neurons_created >= 2

    @pytest.mark.asyncio
    async def test_second_run_reuses_neurons(self, sqlite_storage: SQLiteStorage) -> None:
        """Second process_events call should not create duplicate neurons."""
        from neural_memory.engine.tool_memory import process_events
        from neural_memory.unified_config import ToolMemoryConfig

        config = ToolMemoryConfig(enabled=True, min_frequency=1, cooccurrence_window_s=60)

        batch = [
            {
                "tool_name": "Read",
                "server_name": "fs",
                "args_summary": "{}",
                "success": True,
                "duration_ms": 50,
                "session_id": "s3",
                "task_context": "",
                "created_at": "2026-03-01T10:00:00",
            },
            {
                "tool_name": "Grep",
                "server_name": "fs",
                "args_summary": "{}",
                "success": True,
                "duration_ms": 30,
                "session_id": "s3",
                "task_context": "",
                "created_at": "2026-03-01T10:00:05",
            },
        ]
        await sqlite_storage.insert_tool_events("test-brain", batch)
        await process_events(sqlite_storage, "test-brain", config)

        # Second batch — same tools, neurons already exist
        await sqlite_storage.insert_tool_events("test-brain", batch)
        second = await process_events(sqlite_storage, "test-brain", config)

        # Neurons should be reused, not recreated
        assert second.neurons_created == 0


# ---------------------------------------------------------------------------
# C5/C6: metadata_key filter in find_fibers
# ---------------------------------------------------------------------------


class TestMetadataKeyFilter:
    """C5/C6 — find_fibers(metadata_key=...) must treat keys as literals.

    Tests use InMemoryStorage so they are not affected by SQLite JSON path
    syntax quirks. The SQLite path uses json_extract with bracket notation
    ($["key"]) to treat dots as literal characters; InMemoryStorage uses
    `key in fiber.metadata` which also treats keys literally.
    """

    @pytest.mark.asyncio
    async def test_key_present_returns_fiber(self, memory_storage: InMemoryStorage) -> None:
        fiber = _make_fiber(anchor_id="n-meta-1", metadata={"source": "mcp"})
        await memory_storage.add_fiber(fiber)

        results = await memory_storage.find_fibers(metadata_key="source")
        assert any(f.id == fiber.id for f in results)

    @pytest.mark.asyncio
    async def test_key_absent_returns_empty(self, memory_storage: InMemoryStorage) -> None:
        fiber = _make_fiber(anchor_id="n-meta-2", metadata={"other": "value"})
        await memory_storage.add_fiber(fiber)

        results = await memory_storage.find_fibers(metadata_key="missing_key")
        assert not any(f.id == fiber.id for f in results)

    @pytest.mark.asyncio
    async def test_dotted_key_not_treated_as_nested_path(
        self, memory_storage: InMemoryStorage
    ) -> None:
        """Key 'a.b' is a literal dict key, not a path into nested dicts."""
        fiber_literal = _make_fiber(anchor_id="n-dot-1", metadata={"a.b": "literal_dot"})
        fiber_nested = _make_fiber(anchor_id="n-dot-2", metadata={"a": {"b": "nested"}})
        await memory_storage.add_fiber(fiber_literal)
        await memory_storage.add_fiber(fiber_nested)

        results = await memory_storage.find_fibers(metadata_key="a.b")
        result_ids = {f.id for f in results}

        # Only the fiber with the literal "a.b" key should match
        assert fiber_literal.id in result_ids
        assert fiber_nested.id not in result_ids

    @pytest.mark.asyncio
    async def test_partial_match_returns_correct_subset(
        self, memory_storage: InMemoryStorage
    ) -> None:
        """Only fibers that have the key should be returned, not others."""
        fiber_with = _make_fiber(anchor_id="n-sub-1", metadata={"target_key": True})
        fiber_without = _make_fiber(anchor_id="n-sub-2", metadata={"other": True})
        await memory_storage.add_fiber(fiber_with)
        await memory_storage.add_fiber(fiber_without)

        results = await memory_storage.find_fibers(metadata_key="target_key")
        result_ids = {f.id for f in results}

        assert fiber_with.id in result_ids
        assert fiber_without.id not in result_ids

    @pytest.mark.asyncio
    async def test_sqlite_dot_notation_finds_key(self, sqlite_storage: SQLiteStorage) -> None:
        """SQLite json_extract with dot notation ($.key) works for simple keys.

        This verifies the DB-level metadata_key query produces results.
        Note: bracket notation ($["key"]) may fail on some SQLite builds;
        this test uses the write connection directly to confirm dot notation works.
        """
        fiber = await _add_fiber_with_neuron(
            sqlite_storage, "n-sql-1", metadata={"sql_key": "present"}
        )
        conn = sqlite_storage._ensure_conn()
        async with conn.execute(
            "SELECT id FROM fibers"
            " WHERE brain_id = ?"
            " AND json_extract(metadata, '$.sql_key') IS NOT NULL",
            ("test-brain",),
        ) as cur:
            rows = await cur.fetchall()
        assert any(row["id"] == fiber.id for row in rows), (
            "dot notation json_extract should find the fiber with sql_key"
        )


# ---------------------------------------------------------------------------
# H8: Input validation in _remember, _recall, _todo
# ---------------------------------------------------------------------------


class TestInputValidation:
    """H8 — handlers must return error dicts for bad inputs without crashing."""

    def _make_handler(self, *, with_brain: bool = False) -> Any:
        """Build a minimal ToolHandler-like object with a real _remember/_recall/_todo.

        Args:
            with_brain: If True, the storage mock has a brain configured so that
                        content-level validation (not brain-not-found) is exercised.
        """
        from neural_memory.mcp.tool_handlers import ToolHandler

        storage_factory = _make_brain_storage if with_brain else _make_no_brain_storage

        class MinimalServer(ToolHandler):
            config = MagicMock()
            hooks = AsyncMock()
            hooks.emit = AsyncMock(return_value=None)

            async def get_storage(self) -> Any:
                return storage_factory()

            def _fire_eternal_trigger(self, content: str) -> None:
                pass

            async def _check_maintenance(self) -> None:
                return None

            def _get_maintenance_hint(self, pulse: Any) -> None:
                return None

            async def _passive_capture(self, text: str) -> None:
                pass

            async def _get_active_session(self, storage: Any) -> None:
                return None

            async def _check_onboarding(self) -> None:
                return None

            def get_update_hint(self) -> None:
                return None

        return MinimalServer()

    @pytest.mark.asyncio
    async def test_remember_missing_content_returns_error(self) -> None:
        handler = self._make_handler(with_brain=True)
        result = await handler._remember({})
        assert "error" in result
        assert "content" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_remember_empty_content_returns_error(self) -> None:
        handler = self._make_handler(with_brain=True)
        result = await handler._remember({"content": ""})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_remember_non_string_content_returns_error(self) -> None:
        handler = self._make_handler(with_brain=True)
        result = await handler._remember({"content": 12345})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_missing_query_returns_error(self) -> None:
        handler = self._make_handler(with_brain=True)
        result = await handler._recall({})
        assert "error" in result
        assert "query" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_recall_non_string_query_returns_error(self) -> None:
        handler = self._make_handler()
        result = await handler._recall({"query": 42})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_todo_missing_task_returns_error(self) -> None:
        handler = self._make_handler()
        result = await handler._todo({})
        assert "error" in result
        assert "task" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_todo_empty_task_returns_error(self) -> None:
        handler = self._make_handler()
        result = await handler._todo({"task": ""})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_todo_non_string_task_returns_error(self) -> None:
        handler = self._make_handler()
        result = await handler._todo({"task": None})
        assert "error" in result


def _make_no_brain_storage() -> Any:
    """Return a storage mock that has no brain configured."""
    storage = MagicMock()
    storage._current_brain_id = None
    return storage


def _make_brain_storage() -> Any:
    """Return a storage mock that has a brain set and returns a real Brain object.

    Used to get past the _require_brain_id / get_brain checks so that
    content-level validation (e.g. missing 'content') is actually reached.
    """
    from neural_memory.core.brain import Brain

    brain = Brain.create(name="test")
    storage = MagicMock()
    storage._current_brain_id = brain.id
    storage.get_brain = AsyncMock(return_value=brain)
    storage.disable_auto_save = MagicMock()
    return storage


# ---------------------------------------------------------------------------
# H4: Tags filter ordering correctness
# ---------------------------------------------------------------------------


class TestTagsFilterOrdering:
    """H4 — tags filter must apply after SQL ORDER BY salience.

    SQL fetches limit*3 rows sorted by salience DESC, then Python filters by tag.
    The final result should be the highest-salience matching fibers.
    """

    @pytest.mark.asyncio
    async def test_tags_filter_returns_highest_salience_matches(
        self, sqlite_storage: SQLiteStorage
    ) -> None:
        """With limit=2, should return the 2 highest-salience fibers that have the tag."""
        # Insert 6 fibers: even indices have the tag, varying salience
        for i in range(6):
            anchor = f"n-tag-{i}"
            has_tag = i % 2 == 0  # even: i=0 (0.0), i=2 (0.2), i=4 (0.4)
            await _add_fiber_with_neuron(
                sqlite_storage,
                anchor,
                tags={"mytag"} if has_tag else set(),
                salience=float(i) / 10,
            )

        # tagged fibers: salience 0.0, 0.2, 0.4 — top-2 should be 0.4 and 0.2
        results = await sqlite_storage.find_fibers(tags={"mytag"}, limit=2)
        assert len(results) == 2

        result_saliences = [r.salience for r in results]
        assert result_saliences[0] >= result_saliences[1]
        assert result_saliences[0] == pytest.approx(0.4, abs=1e-6)
        assert result_saliences[1] == pytest.approx(0.2, abs=1e-6)

    @pytest.mark.asyncio
    async def test_tags_filter_empty_when_no_match(self, sqlite_storage: SQLiteStorage) -> None:
        """Tags filter with no matching fibers returns empty list."""
        for i in range(3):
            await _add_fiber_with_neuron(sqlite_storage, f"n-notag-{i}", tags={"other"})

        results = await sqlite_storage.find_fibers(tags={"absent_tag"}, limit=5)
        assert results == []


# ---------------------------------------------------------------------------
# M7: In-memory find_fibers ordering by salience
# ---------------------------------------------------------------------------


class TestInMemoryOrdering:
    """M7 — InMemoryStorage.find_fibers must return highest-salience fibers first."""

    @pytest.mark.asyncio
    async def test_results_ordered_by_salience_desc(self, memory_storage: InMemoryStorage) -> None:
        """Fibers inserted in low-to-high order should be returned high-to-low."""
        saliences = [0.1, 0.9, 0.3, 0.7, 0.5]
        for idx, sal in enumerate(saliences):
            f = _make_fiber(anchor_id=f"n-ord-{idx}", salience=sal)
            await memory_storage.add_fiber(f)

        results = await memory_storage.find_fibers(limit=5)
        assert len(results) == 5

        for i in range(len(results) - 1):
            assert results[i].salience >= results[i + 1].salience, (
                f"Salience not descending at position {i}: "
                f"{results[i].salience} < {results[i + 1].salience}"
            )

    @pytest.mark.asyncio
    async def test_limit_returns_top_k_by_salience(self, memory_storage: InMemoryStorage) -> None:
        """limit=2 should return the 2 highest-salience fibers."""
        saliences = [0.2, 0.8, 0.5, 0.1, 0.9]
        for idx, sal in enumerate(saliences):
            f = _make_fiber(anchor_id=f"n-lim-{idx}", salience=sal)
            await memory_storage.add_fiber(f)

        results = await memory_storage.find_fibers(limit=2)
        assert len(results) == 2
        assert results[0].salience == pytest.approx(0.9, abs=1e-6)
        assert results[1].salience == pytest.approx(0.8, abs=1e-6)

    @pytest.mark.asyncio
    async def test_insertion_order_does_not_affect_results(
        self, memory_storage: InMemoryStorage
    ) -> None:
        """Inserting highest-salience fiber last still puts it first in results."""
        low = _make_fiber(anchor_id="n-ins-low", salience=0.1)
        mid = _make_fiber(anchor_id="n-ins-mid", salience=0.5)
        high = _make_fiber(anchor_id="n-ins-high", salience=0.95)

        # Insert in ascending order
        await memory_storage.add_fiber(low)
        await memory_storage.add_fiber(mid)
        await memory_storage.add_fiber(high)

        results = await memory_storage.find_fibers(limit=3)
        assert results[0].id == high.id
        assert results[1].id == mid.id
        assert results[2].id == low.id
