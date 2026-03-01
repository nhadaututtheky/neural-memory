"""Tests for tool memory engine (v2.17.0)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from neural_memory.engine.tool_memory import (
    IngestResult,
    ProcessResult,
    _parse_buffer_line,
    _tool_neuron_content,
    ingest_buffer,
    process_events,
)
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.unified_config import ToolMemoryConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def storage(tmp_path: Path) -> SQLiteStorage:
    """Create an initialized SQLiteStorage for testing."""
    db_path = tmp_path / "test.db"
    store = SQLiteStorage(db_path)
    await store.initialize()
    # Create a test brain with all required columns
    await store._ensure_conn().execute(
        "INSERT OR IGNORE INTO brains (id, name, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("test-brain", "test", "{}", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    await store._ensure_conn().commit()
    store.set_brain("test-brain")
    return store


def _make_jsonl(events: list[dict]) -> str:
    """Create JSONL content from a list of event dicts."""
    return "\n".join(json.dumps(ev) for ev in events) + "\n"


def _make_event(
    tool_name: str = "nmem_recall",
    server_name: str = "neural-memory",
    success: bool = True,
    duration_ms: int = 100,
    session_id: str = "session-1",
    task_context: str = "",
    created_at: str = "2026-03-01T10:00:00",
) -> dict:
    return {
        "tool_name": tool_name,
        "server_name": server_name,
        "args_summary": '{"query": "test"}',
        "success": success,
        "duration_ms": duration_ms,
        "session_id": session_id,
        "task_context": task_context,
        "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_ingest_result_frozen(self) -> None:
        r = IngestResult(events_ingested=5, events_skipped=1)
        assert r.events_ingested == 5
        with pytest.raises(FrozenInstanceError):
            r.events_ingested = 10  # type: ignore[misc]

    def test_process_result_frozen(self) -> None:
        r = ProcessResult(
            neurons_created=2, synapses_created=1, synapses_reinforced=0, events_processed=5
        )
        assert r.events_processed == 5
        with pytest.raises(FrozenInstanceError):
            r.events_processed = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Buffer parsing
# ---------------------------------------------------------------------------


class TestParseBufferLine:
    def test_valid_line(self) -> None:
        line = json.dumps({"tool_name": "Read", "success": True})
        result = _parse_buffer_line(line)
        assert result is not None
        assert result["tool_name"] == "Read"

    def test_missing_tool_name(self) -> None:
        line = json.dumps({"success": True})
        assert _parse_buffer_line(line) is None

    def test_malformed_json(self) -> None:
        assert _parse_buffer_line("not json") is None

    def test_empty_string(self) -> None:
        assert _parse_buffer_line("") is None


class TestToolNeuronContent:
    def test_format(self) -> None:
        assert _tool_neuron_content("nmem_recall") == "tool:nmem_recall"


# ---------------------------------------------------------------------------
# Ingest buffer
# ---------------------------------------------------------------------------


class TestIngestBuffer:
    @pytest.mark.asyncio
    async def test_ingest_valid_events(self, storage: SQLiteStorage, tmp_path: Path) -> None:
        events = [_make_event(tool_name="Read"), _make_event(tool_name="Grep")]
        buf = tmp_path / "tool_events.jsonl"
        buf.write_text(_make_jsonl(events))

        result = await ingest_buffer(storage, "test-brain", buf)
        assert result.events_ingested == 2
        assert result.events_skipped == 0
        # Buffer should be truncated
        assert buf.read_text() == ""

    @pytest.mark.asyncio
    async def test_ingest_skips_malformed(self, storage: SQLiteStorage, tmp_path: Path) -> None:
        content = json.dumps({"tool_name": "Read"}) + "\nnot-json\n"
        buf = tmp_path / "tool_events.jsonl"
        buf.write_text(content)

        result = await ingest_buffer(storage, "test-brain", buf)
        assert result.events_ingested == 1
        assert result.events_skipped == 1

    @pytest.mark.asyncio
    async def test_ingest_missing_file(self, storage: SQLiteStorage, tmp_path: Path) -> None:
        buf = tmp_path / "nonexistent.jsonl"
        result = await ingest_buffer(storage, "test-brain", buf)
        assert result.events_ingested == 0

    @pytest.mark.asyncio
    async def test_ingest_empty_file(self, storage: SQLiteStorage, tmp_path: Path) -> None:
        buf = tmp_path / "tool_events.jsonl"
        buf.write_text("")
        result = await ingest_buffer(storage, "test-brain", buf)
        assert result.events_ingested == 0


# ---------------------------------------------------------------------------
# Storage mixin
# ---------------------------------------------------------------------------


class TestToolEventsStorage:
    @pytest.mark.asyncio
    async def test_insert_and_get_unprocessed(self, storage: SQLiteStorage) -> None:
        events = [_make_event(tool_name="Read"), _make_event(tool_name="Grep")]
        count = await storage.insert_tool_events("test-brain", events)
        assert count == 2

        unprocessed = await storage.get_unprocessed_events("test-brain")
        assert len(unprocessed) == 2
        assert unprocessed[0]["tool_name"] == "Read"

    @pytest.mark.asyncio
    async def test_mark_processed(self, storage: SQLiteStorage) -> None:
        await storage.insert_tool_events("test-brain", [_make_event()])
        events = await storage.get_unprocessed_events("test-brain")
        assert len(events) == 1

        await storage.mark_events_processed("test-brain", [events[0]["id"]])
        remaining = await storage.get_unprocessed_events("test-brain")
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_get_tool_stats(self, storage: SQLiteStorage) -> None:
        events = [
            _make_event(tool_name="Read", success=True),
            _make_event(tool_name="Read", success=True),
            _make_event(tool_name="Read", success=False),
            _make_event(tool_name="Grep", success=True),
        ]
        await storage.insert_tool_events("test-brain", events)
        stats = await storage.get_tool_stats("test-brain")
        assert stats["total_events"] == 4
        assert len(stats["top_tools"]) == 2
        # Read: 2/3 success rate
        read_tool = next(t for t in stats["top_tools"] if t["tool_name"] == "Read")
        assert read_tool["count"] == 3
        assert read_tool["success_rate"] == pytest.approx(0.67, abs=0.01)

    @pytest.mark.asyncio
    async def test_empty_stats(self, storage: SQLiteStorage) -> None:
        stats = await storage.get_tool_stats("test-brain")
        assert stats["total_events"] == 0
        assert stats["top_tools"] == []


# ---------------------------------------------------------------------------
# Process events — pattern detection
# ---------------------------------------------------------------------------


class TestProcessEvents:
    @pytest.mark.asyncio
    async def test_no_events(self, storage: SQLiteStorage) -> None:
        config = ToolMemoryConfig(enabled=True, min_frequency=1)
        result = await process_events(storage, "test-brain", config)
        assert result.events_processed == 0

    @pytest.mark.asyncio
    async def test_used_with_detection(self, storage: SQLiteStorage) -> None:
        """Tools used within time window in same session create USED_WITH synapse."""
        events = [
            _make_event(tool_name="Grep", session_id="s1", created_at="2026-03-01T10:00:00"),
            _make_event(tool_name="Read", session_id="s1", created_at="2026-03-01T10:00:30"),
            _make_event(tool_name="Grep", session_id="s1", created_at="2026-03-01T10:01:00"),
            _make_event(tool_name="Read", session_id="s1", created_at="2026-03-01T10:01:30"),
        ]
        await storage.insert_tool_events("test-brain", events)
        config = ToolMemoryConfig(enabled=True, min_frequency=2, cooccurrence_window_s=60)
        result = await process_events(storage, "test-brain", config)

        assert result.events_processed == 4
        assert result.synapses_created >= 1

        # Verify USED_WITH synapse exists
        from neural_memory.core.synapse import SynapseType

        synapses = await storage.get_synapses(type=SynapseType.USED_WITH)
        assert len(synapses) >= 1

    @pytest.mark.asyncio
    async def test_frequency_threshold(self, storage: SQLiteStorage) -> None:
        """Tools below frequency threshold don't create neurons."""
        events = [
            _make_event(tool_name="Rare", session_id="s1", created_at="2026-03-01T10:00:00"),
            _make_event(tool_name="Common", session_id="s1", created_at="2026-03-01T10:00:05"),
            _make_event(tool_name="Common", session_id="s1", created_at="2026-03-01T10:00:10"),
            _make_event(tool_name="Common", session_id="s1", created_at="2026-03-01T10:00:15"),
        ]
        await storage.insert_tool_events("test-brain", events)
        config = ToolMemoryConfig(enabled=True, min_frequency=3)
        result = await process_events(storage, "test-brain", config)

        # "Rare" doesn't meet frequency threshold, so no USED_WITH pairing
        assert result.synapses_created == 0
        assert result.events_processed == 4

    @pytest.mark.asyncio
    async def test_effective_for_detection(self, storage: SQLiteStorage) -> None:
        """Successful tool with task context creates EFFECTIVE_FOR synapse."""
        events = [
            _make_event(
                tool_name="Read", task_context="debug auth", created_at="2026-03-01T10:00:00"
            ),
            _make_event(
                tool_name="Read", task_context="debug auth", created_at="2026-03-01T10:00:05"
            ),
            _make_event(
                tool_name="Read", task_context="debug auth", created_at="2026-03-01T10:00:10"
            ),
        ]
        await storage.insert_tool_events("test-brain", events)
        config = ToolMemoryConfig(enabled=True, min_frequency=3)
        result = await process_events(storage, "test-brain", config)

        assert result.events_processed == 3
        assert result.synapses_created >= 1

        from neural_memory.core.synapse import SynapseType

        synapses = await storage.get_synapses(type=SynapseType.EFFECTIVE_FOR)
        assert len(synapses) >= 1

    @pytest.mark.asyncio
    async def test_failed_tool_no_effective_for(self, storage: SQLiteStorage) -> None:
        """Failed tool calls don't create EFFECTIVE_FOR synapses."""
        events = [
            _make_event(
                tool_name="Bash",
                success=False,
                task_context="deploy",
                created_at="2026-03-01T10:00:00",
            ),
            _make_event(
                tool_name="Bash",
                success=False,
                task_context="deploy",
                created_at="2026-03-01T10:00:05",
            ),
            _make_event(
                tool_name="Bash",
                success=False,
                task_context="deploy",
                created_at="2026-03-01T10:00:10",
            ),
        ]
        await storage.insert_tool_events("test-brain", events)
        config = ToolMemoryConfig(enabled=True, min_frequency=3)
        await process_events(storage, "test-brain", config)

        from neural_memory.core.synapse import SynapseType

        synapses = await storage.get_synapses(type=SynapseType.EFFECTIVE_FOR)
        assert len(synapses) == 0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestToolMemoryConfig:
    def test_defaults(self) -> None:
        c = ToolMemoryConfig()
        assert c.enabled is False
        assert c.min_frequency == 3
        assert c.cooccurrence_window_s == 60

    def test_from_dict(self) -> None:
        c = ToolMemoryConfig.from_dict(
            {
                "enabled": True,
                "blacklist": ["nmem_"],
                "min_frequency": 5,
            }
        )
        assert c.enabled is True
        assert c.blacklist == ("nmem_",)
        assert c.min_frequency == 5

    def test_to_dict_roundtrip(self) -> None:
        c = ToolMemoryConfig(enabled=True, blacklist=("foo",), min_frequency=10)
        d = c.to_dict()
        c2 = ToolMemoryConfig.from_dict(d)
        assert c2.enabled == c.enabled
        assert c2.blacklist == c.blacklist
        assert c2.min_frequency == c.min_frequency

    def test_from_dict_invalid_values(self) -> None:
        """Invalid values fall back to defaults."""
        c = ToolMemoryConfig.from_dict(
            {
                "min_frequency": "not_a_number",
                "cooccurrence_window_s": -5,
            }
        )
        assert c.min_frequency == 3  # default
        assert c.cooccurrence_window_s == 1  # clamped to min 1
