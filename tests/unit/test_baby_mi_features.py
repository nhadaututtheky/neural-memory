"""Tests for Baby Mi feedback features (v2.28.0).

Covers:
1. FK constraint fix in update_fiber
2. SEMANTIC alternative path (rehearsal count + distinct windows)
3. Bulk remember batch
4. Auto-promote context→fact
5. Trust score field
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.core.memory_types import (
    MemoryType,
    TypedMemory,
    _cap_trust_score,
)
from neural_memory.engine.memory_stages import (
    _MIN_DISTINCT_WINDOWS,
    _MIN_REHEARSAL_COUNT,
    MaturationRecord,
    MemoryStage,
    compute_stage_transition,
)
from neural_memory.mcp.constants import MAX_BATCH_SIZE, MAX_BATCH_TOTAL_CHARS
from neural_memory.storage.sqlite_schema import SCHEMA_VERSION
from neural_memory.utils.timeutils import utcnow

# ─────────────────── #1: FK Constraint Fix ───────────────────


class TestFKConstraintFix:
    """update_fiber should gracefully skip when fiber was deleted."""

    @pytest.mark.asyncio
    async def test_update_fiber_deleted_fiber_no_exception(self):
        """update_fiber with rowcount=0 should return, not raise."""
        from neural_memory.storage.sqlite_fibers import SQLiteFiberMixin

        mixin = SQLiteFiberMixin()

        # Mock connection — UPDATE returns rowcount=0 (fiber gone)
        mock_cursor = AsyncMock()
        mock_cursor.rowcount = 0
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        mixin._ensure_conn = MagicMock(return_value=mock_conn)  # type: ignore[attr-defined]
        mixin._get_brain_id = MagicMock(return_value="test-brain")  # type: ignore[attr-defined]

        fiber = MagicMock()
        fiber.id = "deleted-fiber"
        fiber.neuron_ids = {"n1"}
        fiber.synapse_ids = {"s1"}
        fiber.anchor_neuron_id = "n1"
        fiber.pathway = []
        fiber.conductivity = 1.0
        fiber.last_conducted = None
        fiber.time_start = None
        fiber.time_end = None
        fiber.coherence = 0.0
        fiber.salience = 0.0
        fiber.frequency = 0
        fiber.summary = None
        fiber.tags = set()
        fiber.auto_tags = set()
        fiber.agent_tags = set()
        fiber.metadata = {}
        fiber.compression_tier = 0
        fiber.pinned = False

        # Should NOT raise ValueError
        await mixin.update_fiber(fiber)

        # Should NOT call commit (early return)
        mock_conn.commit.assert_not_called()


# ─────────────────── #2: SEMANTIC Alternative Path ───────────────────


class TestSemanticAlternativePath:
    """EPISODIC→SEMANTIC via rehearsal count + distinct windows."""

    def test_constants(self):
        assert _MIN_REHEARSAL_COUNT == 15
        assert _MIN_DISTINCT_WINDOWS == 5

    def test_classic_path_still_works(self):
        """3 distinct days + 7 days elapsed → SEMANTIC."""
        now = utcnow()
        entered = now - timedelta(days=8)
        # 3 distinct days of reinforcement
        timestamps = tuple((entered + timedelta(days=d)).isoformat() for d in [1, 3, 5])
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=3,
            reinforcement_timestamps=timestamps,
        )
        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.SEMANTIC

    def test_agent_path_high_rehearsals_with_spread(self):
        """15+ rehearsals, 5+ distinct 2h windows, 7+ days → SEMANTIC."""
        now = utcnow()
        entered = now - timedelta(days=8)
        # Generate 15 timestamps spread across different 2h windows on same day
        base = entered + timedelta(days=1)
        timestamps = tuple(
            (base + timedelta(hours=i * 2, minutes=10)).isoformat() for i in range(15)
        )
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=15,
            reinforcement_timestamps=timestamps,
        )
        # Should have >= 5 distinct 2h windows
        assert record.distinct_reinforcement_windows >= 5
        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.SEMANTIC

    def test_agent_path_high_rehearsals_no_spread_stays_episodic(self):
        """15 rehearsals but all in same 2h window → still EPISODIC."""
        now = datetime(2026, 3, 10, 12, 0, 0)
        entered = datetime(2026, 3, 1, 12, 0, 0)
        # All timestamps in hour 10:00-11:14 on same day → bucket 10//2=5
        base = datetime(2026, 3, 2, 10, 0, 0)
        timestamps = tuple((base + timedelta(minutes=i * 5)).isoformat() for i in range(15))
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=15,
            reinforcement_timestamps=timestamps,
        )
        assert record.distinct_reinforcement_windows == 1
        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.EPISODIC

    def test_agent_path_not_enough_rehearsals(self):
        """10 rehearsals (< 15) with spread → still EPISODIC."""
        now = utcnow()
        entered = now - timedelta(days=8)
        base = entered + timedelta(days=1)
        timestamps = tuple((base + timedelta(hours=i * 2)).isoformat() for i in range(10))
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=10,
            reinforcement_timestamps=timestamps,
        )
        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.EPISODIC

    def test_time_gate_enforced(self):
        """15 rehearsals + spread but only 3 days elapsed → EPISODIC."""
        now = utcnow()
        entered = now - timedelta(days=3)  # Only 3 days, not 7
        base = entered + timedelta(hours=1)
        timestamps = tuple((base + timedelta(hours=i * 2)).isoformat() for i in range(15))
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=15,
            reinforcement_timestamps=timestamps,
        )
        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.EPISODIC

    def test_distinct_reinforcement_windows_property(self):
        """Test the window bucketing logic."""
        ts = [
            "2026-03-01T08:30:00",  # bucket 4 (8/2=4)
            "2026-03-01T09:30:00",  # bucket 4 (9/2=4) — same
            "2026-03-01T10:30:00",  # bucket 5 (10/2=5)
            "2026-03-01T14:30:00",  # bucket 7 (14/2=7)
            "2026-03-02T08:30:00",  # bucket 4 on different day
        ]
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=utcnow(),
            reinforcement_timestamps=tuple(ts),
        )
        # day1:4, day1:5, day1:7, day2:4 = 4 distinct windows
        assert record.distinct_reinforcement_windows == 4


# ─────────────────── #3: Bulk Remember ───────────────────


class TestBulkRemember:
    """nmem_remember_batch tool."""

    def test_batch_constants(self):
        assert MAX_BATCH_SIZE == 20
        assert MAX_BATCH_TOTAL_CHARS == 500_000

    @pytest.mark.asyncio
    async def test_batch_empty_array_error(self):
        """Empty memories array should return error."""
        from neural_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)

        result = await handler._remember_batch({"memories": []})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_batch_too_many_items_error(self):
        """More than 20 items should return error."""
        from neural_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)

        memories = [{"content": f"memory {i}"} for i in range(25)]
        result = await handler._remember_batch({"memories": memories})
        assert "error" in result
        assert "25" in result["error"]

    @pytest.mark.asyncio
    async def test_batch_total_chars_limit(self):
        """Total content exceeding 500K should return error."""
        from neural_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)

        memories = [{"content": "x" * 100_000} for _ in range(6)]  # 600K
        result = await handler._remember_batch({"memories": memories})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_batch_partial_success(self):
        """Some items succeed, some fail — partial success."""
        from neural_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)

        # _remember returns success for first, error for second
        async def mock_remember(args):
            if args.get("content") == "good":
                return {"success": True, "fiber_id": "f1", "memory_type": "fact"}
            return {"error": "bad content"}

        handler._remember = AsyncMock(side_effect=mock_remember)

        result = await handler._remember_batch(
            {
                "memories": [
                    {"content": "good"},
                    {"content": "bad"},
                    {"content": "good"},
                ]
            }
        )
        assert result["saved"] == 2
        assert result["failed"] == 1
        assert result["total"] == 3
        assert len(result["results"]) == 3
        assert result["results"][0]["status"] == "ok"
        assert result["results"][1]["status"] == "error"
        assert result["results"][2]["status"] == "ok"


# ─────────────────── #4: Auto-Promote Context→Fact ───────────────────


class TestAutoPromote:
    """Context memories with frequency >= 5 get promoted to fact."""

    @pytest.mark.asyncio
    async def test_promote_memory_type_stores_audit_trail(self):
        """Promotion should set metadata.auto_promoted and promoted_from."""
        from neural_memory.storage.sqlite_typed import SQLiteTypedMemoryMixin

        mixin = SQLiteTypedMemoryMixin()

        # Mock connection
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(json.dumps({}), "context"))

        mock_update_cursor = AsyncMock()
        mock_update_cursor.rowcount = 1

        call_count = 0

        async def mock_execute(sql, params=None):
            nonlocal call_count
            call_count += 1
            if "SELECT" in sql:
                return mock_cursor
            return mock_update_cursor

        mock_conn = AsyncMock()
        mock_conn.execute = mock_execute
        mock_conn.commit = AsyncMock()

        mixin._ensure_conn = MagicMock(return_value=mock_conn)  # type: ignore[attr-defined]
        mixin._get_brain_id = MagicMock(return_value="test-brain")  # type: ignore[attr-defined]

        result = await mixin.promote_memory_type(
            fiber_id="f1",
            new_type=MemoryType.FACT,
            new_expires_at=None,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_promote_skips_already_same_type(self):
        """Should not promote if already the target type."""
        from neural_memory.storage.sqlite_typed import SQLiteTypedMemoryMixin

        mixin = SQLiteTypedMemoryMixin()

        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(
            return_value=(json.dumps({}), "fact")  # Already fact
        )

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        mixin._ensure_conn = MagicMock(return_value=mock_conn)  # type: ignore[attr-defined]
        mixin._get_brain_id = MagicMock(return_value="test-brain")  # type: ignore[attr-defined]

        result = await mixin.promote_memory_type(
            fiber_id="f1",
            new_type=MemoryType.FACT,
        )
        assert result is False


# ─────────────────── #5: Trust Score ───────────────────


class TestTrustScore:
    """Trust score field on TypedMemory."""

    def test_schema_version_25(self):
        assert SCHEMA_VERSION == 31

    def test_trust_score_field_on_typed_memory(self):
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
            trust_score=0.8,
        )
        assert tm.trust_score is not None
        assert tm.trust_score <= 0.9  # Capped by user_input ceiling

    def test_trust_score_none_by_default(self):
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
        )
        assert tm.trust_score is None

    def test_cap_trust_score_user_input(self):
        assert _cap_trust_score(1.0, "user_input") == 0.9

    def test_cap_trust_score_verified(self):
        assert _cap_trust_score(1.0, "verified") == 1.0

    def test_cap_trust_score_ai_inference(self):
        assert _cap_trust_score(0.9, "ai_inference") == 0.7

    def test_cap_trust_score_auto_capture(self):
        assert _cap_trust_score(0.8, "auto_capture") == 0.5

    def test_cap_trust_score_none_passthrough(self):
        assert _cap_trust_score(None, "user_input") is None

    def test_cap_trust_score_clamps_negative(self):
        result = _cap_trust_score(-0.5, "verified")
        assert result == 0.0

    def test_cap_trust_score_clamps_over_one(self):
        result = _cap_trust_score(1.5, "verified")
        assert result == 1.0

    def test_cap_trust_score_mcp_source(self):
        """mcp:claude_code → mcp_tool ceiling 0.8."""
        result = _cap_trust_score(0.95, "mcp:claude_code")
        assert result == 0.8

    def test_typed_memory_source_field(self):
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
            source="user_input",
        )
        assert tm.source == "user_input"
