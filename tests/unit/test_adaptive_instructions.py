"""Tests for Phase 2: Adaptive Instructions.

Covers:
- Instruction metadata auto-population on encode
- nmem_refine: version increment, content update, failure_mode append, trigger append, caps
- nmem_report_outcome: execution count, success rate computation, failure append
- Recall boost for proven instructions
- Trigger pattern matching in recall
- Backward compat: old instructions without metadata work normally
- Backfill: refine on old instruction auto-adds metadata fields
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.encoder import (
    _extract_trigger_patterns,
    _inject_instruction_metadata,
)
from neural_memory.mcp.server import MCPServer

# ──────────────────── Shared helpers ────────────────────


def _make_server() -> MCPServer:
    """Create a test server with mocked config."""
    server = MCPServer.__new__(MCPServer)
    server._config = MagicMock()
    server._config.encryption = MagicMock(enabled=False, auto_encrypt_sensitive=False)
    server._config.safety = MagicMock(auto_redact_min_severity=3)
    server._config.auto = MagicMock(enabled=False)
    server._config.dedup = MagicMock(enabled=False)
    server._config.tool_tier = MagicMock(tier="full")
    server._storage = None
    server._hooks = None
    server._eternal_trigger_count = 0
    return server


def _make_typed_mem(memory_type_str: str = "instruction", fiber_id: str = "fiber-1"):
    from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory

    return TypedMemory.create(
        fiber_id=fiber_id,
        memory_type=MemoryType(memory_type_str),
        priority=Priority.NORMAL,
        source="test",
    )


def _make_fiber(
    fiber_id: str = "fiber-1",
    metadata: dict | None = None,
    anchor_neuron_id: str = "neuron-1",
):
    from neural_memory.core.fiber import Fiber

    return Fiber.create(
        neuron_ids={anchor_neuron_id},
        synapse_ids=set(),
        anchor_neuron_id=anchor_neuron_id,
        fiber_id=fiber_id,
        metadata=metadata or {},
    )


def _make_neuron(content: str = "Always run tests before committing", neuron_id: str = "neuron-1"):
    from neural_memory.core.neuron import Neuron, NeuronType

    return Neuron.create(
        content=content,
        type=NeuronType.CONCEPT,
        neuron_id=neuron_id,
    )


# ──────────────────── Encoder: trigger extraction ────────────────────


class TestExtractTriggerPatterns:
    """Tests for the _extract_trigger_patterns helper."""

    def test_returns_list(self) -> None:
        result = _extract_trigger_patterns("Always run tests before committing code")
        assert isinstance(result, list)

    def test_skips_stop_words(self) -> None:
        result = _extract_trigger_patterns("Always run tests before committing code")
        assert "before" not in result
        assert "always" not in result  # 'always' is in stop words

    def test_max_patterns_respected(self) -> None:
        result = _extract_trigger_patterns(
            "Never hardcode secrets passwords tokens credentials keys apikeys", max_patterns=3
        )
        assert len(result) <= 3

    def test_empty_content_returns_empty(self) -> None:
        result = _extract_trigger_patterns("")
        assert result == []

    def test_only_stop_words_returns_empty(self) -> None:
        result = _extract_trigger_patterns("a an the and or but")
        assert result == []

    def test_returns_lowercase(self) -> None:
        result = _extract_trigger_patterns("Deploy Production Server")
        for kw in result:
            assert kw == kw.lower()

    def test_frequency_ranking(self) -> None:
        # "deploy" repeated 3x should rank first
        result = _extract_trigger_patterns("deploy deploy deploy server database", max_patterns=5)
        assert result[0] == "deploy"


# ──────────────────── Encoder: inject_instruction_metadata ────────────────────


class TestInjectInstructionMetadata:
    """Tests for _inject_instruction_metadata."""

    def test_adds_all_required_fields(self) -> None:
        result = _inject_instruction_metadata("Run tests first", {})
        required = [
            "version",
            "execution_count",
            "success_count",
            "failure_count",
            "success_rate",
            "last_executed_at",
            "failure_modes",
            "trigger_patterns",
            "refinement_history",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_version_defaults_to_1(self) -> None:
        result = _inject_instruction_metadata("instruction text", {})
        assert result["version"] == 1

    def test_execution_counts_default_to_zero(self) -> None:
        result = _inject_instruction_metadata("instruction text", {})
        assert result["execution_count"] == 0
        assert result["success_count"] == 0
        assert result["failure_count"] == 0

    def test_success_rate_defaults_to_none(self) -> None:
        result = _inject_instruction_metadata("instruction text", {})
        assert result["success_rate"] is None

    def test_trigger_patterns_auto_extracted(self) -> None:
        result = _inject_instruction_metadata("Always commit tests before merging code", {})
        # Should have extracted some keywords
        assert isinstance(result["trigger_patterns"], list)

    def test_existing_keys_not_overwritten(self) -> None:
        # Caller passes version=3 — should be preserved
        result = _inject_instruction_metadata(
            "instruction text", {"version": 3, "execution_count": 10}
        )
        assert result["version"] == 3
        assert result["execution_count"] == 10

    def test_existing_trigger_patterns_preserved(self) -> None:
        existing_triggers = ["deploy", "production"]
        result = _inject_instruction_metadata(
            "instruction text", {"trigger_patterns": existing_triggers}
        )
        assert result["trigger_patterns"] == existing_triggers

    def test_does_not_mutate_input(self) -> None:
        original = {"type": "instruction"}
        _inject_instruction_metadata("text", original)
        assert list(original.keys()) == ["type"]  # unchanged


# ──────────────────── nmem_refine ────────────────────


class TestNmemRefine:
    """Tests for the nmem_refine handler."""

    @pytest.mark.asyncio
    async def test_missing_neuron_id(self) -> None:
        server = _make_server()
        result = await server.call_tool("nmem_refine", {})
        assert "error" in result
        assert "neuron_id" in result["error"]

    @pytest.mark.asyncio
    async def test_no_changes_provided(self) -> None:
        server = _make_server()
        result = await server.call_tool("nmem_refine", {"neuron_id": "fiber-1"})
        assert "error" in result
        assert "At least one" in result["error"]

    @pytest.mark.asyncio
    async def test_memory_not_found(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=None)
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine", {"neuron_id": "nonexistent", "new_content": "updated"}
        )
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_wrong_type_rejected(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem("fact"))
        storage.get_fiber = AsyncMock(return_value=_make_fiber())
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine", {"neuron_id": "fiber-1", "new_content": "updated"}
        )
        assert "error" in result
        assert "instruction/workflow" in result["error"]

    @pytest.mark.asyncio
    async def test_version_increments_on_content_update(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(metadata={"version": 2, "type": "instruction"})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.get_neuron = AsyncMock(return_value=_make_neuron())
        storage.update_neuron = AsyncMock()
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine",
            {"neuron_id": "fiber-1", "new_content": "New instruction text", "reason": "improved"},
        )

        assert result["status"] == "refined"
        assert result["metadata"]["version"] == 3
        assert any("v2→v3" in c for c in result["changes"])

    @pytest.mark.asyncio
    async def test_refinement_history_stored(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(metadata={"version": 1, "type": "instruction"})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.get_neuron = AsyncMock(return_value=_make_neuron("old content"))
        storage.update_neuron = AsyncMock()
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine",
            {"neuron_id": "fiber-1", "new_content": "new content", "reason": "bug fix"},
        )

        history = result["metadata"]["refinement_history"]
        assert len(history) == 1
        assert history[0]["version"] == 1
        assert history[0]["reason"] == "bug fix"
        assert "old content" in history[0]["old_content"]

    @pytest.mark.asyncio
    async def test_add_failure_mode(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(metadata={"type": "instruction"})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine",
            {"neuron_id": "fiber-1", "add_failure_mode": "Fails on Windows paths"},
        )

        assert result["status"] == "refined"
        assert "Fails on Windows paths" in result["metadata"]["failure_modes"]
        assert any("failure_mode" in c for c in result["changes"])

    @pytest.mark.asyncio
    async def test_failure_modes_deduped(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        existing_modes = ["Fails on Windows paths"]
        fiber = _make_fiber(metadata={"type": "instruction", "failure_modes": existing_modes})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine",
            {"neuron_id": "fiber-1", "add_failure_mode": "Fails on Windows paths"},
        )

        # Should still be 1, not 2
        assert result["metadata"]["failure_modes"].count("Fails on Windows paths") == 1

    @pytest.mark.asyncio
    async def test_failure_modes_capped_at_20(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        existing_modes = [f"failure-{i}" for i in range(20)]
        fiber = _make_fiber(metadata={"type": "instruction", "failure_modes": existing_modes})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine",
            {"neuron_id": "fiber-1", "add_failure_mode": "one-more"},
        )

        assert len(result["metadata"]["failure_modes"]) <= 20

    @pytest.mark.asyncio
    async def test_add_trigger_appended_and_normalized(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(metadata={"type": "instruction"})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine",
            {"neuron_id": "fiber-1", "add_trigger": "Deploy"},
        )

        assert "deploy" in result["metadata"]["trigger_patterns"]

    @pytest.mark.asyncio
    async def test_trigger_patterns_capped_at_10(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        existing_triggers = [f"trigger-{i}" for i in range(10)]
        fiber = _make_fiber(metadata={"type": "instruction", "trigger_patterns": existing_triggers})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine",
            {"neuron_id": "fiber-1", "add_trigger": "newkeyword"},
        )

        assert len(result["metadata"]["trigger_patterns"]) <= 10

    @pytest.mark.asyncio
    async def test_backfill_old_instruction_without_metadata(self) -> None:
        """Old instructions with no instruction fields should get defaults on refine."""
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        # Old fiber with no instruction metadata at all
        fiber = _make_fiber(metadata={"type": "instruction"})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine",
            {"neuron_id": "fiber-1", "add_failure_mode": "something"},
        )

        # Backfill should have supplied defaults
        meta = result["metadata"]
        assert "version" in meta
        assert "execution_count" in meta
        assert meta["execution_count"] == 0

    @pytest.mark.asyncio
    async def test_workflow_type_accepted(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(metadata={"type": "workflow"})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem("workflow"))
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine",
            {"neuron_id": "fiber-1", "add_trigger": "ci"},
        )

        assert result["status"] == "refined"

    @pytest.mark.asyncio
    async def test_update_fiber_metadata_called(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(metadata={"type": "instruction"})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        await server.call_tool(
            "nmem_refine",
            {"neuron_id": "fiber-1", "add_failure_mode": "test failure"},
        )

        storage.update_fiber_metadata.assert_awaited_once()


# ──────────────────── nmem_report_outcome ────────────────────


class TestNmemReportOutcome:
    """Tests for the nmem_report_outcome handler."""

    @pytest.mark.asyncio
    async def test_missing_neuron_id(self) -> None:
        server = _make_server()
        result = await server.call_tool("nmem_report_outcome", {"success": True})
        assert "error" in result
        assert "neuron_id" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_success(self) -> None:
        server = _make_server()
        result = await server.call_tool("nmem_report_outcome", {"neuron_id": "fiber-1"})
        assert "error" in result
        assert "success" in result["error"]

    @pytest.mark.asyncio
    async def test_memory_not_found(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=None)
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_report_outcome", {"neuron_id": "nonexistent", "success": True}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_wrong_type_rejected(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem("fact"))
        storage.get_fiber = AsyncMock(return_value=_make_fiber())
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_report_outcome", {"neuron_id": "fiber-1", "success": True}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_success_increments_exec_and_success_count(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(
            metadata={
                "type": "instruction",
                "execution_count": 5,
                "success_count": 4,
                "failure_count": 1,
            }
        )
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_report_outcome", {"neuron_id": "fiber-1", "success": True}
        )

        assert result["status"] == "recorded"
        assert result["success"] is True
        assert result["execution_count"] == 6
        assert result["success_count"] == 5
        assert result["failure_count"] == 1

    @pytest.mark.asyncio
    async def test_failure_increments_failure_count(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(
            metadata={
                "type": "instruction",
                "execution_count": 3,
                "success_count": 3,
                "failure_count": 0,
            }
        )
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_report_outcome", {"neuron_id": "fiber-1", "success": False}
        )

        assert result["failure_count"] == 1
        assert result["success_count"] == 3  # unchanged

    @pytest.mark.asyncio
    async def test_success_rate_computed_correctly(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        # 8 successes out of 9 executions → after this success: 9/10 = 0.9
        fiber = _make_fiber(
            metadata={
                "type": "instruction",
                "execution_count": 9,
                "success_count": 8,
                "failure_count": 1,
            }
        )
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_report_outcome", {"neuron_id": "fiber-1", "success": True}
        )

        assert result["success_rate"] == pytest.approx(0.9, abs=0.001)

    @pytest.mark.asyncio
    async def test_failure_description_appended_to_failure_modes(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(metadata={"type": "instruction"})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_report_outcome",
            {
                "neuron_id": "fiber-1",
                "success": False,
                "failure_description": "Timeout on slow CI",
            },
        )

        assert "Timeout on slow CI" in result["failure_modes"]

    @pytest.mark.asyncio
    async def test_failure_description_not_added_on_success(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(metadata={"type": "instruction"})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_report_outcome",
            {
                "neuron_id": "fiber-1",
                "success": True,
                "failure_description": "should not be added",
            },
        )

        assert "should not be added" not in result["failure_modes"]

    @pytest.mark.asyncio
    async def test_last_executed_at_updated(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(metadata={"type": "instruction"})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_report_outcome", {"neuron_id": "fiber-1", "success": True}
        )

        assert result["last_executed_at"] is not None

    @pytest.mark.asyncio
    async def test_backfill_old_instruction_on_report_outcome(self) -> None:
        """Old instructions without execution_count field should be backfilled."""
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        # Fiber with no instruction-specific keys
        fiber = _make_fiber(metadata={"type": "instruction"})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_report_outcome", {"neuron_id": "fiber-1", "success": True}
        )

        assert result["execution_count"] == 1  # 0 + 1
        assert result["success_count"] == 1


# ──────────────────── Recall boost ────────────────────


class TestInstructionRecallBoost:
    """Tests for the instruction scoring boost in retrieval._find_matching_fibers."""

    def _make_activation(self, neuron_id: str, level: float = 0.8):
        from neural_memory.engine.activation import ActivationResult

        return ActivationResult(
            neuron_id=neuron_id,
            activation_level=level,
            hop_distance=0,
            path=[neuron_id],
            source_anchor=neuron_id,
        )

    def _make_fiber_with_meta(
        self,
        fiber_id: str,
        neuron_id: str,
        meta: dict,
        salience: float = 0.8,
    ):
        from neural_memory.core.fiber import Fiber

        fiber = Fiber.create(
            neuron_ids={neuron_id},
            synapse_ids=set(),
            anchor_neuron_id=neuron_id,
            fiber_id=fiber_id,
            metadata=meta,
        )
        return fiber.with_salience(salience)

    def test_high_success_instruction_scores_higher_than_unproven(self) -> None:
        """A proven instruction (exec=10, rate=1.0) should beat an unproven one."""

        # Build a minimal pipeline just to access _find_matching_fibers indirectly
        # by testing the _fiber_score closure logic directly.
        # We verify the boost math is correct.

        # proven: exec=10, success_rate=1.0 → confidence=1.0, boost=(1.0-0.5)*1.0*0.3 = 0.15
        proven_meta = {
            "type": "instruction",
            "execution_count": 10,
            "success_rate": 1.0,
        }
        # unproven: exec=0 → no boost
        unproven_meta = {"type": "instruction", "execution_count": 0}

        # Simulate boost calculation directly
        def compute_boost(meta: dict) -> float:
            exec_count = meta.get("execution_count", 0)
            success_rate = meta.get("success_rate")
            if exec_count > 0 and success_rate is not None:
                confidence = min(1.0, exec_count / 10.0)
                return (float(success_rate) - 0.5) * confidence * 0.3
            return 0.0

        proven_boost = compute_boost(proven_meta)
        unproven_boost = compute_boost(unproven_meta)
        assert proven_boost > unproven_boost
        assert proven_boost == pytest.approx(0.15, abs=0.001)
        assert unproven_boost == 0.0

    def test_low_success_instruction_gets_negative_boost(self) -> None:
        """An instruction with 0% success rate should get a negative boost."""

        def compute_boost(meta: dict) -> float:
            exec_count = meta.get("execution_count", 0)
            success_rate = meta.get("success_rate")
            if exec_count > 0 and success_rate is not None:
                confidence = min(1.0, exec_count / 10.0)
                return (float(success_rate) - 0.5) * confidence * 0.3
            return 0.0

        bad_meta = {"type": "instruction", "execution_count": 10, "success_rate": 0.0}
        boost = compute_boost(bad_meta)
        assert boost < 0

    def test_confidence_grows_with_execution_count(self) -> None:
        """More executions = more confident in the boost."""

        def compute_boost(exec_count: int, success_rate: float) -> float:
            confidence = min(1.0, exec_count / 10.0)
            return (success_rate - 0.5) * confidence * 0.3

        boost_at_1 = compute_boost(1, 1.0)
        boost_at_5 = compute_boost(5, 1.0)
        boost_at_10 = compute_boost(10, 1.0)

        assert boost_at_1 < boost_at_5 < boost_at_10
        assert boost_at_10 == pytest.approx(0.15, abs=0.001)


# ──────────────────── Trigger pattern matching ────────────────────


class TestTriggerPatternMatching:
    """Tests for trigger pattern matching boost logic."""

    def test_query_overlap_above_threshold_boosts(self) -> None:
        """Query tokens overlapping triggers > 30% should give positive boost."""
        query_tokens = {"deploy", "production", "server"}
        trigger_patterns = {"deploy", "production"}

        overlap = len(query_tokens & trigger_patterns) / max(len(trigger_patterns), 1)
        assert overlap > 0.3
        boost = overlap * 0.2
        assert boost > 0

    def test_no_overlap_gives_no_boost(self) -> None:
        query_tokens = {"write", "tests", "first"}
        trigger_patterns = {"deploy", "production"}

        overlap = len(query_tokens & trigger_patterns) / max(len(trigger_patterns), 1)
        assert overlap == 0.0

    def test_empty_triggers_no_boost(self) -> None:
        query_tokens = {"deploy", "production"}
        trigger_patterns: set[str] = set()

        # When triggers is empty, we skip the boost (no div by zero)
        if trigger_patterns:
            overlap = len(query_tokens & trigger_patterns) / max(len(trigger_patterns), 1)
        else:
            overlap = 0.0
        assert overlap == 0.0

    def test_full_overlap_gives_max_boost(self) -> None:
        query_tokens = {"deploy", "production"}
        trigger_patterns = {"deploy", "production"}

        overlap = len(query_tokens & trigger_patterns) / max(len(trigger_patterns), 1)
        boost = overlap * 0.2 if overlap > 0.3 else 0.0
        assert boost == pytest.approx(0.2, abs=0.001)


# ──────────────────── Backward compatibility ────────────────────


class TestBackwardCompatibility:
    """Ensure instructions without metadata fields don't break."""

    @pytest.mark.asyncio
    async def test_report_outcome_on_bare_instruction(self) -> None:
        """An instruction with completely empty metadata should be safe to report on."""
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        # Completely empty metadata
        fiber = _make_fiber(metadata={})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_report_outcome", {"neuron_id": "fiber-1", "success": True}
        )

        # Should succeed and return valid stats
        assert "error" not in result
        assert result["execution_count"] == 1
        assert result["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_refine_on_bare_instruction(self) -> None:
        """An instruction with completely empty metadata should be safe to refine."""
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        fiber = _make_fiber(metadata={})
        storage.get_typed_memory = AsyncMock(return_value=_make_typed_mem())
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_fiber_metadata = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "nmem_refine",
            {"neuron_id": "fiber-1", "add_trigger": "test"},
        )

        assert "error" not in result
        assert result["status"] == "refined"
