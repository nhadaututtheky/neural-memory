"""Tests for the eternal context system.

Tests cover:
- TriggerEngine: pattern detection for all trigger types
- EternalContext: async query layer over neural graph
"""

from __future__ import annotations

from datetime import datetime

import pytest

from neural_memory.core.eternal_context import EternalContext
from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.core.trigger_engine import (
    TriggerType,
    check_triggers,
    estimate_session_tokens,
)

# ──────────────────── TriggerEngine Tests ────────────────────


class TestTriggerEngine:
    """Tests for auto-save trigger detection."""

    def test_no_trigger_on_normal_text(self) -> None:
        """Normal conversation text should not trigger."""
        result = check_triggers("How do I implement a login form?", message_count=3)
        assert result.triggered is False

    def test_no_trigger_on_short_text(self) -> None:
        """Very short text should not trigger patterns."""
        result = check_triggers("ok", message_count=3)
        assert result.triggered is False

    def test_user_leaving_english(self) -> None:
        """Detect English user-leaving patterns."""
        result = check_triggers("Alright, bye! I'll continue tomorrow.", message_count=10)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.USER_LEAVING
        assert 1 in result.save_tiers and 2 in result.save_tiers and 3 in result.save_tiers

    def test_user_leaving_vietnamese(self) -> None:
        """Detect Vietnamese user-leaving patterns."""
        result = check_triggers("ok tạm nghỉ nhé, mai làm tiếp", message_count=10)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.USER_LEAVING

    def test_milestone_english(self) -> None:
        """Detect English milestone/workflow completion."""
        result = check_triggers("All tests pass, feature complete!", message_count=10)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.WORKFLOW_END
        assert 1 in result.save_tiers and 2 in result.save_tiers

    def test_milestone_vietnamese(self) -> None:
        """Detect Vietnamese milestone patterns."""
        result = check_triggers("hoàn thành rồi, deploy xong", message_count=10)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.WORKFLOW_END

    def test_error_fixed(self) -> None:
        """Detect error-fixed patterns."""
        result = check_triggers("Fixed the CORS issue by adding the right headers", message_count=5)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.ERROR_FIXED

    def test_decision_made(self) -> None:
        """Detect decision patterns."""
        result = check_triggers(
            "We decided to use PostgreSQL instead of MongoDB for this project",
            message_count=5,
        )
        assert result.triggered is True
        assert result.trigger_type == TriggerType.DECISION_MADE

    def test_checkpoint_every_n_messages(self) -> None:
        """Checkpoint triggers at multiples of interval."""
        result = check_triggers("just a normal message", message_count=15)
        assert result.triggered is True
        assert result.trigger_type == TriggerType.CHECKPOINT
        assert 2 in result.save_tiers

    def test_no_checkpoint_at_non_multiple(self) -> None:
        """No checkpoint at non-multiples."""
        result = check_triggers("just a normal message", message_count=14)
        assert result.triggered is False

    def test_context_warning(self) -> None:
        """Context capacity warning when tokens > threshold."""
        result = check_triggers(
            "some text here",
            message_count=5,
            token_estimate=110_000,
            max_tokens=128_000,
        )
        assert result.triggered is True
        assert result.trigger_type == TriggerType.CONTEXT_WARNING
        assert 1 in result.save_tiers and 2 in result.save_tiers and 3 in result.save_tiers

    def test_context_warning_priority_over_patterns(self) -> None:
        """Context warning has highest priority even if text matches other patterns."""
        result = check_triggers(
            "bye, I'm done for today",
            message_count=15,
            token_estimate=120_000,
            max_tokens=128_000,
        )
        assert result.trigger_type == TriggerType.CONTEXT_WARNING

    def test_estimate_session_tokens(self) -> None:
        """Token estimation formula: messages*150 + code*5 + errors*300."""
        tokens = estimate_session_tokens(message_count=100, code_lines=200, error_count=2)
        assert tokens == 100 * 150 + 200 * 5 + 2 * 300
        assert tokens == 16600

    def test_estimate_session_tokens_messages_only(self) -> None:
        """Token estimation with messages only."""
        tokens = estimate_session_tokens(message_count=50)
        assert tokens == 7500


# ──────────────────── Helper to populate storage ────────────────────


async def _encode_typed_memory(
    storage,
    content: str,
    memory_type: MemoryType,
    priority: int = 5,
    tags: set[str] | None = None,
    metadata: dict | None = None,
) -> str:
    """Encode content as a fiber + typed memory. Returns fiber_id."""
    from neural_memory.core.brain import BrainConfig
    from neural_memory.engine.encoder import MemoryEncoder

    brain = await storage.get_brain(storage._current_brain_id)
    if brain is None:
        msg = "No brain configured"
        raise RuntimeError(msg)
    encoder = MemoryEncoder(storage, brain.config if brain.config else BrainConfig())
    result = await encoder.encode(
        content=content,
        timestamp=datetime.now(),
        tags=tags or set(),
    )
    typed_mem = TypedMemory.create(
        fiber_id=result.fiber.id,
        memory_type=memory_type,
        priority=Priority.from_int(priority),
        source="test",
        tags=tags,
        metadata=metadata,
    )
    await storage.add_typed_memory(typed_mem)
    return result.fiber.id


# ──────────────────── EternalContext Tests ────────────────────


class TestEternalContext:
    """Tests for the async eternal context query layer."""

    @pytest.mark.asyncio
    async def test_empty_injection_level1(self, storage) -> None:
        """Level 1 injection on empty brain returns header."""
        ctx = EternalContext(storage, "test")
        injection = await ctx.get_injection(level=1)
        assert "## Project Context" in injection

    @pytest.mark.asyncio
    async def test_injection_with_project_context(self, storage) -> None:
        """Level 1 shows project context FACT."""
        await _encode_typed_memory(
            storage,
            "Project: MyApp. Tech stack: Python, FastAPI",
            MemoryType.FACT,
            priority=10,
            tags={"project_context"},
        )
        ctx = EternalContext(storage, "test")
        injection = await ctx.get_injection(level=1)
        assert "MyApp" in injection

    @pytest.mark.asyncio
    async def test_injection_with_instructions(self, storage) -> None:
        """Level 1 shows high-priority instructions."""
        await _encode_typed_memory(
            storage,
            "Always use PostgreSQL, never SQLite",
            MemoryType.INSTRUCTION,
            priority=9,
        )
        ctx = EternalContext(storage, "test")
        injection = await ctx.get_injection(level=1)
        assert "PostgreSQL" in injection

    @pytest.mark.asyncio
    async def test_injection_with_session_state(self, storage) -> None:
        """Level 1 shows current session feature/task."""
        await _encode_typed_memory(
            storage,
            "Session: working on Authentication, task: Login form",
            MemoryType.CONTEXT,
            priority=7,
            tags={"session_state"},
            metadata={
                "active": True,
                "feature": "Authentication",
                "task": "Login form",
                "progress": 0.65,
                "branch": "feat/auth",
            },
        )
        ctx = EternalContext(storage, "test")
        injection = await ctx.get_injection(level=1)
        assert "Authentication" in injection
        assert "Login form" in injection
        assert "65%" in injection
        assert "feat/auth" in injection

    @pytest.mark.asyncio
    async def test_injection_level2_decisions(self, storage) -> None:
        """Level 2 includes decisions."""
        await _encode_typed_memory(
            storage,
            "Decision: Use NextAuth for authentication",
            MemoryType.DECISION,
            priority=7,
        )
        ctx = EternalContext(storage, "test")
        injection = await ctx.get_injection(level=2)
        assert "NextAuth" in injection
        assert "Key Decisions" in injection

    @pytest.mark.asyncio
    async def test_injection_level2_todos(self, storage) -> None:
        """Level 2 includes pending todos."""
        await _encode_typed_memory(
            storage,
            "Add unit tests for auth module",
            MemoryType.TODO,
            priority=5,
        )
        ctx = EternalContext(storage, "test")
        injection = await ctx.get_injection(level=2)
        assert "unit tests" in injection
        assert "Pending Tasks" in injection

    @pytest.mark.asyncio
    async def test_injection_level3_summaries(self, storage) -> None:
        """Level 3 includes session summaries."""
        await _encode_typed_memory(
            storage,
            "Discussed auth options, picked NextAuth",
            MemoryType.CONTEXT,
            priority=5,
            tags={"session_summary"},
        )
        ctx = EternalContext(storage, "test")
        injection = await ctx.get_injection(level=3)
        assert "Session History" in injection

    @pytest.mark.asyncio
    async def test_injection_level1_does_not_include_decisions(self, storage) -> None:
        """Level 1 should NOT include level 2 content."""
        await _encode_typed_memory(
            storage,
            "Decision: Use Redis for caching",
            MemoryType.DECISION,
            priority=7,
        )
        ctx = EternalContext(storage, "test")
        injection = await ctx.get_injection(level=1)
        assert "Key Decisions" not in injection

    @pytest.mark.asyncio
    async def test_get_status(self, storage) -> None:
        """Status returns memory counts and session info."""
        await _encode_typed_memory(storage, "Fact 1", MemoryType.FACT, priority=5)
        await _encode_typed_memory(storage, "Fact 2", MemoryType.FACT, priority=5)
        await _encode_typed_memory(
            storage,
            "Decision: Use X",
            MemoryType.DECISION,
            priority=7,
        )

        ctx = EternalContext(storage, "test")
        status = await ctx.get_status()
        assert status["memory_counts"]["fact"] == 2
        assert status["memory_counts"]["decision"] == 1
        assert status["message_count"] == 0

    @pytest.mark.asyncio
    async def test_get_status_with_session(self, storage) -> None:
        """Status includes active session info."""
        await _encode_typed_memory(
            storage,
            "Session: working on Auth",
            MemoryType.CONTEXT,
            tags={"session_state"},
            metadata={"active": True, "feature": "Auth", "task": "Login"},
        )
        ctx = EternalContext(storage, "test")
        status = await ctx.get_status()
        assert status["session"]["feature"] == "Auth"
        assert status["session"]["task"] == "Login"

    @pytest.mark.asyncio
    async def test_estimate_context_usage_empty(self, storage) -> None:
        """Empty context has near-zero usage."""
        ctx = EternalContext(storage, "test")
        usage = await ctx.estimate_context_usage(max_tokens=128_000)
        assert 0.0 <= usage <= 0.01

    @pytest.mark.asyncio
    async def test_estimate_context_usage_zero_max(self, storage) -> None:
        """Zero max_tokens returns 0."""
        ctx = EternalContext(storage, "test")
        usage = await ctx.estimate_context_usage(max_tokens=0)
        assert usage == 0.0

    def test_increment_message_count(self) -> None:
        """Message counter increments correctly."""
        # EternalContext can be created without storage for counter-only ops
        # but we need a mock. Use None and only test counter.
        from unittest.mock import MagicMock

        ctx = EternalContext(MagicMock(), "test")
        assert ctx.increment_message_count() == 1
        assert ctx.increment_message_count() == 2
        assert ctx.message_count == 2

    @pytest.mark.asyncio
    async def test_get_memory_content_missing_fiber(self, storage) -> None:
        """Missing fiber returns None."""
        ctx = EternalContext(storage, "test")
        content = await ctx._get_memory_content("nonexistent-id")
        assert content is None
