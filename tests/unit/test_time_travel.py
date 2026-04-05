"""Tests for time-travel queries — reconstruct memory state at past timestamps."""

from __future__ import annotations

from datetime import datetime, timedelta

from neural_memory.engine.memory_stages import MemoryStage
from neural_memory.engine.time_travel import reconstruct_stage


class TestReconstructStage:
    """Tests for reconstruct_stage()."""

    def test_stage_entered_before_as_of(self) -> None:
        """If stage was entered before as_of, return current stage."""
        entered = datetime(2026, 3, 1, 12, 0)
        as_of = datetime(2026, 3, 5, 12, 0)
        result = reconstruct_stage(MemoryStage.EPISODIC, entered, as_of)
        assert result == MemoryStage.EPISODIC

    def test_stage_entered_after_as_of(self) -> None:
        """If stage was entered after as_of, return previous stage."""
        entered = datetime(2026, 3, 10, 12, 0)
        as_of = datetime(2026, 3, 5, 12, 0)
        result = reconstruct_stage(MemoryStage.EPISODIC, entered, as_of)
        assert result == MemoryStage.WORKING  # Previous stage

    def test_semantic_entered_after_as_of(self) -> None:
        """SEMANTIC entered after as_of should return EPISODIC."""
        entered = datetime(2026, 3, 15, 12, 0)
        as_of = datetime(2026, 3, 5, 12, 0)
        result = reconstruct_stage(MemoryStage.SEMANTIC, entered, as_of)
        assert result == MemoryStage.EPISODIC

    def test_working_entered_after_as_of(self) -> None:
        """WORKING entered after as_of should return SHORT_TERM."""
        entered = datetime(2026, 3, 10, 12, 0)
        as_of = datetime(2026, 3, 5, 12, 0)
        result = reconstruct_stage(MemoryStage.WORKING, entered, as_of)
        assert result == MemoryStage.SHORT_TERM

    def test_stm_entered_after_as_of(self) -> None:
        """STM entered after as_of should return STM (can't go lower)."""
        entered = datetime(2026, 3, 10, 12, 0)
        as_of = datetime(2026, 3, 5, 12, 0)
        result = reconstruct_stage(MemoryStage.SHORT_TERM, entered, as_of)
        assert result == MemoryStage.SHORT_TERM

    def test_no_stage_entered_at(self) -> None:
        """Legacy data with no stage_entered_at should return current stage."""
        as_of = datetime(2026, 3, 5, 12, 0)
        result = reconstruct_stage(MemoryStage.SEMANTIC, None, as_of)
        assert result == MemoryStage.SEMANTIC

    def test_exact_boundary(self) -> None:
        """Stage entered at exactly as_of should be included (<=)."""
        entered = datetime(2026, 3, 5, 12, 0)
        as_of = datetime(2026, 3, 5, 12, 0)
        result = reconstruct_stage(MemoryStage.EPISODIC, entered, as_of)
        assert result == MemoryStage.EPISODIC

    def test_stage_entered_one_second_after(self) -> None:
        """Stage entered 1 second after as_of should walk back."""
        entered = datetime(2026, 3, 5, 12, 0, 1)
        as_of = datetime(2026, 3, 5, 12, 0, 0)
        result = reconstruct_stage(MemoryStage.EPISODIC, entered, as_of)
        assert result == MemoryStage.WORKING
