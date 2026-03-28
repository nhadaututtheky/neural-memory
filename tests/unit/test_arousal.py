"""Tests for arousal detection — emotional intensity scoring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.arousal import ArousalStep, compute_arousal
from neural_memory.engine.pipeline import PipelineContext


class TestComputeArousal:
    """Pure function tests for compute_arousal()."""

    def test_empty_content(self) -> None:
        assert compute_arousal("") == 0.0

    def test_neutral_content(self) -> None:
        """Mundane content → zero arousal."""
        assert compute_arousal("updated the readme file") == 0.0

    def test_critical_production_outage(self) -> None:
        """Multiple high-arousal signals → high arousal."""
        score = compute_arousal("CRITICAL production outage, data loss!!")
        assert score >= 0.7

    def test_security_vulnerability(self) -> None:
        """Security breach → high arousal."""
        score = compute_arousal("security vulnerability found in auth module")
        assert score > 0.3

    def test_positive_breakthrough(self) -> None:
        """Breakthrough fix → positive arousal."""
        score = compute_arousal("finally solved the bug, breakthrough fix")
        assert score > 0.3

    def test_bug_crash_fail(self) -> None:
        """Bug + crash → moderate arousal."""
        score = compute_arousal("bug caused the server to crash")
        assert score > 0.3

    def test_mixed_signals(self) -> None:
        """Fixed but broke → moderate arousal (both positive + negative)."""
        score = compute_arousal("fixed the login bug but broke the signup flow")
        assert 0.2 < score < 0.8

    def test_single_positive(self) -> None:
        """Single positive signal → low-moderate arousal."""
        score = compute_arousal("this solution is elegant")
        assert 0.1 <= score <= 0.5

    def test_capped_at_one(self) -> None:
        """Many signals → capped at 1.0."""
        content = "CRITICAL URGENT!! bug crashed the server, data loss, security breach, broken dangerous failure"
        score = compute_arousal(content)
        assert score <= 1.0

    def test_case_insensitive(self) -> None:
        """Patterns work regardless of case."""
        assert compute_arousal("Bug found") == compute_arousal("bug found")

    def test_hotfix_rollback(self) -> None:
        """Hotfix/rollback → high-arousal amplifier."""
        score = compute_arousal("deploying hotfix for the rollback issue")
        assert score > 0.3


class TestVietnameseArousal:
    """Vietnamese language arousal detection tests (issue #119)."""

    def test_vietnamese_positive_thanh_cong(self) -> None:
        """Vietnamese 'thành công' (success) → positive arousal."""
        score = compute_arousal("Deploy thành công lên production rồi")
        assert score > 0.0

    def test_vietnamese_negative_loi(self) -> None:
        """Vietnamese 'lỗi' (error) → negative arousal."""
        score = compute_arousal("Phát hiện lỗi nghiêm trọng trong module auth")
        assert score > 0.3

    def test_vietnamese_high_arousal_su_co(self) -> None:
        """Vietnamese 'SỰ CỐ' (incident) → high arousal."""
        score = compute_arousal("SỰ CỐ NGHIÊM TRỌNG: mất dữ liệu production!!")
        assert score >= 0.5

    def test_vietnamese_breakthrough(self) -> None:
        """Vietnamese 'đột phá' (breakthrough) → positive arousal."""
        score = compute_arousal("Giải pháp đột phá cho bài toán tối ưu hiệu suất")
        assert score > 0.0

    def test_vietnamese_neutral(self) -> None:
        """Vietnamese neutral content → zero or near-zero arousal."""
        score = compute_arousal("Cập nhật file readme cho dự án")
        assert score < 0.2

    def test_mixed_vi_en_crash(self) -> None:
        """Mixed Vietnamese/English with borrowed 'crash' → arousal detected."""
        score = compute_arousal("Server bị crash liên tục, sự cố nghiêm trọng")
        assert score > 0.3


class TestLanguageAgnosticArousal:
    """Language-agnostic heuristics tests (issue #119)."""

    def test_exclamation_clusters(self) -> None:
        """Multiple exclamation marks → arousal detected."""
        score = compute_arousal("Ceci est très important!!!")
        assert score > 0.0

    def test_caps_heavy_content(self) -> None:
        """ALL CAPS heavy content → arousal detected."""
        score = compute_arousal("THIS IS ABSOLUTELY TERRIBLE AND COMPLETELY BROKEN BEYOND REPAIR")
        assert score > 0.0

    def test_punctuation_only_no_keywords(self) -> None:
        """Non-English with exclamation clusters → agnostic arousal."""
        score = compute_arousal("これは大変です!!! 問題が発生しました!!")
        assert score > 0.0


class TestArousalStep:
    """Pipeline step integration tests."""

    def _make_config(self, *, enabled: bool = True) -> MagicMock:
        config = MagicMock()
        config.arousal_enabled = enabled
        return config

    def _make_ctx(self, content: str = "test") -> PipelineContext:
        return PipelineContext(
            content=content,
            timestamp=__import__("datetime").datetime(2026, 3, 26, 12, 0, 0),
            metadata={},
            tags=set(),
            language="en",
        )

    @pytest.mark.asyncio
    async def test_adds_arousal_to_metadata(self) -> None:
        """ArousalStep stores _arousal in effective_metadata."""
        step = ArousalStep()
        ctx = self._make_ctx("CRITICAL production outage!!")
        storage = AsyncMock()
        config = self._make_config()

        result = await step.execute(ctx, storage, config)
        assert "_arousal" in result.effective_metadata
        assert result.effective_metadata["_arousal"] > 0.5

    @pytest.mark.asyncio
    async def test_no_arousal_for_neutral(self) -> None:
        """Neutral content → no _arousal key added."""
        step = ArousalStep()
        ctx = self._make_ctx("updated the readme")
        storage = AsyncMock()
        config = self._make_config()

        result = await step.execute(ctx, storage, config)
        assert "_arousal" not in result.effective_metadata

    @pytest.mark.asyncio
    async def test_disabled_via_config(self) -> None:
        """When disabled, no metadata added even for high-arousal content."""
        step = ArousalStep()
        ctx = self._make_ctx("CRITICAL data loss!!")
        storage = AsyncMock()
        config = self._make_config(enabled=False)

        result = await step.execute(ctx, storage, config)
        assert "_arousal" not in result.effective_metadata

    @pytest.mark.asyncio
    async def test_step_name(self) -> None:
        """Step name is 'arousal'."""
        assert ArousalStep().name == "arousal"

    @pytest.mark.asyncio
    async def test_arousal_value_rounded(self) -> None:
        """Arousal value is rounded to 3 decimal places."""
        step = ArousalStep()
        ctx = self._make_ctx("bug crashed the server, data loss")
        storage = AsyncMock()
        config = self._make_config()

        result = await step.execute(ctx, storage, config)
        arousal = result.effective_metadata.get("_arousal", 0.0)
        assert arousal == round(arousal, 3)
