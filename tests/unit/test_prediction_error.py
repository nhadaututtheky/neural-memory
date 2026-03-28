"""Tests for prediction error encoding — Phase 2 Neuro Engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.prediction_error import (
    PredictionErrorStep,
    _detects_reversal,
    compute_surprise_bonus,
)
from neural_memory.utils.simhash import simhash


class TestDetectsReversal:
    """Test reversal detection between two content strings."""

    def test_negation_flip(self) -> None:
        assert _detects_reversal("API is working well", "API is not working well")

    def test_negation_doesnt(self) -> None:
        assert _detects_reversal("database works fine", "database doesn't work fine")

    def test_opposite_adjectives_fast_slow(self) -> None:
        assert _detects_reversal("PostgreSQL is fast for OLTP", "PostgreSQL is slow for analytics")

    def test_opposite_adjectives_good_bad(self) -> None:
        assert _detects_reversal("the new API is good", "the new API is bad")

    def test_no_reversal_different_topics(self) -> None:
        assert not _detects_reversal("Python is fast", "JavaScript has async")

    def test_no_reversal_same_sentiment(self) -> None:
        assert not _detects_reversal("API works great", "API performs well")

    def test_opposite_stable_unstable(self) -> None:
        assert _detects_reversal("the service is stable", "the service is unstable")


class TestVietnameseReversal:
    """Vietnamese reversal detection tests (issue #119)."""

    def test_vietnamese_negation_khong(self) -> None:
        """Vietnamese 'không' negation flip → reversal detected."""
        assert _detects_reversal(
            "Service hoạt động tốt",
            "Service không hoạt động tốt",
        )

    def test_vietnamese_opposite_nhanh_cham(self) -> None:
        """Vietnamese 'nhanh/chậm' (fast/slow) → reversal detected."""
        assert _detects_reversal(
            "Query chạy nhanh trên PostgreSQL",
            "Query chạy chậm trên PostgreSQL",
        )

    def test_vietnamese_opposite_tot_xau(self) -> None:
        """Vietnamese 'tốt/xấu' (good/bad) → reversal detected."""
        assert _detects_reversal(
            "Kết quả tốt sau khi deploy",
            "Kết quả xấu sau khi deploy",
        )

    def test_vietnamese_thanh_cong_that_bai(self) -> None:
        """Vietnamese 'thành công/thất bại' (success/failure) → reversal."""
        assert _detects_reversal(
            "Deploy thành công lên staging",
            "Deploy thất bại trên staging",
        )

    def test_vietnamese_no_reversal(self) -> None:
        """Different Vietnamese topics → no reversal."""
        assert not _detects_reversal(
            "Database hoạt động ổn định",
            "Frontend cần thêm responsive design",
        )

    def test_mixed_vi_en_reversal(self) -> None:
        """Mixed Vietnamese/English with English 'works/broken' → reversal."""
        assert _detects_reversal(
            "API works tốt trên staging environment",
            "API broken trên staging environment",
        )

    def test_vietnamese_negation_chua(self) -> None:
        """Vietnamese 'chưa' (not yet) negation → reversal detected."""
        assert _detects_reversal(
            "Migration đã hoàn thành rồi",
            "Migration chưa hoàn thành rồi",
        )


class TestComputeSurpriseBonus:
    """Test surprise bonus computation."""

    @pytest.fixture
    def mock_storage(self) -> AsyncMock:
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])
        return storage

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.prediction_error_enabled = True
        return config

    @pytest.mark.asyncio
    async def test_novel_topic_no_existing(
        self, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Completely novel topic → surprise 1.5."""
        result = await compute_surprise_bonus(
            content="quantum computing breakthroughs",
            tags={"quantum", "computing"},
            content_hash=simhash("quantum computing breakthroughs"),
            storage=mock_storage,
            config=mock_config,
        )
        assert result == 1.5

    @pytest.mark.asyncio
    async def test_no_tags_moderate(self, mock_storage: AsyncMock, mock_config: MagicMock) -> None:
        """No tags → moderate novelty (1.0)."""
        result = await compute_surprise_bonus(
            content="some content",
            tags=set(),
            content_hash=simhash("some content"),
            storage=mock_storage,
            config=mock_config,
        )
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_redundant_content(self, mock_storage: AsyncMock, mock_config: MagicMock) -> None:
        """Near-duplicate content → surprise ~0."""
        content = "PostgreSQL is fast for OLTP workloads"
        h = simhash(content)

        existing = MagicMock()
        existing.id = "existing-1"
        existing.content = content
        existing.content_hash = h

        mock_storage.find_neurons = AsyncMock(return_value=[existing])

        result = await compute_surprise_bonus(
            content=content,
            tags={"postgresql", "oltp"},
            content_hash=h,
            storage=mock_storage,
            config=mock_config,
        )
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_contradiction_high_surprise(
        self, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Contradictory content → surprise 2.5."""
        existing = MagicMock()
        existing.id = "existing-1"
        existing.content = "PostgreSQL is fast for analytics"
        existing.content_hash = simhash("PostgreSQL is fast for analytics")

        mock_storage.find_neurons = AsyncMock(return_value=[existing])

        result = await compute_surprise_bonus(
            content="PostgreSQL is slow for analytics",
            tags={"postgresql", "analytics"},
            content_hash=simhash("PostgreSQL is slow for analytics"),
            storage=mock_storage,
            config=mock_config,
        )
        assert result == 2.5

    @pytest.mark.asyncio
    async def test_moderate_novelty(self, mock_storage: AsyncMock, mock_config: MagicMock) -> None:
        """Different but related content → moderate surprise."""
        existing = MagicMock()
        existing.id = "existing-1"
        existing.content = "Python is great for data science and machine learning"
        existing.content_hash = simhash("Python is great for data science and machine learning")

        mock_storage.find_neurons = AsyncMock(return_value=[existing])

        result = await compute_surprise_bonus(
            content="Rust is great for systems programming and performance",
            tags={"rust", "programming"},
            content_hash=simhash("Rust is great for systems programming and performance"),
            storage=mock_storage,
            config=mock_config,
        )
        assert result > 0.0  # Some novelty detected


class TestPredictionErrorStep:
    """Test the pipeline step integration."""

    @pytest.fixture
    def mock_ctx(self) -> MagicMock:
        ctx = MagicMock()
        ctx.content = "test content about databases"
        ctx.merged_tags = {"database", "test"}
        ctx.auto_tags = {"database", "test"}
        ctx.content_hash = simhash("test content about databases")
        ctx.effective_metadata = {}
        return ctx

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        config = MagicMock()
        config.prediction_error_enabled = True
        return config

    @pytest.fixture
    def mock_storage(self) -> AsyncMock:
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])
        return storage

    def test_step_name(self) -> None:
        assert PredictionErrorStep().name == "prediction_error"

    @pytest.mark.asyncio
    async def test_disabled_via_config(self, mock_ctx: MagicMock, mock_storage: AsyncMock) -> None:
        config = MagicMock()
        config.prediction_error_enabled = False
        step = PredictionErrorStep()
        result = await step.execute(mock_ctx, mock_storage, config)
        assert result is mock_ctx
        mock_storage.find_neurons.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_dedup_reused(
        self, mock_ctx: MagicMock, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Skip prediction error if dedup already reused an anchor."""
        mock_ctx.effective_metadata["_dedup_reused_anchor"] = MagicMock()
        step = PredictionErrorStep()
        await step.execute(mock_ctx, mock_storage, mock_config)
        mock_storage.find_neurons.assert_not_called()

    @pytest.mark.asyncio
    async def test_adds_surprise_bonus(
        self, mock_ctx: MagicMock, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Novel topic → surprise bonus added to effective_metadata."""
        step = PredictionErrorStep()
        result = await step.execute(mock_ctx, mock_storage, mock_config)
        assert "_surprise_bonus" in result.effective_metadata
        assert result.effective_metadata["_surprise_bonus"] > 0

    @pytest.mark.asyncio
    async def test_priority_clamped(
        self, mock_ctx: MagicMock, mock_storage: AsyncMock, mock_config: MagicMock
    ) -> None:
        """Priority should be clamped to [1, 10]."""
        mock_ctx.effective_metadata["auto_priority"] = 9

        # Force high surprise via contradiction
        existing = MagicMock()
        existing.id = "e1"
        existing.content = "service is stable and reliable"
        existing.content_hash = simhash("service is stable and reliable")
        mock_storage.find_neurons = AsyncMock(return_value=[existing])

        mock_ctx.content = "service is unstable and unreliable"
        mock_ctx.content_hash = simhash("service is unstable and unreliable")
        mock_ctx.merged_tags = {"service"}

        step = PredictionErrorStep()
        result = await step.execute(mock_ctx, mock_storage, mock_config)
        assert result.effective_metadata["auto_priority"] <= 10
