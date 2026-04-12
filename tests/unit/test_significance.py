"""Tests for significance weighting engine (amygdala boost).

Covers:
- is_correction() detection (English + Vietnamese)
- SignificanceResult dataclass and metadata
- score_significance() core logic
- Priority boost/penalty scenarios
- Near-duplicate filtering
- Tag extraction
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.engine.significance import (
    SignificanceResult,
    _extract_tags,
    is_correction,
    score_significance,
)

# ── is_correction ────────────────────────────────────────────────────


class TestIsCorrection:
    def test_thats_wrong(self) -> None:
        assert is_correction("that's wrong, you should use X")

    def test_its_incorrect(self) -> None:
        assert is_correction("It's incorrect — the right approach is Y")

    def test_actually_should_be(self) -> None:
        assert is_correction("actually, it should be snake_case")

    def test_no_should_have(self) -> None:
        assert is_correction("no, it should have a type annotation")

    def test_change_it_to(self) -> None:
        assert is_correction("change it to use async/await")

    def test_fix_to(self) -> None:
        assert is_correction("fix that to return a list")

    def test_instead_of_use(self) -> None:
        assert is_correction("instead of map, use a list comprehension")

    def test_vietnamese_sai_roi(self) -> None:
        assert is_correction("sai rồi, phải làm như thế này")

    def test_vietnamese_phai_la(self) -> None:
        assert is_correction("phải là snake_case mới đúng")

    def test_vietnamese_sua_lai(self) -> None:
        assert is_correction("sửa lại thành async function")

    def test_not_correction_preference(self) -> None:
        assert not is_correction("I prefer using Python")

    def test_not_correction_fact(self) -> None:
        assert not is_correction("The solution is to use caching")

    def test_not_correction_empty(self) -> None:
        assert not is_correction("")


# ── SignificanceResult ───────────────────────────────────────────────


class TestSignificanceResult:
    def test_frozen(self) -> None:
        r = SignificanceResult(
            surprise_bonus=1.5,
            is_correction=False,
            is_contradiction=False,
            is_novel=True,
            adjusted_priority=7,
            boost_applied=1.5,
        )
        with pytest.raises(AttributeError):
            r.adjusted_priority = 10  # type: ignore[misc]

    def test_to_metadata(self) -> None:
        r = SignificanceResult(
            surprise_bonus=2.5,
            is_correction=True,
            is_contradiction=True,
            is_novel=True,
            adjusted_priority=9,
            boost_applied=2.5,
        )
        meta = r.to_metadata()
        assert meta["surprise"] == 2.5
        assert meta["correction"] is True
        assert meta["contradiction"] is True
        assert meta["novel"] is True
        assert meta["boost"] == 2.5


# ── _extract_tags ────────────────────────────────────────────────────


class TestExtractTags:
    def test_camel_case(self) -> None:
        tags = _extract_tags("Using FastAPI with Pydantic validation")
        assert "fastapi" in tags
        assert "pydantic" in tags

    def test_capitalized_words(self) -> None:
        tags = _extract_tags("Chose Redis over Memcached for caching")
        assert "redis" in tags
        assert "memcached" in tags

    def test_tech_terms(self) -> None:
        tags = _extract_tags("The PostgreSQL database with GraphQL API")
        assert "postgresql" in tags

    def test_short_words_excluded(self) -> None:
        tags = _extract_tags("If A or B then do it")
        # "A" and "B" are < 3 chars, should be excluded
        assert "a" not in tags
        assert "b" not in tags

    def test_empty_string(self) -> None:
        assert _extract_tags("") == set()


# ── score_significance ───────────────────────────────────────────────


class TestScoreSignificance:
    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        storage = MagicMock()
        storage.brain_id = "test-brain"
        storage.find_neurons = AsyncMock(return_value=[])
        return storage

    @pytest.fixture
    def mock_brain_config(self) -> MagicMock:
        return MagicMock()

    async def test_novel_topic_boost(
        self, mock_storage: MagicMock, mock_brain_config: MagicMock
    ) -> None:
        """Novel topics (no matching neurons) get novelty boost."""
        with patch(
            "neural_memory.engine.prediction_error.compute_surprise_bonus",
            new_callable=AsyncMock,
            return_value=1.5,
        ):
            result = await score_significance(
                content="Completely new topic about quantum computing",
                detected_type="fact",
                base_priority=5,
                storage=mock_storage,
                brain_config=mock_brain_config,
            )
        assert result.is_novel
        assert not result.is_contradiction
        assert result.adjusted_priority > 5
        assert result.boost_applied == 1.5  # novelty_boost default

    async def test_contradiction_boost(
        self, mock_storage: MagicMock, mock_brain_config: MagicMock
    ) -> None:
        """Contradictions get highest boost."""
        with patch(
            "neural_memory.engine.prediction_error.compute_surprise_bonus",
            new_callable=AsyncMock,
            return_value=2.5,
        ):
            result = await score_significance(
                content="FastAPI is not suitable for this use case",
                detected_type="decision",
                base_priority=6,
                storage=mock_storage,
                brain_config=mock_brain_config,
            )
        assert result.is_contradiction
        assert result.is_novel  # contradiction implies novel
        assert result.adjusted_priority >= 8
        assert result.boost_applied == 2.5  # contradiction_boost default

    async def test_correction_boost(
        self, mock_storage: MagicMock, mock_brain_config: MagicMock
    ) -> None:
        """User corrections get correction boost."""
        with patch(
            "neural_memory.engine.prediction_error.compute_surprise_bonus",
            new_callable=AsyncMock,
            return_value=1.0,
        ):
            result = await score_significance(
                content="that's wrong, it should be snake_case",
                detected_type="preference",
                base_priority=7,
                storage=mock_storage,
                brain_config=mock_brain_config,
            )
        assert result.is_correction
        assert result.adjusted_priority >= 9  # 7 + 2.0 correction_boost
        assert result.boost_applied == 2.0

    async def test_near_duplicate_penalty(
        self, mock_storage: MagicMock, mock_brain_config: MagicMock
    ) -> None:
        """Near-duplicates (surprise=0) get deprioritized."""
        with patch(
            "neural_memory.engine.prediction_error.compute_surprise_bonus",
            new_callable=AsyncMock,
            return_value=0.0,
        ):
            result = await score_significance(
                content="Already known fact about Python",
                detected_type="fact",
                base_priority=5,
                storage=mock_storage,
                brain_config=mock_brain_config,
            )
        assert not result.is_novel
        assert result.adjusted_priority <= 3  # 5 + (-2.0) = 3
        assert result.boost_applied == -2.0

    async def test_priority_capped_at_10(
        self, mock_storage: MagicMock, mock_brain_config: MagicMock
    ) -> None:
        """Priority never exceeds 10."""
        with patch(
            "neural_memory.engine.prediction_error.compute_surprise_bonus",
            new_callable=AsyncMock,
            return_value=2.5,
        ):
            result = await score_significance(
                content="that's wrong, should use Redis instead of Memcached",
                detected_type="decision",
                base_priority=9,
                storage=mock_storage,
                brain_config=mock_brain_config,
            )
        assert result.adjusted_priority == 10

    async def test_priority_minimum_1(
        self, mock_storage: MagicMock, mock_brain_config: MagicMock
    ) -> None:
        """Priority never goes below 1."""
        with patch(
            "neural_memory.engine.prediction_error.compute_surprise_bonus",
            new_callable=AsyncMock,
            return_value=0.0,
        ):
            result = await score_significance(
                content="duplicate content",
                detected_type="fact",
                base_priority=1,
                storage=mock_storage,
                brain_config=mock_brain_config,
            )
        assert result.adjusted_priority >= 1

    async def test_surprise_failure_defaults_moderate(
        self, mock_storage: MagicMock, mock_brain_config: MagicMock
    ) -> None:
        """If compute_surprise_bonus fails, defaults to moderate novelty."""
        with patch(
            "neural_memory.engine.prediction_error.compute_surprise_bonus",
            new_callable=AsyncMock,
            side_effect=RuntimeError("storage error"),
        ):
            result = await score_significance(
                content="Some content",
                detected_type="fact",
                base_priority=5,
                storage=mock_storage,
                brain_config=mock_brain_config,
            )
        # Should not raise, should default to moderate
        assert result.surprise_bonus == 1.0
        assert result.is_novel

    async def test_custom_boost_values(
        self, mock_storage: MagicMock, mock_brain_config: MagicMock
    ) -> None:
        """Custom boost values are respected."""
        with patch(
            "neural_memory.engine.prediction_error.compute_surprise_bonus",
            new_callable=AsyncMock,
            return_value=1.5,
        ):
            result = await score_significance(
                content="that's wrong, fix it to use lists",
                detected_type="preference",
                base_priority=5,
                storage=mock_storage,
                brain_config=mock_brain_config,
                correction_boost=3.0,
            )
        # Correction boost (3.0) should win over novelty (1.5)
        assert result.boost_applied == 3.0
        assert result.adjusted_priority == 8  # 5 + 3.0


# ── ProactiveConfig significance fields ──────────────────────────────


class TestProactiveConfigSignificance:
    def test_significance_defaults(self) -> None:
        from neural_memory.unified_config import ProactiveConfig

        config = ProactiveConfig()
        assert config.significance_enabled is True
        assert config.correction_boost == 2.0
        assert config.contradiction_boost == 2.5
        assert config.novelty_boost == 1.5

    def test_significance_from_dict(self) -> None:
        from neural_memory.unified_config import ProactiveConfig

        config = ProactiveConfig.from_dict(
            {
                "significance_enabled": False,
                "correction_boost": 3.0,
            }
        )
        assert config.significance_enabled is False
        assert config.correction_boost == 3.0
        assert config.novelty_boost == 1.5  # default

    def test_significance_to_dict_roundtrip(self) -> None:
        from neural_memory.unified_config import ProactiveConfig

        original = ProactiveConfig(
            significance_enabled=True,
            correction_boost=2.5,
            contradiction_boost=3.0,
            novelty_boost=2.0,
        )
        restored = ProactiveConfig.from_dict(original.to_dict())
        assert restored.correction_boost == 2.5
        assert restored.contradiction_boost == 3.0
        assert restored.novelty_boost == 2.0
