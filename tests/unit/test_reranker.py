"""Tests for cross-encoder reranker — score blending, fallback, graceful skip."""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from neural_memory.engine.reranker import (
    CrossEncoderReranker,
    RerankedResult,
    _normalize_scores,
    _sigmoid,
    rerank_activations,
    reranker_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeActivationResult:
    neuron_id: str
    activation_level: float
    hop_distance: int = 1
    path: list[str] | None = None
    source_anchor: str = ""

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = []


def _make_candidates(
    n: int, *, base_activation: float = 0.5
) -> list[tuple[str, str, float]]:
    """Build candidate tuples (id, content, activation)."""
    return [
        (f"n{i}", f"content about topic {i}", base_activation - i * 0.01)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Sigmoid / normalization
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_sigmoid_zero(self) -> None:
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self) -> None:
        assert _sigmoid(10.0) == pytest.approx(1.0, abs=0.001)

    def test_sigmoid_large_negative(self) -> None:
        assert _sigmoid(-10.0) == pytest.approx(0.0, abs=0.001)

    def test_sigmoid_monotonic(self) -> None:
        values = [-5, -2, 0, 2, 5]
        results = [_sigmoid(v) for v in values]
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]


class TestNormalize:
    def test_normalize_empty(self) -> None:
        assert _normalize_scores([]) == []

    def test_normalize_all_zero(self) -> None:
        result = _normalize_scores([0.0, 0.0])
        assert all(v == pytest.approx(0.5) for v in result)

    def test_normalize_range(self) -> None:
        result = _normalize_scores([-10, 0, 10])
        assert result[0] < 0.01
        assert result[1] == pytest.approx(0.5)
        assert result[2] > 0.99


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker:
    @patch("neural_memory.engine.reranker._check_cross_encoder", return_value=True)
    def test_rerank_empty_candidates(self, _mock: MagicMock) -> None:
        reranker = CrossEncoderReranker()
        result = reranker.rerank("test query", [], limit=5)
        assert result == []

    @patch("neural_memory.engine.reranker._check_cross_encoder", return_value=True)
    def test_rerank_scores_blended(self, _mock: MagicMock) -> None:
        """Reranked results should blend cross-encoder score with activation."""
        import numpy as np

        mock_model = MagicMock()
        # Higher score for candidate 2 (index 1) than candidate 1 (index 0)
        mock_model.predict.return_value = np.array([0.5, 3.0, -1.0])

        reranker = CrossEncoderReranker(blend_weight=0.7)
        reranker._model = mock_model

        candidates = [
            ("n0", "first topic", 0.9),
            ("n1", "second topic", 0.5),
            ("n2", "third topic", 0.3),
        ]
        results = reranker.rerank("query", candidates, limit=3)

        assert len(results) == 3
        # n1 should rank highest (high reranker score despite lower activation)
        assert results[0].neuron_id == "n1"
        # All have blended scores
        for r in results:
            assert 0.0 <= r.blended_score <= 1.0

    @patch("neural_memory.engine.reranker._check_cross_encoder", return_value=True)
    def test_rerank_limit_respected(self, _mock: MagicMock) -> None:
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2.0, 1.5, 1.0, 0.5, 0.0])

        reranker = CrossEncoderReranker()
        reranker._model = mock_model

        candidates = _make_candidates(5)
        results = reranker.rerank("query", candidates, limit=2)

        assert len(results) <= 2

    @patch("neural_memory.engine.reranker._check_cross_encoder", return_value=True)
    def test_rerank_min_score_filter(self, _mock: MagicMock) -> None:
        """Candidates with very low reranker scores should be filtered."""
        import numpy as np

        mock_model = MagicMock()
        # All very negative scores → sigmoid < min_score
        mock_model.predict.return_value = np.array([-10.0, -10.0, -10.0])

        reranker = CrossEncoderReranker(min_score=0.5)
        reranker._model = mock_model

        candidates = _make_candidates(3)
        results = reranker.rerank("query", candidates, limit=10)

        # Should fallback to top 3
        assert len(results) == 3

    @patch("neural_memory.engine.reranker._check_cross_encoder", return_value=True)
    def test_rerank_max_candidates_cap(self, _mock: MagicMock) -> None:
        """Should cap candidates at max_candidates."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict.return_value = np.ones(5)

        reranker = CrossEncoderReranker(max_candidates=5)
        reranker._model = mock_model

        candidates = _make_candidates(50)
        reranker.rerank("query", candidates, limit=10)

        # Model should only receive max_candidates pairs
        called_pairs = mock_model.predict.call_args[0][0]
        assert len(called_pairs) == 5

    @patch("neural_memory.engine.reranker._check_cross_encoder", return_value=True)
    def test_blend_weight_bounds(self, _mock: MagicMock) -> None:
        """Blend weight should be clamped to [0, 1]."""
        r1 = CrossEncoderReranker(blend_weight=-0.5)
        assert r1._blend_weight == 0.0

        r2 = CrossEncoderReranker(blend_weight=1.5)
        assert r2._blend_weight == 1.0


# ---------------------------------------------------------------------------
# reranker_available
# ---------------------------------------------------------------------------

class TestRerankerAvailable:
    @patch("neural_memory.engine.reranker._CROSS_ENCODER_AVAILABLE", None)
    @patch("neural_memory.engine.reranker._check_cross_encoder", return_value=False)
    def test_not_available(self, _mock: MagicMock) -> None:
        assert reranker_available() is False

    @patch("neural_memory.engine.reranker._CROSS_ENCODER_AVAILABLE", True)
    def test_available(self) -> None:
        assert reranker_available() is True


# ---------------------------------------------------------------------------
# rerank_activations convenience function
# ---------------------------------------------------------------------------

class TestRerankActivations:
    @patch("neural_memory.engine.reranker.reranker_available", return_value=False)
    def test_graceful_skip_when_unavailable(self, _mock: MagicMock) -> None:
        """When reranker not installed, return activations unchanged."""
        activations = {
            "n1": FakeActivationResult(neuron_id="n1", activation_level=0.8),
            "n2": FakeActivationResult(neuron_id="n2", activation_level=0.5),
        }
        result = rerank_activations("test", activations, {"n1": "content1", "n2": "content2"})
        assert result is activations  # same object, unchanged

    @patch("neural_memory.engine.reranker.reranker_available", return_value=True)
    def test_rerank_updates_activation_levels(self, _mock: MagicMock) -> None:
        """Reranked activations should have blended scores."""
        import numpy as np

        activations = {
            "n1": FakeActivationResult(neuron_id="n1", activation_level=0.8),
            "n2": FakeActivationResult(neuron_id="n2", activation_level=0.5),
        }
        contents = {"n1": "topic one", "n2": "topic two"}

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 2.0])

        with patch(
            "neural_memory.engine.reranker.CrossEncoderReranker._ensure_model",
            return_value=mock_model,
        ):
            result = rerank_activations("test query", activations, contents)

        assert len(result) <= 2
        # Activation levels should now be blended scores
        for nid, ar in result.items():
            assert 0.0 <= ar.activation_level <= 1.0

    @patch("neural_memory.engine.reranker.reranker_available", return_value=True)
    def test_empty_contents_returns_unchanged(self, _mock: MagicMock) -> None:
        """If no neuron contents, return activations unchanged."""
        activations = {
            "n1": FakeActivationResult(neuron_id="n1", activation_level=0.8),
        }
        result = rerank_activations("test", activations, {})
        assert result is activations


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestRerankerConfig:
    def test_brain_config_defaults(self) -> None:
        from neural_memory.core.brain import BrainConfig

        config = BrainConfig()
        assert config.reranker_enabled is False
        assert config.reranker_model == "BAAI/bge-reranker-v2-m3"
        assert config.reranker_overfetch_multiplier == 3
        assert config.reranker_blend_weight == 0.7
        assert config.reranker_min_score == 0.15
        assert config.reranker_max_candidates == 30

    def test_brain_config_with_updates(self) -> None:
        from neural_memory.core.brain import BrainConfig

        config = BrainConfig()
        updated = config.with_updates(reranker_enabled=True, reranker_blend_weight=0.8)
        assert updated.reranker_enabled is True
        assert updated.reranker_blend_weight == 0.8
        assert config.reranker_enabled is False  # original unchanged

    def test_unified_reranker_config_defaults(self) -> None:
        from neural_memory.unified_config import RerankerConfig

        cfg = RerankerConfig()
        assert cfg.enabled is False
        assert cfg.model_name == "BAAI/bge-reranker-v2-m3"
        assert cfg.overfetch_multiplier == 3

    def test_unified_reranker_config_roundtrip(self) -> None:
        from neural_memory.unified_config import RerankerConfig

        cfg = RerankerConfig(enabled=True, blend_weight=0.8, max_candidates=20)
        d = cfg.to_dict()
        restored = RerankerConfig.from_dict(d)
        assert restored.enabled is True
        assert restored.blend_weight == 0.8
        assert restored.max_candidates == 20

    def test_unified_reranker_config_from_empty_dict(self) -> None:
        from neural_memory.unified_config import RerankerConfig

        cfg = RerankerConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.model_name == "BAAI/bge-reranker-v2-m3"


# ---------------------------------------------------------------------------
# RerankedResult dataclass
# ---------------------------------------------------------------------------

class TestRerankedResult:
    def test_frozen(self) -> None:
        r = RerankedResult(
            neuron_id="n1",
            activation_level=0.5,
            rerank_score=1.0,
            blended_score=0.7,
        )
        with pytest.raises(AttributeError):
            r.neuron_id = "n2"  # type: ignore[misc]

    def test_fields(self) -> None:
        r = RerankedResult(
            neuron_id="n1",
            activation_level=0.5,
            rerank_score=1.0,
            blended_score=0.7,
        )
        assert r.neuron_id == "n1"
        assert r.activation_level == 0.5
        assert r.rerank_score == 1.0
        assert r.blended_score == 0.7
