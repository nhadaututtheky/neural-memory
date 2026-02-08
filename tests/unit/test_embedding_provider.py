"""Tests for embedding provider protocol and config."""

from __future__ import annotations

import math

import pytest

from neural_memory.engine.embedding.config import EmbeddingConfig
from neural_memory.engine.embedding.provider import EmbeddingProvider


# ── Mock provider for testing ────────────────────────────────────


class MockEmbeddingProvider(EmbeddingProvider):
    """Simple mock embedding provider for unit tests."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        """Simple deterministic embedding based on text hash."""
        h = hash(text) % 1000
        vec = [(h + i) / 1000.0 for i in range(self._dim)]
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @property
    def dimension(self) -> int:
        return self._dim


# ── Config tests ─────────────────────────────────────────────────


class TestEmbeddingConfig:
    """Test EmbeddingConfig defaults and immutability."""

    def test_defaults(self) -> None:
        """Config should have sensible defaults."""
        config = EmbeddingConfig()
        assert config.enabled is False
        assert config.provider == "sentence_transformer"
        assert config.model == "all-MiniLM-L6-v2"
        assert config.similarity_threshold == 0.7
        assert config.activation_boost == 0.15

    def test_frozen(self) -> None:
        """Config should be immutable."""
        config = EmbeddingConfig()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore[misc]

    def test_custom_config(self) -> None:
        """Should support custom configuration."""
        config = EmbeddingConfig(
            enabled=True,
            provider="openai",
            model="text-embedding-3-small",
            similarity_threshold=0.8,
            activation_boost=0.2,
        )
        assert config.enabled is True
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"


# ── Provider protocol tests ──────────────────────────────────────


class TestEmbeddingProvider:
    """Test EmbeddingProvider ABC and default implementations."""

    @pytest.mark.asyncio
    async def test_embed_returns_list(self) -> None:
        """embed() should return a list of floats."""
        provider = MockEmbeddingProvider(dim=4)
        result = await provider.embed("test text")
        assert isinstance(result, list)
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_dimension_property(self) -> None:
        """dimension should return the correct dimensionality."""
        provider = MockEmbeddingProvider(dim=8)
        assert provider.dimension == 8

    @pytest.mark.asyncio
    async def test_embed_batch_default(self) -> None:
        """Default embed_batch should call embed sequentially."""
        provider = MockEmbeddingProvider(dim=4)
        texts = ["hello", "world", "test"]
        results = await provider.embed_batch(texts)
        assert len(results) == 3
        for vec in results:
            assert len(vec) == 4

    @pytest.mark.asyncio
    async def test_embed_deterministic(self) -> None:
        """Same text should produce same embedding."""
        provider = MockEmbeddingProvider(dim=4)
        v1 = await provider.embed("same text")
        v2 = await provider.embed("same text")
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_different_texts_differ(self) -> None:
        """Different texts should produce different embeddings."""
        provider = MockEmbeddingProvider(dim=4)
        v1 = await provider.embed("text a")
        v2 = await provider.embed("text b")
        assert v1 != v2


# ── Cosine similarity tests ─────────────────────────────────────


class TestCosineSimilarity:
    """Test default cosine similarity implementation."""

    @pytest.mark.asyncio
    async def test_identical_vectors(self) -> None:
        """Identical vectors should have similarity 1.0."""
        provider = MockEmbeddingProvider()
        vec = [1.0, 2.0, 3.0, 4.0]
        sim = await provider.similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        provider = MockEmbeddingProvider()
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0]
        sim = await provider.similarity(v1, v2)
        assert sim == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_opposite_vectors(self) -> None:
        """Opposite vectors should have similarity -1.0."""
        provider = MockEmbeddingProvider()
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0, 0.0]
        sim = await provider.similarity(v1, v2)
        assert sim == pytest.approx(-1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_zero_vector(self) -> None:
        """Zero vector should return similarity 0.0."""
        provider = MockEmbeddingProvider()
        v1 = [1.0, 2.0, 3.0, 4.0]
        v2 = [0.0, 0.0, 0.0, 0.0]
        sim = await provider.similarity(v1, v2)
        assert sim == 0.0

    @pytest.mark.asyncio
    async def test_similarity_range(self) -> None:
        """Cosine similarity should be in [-1, 1]."""
        provider = MockEmbeddingProvider(dim=4)
        v1 = await provider.embed("first text")
        v2 = await provider.embed("second text")
        sim = await provider.similarity(v1, v2)
        assert -1.0 <= sim <= 1.0

    @pytest.mark.asyncio
    async def test_self_similarity(self) -> None:
        """Embedding of same text should have similarity 1.0 with itself."""
        provider = MockEmbeddingProvider(dim=4)
        vec = await provider.embed("test text")
        sim = await provider.similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=1e-6)


# ── BrainConfig embedding fields ────────────────────────────────


class TestBrainConfigEmbeddingFields:
    """Test that BrainConfig has embedding configuration fields."""

    def test_defaults(self) -> None:
        """BrainConfig should have embedding fields with defaults."""
        from neural_memory.core.brain import BrainConfig

        config = BrainConfig()
        assert config.embedding_enabled is False
        assert config.embedding_provider == "sentence_transformer"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_similarity_threshold == 0.7
        assert config.embedding_activation_boost == 0.15

    def test_with_updates(self) -> None:
        """with_updates should propagate embedding fields."""
        from neural_memory.core.brain import BrainConfig

        config = BrainConfig()
        updated = config.with_updates(embedding_enabled=True)
        assert updated.embedding_enabled is True
        assert updated.embedding_provider == "sentence_transformer"
