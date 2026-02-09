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


# ── SentenceTransformerEmbedding tests ──────────────────────────


class TestSentenceTransformerEmbedding:
    """Test SentenceTransformerEmbedding with mocked sentence_transformers."""

    @pytest.mark.asyncio
    async def test_lazy_import_error(self) -> None:
        """Should raise ImportError with helpful message when sentence-transformers not installed."""
        import unittest.mock

        from neural_memory.engine.embedding.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        provider = SentenceTransformerEmbedding()
        with unittest.mock.patch.dict("sys.modules", {"sentence_transformers": None}):
            # Reset the model so it tries to re-import
            provider._model = None
            with pytest.raises(ImportError, match="sentence-transformers"):
                await provider.embed("test")

    @pytest.mark.asyncio
    async def test_embed_with_mock(self) -> None:
        """Should call model.encode and return result."""
        import unittest.mock

        import numpy as np

        from neural_memory.engine.embedding.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        provider = SentenceTransformerEmbedding(model_name="test-model")

        # Create a mock model that returns a numpy array
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_model.get_sentence_embedding_dimension.return_value = 3

        # Inject the mock model directly
        provider._model = mock_model
        provider._dimension = 3

        result = await provider.embed("hello world")
        assert result == pytest.approx([0.1, 0.2, 0.3])
        mock_model.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch_with_mock(self) -> None:
        """Should call model.encode with a list and return list of lists."""
        import unittest.mock

        import numpy as np

        from neural_memory.engine.embedding.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        provider = SentenceTransformerEmbedding()

        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        provider._model = mock_model
        provider._dimension = 2

        result = await provider.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.3, 0.4])

    def test_default_dimension(self) -> None:
        """Should default to 384 for all-MiniLM-L6-v2."""
        from neural_memory.engine.embedding.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        provider = SentenceTransformerEmbedding()
        assert provider.dimension == 384

    @pytest.mark.asyncio
    async def test_ensure_model_sets_dimension_from_model(self) -> None:
        """_ensure_model should update dimension from model introspection."""
        import sys
        import types
        import unittest.mock

        from neural_memory.engine.embedding.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        provider = SentenceTransformerEmbedding(model_name="custom-model")

        # Create a fake sentence_transformers module with SentenceTransformer class
        mock_st_module = types.ModuleType("sentence_transformers")
        mock_model_instance = unittest.mock.MagicMock()
        mock_model_instance.get_sentence_embedding_dimension.return_value = 768
        mock_st_class = unittest.mock.MagicMock(return_value=mock_model_instance)
        mock_st_module.SentenceTransformer = mock_st_class

        with unittest.mock.patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            provider._model = None
            provider._ensure_model()

        assert provider.dimension == 768
        mock_st_class.assert_called_once_with("custom-model")


# ── OpenAIEmbedding tests ───────────────────────────────────────


class TestOpenAIEmbedding:
    """Test OpenAIEmbedding with mocked openai."""

    def test_requires_api_key(self) -> None:
        """Should raise ValueError when no API key provided."""
        import os
        import unittest.mock

        from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        # Clear environment so OPENAI_API_KEY is not set
        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with unittest.mock.patch.dict(os.environ, env_without_key, clear=True):
            with pytest.raises(ValueError, match="API key"):
                OpenAIEmbedding()

    def test_accepts_explicit_api_key(self) -> None:
        """Should accept an explicit API key parameter."""
        from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        provider = OpenAIEmbedding(api_key="test-key-123")
        assert provider._api_key == "test-key-123"

    def test_dimension_default_model(self) -> None:
        """Should return 1536 for default model text-embedding-3-small."""
        from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        provider = OpenAIEmbedding(api_key="test-key")
        assert provider.dimension == 1536

    def test_dimension_large_model(self) -> None:
        """Should return 3072 for text-embedding-3-large."""
        from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        provider = OpenAIEmbedding(api_key="test-key", model="text-embedding-3-large")
        assert provider.dimension == 3072

    def test_dimension_ada_model(self) -> None:
        """Should return 1536 for text-embedding-ada-002."""
        from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        provider = OpenAIEmbedding(api_key="test-key", model="text-embedding-ada-002")
        assert provider.dimension == 1536

    def test_dimension_unknown_model_defaults_to_1536(self) -> None:
        """Should fallback to 1536 for unknown models."""
        from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        provider = OpenAIEmbedding(api_key="test-key", model="some-future-model")
        assert provider.dimension == 1536

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self) -> None:
        """embed_batch with empty list should return empty list."""
        from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        provider = OpenAIEmbedding(api_key="test-key")
        result = await provider.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_with_mock_client(self) -> None:
        """Should call the OpenAI client and return embedding."""
        import unittest.mock

        from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        provider = OpenAIEmbedding(api_key="test-key")

        # Mock the async client
        mock_embedding = unittest.mock.MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_embedding.index = 0

        mock_response = unittest.mock.MagicMock()
        mock_response.data = [mock_embedding]

        mock_client = unittest.mock.AsyncMock()
        mock_client.embeddings.create = unittest.mock.AsyncMock(return_value=mock_response)

        provider._client = mock_client

        result = await provider.embed("hello")
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            input=["hello"],
            model="text-embedding-3-small",
        )

    @pytest.mark.asyncio
    async def test_embed_batch_with_mock_client(self) -> None:
        """Should call the OpenAI client with batch input and sort by index."""
        import unittest.mock

        from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        provider = OpenAIEmbedding(api_key="test-key")

        # Mock response with out-of-order indices to test sorting
        mock_emb_0 = unittest.mock.MagicMock()
        mock_emb_0.embedding = [0.1, 0.2]
        mock_emb_0.index = 0

        mock_emb_1 = unittest.mock.MagicMock()
        mock_emb_1.embedding = [0.3, 0.4]
        mock_emb_1.index = 1

        mock_response = unittest.mock.MagicMock()
        # Return in reverse order to test sorting
        mock_response.data = [mock_emb_1, mock_emb_0]

        mock_client = unittest.mock.AsyncMock()
        mock_client.embeddings.create = unittest.mock.AsyncMock(return_value=mock_response)

        provider._client = mock_client

        result = await provider.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_lazy_import_error(self) -> None:
        """Should raise ImportError with helpful message when openai not installed."""
        import unittest.mock

        from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        provider = OpenAIEmbedding(api_key="test-key")
        provider._client = None  # Force re-import

        with unittest.mock.patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                await provider.embed("test")


# ── Embedding provider edge cases ───────────────────────────────


class TestEmbeddingProviderEdgeCases:
    """Test edge cases for EmbeddingProvider base class."""

    @pytest.mark.asyncio
    async def test_similarity_mismatched_lengths(self) -> None:
        """Should raise ValueError for vectors of different lengths."""
        provider = MockEmbeddingProvider(dim=4)
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0, 4.0]
        with pytest.raises(ValueError):
            await provider.similarity(v1, v2)

    @pytest.mark.asyncio
    async def test_similarity_both_zero_vectors(self) -> None:
        """Both zero vectors should return 0.0."""
        provider = MockEmbeddingProvider(dim=4)
        v1 = [0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 0.0, 0.0, 0.0]
        sim = await provider.similarity(v1, v2)
        assert sim == 0.0

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self) -> None:
        """embed_batch with empty list should return empty list."""
        provider = MockEmbeddingProvider(dim=4)
        result = await provider.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_single_item(self) -> None:
        """embed_batch with single item should return one vector."""
        provider = MockEmbeddingProvider(dim=4)
        result = await provider.embed_batch(["only one"])
        assert len(result) == 1
        assert len(result[0]) == 4

    @pytest.mark.asyncio
    async def test_similarity_unit_vectors(self) -> None:
        """Unit vectors at known angle should produce correct similarity."""
        provider = MockEmbeddingProvider(dim=4)
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 0.0, 0.0, 1.0]
        sim = await provider.similarity(v1, v2)
        assert sim == pytest.approx(0.0, abs=1e-6)
