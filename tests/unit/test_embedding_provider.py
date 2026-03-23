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

    def test_gemini_provider_valid(self) -> None:
        """Should accept 'gemini' as a valid provider."""
        config = EmbeddingConfig(provider="gemini", model="gemini-embedding-001")
        assert config.provider == "gemini"

    def test_openrouter_provider_valid(self) -> None:
        """Should accept 'openrouter' as a valid provider."""
        config = EmbeddingConfig(
            provider="openrouter",
            model="openai/text-embedding-3-small",
        )
        assert config.provider == "openrouter"


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

        from neural_memory.engine.embedding.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        provider = SentenceTransformerEmbedding(model_name="test-model")

        # Create a mock model that returns an object with .tolist()
        mock_result = unittest.mock.MagicMock()
        mock_result.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = mock_result
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

        from neural_memory.engine.embedding.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        provider = SentenceTransformerEmbedding()

        # Mock model returning iterable of objects with .tolist() (like numpy rows)
        row1 = unittest.mock.MagicMock()
        row1.tolist.return_value = [0.1, 0.2]
        row2 = unittest.mock.MagicMock()
        row2.tolist.return_value = [0.3, 0.4]
        mock_model = unittest.mock.MagicMock()
        mock_model.encode.return_value = [row1, row2]
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


# ── GeminiEmbedding tests ────────────────────────────────────────


class TestGeminiEmbedding:
    """Test GeminiEmbedding with mocked google-genai."""

    def test_requires_api_key(self) -> None:
        """Should raise ValueError when no API key provided."""
        import os
        import unittest.mock

        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        # Clear environment so GEMINI_API_KEY and GOOGLE_API_KEY are not set
        env_without_key = {
            k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        }
        with unittest.mock.patch.dict(os.environ, env_without_key, clear=True):
            with pytest.raises(ValueError, match="API key"):
                GeminiEmbedding()

    def test_accepts_explicit_api_key(self) -> None:
        """Should accept an explicit API key parameter."""
        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(api_key="test-key-123")
        assert provider._api_key == "test-key-123"

    def test_accepts_gemini_env_key(self) -> None:
        """Should pick up GEMINI_API_KEY from environment."""
        import os
        import unittest.mock

        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        env["GEMINI_API_KEY"] = "env-gemini-key"
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            provider = GeminiEmbedding()
            assert provider._api_key == "env-gemini-key"

    def test_accepts_google_env_key(self) -> None:
        """Should fall back to GOOGLE_API_KEY from environment."""
        import os
        import unittest.mock

        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        env["GOOGLE_API_KEY"] = "env-google-key"
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            provider = GeminiEmbedding()
            assert provider._api_key == "env-google-key"

    def test_dimension_default_model(self) -> None:
        """Should return 3072 for default model gemini-embedding-001."""
        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(api_key="test-key")
        assert provider.dimension == 3072

    def test_dimension_text_embedding_004(self) -> None:
        """Should return 768 for text-embedding-004."""
        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(api_key="test-key", model="text-embedding-004")
        assert provider.dimension == 768

    def test_dimension_unknown_model_defaults_to_3072(self) -> None:
        """Should fallback to 3072 for unknown models."""
        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(api_key="test-key", model="some-future-model")
        assert provider.dimension == 3072

    def test_task_type_default(self) -> None:
        """Should default to RETRIEVAL_QUERY task type."""
        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(api_key="test-key")
        assert provider._task_type == "RETRIEVAL_QUERY"

    def test_task_type_custom(self) -> None:
        """Should accept custom task type."""
        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(api_key="test-key", task_type="RETRIEVAL_DOCUMENT")
        assert provider._task_type == "RETRIEVAL_DOCUMENT"

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self) -> None:
        """embed_batch with empty list should return empty list."""
        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(api_key="test-key")
        result = await provider.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_with_mock_client(self) -> None:
        """Should call the Gemini client and return embedding."""
        import unittest.mock

        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(api_key="test-key")

        # Mock embedding result
        mock_embedding = unittest.mock.MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]

        mock_response = unittest.mock.MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_aio_models = unittest.mock.AsyncMock()
        mock_aio_models.embed_content = unittest.mock.AsyncMock(return_value=mock_response)

        mock_aio = unittest.mock.MagicMock()
        mock_aio.models = mock_aio_models

        mock_client = unittest.mock.MagicMock()
        mock_client.aio = mock_aio

        provider._client = mock_client

        result = await provider.embed("hello")
        assert result == [0.1, 0.2, 0.3]
        mock_aio_models.embed_content.assert_called_once_with(
            model="gemini-embedding-001",
            contents="hello",
            config={"task_type": "RETRIEVAL_QUERY"},
        )

    @pytest.mark.asyncio
    async def test_embed_batch_with_mock_client(self) -> None:
        """Should call the Gemini client with batch input."""
        import unittest.mock

        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(api_key="test-key")

        mock_emb_0 = unittest.mock.MagicMock()
        mock_emb_0.values = [0.1, 0.2]

        mock_emb_1 = unittest.mock.MagicMock()
        mock_emb_1.values = [0.3, 0.4]

        mock_response = unittest.mock.MagicMock()
        mock_response.embeddings = [mock_emb_0, mock_emb_1]

        mock_aio_models = unittest.mock.AsyncMock()
        mock_aio_models.embed_content = unittest.mock.AsyncMock(return_value=mock_response)

        mock_aio = unittest.mock.MagicMock()
        mock_aio.models = mock_aio_models

        mock_client = unittest.mock.MagicMock()
        mock_client.aio = mock_aio

        provider._client = mock_client

        result = await provider.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_lazy_import_error(self) -> None:
        """Should raise ImportError with helpful message when google-genai not installed."""
        import unittest.mock

        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(api_key="test-key")
        provider._client = None  # Force re-import

        with unittest.mock.patch.dict("sys.modules", {"google": None, "google.genai": None}):
            with pytest.raises(ImportError, match="google-genai"):
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


# ── EmbeddingSettings validation tests ───────────────────────────


class TestEmbeddingSettings:
    """Test EmbeddingSettings from unified_config."""

    def test_defaults(self) -> None:
        """EmbeddingSettings should have sensible defaults."""
        from neural_memory.unified_config import EmbeddingSettings

        settings = EmbeddingSettings()
        assert settings.enabled is False
        assert settings.provider == "sentence_transformer"
        assert settings.model == "all-MiniLM-L6-v2"
        assert settings.similarity_threshold == 0.7

    def test_frozen(self) -> None:
        """EmbeddingSettings should be immutable."""
        from dataclasses import FrozenInstanceError

        from neural_memory.unified_config import EmbeddingSettings

        settings = EmbeddingSettings()
        with pytest.raises(FrozenInstanceError):
            settings.enabled = True  # type: ignore[misc]

    def test_valid_providers(self) -> None:
        """Should accept all valid provider names."""
        from neural_memory.unified_config import EmbeddingSettings

        for provider in ("sentence_transformer", "openai", "openrouter", "gemini", "ollama", ""):
            s = EmbeddingSettings(provider=provider)
            assert s.provider == provider

    def test_invalid_provider_falls_back(self) -> None:
        """Should fall back to disabled on invalid provider names."""
        from neural_memory.unified_config import EmbeddingSettings

        s = EmbeddingSettings(provider="gremlin")
        assert s.provider == ""
        assert s.enabled is False

    def test_from_dict_round_trip(self) -> None:
        """from_dict/to_dict should round-trip correctly."""
        from neural_memory.unified_config import EmbeddingSettings

        original = EmbeddingSettings(
            enabled=True,
            provider="gemini",
            model="gemini-embedding-001",
            similarity_threshold=0.8,
        )
        data = original.to_dict()
        restored = EmbeddingSettings.from_dict(data)
        assert restored.enabled == original.enabled
        assert restored.provider == original.provider
        assert restored.model == original.model
        assert restored.similarity_threshold == original.similarity_threshold

    def test_from_dict_invalid_provider_falls_back(self) -> None:
        """from_dict with invalid provider should fall back to disabled."""
        from neural_memory.unified_config import EmbeddingSettings

        s = EmbeddingSettings.from_dict({"provider": "bad_provider"})
        assert s.provider == ""
        assert s.enabled is False


# ── OllamaEmbedding tests ────────────────────────────────────────


class TestOllamaEmbedding:
    """Test OllamaEmbedding with mocked httpx."""

    def test_default_model(self) -> None:
        """Should default to bge-m3 model."""
        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding()
        assert provider._model == "bge-m3"

    def test_custom_model_and_url(self) -> None:
        """Should accept custom model and base URL."""
        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding(model="nomic-embed-text", base_url="http://gpu-server:11434")
        assert provider._model == "nomic-embed-text"
        assert provider._base_url == "http://gpu-server:11434"

    def test_strips_trailing_slash(self) -> None:
        """Should strip trailing slash from base URL."""
        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding(base_url="http://localhost:11434/")
        assert provider._base_url == "http://localhost:11434"

    def test_dimension_known_models(self) -> None:
        """Should return correct dimensions for known models."""
        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        assert OllamaEmbedding(model="bge-m3").dimension == 1024
        assert OllamaEmbedding(model="nomic-embed-text").dimension == 768
        assert OllamaEmbedding(model="mxbai-embed-large").dimension == 1024
        assert OllamaEmbedding(model="all-minilm").dimension == 384

    def test_dimension_unknown_model_defaults_to_1024(self) -> None:
        """Should fallback to 1024 for unknown models."""
        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding(model="some-future-model")
        assert provider.dimension == 1024

    def test_dimension_uses_cached_after_embed(self) -> None:
        """Should prefer cached dimension from actual API response."""
        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding(model="bge-m3")
        provider._cached_dimension = 512  # Simulate cache from embed()
        assert provider.dimension == 512

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self) -> None:
        """embed_batch with empty list should return empty list."""
        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding()
        result = await provider.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_with_mock_client(self) -> None:
        """Should call Ollama /api/embed and return embedding."""
        import unittest.mock

        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding()

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        mock_client = unittest.mock.AsyncMock()
        mock_client.post = unittest.mock.AsyncMock(return_value=mock_response)

        provider._client = mock_client

        result = await provider.embed("hello")
        assert result == [0.1, 0.2, 0.3]
        mock_client.post.assert_called_once_with(
            "/api/embed",
            json={"model": "bge-m3", "input": "hello"},
        )

    @pytest.mark.asyncio
    async def test_embed_caches_dimension(self) -> None:
        """Should cache dimension from first API response."""
        import unittest.mock

        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding()
        assert provider._cached_dimension is None

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1] * 1024]}

        mock_client = unittest.mock.AsyncMock()
        mock_client.post = unittest.mock.AsyncMock(return_value=mock_response)
        provider._client = mock_client

        await provider.embed("test")
        assert provider._cached_dimension == 1024

    @pytest.mark.asyncio
    async def test_embed_batch_with_mock_client(self) -> None:
        """Should call Ollama /api/embed with batch input."""
        import unittest.mock

        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding()

        mock_response = unittest.mock.MagicMock()
        mock_response.raise_for_status = unittest.mock.MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }

        mock_client = unittest.mock.AsyncMock()
        mock_client.post = unittest.mock.AsyncMock(return_value=mock_response)
        provider._client = mock_client

        result = await provider.embed_batch(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_embed_batch_chunks_large_input(self) -> None:
        """Should chunk large batches into _BATCH_SIZE per request."""
        import unittest.mock

        from neural_memory.engine.embedding.ollama_embedding import (
            _BATCH_SIZE,
            OllamaEmbedding,
        )

        provider = OllamaEmbedding()

        # Create texts larger than batch size
        texts = [f"text_{i}" for i in range(_BATCH_SIZE + 5)]

        def make_response(embeddings: list[list[float]]) -> unittest.mock.MagicMock:
            resp = unittest.mock.MagicMock()
            resp.raise_for_status = unittest.mock.MagicMock()
            resp.json.return_value = {"embeddings": embeddings}
            return resp

        # First call returns _BATCH_SIZE embeddings, second returns 5
        mock_client = unittest.mock.AsyncMock()
        mock_client.post = unittest.mock.AsyncMock(
            side_effect=[
                make_response([[0.1]] * _BATCH_SIZE),
                make_response([[0.2]] * 5),
            ]
        )
        provider._client = mock_client

        result = await provider.embed_batch(texts)
        assert len(result) == _BATCH_SIZE + 5
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_lazy_import_error(self) -> None:
        """Should raise ImportError with helpful message when httpx not installed."""
        import unittest.mock

        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding()
        provider._client = None  # Force re-import

        with unittest.mock.patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx"):
                await provider.embed("test")

    def test_ollama_in_valid_providers(self) -> None:
        """'ollama' should be a valid embedding provider."""
        config = EmbeddingConfig(provider="ollama", model="bge-m3")
        assert config.provider == "ollama"

    def test_factory_creates_ollama_provider(self) -> None:
        """_create_provider should create OllamaEmbedding for 'ollama' provider."""
        from neural_memory.core.brain import BrainConfig
        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding
        from neural_memory.engine.semantic_discovery import _create_provider

        config = BrainConfig(embedding_provider="ollama", embedding_model="bge-m3")
        provider = _create_provider(config)
        assert isinstance(provider, OllamaEmbedding)
        assert provider._model == "bge-m3"


# ── OpenRouterEmbedding tests ────────────────────────────────────


class TestOpenRouterEmbedding:
    """Test OpenRouterEmbedding with mocked OpenAI client."""

    def test_requires_api_key(self) -> None:
        """Should raise ValueError when no API key is provided."""
        import os
        import unittest.mock

        from neural_memory.engine.embedding.openrouter_embedding import OpenRouterEmbedding

        env_without_key = {
            k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"
        }
        with unittest.mock.patch.dict(os.environ, env_without_key, clear=True):
            with pytest.raises(ValueError, match="API key"):
                OpenRouterEmbedding()

    def test_accepts_openrouter_env_key(self) -> None:
        """Should pick up OPENROUTER_API_KEY from environment."""
        import os
        import unittest.mock

        from neural_memory.engine.embedding.openrouter_embedding import OpenRouterEmbedding

        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        env["OPENROUTER_API_KEY"] = "env-openrouter-key"
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            provider = OpenRouterEmbedding()
            assert provider._api_key == "env-openrouter-key"

    def test_defaults(self) -> None:
        """Should use the documented OpenRouter defaults."""
        from neural_memory.engine.embedding.openrouter_embedding import OpenRouterEmbedding

        provider = OpenRouterEmbedding(api_key="test-key")
        assert provider._model == "openai/text-embedding-3-small"
        assert provider._base_url == "https://openrouter.ai/api/v1"

    def test_dimension_known_model(self) -> None:
        """Should return 1536 for OpenRouter's OpenAI small embedding model."""
        from neural_memory.engine.embedding.openrouter_embedding import OpenRouterEmbedding

        provider = OpenRouterEmbedding(
            api_key="test-key",
            model="openai/text-embedding-3-small",
        )
        assert provider.dimension == 1536

    @pytest.mark.asyncio
    async def test_embed_with_mock_client(self) -> None:
        """Should call the OpenRouter client and return embedding."""
        import unittest.mock

        from neural_memory.engine.embedding.openrouter_embedding import OpenRouterEmbedding

        provider = OpenRouterEmbedding(api_key="test-key")

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
            model="openai/text-embedding-3-small",
        )

    def test_openrouter_in_valid_providers(self) -> None:
        """'openrouter' should be a valid embedding provider."""
        config = EmbeddingConfig(
            provider="openrouter",
            model="openai/text-embedding-3-small",
        )
        assert config.provider == "openrouter"

    def test_factory_creates_openrouter_provider(self) -> None:
        """_create_provider should create OpenRouterEmbedding for 'openrouter' provider."""
        import os
        import unittest.mock

        from neural_memory.core.brain import BrainConfig
        from neural_memory.engine.embedding.openrouter_embedding import OpenRouterEmbedding
        from neural_memory.engine.semantic_discovery import _create_provider, _provider_cache

        config = BrainConfig(
            embedding_provider="openrouter",
            embedding_model="openai/text-embedding-3-small",
        )
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        env["OPENROUTER_API_KEY"] = "test-openrouter-key"

        with unittest.mock.patch.dict(os.environ, env, clear=True):
            provider = _create_provider(config)
            assert isinstance(provider, OpenRouterEmbedding)
            assert provider._model == "openai/text-embedding-3-small"

        _provider_cache.clear()
