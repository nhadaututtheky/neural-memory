"""Sentence-transformer embedding provider with lazy import."""

from __future__ import annotations

import asyncio
from typing import Any

from neural_memory.engine.embedding.provider import EmbeddingProvider

# Default dimension for "all-MiniLM-L6-v2"
_DEFAULT_DIMENSION = 384


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Embedding provider backed by ``sentence-transformers``.

    The heavy ``sentence_transformers`` import and model loading are
    deferred until the first call to :meth:`embed`, keeping startup
    cost at zero when the embedding layer is disabled.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._dimension: int = _DEFAULT_DIMENSION

    def _ensure_model(self) -> Any:
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for "
                    "SentenceTransformerEmbedding. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

            self._model = SentenceTransformer(self._model_name)
            embedding_dim: int | None = getattr(
                self._model, "get_sentence_embedding_dimension", lambda: None
            )()
            if embedding_dim is not None:
                self._dimension = embedding_dim

        return self._model

    async def embed(self, text: str) -> list[float]:
        """Encode *text* into a dense vector.

        The synchronous ``model.encode`` call is dispatched to a
        thread-pool executor so it does not block the event loop.
        """
        model = self._ensure_model()
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, lambda: model.encode(text, convert_to_numpy=True))
        return vector.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts using the model's native batching."""
        model = self._ensure_model()
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(
            None, lambda: model.encode(texts, convert_to_numpy=True)
        )
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dimension
