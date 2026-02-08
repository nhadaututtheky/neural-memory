"""OpenAI embedding provider with lazy import."""

from __future__ import annotations

import os
from typing import Any

from neural_memory.engine.embedding.provider import EmbeddingProvider

# Known dimensions per model
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbedding(EmbeddingProvider):
    """Embedding provider backed by the OpenAI Embeddings API.

    The ``openai`` package is imported lazily on first use so that the
    dependency is only required when this provider is actually selected.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "An OpenAI API key is required. Pass it directly or set "
                "the OPENAI_API_KEY environment variable."
            )
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        """Lazy-initialise the async OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai is required for OpenAIEmbedding. Install it with: pip install openai"
                ) from exc

            self._client = AsyncOpenAI(api_key=self._api_key)

        return self._client

    async def embed(self, text: str) -> list[float]:
        """Embed a single text via the OpenAI API."""
        client = self._ensure_client()
        response = await client.embeddings.create(
            input=[text],
            model=self._model,
        )
        return list(response.data[0].embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        The OpenAI API natively supports batch input, making this more
        efficient than the default sequential fallback.
        """
        if not texts:
            return []

        client = self._ensure_client()
        response = await client.embeddings.create(
            input=texts,
            model=self._model,
        )
        # The API returns embeddings in the same order as the input
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [list(item.embedding) for item in sorted_data]

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors for the configured model."""
        return _MODEL_DIMENSIONS.get(self._model, 1536)
