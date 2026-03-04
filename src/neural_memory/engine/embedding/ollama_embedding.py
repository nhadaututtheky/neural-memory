"""Ollama embedding provider — fully local, zero API cost."""

from __future__ import annotations

import os
from typing import Any

from neural_memory.engine.embedding.provider import EmbeddingProvider

# Known dimensions per model
_MODEL_DIMENSIONS: dict[str, int] = {
    "bge-m3": 1024,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
    "all-minilm": 384,
}

_DEFAULT_MODEL = "bge-m3"
_DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama runs locally — use small batches to avoid timeouts on large models
_BATCH_SIZE = 10


class OllamaEmbedding(EmbeddingProvider):
    """Embedding provider backed by a local Ollama server.

    Uses the ``/api/embed`` endpoint (batch-capable) introduced in Ollama 0.4+.
    The ``httpx`` package is imported lazily on first use.

    No API key required — Ollama runs locally by default.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client: Any | None = None
        # Cache dimension after first successful embed
        self._cached_dimension: int | None = None

    def _ensure_client(self) -> Any:
        """Lazy-initialise the httpx async client."""
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise ImportError(
                    "httpx is required for OllamaEmbedding. Install it with: pip install httpx"
                ) from exc

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=300.0,
            )

        return self._client

    async def embed(self, text: str) -> list[float]:
        """Embed a single text via the Ollama /api/embed endpoint."""
        client = self._ensure_client()
        response = await client.post(
            "/api/embed",
            json={"model": self._model, "input": text},
        )
        response.raise_for_status()
        data = response.json()
        embedding = data["embeddings"][0]

        # Cache dimension from first real response
        if self._cached_dimension is None:
            self._cached_dimension = len(embedding)

        return list(embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts via the Ollama /api/embed endpoint.

        Ollama supports batch input natively. Batches are capped
        at _BATCH_SIZE items per request to avoid timeouts.
        """
        if not texts:
            return []

        client = self._ensure_client()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            chunk = texts[i : i + _BATCH_SIZE]
            response = await client.post(
                "/api/embed",
                json={"model": self._model, "input": chunk},
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend(list(emb) for emb in data["embeddings"])

        # Cache dimension from first batch
        if self._cached_dimension is None and all_embeddings:
            self._cached_dimension = len(all_embeddings[0])

        return all_embeddings

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors for the configured model."""
        if self._cached_dimension is not None:
            return self._cached_dimension
        return _MODEL_DIMENSIONS.get(self._model, 1024)
