"""Abstract base class for embedding providers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Base class for all embedding providers.

    Subclasses must implement ``embed`` and ``dimension``.  The default
    ``embed_batch`` falls back to sequential ``embed`` calls and
    ``similarity`` computes cosine similarity between two vectors.
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for each entry in *texts*.

        The default implementation calls :meth:`embed` sequentially.
        Providers that support native batching should override this.
        """
        return [await self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the vectors returned by :meth:`embed`."""

    async def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 when either vector has zero magnitude.
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)
