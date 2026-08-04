"""EmbeddingStep — generates and indexes vector embeddings during encode.

Runs as the last pipeline step. Embeds the anchor neuron's content and
stores the vector in both the storage vector index (for knn_search) and
in neuron metadata (for backward compatibility / migration).

Non-blocking: if the embedding provider is unavailable or fails, encoding
continues without embeddings.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.engine.embedding.provider import EmbeddingProvider
    from neural_memory.engine.pipeline import PipelineContext
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


class EmbeddingStep:
    """Pipeline step that embeds anchor neuron content into a vector index."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._provider = embedding_provider

    @property
    def name(self) -> str:
        return "EmbeddingStep"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        """Embed anchor neuron and store vector."""
        anchor = ctx.anchor_neuron
        if anchor is None:
            return ctx

        try:
            vector = await self._provider.embed(anchor.content)

            # Store in vector index (if storage supports it)
            if hasattr(storage, "vector_index_add"):
                await storage.vector_index_add(anchor.id, vector)

        except Exception:
            if os.environ.get("NMEM_REQUIRE_EMBEDDING") == "1":
                raise
            logger.debug(
                "EmbeddingStep failed for neuron %s — continuing without embedding",
                anchor.id,
                exc_info=True,
            )

        return ctx
