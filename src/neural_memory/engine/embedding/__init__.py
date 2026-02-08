"""Optional embedding layer for semantic similarity search.

OFF by default — zero-LLM core preserved.
Enable via BrainConfig(embedding_enabled=True).
"""

from neural_memory.engine.embedding.config import EmbeddingConfig
from neural_memory.engine.embedding.provider import EmbeddingProvider

__all__ = ["EmbeddingConfig", "EmbeddingProvider"]
