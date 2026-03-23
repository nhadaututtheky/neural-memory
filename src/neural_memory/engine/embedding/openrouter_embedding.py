"""OpenRouter embedding provider using the OpenAI-compatible API."""

from __future__ import annotations

from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

_MODEL_DIMENSIONS: dict[str, int] = {
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-ada-002": 1536,
}

_DEFAULT_MODEL = "openai/text-embedding-3-small"
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEmbedding(OpenAIEmbedding):
    """Embedding provider backed by the OpenRouter embeddings API."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        *,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_key_env="OPENROUTER_API_KEY",
            provider_label="OpenRouter",
        )

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors for the configured model."""
        return _MODEL_DIMENSIONS.get(self._model, 1536)
