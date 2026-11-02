"""Cheaper Inference embedding provider using the OpenAI-compatible API."""

from __future__ import annotations

from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_BASE_URL = "https://api.cheaperinference.com/v1"


class CheaperInferenceEmbedding(OpenAIEmbedding):
    """Embedding provider backed by the Cheaper Inference embeddings API.

    Model ids are the bare OpenAI names, so the known dimensions are
    inherited from :class:`OpenAIEmbedding`.
    """

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
            api_key_env="CHEAPER_INFERENCE_API_KEY",
            provider_label="Cheaper Inference",
        )
