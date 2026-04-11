"""Semantic synapse discovery — offline consolidation via embeddings.

Discovers SIMILAR_TO synapses between unconnected CONCEPT and ENTITY
neurons by computing cosine similarity on their embedding vectors.

This is an **offline consolidation** step, not a recall-time operation.
It enriches the neural graph so that spreading activation can later
traverse the discovered semantic links.

Optional: silently skips if sentence-transformers is not installed.
Discovered synapses decay 2x faster during pruning unless reinforced,
preventing stale semantic links from accumulating.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Hard caps to prevent runaway
MAX_NEURONS_TO_EMBED = 500
MAX_PAIRS_HARD_CAP = 200


@dataclass(frozen=True)
class SemanticDiscoveryResult:
    """Result of a semantic discovery run."""

    neurons_embedded: int = 0
    pairs_evaluated: int = 0
    synapses_created: int = 0
    skipped_existing: int = 0
    synapses: list[Synapse] = field(default_factory=list)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _auto_detect_provider() -> tuple[str, str]:
    """Auto-detect the best available embedding provider.

    Detection order (first available wins):
    1. Ollama running locally (free, no install needed if Ollama app present)
    2. sentence-transformers installed (free, local, but ~440MB download)
    3. Gemini API key set (free tier available)
    4. OpenAI API key set (paid)
    5. OpenRouter API key set (OpenAI-compatible)

    Returns:
        Tuple of (provider_name, model_name).

    Raises:
        RuntimeError: If no provider is available.
    """
    # 1. Check Ollama (local server)
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            # Prefer multilingual embedding models
            preferred = ["bge-m3", "nomic-embed-text", "mxbai-embed-large", "all-minilm"]
            available_names = [m.get("name", "").split(":")[0] for m in models]
            for pref in preferred:
                if pref in available_names:
                    return ("ollama", pref)
            # Any model works as fallback
            if models:
                return ("ollama", "bge-m3")
    except Exception:
        logger.debug("Ollama embedding probe failed", exc_info=True)

    # 2. Check sentence-transformers (local)
    try:
        import sentence_transformers

        return ("sentence_transformer", "paraphrase-multilingual-MiniLM-L12-v2")
    except ImportError:
        pass

    # 3. Check Gemini API key (free tier)
    import os

    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return ("gemini", "models/text-embedding-004")

    # 4. Check OpenAI API key (paid)
    if os.environ.get("OPENAI_API_KEY"):
        return ("openai", "text-embedding-3-small")

    # 5. Check OpenRouter API key (OpenAI-compatible)
    if os.environ.get("OPENROUTER_API_KEY"):
        return ("openrouter", "openai/text-embedding-3-small")

    raise RuntimeError(
        "No embedding provider available. Install one of:\n"
        "  pip install neural-memory[embeddings]    # sentence-transformers (local, free)\n"
        "  ollama pull bge-m3                       # Ollama (local, free)\n"
        "  export GEMINI_API_KEY=...                # Google Gemini (free tier)\n"
        "  export OPENAI_API_KEY=...                # OpenAI (paid)\n"
        "  export OPENROUTER_API_KEY=...            # OpenRouter (OpenAI-compatible)"
    )


# Module-level singleton cache — avoids reloading models per tool call (#100)
_provider_cache: dict[tuple[str, str], Any] = {}


def _create_provider(config: BrainConfig, task_type: str = "RETRIEVAL_QUERY") -> Any:
    """Create or retrieve a cached embedding provider from BrainConfig.

    Providers are cached by (provider_name, model_name) so the model is loaded
    once per MCP process lifetime instead of once per tool call.

    Args:
        config: Brain configuration with embedding_provider and embedding_model.
        task_type: Task type hint for providers that support it (e.g. Gemini).

    Raises ImportError if the required package is not installed.
    """
    provider_name = config.embedding_provider
    model_name = config.embedding_model

    # Auto-detect best available provider
    if provider_name == "auto":
        provider_name, model_name = _auto_detect_provider()
        logger.info("Auto-detected embedding provider: %s (model: %s)", provider_name, model_name)

    cache_key = (provider_name, model_name)
    if cache_key in _provider_cache:
        return _provider_cache[cache_key]

    provider: Any
    if provider_name == "sentence_transformer":
        from neural_memory.engine.embedding.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        provider = SentenceTransformerEmbedding(model_name=model_name)
    elif provider_name == "openai":
        from neural_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        provider = OpenAIEmbedding(model=model_name)
    elif provider_name == "openrouter":
        from neural_memory.engine.embedding.openrouter_embedding import OpenRouterEmbedding

        provider = OpenRouterEmbedding(model=model_name)
    elif provider_name == "gemini":
        from neural_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(model=model_name, task_type=task_type)
    elif provider_name == "ollama":
        from neural_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding(model=model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")

    _provider_cache[cache_key] = provider
    return provider


async def discover_semantic_synapses(
    storage: NeuralStorage,
    config: BrainConfig,
) -> SemanticDiscoveryResult:
    """Discover SIMILAR_TO synapses between unconnected neurons.

    Steps:
        1. Fetch CONCEPT + ENTITY neurons (capped at MAX_NEURONS_TO_EMBED)
        2. Batch-embed their content via EmbeddingProvider
        3. Compute pairwise cosine similarity
        4. Create SIMILAR_TO synapses for pairs above threshold
        5. Skip pairs that already have a synapse connection

    Args:
        storage: Neural storage backend
        config: Brain configuration (uses embedding_* and semantic_discovery_* fields)

    Returns:
        SemanticDiscoveryResult with created synapses
    """
    if not config.embedding_enabled:
        logger.debug("Embedding disabled — skipping semantic discovery")
        return SemanticDiscoveryResult()

    # Try to create embedding provider
    try:
        provider = _create_provider(config, task_type="RETRIEVAL_DOCUMENT")
    except Exception:
        logger.debug("Embedding provider unavailable — skipping semantic discovery")
        return SemanticDiscoveryResult()

    # Paginate through all neurons to collect CONCEPT+ENTITY (avoid OOM)
    batch_size = 5000
    offset = 0
    eligible: list[Neuron] = []
    while True:
        batch = await storage.find_neurons(limit=batch_size, offset=offset)
        if not batch:
            break
        eligible.extend(
            n
            for n in batch
            if n.type in (NeuronType.CONCEPT, NeuronType.ENTITY) and n.content.strip()
        )
        offset += len(batch)
        if len(batch) < batch_size:
            break

    if len(eligible) < 2:
        return SemanticDiscoveryResult()

    # Cap to prevent embedding too many
    eligible = eligible[:MAX_NEURONS_TO_EMBED]

    # Batch embed
    texts = [n.content for n in eligible]
    try:
        embeddings = await provider.embed_batch(texts)
    except Exception:
        logger.debug("Embedding batch failed — skipping semantic discovery", exc_info=True)
        return SemanticDiscoveryResult()

    neurons_embedded = len(embeddings)

    # Build set of existing synapse pairs for fast lookup
    # Need all types to prevent creating duplicates across different synapse types
    all_synapses = await storage.get_synapses()
    existing_pairs: set[frozenset[str]] = set()
    for syn in all_synapses:
        existing_pairs.add(frozenset({syn.source_id, syn.target_id}))

    # Compute pairwise cosine similarity
    threshold = config.semantic_discovery_similarity_threshold
    max_pairs = min(config.semantic_discovery_max_pairs, MAX_PAIRS_HARD_CAP)

    candidates: list[tuple[int, int, float]] = []
    pairs_evaluated = 0

    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            pairs_evaluated += 1
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                candidates.append((i, j, sim))

    # Sort by similarity descending, take top max_pairs
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:max_pairs]

    # Create synapses
    new_synapses: list[Synapse] = []
    skipped = 0

    for i, j, sim in candidates:
        neuron_a = eligible[i]
        neuron_b = eligible[j]
        pair_key = frozenset({neuron_a.id, neuron_b.id})

        if pair_key in existing_pairs:
            skipped += 1
            continue

        synapse = Synapse.create(
            source_id=neuron_a.id,
            target_id=neuron_b.id,
            type=SynapseType.SIMILAR_TO,
            weight=sim * 0.6,  # Scale down to avoid dominating graph
            metadata={
                "_semantic_discovery": True,
                "cosine_similarity": round(sim, 4),
            },
        )
        new_synapses.append(synapse)
        existing_pairs.add(pair_key)

    return SemanticDiscoveryResult(
        neurons_embedded=neurons_embedded,
        pairs_evaluated=pairs_evaluated,
        synapses_created=len(new_synapses),
        skipped_existing=skipped,
        synapses=new_synapses,
    )
