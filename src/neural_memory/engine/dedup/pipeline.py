"""3-tier dedup pipeline: SimHash -> Embedding -> LLM.

Each tier short-circuits when it reaches a definitive answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.dedup.config import DedupConfig
from neural_memory.utils.simhash import hamming_distance, simhash

if TYPE_CHECKING:
    from neural_memory.engine.dedup.llm_judge import LLMDedupJudge
    from neural_memory.engine.embedding.provider import EmbeddingProvider
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DedupResult:
    """Result of dedup check.

    Attributes:
        is_duplicate: Whether the content is a duplicate of an existing anchor.
        existing_neuron_id: ID of the matching existing anchor (if duplicate).
        tier: Which tier determined the result (1=SimHash, 2=Embedding, 3=LLM, 0=none).
        similarity_score: Similarity score from the determining tier.
        reason: Human-readable explanation.
    """

    is_duplicate: bool
    existing_neuron_id: str = ""
    tier: int = 0
    similarity_score: float = 0.0
    reason: str = ""


class DedupPipeline:
    """3-tier dedup pipeline for anchor neuron deduplication.

    Tier 1: SimHash (zero-dep, fast bitwise comparison).
    Tier 2: Embedding cosine similarity (optional, needs embedding provider).
    Tier 3: LLM judgment (optional, borderline cases only).
    """

    def __init__(
        self,
        config: DedupConfig,
        storage: NeuralStorage,
        embedding_provider: EmbeddingProvider | None = None,
        llm_judge: LLMDedupJudge | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._embedding_provider = embedding_provider
        self._llm_judge = llm_judge

    async def check_duplicate(
        self,
        content: str,
        content_hash: int | None = None,
    ) -> DedupResult:
        """Check if content is a duplicate of an existing anchor neuron.

        Args:
            content: The text content to check.
            content_hash: Pre-computed SimHash (computed if not provided).

        Returns:
            DedupResult indicating whether content is a duplicate.
        """
        if not self._config.enabled:
            return DedupResult(is_duplicate=False, reason="dedup disabled")

        if not content or not content.strip():
            return DedupResult(is_duplicate=False, reason="empty content")

        if content_hash is None:
            content_hash = simhash(content)

        # Fast path: exact content_hash match via indexed column
        hash_candidates = await self._get_candidates_by_hash(content_hash)
        if hash_candidates:
            tier1_result = self._tier1_simhash(content_hash, hash_candidates)
            if tier1_result is not None:
                return tier1_result

        # Fetch candidate anchors from storage (text-based search)
        candidates = await self._get_candidates(content)
        if not candidates:
            return DedupResult(is_duplicate=False, reason="no candidates found")

        # Tier 1: SimHash
        tier1_result = self._tier1_simhash(content_hash, candidates)
        if tier1_result is not None:
            return tier1_result

        # Tier 2: Embedding cosine (if provider available)
        if self._embedding_provider is not None:
            tier2_result = await self._tier2_embedding(content, candidates)
            if tier2_result is not None:
                return tier2_result

        # Tier 3: LLM judgment (if enabled and judge available)
        if self._config.llm_enabled and self._llm_judge is not None:
            tier3_result = await self._tier3_llm(content, candidates)
            if tier3_result is not None:
                return tier3_result

        return DedupResult(is_duplicate=False, reason="no tier found a match")

    async def _get_candidates_by_hash(self, content_hash: int) -> list[Neuron]:
        """Fast path: fetch neurons with matching SimHash via indexed column."""
        if content_hash == 0:
            return []
        try:
            all_hashes = await self._storage.get_neuron_hashes()
            # Find neurons with exact hash match (hamming distance 0)
            matching_ids = [nid for nid, h in all_hashes if h == content_hash]
            if not matching_ids:
                return []
            neurons = []
            for nid in matching_ids[:10]:  # Cap at 10 exact matches
                neuron = await self._storage.get_neuron(nid)
                if neuron and neuron.metadata.get("is_anchor", False):
                    neurons.append(neuron)
            return neurons
        except Exception:
            logger.debug("Hash-based candidate lookup failed", exc_info=True)
            return []

    async def _get_candidates(self, content: str) -> list[Neuron]:
        """Fetch candidate anchor neurons for comparison."""
        max_candidates = min(self._config.max_candidates, 50)

        # Use first significant word as search key
        words = content.split()
        search_term = ""
        for word in words:
            cleaned = word.strip(".,;:!?\"'()[]{}").lower()
            if len(cleaned) >= 3:
                search_term = cleaned
                break

        if not search_term:
            return []

        candidates = await self._storage.find_neurons(
            content_contains=search_term,
            limit=max_candidates,
        )

        # Filter to anchor neurons only
        return [
            n
            for n in candidates
            if n.metadata.get("is_anchor", False) and n.type == NeuronType.CONCEPT
        ]

    def _tier1_simhash(
        self,
        content_hash: int,
        candidates: list[Neuron],
    ) -> DedupResult | None:
        """Tier 1: SimHash near-duplicate detection.

        Returns DedupResult if a definitive match is found, None otherwise.
        """
        threshold = self._config.simhash_threshold

        for candidate in candidates:
            if candidate.content_hash is None or candidate.content_hash == 0:
                continue

            distance = hamming_distance(content_hash, candidate.content_hash)
            if distance <= threshold:
                # Definitive match -- very similar texts
                similarity = 1.0 - (distance / 64.0)
                return DedupResult(
                    is_duplicate=True,
                    existing_neuron_id=candidate.id,
                    tier=1,
                    similarity_score=similarity,
                    reason=f"SimHash match (distance={distance}, threshold={threshold})",
                )

        return None

    async def _tier2_embedding(
        self,
        content: str,
        candidates: list[Neuron],
    ) -> DedupResult | None:
        """Tier 2: Embedding cosine similarity.

        Returns DedupResult if definitive (above threshold or below ambiguous_low).
        Returns None for borderline cases (deferred to Tier 3).
        """
        if self._embedding_provider is None:
            return None

        try:
            content_embedding = await self._embedding_provider.embed(content)
        except Exception:
            logger.debug("Embedding failed for dedup content", exc_info=True)
            return None

        best_score = 0.0
        best_candidate: Neuron | None = None

        for candidate in candidates:
            try:
                candidate_embedding = await self._embedding_provider.embed(candidate.content)
                score = await self._embedding_provider.similarity(
                    content_embedding, candidate_embedding
                )
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            except Exception:
                logger.debug("Embedding comparison failed", exc_info=True)
                continue

        if best_candidate is None:
            return None

        # Definitive duplicate
        if best_score >= self._config.embedding_threshold:
            return DedupResult(
                is_duplicate=True,
                existing_neuron_id=best_candidate.id,
                tier=2,
                similarity_score=best_score,
                reason=f"Embedding match (score={best_score:.3f})",
            )

        # Definitive not-duplicate
        if best_score < self._config.embedding_ambiguous_low:
            return DedupResult(
                is_duplicate=False,
                tier=2,
                similarity_score=best_score,
                reason=f"Embedding mismatch (score={best_score:.3f})",
            )

        # Borderline -- defer to Tier 3
        return None

    async def _tier3_llm(
        self,
        content: str,
        candidates: list[Neuron],
    ) -> DedupResult | None:
        """Tier 3: LLM judgment for borderline cases.

        Limited to max_pairs_per_encode calls.
        """
        if self._llm_judge is None:
            return None

        from neural_memory.engine.dedup.llm_judge import DedupVerdict

        max_pairs = self._config.llm_max_pairs_per_encode

        for candidate in candidates[:max_pairs]:
            try:
                judgment = await self._llm_judge.judge(content, candidate.content)

                if judgment.verdict == DedupVerdict.DUPLICATE:
                    return DedupResult(
                        is_duplicate=True,
                        existing_neuron_id=candidate.id,
                        tier=3,
                        similarity_score=judgment.confidence,
                        reason=f"LLM judge: {judgment.reason}",
                    )
                # DISTINCT for one candidate doesn't mean distinct from all
                # -- continue checking remaining candidates.
                # UNCERTAIN -- also continue to next candidate
            except Exception:
                logger.debug("LLM dedup judge failed", exc_info=True)
                continue

        return None
