"""Deduplication for anchor neurons.

3-tier cascade: SimHash -> Embedding cosine -> LLM judgment.
Each tier short-circuits on definitive answers.

SimHash (Tier 1) always runs — pure Python, zero external deps.
Embedding + LLM tiers require [dedup] enabled = true in config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neural_memory.engine.dedup.config import DedupConfig
from neural_memory.engine.dedup.pipeline import DedupPipeline, DedupResult

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import DedupSettings

__all__ = ["DedupConfig", "DedupPipeline", "DedupResult", "build_dedup_pipeline"]


def build_dedup_pipeline(
    dedup_settings: DedupSettings | Any,
    storage: NeuralStorage | Any,
) -> DedupPipeline | None:
    """Build a dedup pipeline from unified config DedupSettings.

    SimHash (Tier 1) always runs.  Embedding + LLM tiers only activate
    when ``dedup_settings.enabled`` is True.

    Args:
        dedup_settings: A ``DedupSettings`` instance (or any object with
            the same attributes).
        storage: ``NeuralStorage`` backend.

    Returns:
        A ``DedupPipeline``, or ``None`` if construction fails entirely.
    """
    try:
        full_dedup = isinstance(dedup_settings.enabled, bool) and dedup_settings.enabled

        if full_dedup:
            dedup_cfg = DedupConfig(
                enabled=True,
                simhash_threshold=int(dedup_settings.simhash_threshold),
                embedding_threshold=float(dedup_settings.embedding_threshold),
                embedding_ambiguous_low=float(dedup_settings.embedding_ambiguous_low),
                llm_enabled=bool(dedup_settings.llm_enabled),
                llm_provider=str(dedup_settings.llm_provider),
                llm_model=str(dedup_settings.llm_model),
                llm_max_pairs_per_encode=int(dedup_settings.llm_max_pairs_per_encode),
                merge_strategy=str(dedup_settings.merge_strategy),
                max_candidates=int(dedup_settings.max_candidates),
            )

            llm_judge = None
            if dedup_cfg.llm_enabled and dedup_cfg.llm_provider != "none":
                from neural_memory.engine.dedup.llm_judge import create_judge

                llm_judge = create_judge(dedup_cfg.llm_provider, dedup_cfg.llm_model)

            return DedupPipeline(
                config=dedup_cfg,
                storage=storage,
                llm_judge=llm_judge,
            )

        # SimHash-only: no embedding, no LLM — zero external deps
        dedup_cfg = DedupConfig(
            enabled=True,
            simhash_threshold=int(getattr(dedup_settings, "simhash_threshold", 7)),
            llm_enabled=False,
            llm_provider="none",
        )
        return DedupPipeline(config=dedup_cfg, storage=storage)
    except (AttributeError, TypeError, ValueError):
        return None
