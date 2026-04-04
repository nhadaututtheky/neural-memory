"""Directional compression — Pro-grade memory compression.

Multi-axis directional compression preserves semantic direction along
MULTIPLE reference embeddings, not just the primary anchor. This prevents
information loss when a memory relates to several concepts.

Free version: single-axis anisotropic (preserves direction to 1 anchor)
Pro version: multi-axis (preserves direction to top-K related neurons)
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


async def directional_compress(
    content: str,
    level: str,
    embed_fn: Any,
    *,
    reference_embeddings: list[list[float]] | None = None,
    max_axes: int = 3,
) -> str:
    """Multi-axis directional compression.

    Compresses content while preserving semantic direction along
    multiple reference axes (related neurons' embeddings).

    Args:
        content: Source text to compress.
        level: "full", "summary", "essence", or "ghost".
        embed_fn: Async embedding function (str -> list[float]).
        reference_embeddings: Optional pre-computed reference vectors.
        max_axes: Maximum reference axes to preserve (default 3).

    Returns:
        Compressed text preserving multi-directional semantics.
    """
    level_lower = level.lower() if isinstance(level, str) else "summary"

    # FULL level = no compression
    if level_lower == "full":
        return content

    # GHOST level = minimal stub
    if level_lower == "ghost":
        words = content.split()
        return " ".join(words[:5]) + "..." if len(words) > 5 else content

    # Get content embedding
    content_vec = await embed_fn(content)
    if not content_vec:
        return _basic_compress(content, level_lower)

    content_arr = np.array(content_vec, dtype=np.float32)

    # Split content into sentences
    sentences = _split_sentences(content)
    if len(sentences) <= 1:
        return content

    # Score each sentence by directional importance
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        sent_vec = await embed_fn(sentence)
        if not sent_vec:
            scored.append((0.5, sentence))
            continue

        sent_arr = np.array(sent_vec, dtype=np.float32)

        # Primary axis: similarity to full content
        primary_sim = _cosine_sim(sent_arr, content_arr)

        # Reference axes: similarity to related neurons
        ref_score = 0.0
        if reference_embeddings:
            refs = reference_embeddings[:max_axes]
            ref_sims = [_cosine_sim(sent_arr, np.array(ref, dtype=np.float32)) for ref in refs]
            ref_score = max(ref_sims) if ref_sims else 0.0

        # Combined: primary axis + best reference axis
        score = primary_sim * 0.6 + ref_score * 0.4
        scored.append((score, sentence))

    # Sort by importance, keep based on level
    scored.sort(key=lambda x: x[0], reverse=True)

    keep = max(1, len(scored) * 2 // 3) if level_lower == "summary" else max(1, len(scored) // 3)

    kept = scored[:keep]
    # Restore original order
    kept_set = {s for _, s in kept}
    result = [s for s in sentences if s in kept_set]

    return " ".join(result)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two numpy arrays."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple heuristic)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _basic_compress(content: str, level: str) -> str:
    """Fallback compression without embeddings."""
    words = content.split()
    keep = max(1, len(words) * 2 // 3) if level == "summary" else max(1, len(words) // 3)
    return " ".join(words[:keep]) + ("..." if keep < len(words) else "")
