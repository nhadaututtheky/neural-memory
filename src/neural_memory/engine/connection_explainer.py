"""Explain connections between entities via shortest-path graph traversal.

Given two entity names, finds the shortest path through the synapse graph
and returns a human-readable explanation with supporting fiber evidence.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.neuron import Neuron
    from neural_memory.core.synapse import Synapse
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

_MAX_HOPS_LIMIT = 10
_MAX_CANDIDATES = 5
_MAX_EVIDENCE_PER_STEP = 3
_CONFIDENCE_DECAY = 0.4


@dataclass(frozen=True)
class ConnectionStep:
    """One hop in the connection path."""

    neuron_id: str
    content: str
    synapse_type: str
    weight: float
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectionExplanation:
    """Result of an explain_connection query."""

    found: bool
    from_entity: str
    to_entity: str
    steps: tuple[ConnectionStep, ...] = ()
    total_hops: int = 0
    avg_weight: float = 0.0
    confidence: float = 0.0
    strength: str = "none"
    markdown: str = ""


def compute_path_confidence(hops: int, avg_weight: float) -> float:
    """Compute path confidence using exponential decay.

    Combines hop distance decay with average synapse weight to produce
    a single [0, 1] confidence score. Inspired by HyperspaceDB's
    Tribunal scoring: confidence drops exponentially with path length.

    Args:
        hops: Number of hops in the path.
        avg_weight: Average synapse weight along the path.

    Returns:
        Confidence score in [0, 1].
    """
    if hops <= 0 or avg_weight <= 0.0:
        return 0.0
    return math.exp(-_CONFIDENCE_DECAY * hops) * min(avg_weight, 1.0)


def _confidence_to_strength(confidence: float) -> str:
    """Map confidence score to human-readable strength label."""
    if confidence >= 0.7:
        return "strong"
    if confidence >= 0.4:
        return "moderate"
    if confidence >= 0.2:
        return "weak"
    return "tenuous"


async def explain_connection(
    storage: NeuralStorage,
    from_entity: str,
    to_entity: str,
    max_hops: int = 6,
) -> ConnectionExplanation:
    """Find and explain the shortest path between two entities.

    Args:
        storage: Neural storage instance (brain must be set).
        from_entity: Source entity name (fuzzy matched via content_contains).
        to_entity: Target entity name (fuzzy matched via content_contains).
        max_hops: Maximum path length (capped at 10).

    Returns:
        ConnectionExplanation with path steps, evidence, and markdown summary.
    """
    max_hops = min(max_hops, _MAX_HOPS_LIMIT)

    # 1. Find candidate neurons for both entities
    source_neurons = await storage.find_neurons(content_contains=from_entity, limit=_MAX_CANDIDATES)
    target_neurons = await storage.find_neurons(content_contains=to_entity, limit=_MAX_CANDIDATES)

    if not source_neurons or not target_neurons:
        return ConnectionExplanation(
            found=False,
            from_entity=from_entity,
            to_entity=to_entity,
            markdown=_no_path_markdown(from_entity, to_entity, "entity not found"),
        )

    # 2. Try all candidate pairs, pick shortest path
    best_path: list[tuple[Neuron, Synapse]] | None = None
    best_source: Neuron | None = None

    for src in source_neurons:
        for tgt in target_neurons:
            if src.id == tgt.id:
                continue
            path = await storage.get_path(src.id, tgt.id, max_hops=max_hops, bidirectional=True)
            if path is not None and (best_path is None or len(path) < len(best_path)):
                best_path = path
                best_source = src

    if best_path is None or best_source is None:
        return ConnectionExplanation(
            found=False,
            from_entity=from_entity,
            to_entity=to_entity,
            markdown=_no_path_markdown(from_entity, to_entity, "no path found"),
        )

    # 3. Hydrate fiber evidence for path step neurons only (not source)
    all_neuron_ids = [n.id for n, _ in best_path]
    fibers_by_neuron = await _get_evidence(storage, all_neuron_ids)

    # 4. Build steps
    steps: list[ConnectionStep] = []
    for neuron, synapse in best_path:
        evidence = _extract_evidence(fibers_by_neuron.get(neuron.id, []))
        steps.append(
            ConnectionStep(
                neuron_id=neuron.id,
                content=neuron.content,
                synapse_type=synapse.type.value
                if hasattr(synapse.type, "value")
                else str(synapse.type),
                weight=synapse.weight,
                evidence=tuple(evidence),
            )
        )

    total_hops = len(steps)
    avg_weight = sum(s.weight for s in steps) / total_hops if total_hops else 0.0
    confidence = compute_path_confidence(total_hops, avg_weight)
    strength = _confidence_to_strength(confidence)

    frozen_steps = tuple(steps)
    md = _build_markdown(
        best_source, frozen_steps, from_entity, to_entity, avg_weight, confidence, strength
    )

    return ConnectionExplanation(
        found=True,
        from_entity=from_entity,
        to_entity=to_entity,
        steps=frozen_steps,
        total_hops=total_hops,
        avg_weight=round(avg_weight, 3),
        confidence=round(confidence, 3),
        strength=strength,
        markdown=md,
    )


async def _get_evidence(
    storage: NeuralStorage,
    neuron_ids: list[str],
) -> dict[str, list[Fiber]]:
    """Batch-fetch fibers for a list of neuron IDs."""
    fibers = await storage.find_fibers_batch(neuron_ids, limit_per_neuron=_MAX_EVIDENCE_PER_STEP)
    result: dict[str, list[Fiber]] = {}
    for fiber in fibers:
        for nid in neuron_ids:
            if nid in fiber.neuron_ids:
                result.setdefault(nid, []).append(fiber)
    return result


def _extract_evidence(fibers: list[Fiber]) -> list[str]:
    """Extract summary text from fibers as evidence strings."""
    evidence: list[str] = []
    for fiber in fibers[:_MAX_EVIDENCE_PER_STEP]:
        text = fiber.summary or ""
        if text:
            # Truncate long summaries
            evidence.append(text[:200])
    return evidence


def _build_markdown(
    source: Neuron,
    steps: tuple[ConnectionStep, ...],
    from_entity: str,
    to_entity: str,
    avg_weight: float,
    confidence: float = 0.0,
    strength: str = "none",
) -> str:
    """Build human-readable markdown explanation."""
    lines: list[str] = [
        f"## Connection: {from_entity} → {to_entity}",
        "",
        f"**Path length:** {len(steps)} hop(s) | **Avg weight:** {avg_weight:.2f} | **Confidence:** {confidence:.2f} ({strength})",
        "",
        "### Path",
        "",
        f"1. **{source.content}** (start)",
    ]

    for i, step in enumerate(steps, start=2):
        lines.append(
            f"{i}. --[{step.synapse_type}]--> **{step.content}**"
            + (f" (w={step.weight:.2f})" if step.weight < 1.0 else "")
        )

    # Evidence section
    has_evidence = any(step.evidence for step in steps)
    if has_evidence:
        lines.extend(["", "### Evidence", ""])
        for step in steps:
            if step.evidence:
                lines.append(f"**{step.content}:**")
                for ev in step.evidence:
                    lines.append(f"  - {ev}")

    return "\n".join(lines)


def _no_path_markdown(from_entity: str, to_entity: str, reason: str) -> str:
    """Markdown for when no path is found."""
    return (
        f"## Connection: {from_entity} → {to_entity}\n\n"
        f"No connection found ({reason}).\n\n"
        "Try broader entity names or increase max_hops."
    )
