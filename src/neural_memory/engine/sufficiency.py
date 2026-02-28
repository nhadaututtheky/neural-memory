"""Post-stabilization sufficiency check for the retrieval pipeline.

Evaluates whether the activated signal is strong enough to warrant
reconstruction, or whether the pipeline should early-exit. Zero LLM
dependency — pure math on activation statistics.

Gates are conservative: false-INSUFFICIENT (killing good results) is far
worse than false-SUFFICIENT (wasting compute on reconstruction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.engine.activation import ActivationResult


@dataclass(frozen=True)
class SufficiencyMetrics:
    """Raw numeric metrics computed from the activation landscape."""

    anchor_count: int
    anchor_sets_active: int
    neuron_count: int
    intersection_count: int
    top_activation: float
    mean_activation: float
    activation_entropy: float
    activation_mass: float
    coverage_ratio: float
    focus_ratio: float
    proximity_ratio: float
    path_diversity: float
    stab_converged: bool
    stab_neurons_removed: int


@dataclass(frozen=True)
class SufficiencyResult:
    """Result of the sufficiency check gate."""

    sufficient: bool
    confidence: float
    gate: str
    reason: str
    metrics: SufficiencyMetrics


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _shannon_entropy(levels: list[float]) -> float:
    """Shannon entropy (bits) of an activation distribution."""
    total = sum(levels)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for lv in levels:
        p = lv / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _focus_ratio(levels: list[float], top_k: int = 3) -> float:
    """Ratio of top-K activation sum to total. High = focused, low = diffuse."""
    if not levels:
        return 0.0
    sorted_desc = sorted(levels, reverse=True)
    total = sum(sorted_desc)
    if total <= 0:
        return 0.0
    return sum(sorted_desc[:top_k]) / total


def _proximity_ratio(hop_distances: list[int]) -> float:
    """Fraction of activated neurons at hop distance 1 (direct match)."""
    if not hop_distances:
        return 0.0
    hop1_count = sum(1 for h in hop_distances if h <= 1)
    return hop1_count / len(hop_distances)


def _path_diversity(
    source_anchors_in_top5: list[str],
    total_anchor_count: int,
) -> float:
    """Fraction of unique source anchors reaching top-5 neurons."""
    if total_anchor_count <= 0:
        return 0.0
    unique = len(set(source_anchors_in_top5))
    return min(1.0, unique / total_anchor_count)


def _compute_metrics(
    activations: dict[str, ActivationResult],
    anchor_sets: list[list[str]],
    intersections: list[str],
    stab_converged: bool,
    stab_neurons_removed: int,
) -> SufficiencyMetrics:
    """Compute all metrics from the post-stabilization landscape."""
    levels: list[float] = []
    hops: list[int] = []
    source_anchors: list[str] = []

    for act in activations.values():
        levels.append(act.activation_level)
        hops.append(act.hop_distance)
        source_anchors.append(act.source_anchor)

    anchor_count = sum(len(s) for s in anchor_sets)
    all_anchors_flat = {a for s in anchor_sets for a in s}
    active_sources = set(source_anchors)
    anchor_sets_active = sum(
        1 for s in anchor_sets if any(a in active_sources or a in activations for a in s)
    )

    top_activation = max(levels) if levels else 0.0
    total = sum(levels)
    mean_activation = total / len(levels) if levels else 0.0

    # Top-5 source anchors for path diversity
    sorted_pairs = sorted(
        zip(levels, source_anchors, strict=False), key=lambda x: x[0], reverse=True
    )
    top5_anchors = [sa for _, sa in sorted_pairs[:5]]

    return SufficiencyMetrics(
        anchor_count=anchor_count,
        anchor_sets_active=anchor_sets_active,
        neuron_count=len(activations),
        intersection_count=len(intersections),
        top_activation=top_activation,
        mean_activation=mean_activation,
        activation_entropy=_shannon_entropy(levels),
        activation_mass=total,
        coverage_ratio=(anchor_sets_active / len(anchor_sets) if anchor_sets else 0.0),
        focus_ratio=_focus_ratio(levels),
        proximity_ratio=_proximity_ratio(hops),
        path_diversity=_path_diversity(top5_anchors, len(all_anchors_flat)),
        stab_converged=stab_converged,
        stab_neurons_removed=stab_neurons_removed,
    )


def _compute_confidence(m: SufficiencyMetrics, anchor_sets_len: int) -> float:
    """Unified confidence formula from 7 weighted inputs."""
    intersection_ratio = min(1.0, m.intersection_count / max(1, anchor_sets_len))
    stability = 1.0 if m.stab_converged else 0.5

    raw = (
        0.30 * m.top_activation
        + 0.20 * m.focus_ratio
        + 0.15 * m.coverage_ratio
        + 0.15 * intersection_ratio
        + 0.10 * m.proximity_ratio
        + 0.05 * stability
        + 0.05 * m.path_diversity
    )
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------


def check_sufficiency(
    activations: dict[str, ActivationResult],
    anchor_sets: list[list[str]],
    intersections: list[str],
    stab_converged: bool,
    stab_neurons_removed: int,
) -> SufficiencyResult:
    """Evaluate whether retrieval has sufficient signal for reconstruction.

    Gates are evaluated in priority order; first match wins.
    Conservative bias: prefer false-SUFFICIENT over false-INSUFFICIENT.

    Args:
        activations: Neuron activations after stabilization.
        anchor_sets: Groups of anchor neurons matched to query.
        intersections: Neurons reached from multiple anchor groups.
        stab_converged: Whether stabilization converged.
        stab_neurons_removed: Neurons killed by noise floor.

    Returns:
        SufficiencyResult with gate decision, confidence, and metrics.
    """
    m = _compute_metrics(
        activations,
        anchor_sets,
        intersections,
        stab_converged,
        stab_neurons_removed,
    )
    conf = _compute_confidence(m, len(anchor_sets))

    # Gate 1: no_anchors
    if m.anchor_count == 0:
        return SufficiencyResult(
            sufficient=False,
            confidence=0.0,
            gate="no_anchors",
            reason="No anchor neurons found for query",
            metrics=m,
        )

    # Gate 2: empty_landscape
    if m.neuron_count == 0:
        return SufficiencyResult(
            sufficient=False,
            confidence=0.0,
            gate="empty_landscape",
            reason="All activations died during stabilization",
            metrics=m,
        )

    # Gate 3: unstable_noise
    if not m.stab_converged and m.stab_neurons_removed > m.neuron_count and m.top_activation < 0.3:
        return SufficiencyResult(
            sufficient=False,
            confidence=min(conf, 0.1),
            gate="unstable_noise",
            reason=(
                f"Unstable signal: {m.stab_neurons_removed} neurons removed, "
                f"only {m.neuron_count} survived, top activation {m.top_activation:.2f}"
            ),
            metrics=m,
        )

    # Gate 4: ambiguous_spread
    if (
        m.activation_entropy >= 3.0
        and m.focus_ratio < 0.2
        and m.neuron_count >= 15
        and m.top_activation < 0.3
    ):
        return SufficiencyResult(
            sufficient=False,
            confidence=min(conf, 0.1),
            gate="ambiguous_spread",
            reason=(
                f"Diffuse activation: entropy={m.activation_entropy:.1f} bits, "
                f"focus={m.focus_ratio:.2f}, no standout neuron"
            ),
            metrics=m,
        )

    # Gate 5: intersection_convergence
    if m.intersection_count >= 2 and m.top_activation >= 0.4:
        return SufficiencyResult(
            sufficient=True,
            confidence=conf,
            gate="intersection_convergence",
            reason=(
                f"Multi-anchor convergence: {m.intersection_count} intersections, "
                f"top activation {m.top_activation:.2f}"
            ),
            metrics=m,
        )

    # Gate 6: high_coverage_strong_hit
    if m.coverage_ratio >= 0.5 and m.top_activation >= 0.7 and m.focus_ratio >= 0.4:
        return SufficiencyResult(
            sufficient=True,
            confidence=conf,
            gate="high_coverage_strong_hit",
            reason=(
                f"Strong signal: coverage={m.coverage_ratio:.0%}, "
                f"top={m.top_activation:.2f}, focus={m.focus_ratio:.2f}"
            ),
            metrics=m,
        )

    # Gate 7: focused_result
    if m.neuron_count <= 5 and m.top_activation >= 0.5 and m.focus_ratio >= 0.6:
        return SufficiencyResult(
            sufficient=True,
            confidence=conf,
            gate="focused_result",
            reason=(
                f"Focused result: {m.neuron_count} neurons, "
                f"top={m.top_activation:.2f}, focus={m.focus_ratio:.2f}"
            ),
            metrics=m,
        )

    # Gate 8: default_pass
    return SufficiencyResult(
        sufficient=True,
        confidence=conf,
        gate="default_pass",
        reason=f"Default pass: {m.neuron_count} neurons, conf={conf:.2f}",
        metrics=m,
    )
