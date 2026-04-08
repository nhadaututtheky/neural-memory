"""Post-stabilization sufficiency check for the retrieval pipeline.

Evaluates whether the activated signal is strong enough to warrant
reconstruction, or whether the pipeline should early-exit. Zero LLM
dependency — pure math on activation statistics.

Gates are conservative: false-INSUFFICIENT (killing good results) is far
worse than false-SUFFICIENT (wasting compute on reconstruction).

Advanced features:
- Per-query-type threshold profiles (strict / lenient / default)
- EMA calibration adjustment (auto-tune based on historical accuracy)
- Diminishing returns gate (future-proofing for multi-pass retrieval)
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


@dataclass(frozen=True)
class GateCalibration:
    """EMA-derived calibration stats for a single gate.

    Used to optionally adjust gate decisions based on historical accuracy.
    """

    accuracy: float
    avg_confidence: float
    sample_count: int


@dataclass(frozen=True)
class QueryTypeProfile:
    """Threshold adjustments per query intent category.

    Multipliers are applied to base gate thresholds so that:
    - strict queries (factual) demand stronger signal
    - lenient queries (exploratory) allow weaker signal through
    - default queries use standard thresholds (all multipliers = 1.0)
    """

    min_top_activation_factor: float
    entropy_tolerance: float
    min_intersection_count: int


# ---------------------------------------------------------------------------
# Query intent → profile mapping
# ---------------------------------------------------------------------------

_QUERY_PROFILES: dict[str, QueryTypeProfile] = {
    "strict": QueryTypeProfile(
        min_top_activation_factor=1.2,
        entropy_tolerance=0.8,
        min_intersection_count=2,
    ),
    "lenient": QueryTypeProfile(
        min_top_activation_factor=0.7,
        entropy_tolerance=1.5,
        min_intersection_count=1,
    ),
    "default": QueryTypeProfile(
        min_top_activation_factor=1.0,
        entropy_tolerance=1.0,
        min_intersection_count=2,
    ),
}

_INTENT_TO_PROFILE: dict[str, str] = {
    "ask_what": "strict",
    "ask_where": "strict",
    "ask_when": "strict",
    "ask_who": "strict",
    "confirm": "strict",
    "ask_pattern": "lenient",
    "compare": "lenient",
    "ask_why": "lenient",
    "ask_how": "lenient",
    "ask_feeling": "default",
    "recall": "default",
    "unknown": "default",
}


def _get_profile(query_intent: str) -> QueryTypeProfile:
    """Resolve a query intent string to a QueryTypeProfile."""
    profile_name = _INTENT_TO_PROFILE.get(query_intent, "default")
    return _QUERY_PROFILES[profile_name]


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
    query_intent: str = "",
    calibration: dict[str, GateCalibration] | None = None,
    prev_metrics: SufficiencyMetrics | None = None,
    density_scaling: bool = False,
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
        query_intent: Intent string from QueryParser (e.g. "ask_what").
            Used to select strict/lenient/default threshold profile.
        calibration: Optional per-gate EMA stats (from get_gate_ema_stats).
            When provided, gates for SUFFICIENT decisions with low historical
            avg_confidence will be downgraded to INSUFFICIENT.
        prev_metrics: Optional metrics from a previous retrieval pass.
            When provided, the diminishing_returns gate fires if metrics
            have not changed meaningfully (future-proofing for multi-pass).

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

    profile = _get_profile(query_intent)

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
    if not m.stab_converged and m.stab_neurons_removed > m.neuron_count and m.top_activation < 0.2:
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
    # Apply profile entropy_tolerance and min_top_activation_factor
    _entropy_threshold = 4.0 * profile.entropy_tolerance
    _top_act_threshold_ambiguous = 0.3 * profile.min_top_activation_factor
    # Density scaling: normalize entropy by max possible entropy for this neuron count.
    # High absolute entropy with 500 neurons is expected; only flag when RELATIVE
    # entropy is high (homeostatic synaptic scaling — Turrigiano 2008).
    _effective_entropy = m.activation_entropy
    if density_scaling and m.neuron_count >= 4:
        _max_entropy = math.log2(m.neuron_count)
        _relative_entropy = m.activation_entropy / _max_entropy
        # Convert back to absolute scale for threshold comparison:
        # relative 0.8 with 500 neurons → effective = 0.8 * 4.0 = 3.2 (below 4.0, passes)
        _effective_entropy = _relative_entropy * 4.0
    if (
        _effective_entropy >= _entropy_threshold
        and m.focus_ratio < 0.2
        and m.neuron_count >= 15
        and m.top_activation < _top_act_threshold_ambiguous
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

    # Gate 4.5: diminishing_returns
    # When a previous pass produced nearly identical metrics, more passes won't help.
    if prev_metrics is not None:
        _act_delta = abs(m.top_activation - prev_metrics.top_activation)
        _neuron_delta = abs(m.neuron_count - prev_metrics.neuron_count)
        _focus_delta = abs(m.focus_ratio - prev_metrics.focus_ratio)
        if _act_delta < 0.05 and _neuron_delta <= 1 and _focus_delta < 0.05:
            # Take what we have — additional passes won't improve signal
            _dr_conf = max(0.0, min(1.0, conf * 0.85))
            return SufficiencyResult(
                sufficient=True,
                confidence=_dr_conf,
                gate="diminishing_returns",
                reason=(
                    f"Diminishing returns: activation delta={_act_delta:.3f}, "
                    f"neuron delta={_neuron_delta}, focus delta={_focus_delta:.3f}"
                ),
                metrics=m,
            )

    # Gate 5: intersection_convergence
    _top_act_threshold_intersect = 0.4 * profile.min_top_activation_factor
    _min_intersect = profile.min_intersection_count
    if m.intersection_count >= _min_intersect and m.top_activation >= _top_act_threshold_intersect:
        result = SufficiencyResult(
            sufficient=True,
            confidence=conf,
            gate="intersection_convergence",
            reason=(
                f"Multi-anchor convergence: {m.intersection_count} intersections, "
                f"top activation {m.top_activation:.2f}"
            ),
            metrics=m,
        )
        return _apply_calibration(result, calibration)

    # Gate 6: high_coverage_strong_hit
    _top_act_threshold_strong = 0.7 * profile.min_top_activation_factor
    if (
        m.coverage_ratio >= 0.5
        and m.top_activation >= _top_act_threshold_strong
        and m.focus_ratio >= 0.4
    ):
        result = SufficiencyResult(
            sufficient=True,
            confidence=conf,
            gate="high_coverage_strong_hit",
            reason=(
                f"Strong signal: coverage={m.coverage_ratio:.0%}, "
                f"top={m.top_activation:.2f}, focus={m.focus_ratio:.2f}"
            ),
            metrics=m,
        )
        return _apply_calibration(result, calibration)

    # Gate 7: focused_result
    _top_act_threshold_focused = 0.5 * profile.min_top_activation_factor
    if (
        m.neuron_count <= 5
        and m.top_activation >= _top_act_threshold_focused
        and m.focus_ratio >= 0.6
    ):
        result = SufficiencyResult(
            sufficient=True,
            confidence=conf,
            gate="focused_result",
            reason=(
                f"Focused result: {m.neuron_count} neurons, "
                f"top={m.top_activation:.2f}, focus={m.focus_ratio:.2f}"
            ),
            metrics=m,
        )
        return _apply_calibration(result, calibration)

    # Gate 8: default_pass
    result = SufficiencyResult(
        sufficient=True,
        confidence=conf,
        gate="default_pass",
        reason=f"Default pass: {m.neuron_count} neurons, conf={conf:.2f}",
        metrics=m,
    )
    return _apply_calibration(result, calibration)


# ---------------------------------------------------------------------------
# Calibration adjustment
# ---------------------------------------------------------------------------


def _apply_calibration(
    result: SufficiencyResult,
    calibration: dict[str, GateCalibration] | None,
) -> SufficiencyResult:
    """Adjust gate decisions based on historical calibration.

    Three modes:
    1. High-accuracy gate (>0.8, >=10 samples): boost confidence by 10%
       (trust this gate more — it's been reliably correct).
    2. Low-accuracy gate (<0.4, >=10 samples): reduce confidence by 30%
       (this gate often gets it wrong — dampen its signal).
    3. Very low avg_confidence (<0.15, >=10 samples): downgrade to INSUFFICIENT
       (systematic over-prediction — conservative fallback).

    Does not upgrade INSUFFICIENT results (preserves conservative bias).
    """
    if calibration is None or not result.sufficient:
        return result

    gate_cal = calibration.get(result.gate)
    if gate_cal is None:
        return result

    # Downgrade: gate historically predicts sufficient but actual results are weak
    if gate_cal.avg_confidence < 0.15 and gate_cal.sample_count >= 10:
        return SufficiencyResult(
            sufficient=False,
            confidence=max(0.0, min(result.confidence, gate_cal.avg_confidence)),
            gate=result.gate,
            reason=(
                f"{result.reason} [calibration downgrade: "
                f"avg_conf={gate_cal.avg_confidence:.2f}, "
                f"n={gate_cal.sample_count}]"
            ),
            metrics=result.metrics,
        )

    # Low accuracy: dampen confidence (gate often wrong)
    if gate_cal.accuracy < 0.4 and gate_cal.sample_count >= 10:
        dampened = max(0.0, min(1.0, result.confidence * 0.7))
        return SufficiencyResult(
            sufficient=True,
            confidence=dampened,
            gate=result.gate,
            reason=(
                f"{result.reason} [calibration dampen: "
                f"accuracy={gate_cal.accuracy:.2f}, conf {result.confidence:.2f}→{dampened:.2f}]"
            ),
            metrics=result.metrics,
        )

    # High accuracy: boost confidence (gate reliably correct)
    if gate_cal.accuracy > 0.8 and gate_cal.sample_count >= 10:
        boosted = max(0.0, min(1.0, result.confidence * 1.1))
        return SufficiencyResult(
            sufficient=True,
            confidence=boosted,
            gate=result.gate,
            reason=(
                f"{result.reason} [calibration boost: "
                f"accuracy={gate_cal.accuracy:.2f}, conf {result.confidence:.2f}→{boosted:.2f}]"
            ),
            metrics=result.metrics,
        )

    return result
