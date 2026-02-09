"""Activation stabilization — iterative dampening until convergence.

After spreading activation and lateral inhibition, the activation
landscape may still contain noise and imbalanced peaks. Stabilization
settles the network into a stable attractor state through:

1. Noise floor: zero out sub-threshold activations
2. Dampening: global decay factor (like PageRank dampening)
3. Homeostatic normalization: soft-scale mean toward target
4. Convergence check: stop when activations stabilize

This mimics how biological neural networks settle into stable
firing patterns after initial excitation.
"""

from __future__ import annotations

from dataclasses import dataclass

from neural_memory.engine.activation import ActivationResult


@dataclass(frozen=True)
class StabilizationConfig:
    """Configuration for activation stabilization.

    Attributes:
        max_iterations: Safety cap on stabilization rounds
        noise_floor: Activations below this are zeroed
        dampening_factor: Global decay per iteration (like PageRank)
        homeostatic_target: Target mean activation level
        homeostatic_strength: How strongly to pull toward target (0-1)
        convergence_threshold: Max change to consider converged
    """

    max_iterations: int = 10
    noise_floor: float = 0.05
    dampening_factor: float = 0.85
    homeostatic_target: float = 0.5
    homeostatic_strength: float = 0.3
    convergence_threshold: float = 0.01


@dataclass(frozen=True)
class StabilizationReport:
    """Report from a stabilization run.

    Attributes:
        iterations: Number of iterations performed
        converged: Whether convergence was reached
        neurons_removed: Number of neurons dropped by noise floor
        max_delta: Maximum activation change in final iteration
    """

    iterations: int
    converged: bool
    neurons_removed: int
    max_delta: float


def stabilize(
    activations: dict[str, ActivationResult],
    config: StabilizationConfig | None = None,
) -> tuple[dict[str, ActivationResult], StabilizationReport]:
    """Stabilize activation landscape through iterative dampening.

    Process per iteration:
    1. Noise floor: zero activations below threshold
    2. Dampening: multiply all activations by dampening factor
    3. Homeostatic norm: soft-scale mean toward target
    4. Convergence: if max|delta| < threshold, stop

    Args:
        activations: Current activation results from the pipeline
        config: Stabilization parameters (uses defaults if None)

    Returns:
        Tuple of (stabilized activations, stabilization report)
    """
    if config is None:
        config = StabilizationConfig()

    if not activations:
        return {}, StabilizationReport(
            iterations=0,
            converged=True,
            neurons_removed=0,
            max_delta=0.0,
        )

    # Work with mutable levels, rebuild immutable results at end
    levels: dict[str, float] = {nid: act.activation_level for nid, act in activations.items()}
    total_removed = 0
    max_delta = 0.0

    for iteration in range(config.max_iterations):
        prev_levels = dict(levels)

        # Step 1: Noise floor — zero out sub-threshold activations
        to_remove = [nid for nid, lv in levels.items() if lv < config.noise_floor]
        for nid in to_remove:
            del levels[nid]
        total_removed += len(to_remove)

        if not levels:
            return {}, StabilizationReport(
                iterations=iteration + 1,
                converged=True,
                neurons_removed=total_removed,
                max_delta=0.0,
            )

        # Step 2: Dampening — global decay
        levels = {nid: lv * config.dampening_factor for nid, lv in levels.items()}

        # Step 3: Homeostatic normalization — soft-scale mean toward target
        mean_level = sum(levels.values()) / len(levels)
        if mean_level > 0.0:
            scale = 1.0 + config.homeostatic_strength * (
                config.homeostatic_target / mean_level - 1.0
            )
            scale = min(scale, 3.0)  # Cap homeostatic boost
            levels = {nid: max(0.0, min(1.0, lv * scale)) for nid, lv in levels.items()}

        # Step 4: Convergence check
        max_delta = 0.0
        for nid in levels:
            old = prev_levels.get(nid, 0.0)
            delta = abs(levels[nid] - old)
            if delta > max_delta:
                max_delta = delta

        if max_delta < config.convergence_threshold:
            return _rebuild_activations(activations, levels), StabilizationReport(
                iterations=iteration + 1,
                converged=True,
                neurons_removed=total_removed,
                max_delta=max_delta,
            )

    # Did not converge within max_iterations
    return _rebuild_activations(activations, levels), StabilizationReport(
        iterations=config.max_iterations,
        converged=False,
        neurons_removed=total_removed,
        max_delta=max_delta,
    )


def _rebuild_activations(
    original: dict[str, ActivationResult],
    levels: dict[str, float],
) -> dict[str, ActivationResult]:
    """Rebuild immutable ActivationResult objects with updated levels."""
    return {
        nid: ActivationResult(
            neuron_id=nid,
            activation_level=level,
            hop_distance=original[nid].hop_distance,
            path=original[nid].path,
            source_anchor=original[nid].source_anchor,
        )
        for nid, level in levels.items()
        if nid in original
    }
