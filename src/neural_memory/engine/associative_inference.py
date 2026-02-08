"""Associative inference engine — pure-logic module for co-activation analysis.

Given co-activation counts and existing synapse data, produces:
- Inferred synapses (CO_OCCURS type) for frequently co-activated neuron pairs
- Reinforcement candidates for existing synapses
- Associative tags from BFS clustering of inference candidates

Zero storage calls — receives data, returns data. Trivially testable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from neural_memory.core.synapse import Direction, Synapse, SynapseType


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for associative inference."""

    co_activation_threshold: int = 3
    co_activation_window_days: int = 7
    inferred_initial_weight: float = 0.3
    inferred_max_weight: float = 0.8
    weight_scale_factor: float = 0.1
    max_inferences_per_run: int = 50


@dataclass(frozen=True)
class InferenceCandidate:
    """A neuron pair that meets the co-activation threshold for inference."""

    neuron_a: str
    neuron_b: str
    co_activation_count: int
    avg_binding_strength: float
    inferred_weight: float


@dataclass(frozen=True)
class AssociativeTag:
    """A tag inferred from co-activation clusters."""

    tag: str
    source_neuron_ids: frozenset[str]
    confidence: float
    origin: str = "associative"


@dataclass(frozen=True)
class InferenceReport:
    """Summary of an inference run."""

    candidates_evaluated: int
    synapses_created: int
    synapses_reinforced: int
    co_activations_pruned: int


def compute_inferred_weight(
    count: int,
    avg_strength: float,
    config: InferenceConfig,
) -> float:
    """Compute weight for an inferred synapse.

    Linear scaling: base weight + scale_factor * (count - threshold) * avg_strength,
    capped at max_weight.

    Args:
        count: Number of co-activation events
        avg_strength: Average binding strength
        config: Inference configuration

    Returns:
        Weight for the inferred synapse (clamped to [initial, max])
    """
    excess = max(0, count - config.co_activation_threshold)
    weight = config.inferred_initial_weight + config.weight_scale_factor * excess * avg_strength
    return min(config.inferred_max_weight, max(config.inferred_initial_weight, weight))


def identify_candidates(
    counts: list[tuple[str, str, int, float]],
    existing_pairs: set[tuple[str, str]],
    config: InferenceConfig,
) -> tuple[list[InferenceCandidate], list[InferenceCandidate]]:
    """Identify new and reinforcement candidates from co-activation counts.

    Args:
        counts: List of (neuron_a, neuron_b, count, avg_binding_strength)
        existing_pairs: Set of (source_id, target_id) for existing synapses
                       (both directions — callers must include (a,b) and (b,a))
        config: Inference configuration

    Returns:
        Tuple of (new_candidates, reinforce_candidates), each sorted by count desc
        and capped at max_inferences_per_run
    """
    new: list[InferenceCandidate] = []
    reinforce: list[InferenceCandidate] = []

    for neuron_a, neuron_b, count, avg_strength in counts:
        if count < config.co_activation_threshold:
            continue

        weight = compute_inferred_weight(count, avg_strength, config)
        candidate = InferenceCandidate(
            neuron_a=neuron_a,
            neuron_b=neuron_b,
            co_activation_count=count,
            avg_binding_strength=avg_strength,
            inferred_weight=weight,
        )

        pair_exists = (neuron_a, neuron_b) in existing_pairs or (
            neuron_b,
            neuron_a,
        ) in existing_pairs
        if pair_exists:
            reinforce.append(candidate)
        else:
            new.append(candidate)

    new.sort(key=lambda c: c.co_activation_count, reverse=True)
    reinforce.sort(key=lambda c: c.co_activation_count, reverse=True)

    return (
        new[: config.max_inferences_per_run],
        reinforce[: config.max_inferences_per_run],
    )


def create_inferred_synapse(candidate: InferenceCandidate) -> Synapse:
    """Create a CO_OCCURS synapse from an inference candidate.

    The synapse is marked with _inferred metadata for accelerated decay
    during pruning if it goes unreinforced.

    Args:
        candidate: The inference candidate

    Returns:
        A new Synapse instance
    """
    return Synapse.create(
        source_id=candidate.neuron_a,
        target_id=candidate.neuron_b,
        type=SynapseType.CO_OCCURS,
        weight=candidate.inferred_weight,
        direction=Direction.BIDIRECTIONAL,
        metadata={
            "_inferred": True,
            "co_activation_count": candidate.co_activation_count,
            "avg_binding_strength": candidate.avg_binding_strength,
        },
    )


def generate_associative_tags(
    candidates: list[InferenceCandidate],
    neuron_content_map: dict[str, str],
    existing_tags: set[str],
) -> list[AssociativeTag]:
    """Generate associative tags from inference candidates via BFS clustering.

    Groups candidates into connected clusters, then derives a tag name
    from the most common words in neuron content within each cluster.

    Args:
        candidates: Inference candidates (new + reinforce)
        neuron_content_map: Mapping of neuron_id -> content text
        existing_tags: Tags already present (will not be re-generated)

    Returns:
        List of new associative tags
    """
    if not candidates:
        return []

    # Build adjacency graph from candidates
    adjacency: dict[str, set[str]] = defaultdict(set)
    for c in candidates:
        adjacency[c.neuron_a].add(c.neuron_b)
        adjacency[c.neuron_b].add(c.neuron_a)

    # BFS clustering
    visited: set[str] = set()
    clusters: list[set[str]] = []

    for node in adjacency:
        if node in visited:
            continue
        cluster: set[str] = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            cluster.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(cluster) >= 2:
            clusters.append(cluster)

    # Generate tags from clusters
    tags: list[AssociativeTag] = []
    existing_lower = {t.lower() for t in existing_tags}

    for cluster in clusters:
        # Collect content words from neurons in cluster
        words: list[str] = []
        for nid in cluster:
            content = neuron_content_map.get(nid, "")
            words.extend(w.lower().strip() for w in content.split() if len(w.strip()) >= 3)

        if not words:
            continue

        # Pick most frequent word as tag name
        word_counts: dict[str, int] = defaultdict(int)
        for w in words:
            word_counts[w] += 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        tag_name = sorted_words[0][0]

        if tag_name in existing_lower:
            continue

        # Confidence based on cluster size and total candidates
        confidence = min(0.9, 0.3 + 0.1 * len(cluster))

        tags.append(
            AssociativeTag(
                tag=tag_name,
                source_neuron_ids=frozenset(cluster),
                confidence=confidence,
            )
        )

    return tags
