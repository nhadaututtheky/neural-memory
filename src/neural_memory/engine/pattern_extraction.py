"""Pattern extraction — episodic → semantic concept formation.

When consolidation runs with MATURE strategy, this module:
1. Filters fibers in EPISODIC stage with sufficient rehearsal
2. Clusters fibers by tag Jaccard similarity
3. For clusters of 3+ fibers: extracts common entities → CONCEPT neuron

This models how the brain forms semantic memory: recurring episodic
experiences (specific events) generalize into concepts (abstract knowledge).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.clustering import UnionFind
from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber


@dataclass(frozen=True)
class ExtractedPattern:
    """A pattern extracted from recurring episodic fibers.

    Attributes:
        concept_neuron: The new CONCEPT neuron representing the pattern
        synapses: Synapses connecting concept to contributing entities
        source_fiber_ids: IDs of the fibers that contributed
        common_tags: Tags shared across all contributing fibers
        common_entities: Entity content shared across fibers
    """

    concept_neuron: Neuron
    synapses: list[Synapse]
    source_fiber_ids: list[str]
    common_tags: set[str]
    common_entities: list[str]


@dataclass
class ExtractionReport:
    """Report from pattern extraction.

    Attributes:
        fibers_analyzed: Number of eligible fibers examined
        clusters_found: Number of tag-similarity clusters
        patterns_extracted: Number of patterns generated
        concepts_created: Number of CONCEPT neurons created
    """

    fibers_analyzed: int = 0
    clusters_found: int = 0
    patterns_extracted: int = 0
    concepts_created: int = 0


def extract_patterns(
    fibers: list[Fiber],
    maturations: dict[str, MaturationRecord],
    min_rehearsal_count: int = 3,
    min_cluster_size: int = 3,
    tag_overlap_threshold: float = 0.5,
) -> tuple[list[ExtractedPattern], ExtractionReport]:
    """Extract semantic patterns from episodic fibers.

    Finds recurring themes in episodic memories and creates
    CONCEPT neurons representing generalized knowledge.

    Args:
        fibers: All fibers to consider
        maturations: Maturation records keyed by fiber_id
        min_rehearsal_count: Minimum rehearsals to be eligible
        min_cluster_size: Minimum fibers per cluster to extract pattern
        tag_overlap_threshold: Minimum Jaccard similarity for clustering

    Returns:
        Tuple of (extracted patterns, extraction report)
    """
    report = ExtractionReport()

    # Filter to eligible episodic fibers
    eligible = [
        f
        for f in fibers
        if f.id in maturations
        and maturations[f.id].stage == MemoryStage.EPISODIC
        and maturations[f.id].rehearsal_count >= min_rehearsal_count
        and f.tags  # Must have tags for clustering
    ]
    report.fibers_analyzed = len(eligible)

    if len(eligible) < min_cluster_size:
        return [], report

    # Cluster by tag Jaccard similarity (Union-Find)
    clusters = _cluster_by_tags(eligible, tag_overlap_threshold)
    report.clusters_found = len(clusters)

    # Extract patterns from qualifying clusters
    patterns: list[ExtractedPattern] = []
    for cluster_fibers in clusters:
        if len(cluster_fibers) < min_cluster_size:
            continue

        pattern = _extract_from_cluster(cluster_fibers)
        if pattern is not None:
            patterns.append(pattern)
            report.patterns_extracted += 1
            report.concepts_created += 1

    return patterns, report


def _cluster_by_tags(
    fibers: list[Fiber],
    threshold: float,
) -> list[list[Fiber]]:
    """Cluster fibers by tag Jaccard similarity using Union-Find."""
    n = len(fibers)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            tags_a = fibers[i].tags
            tags_b = fibers[j].tags
            intersection = len(tags_a & tags_b)
            union_size = len(tags_a | tags_b)
            if union_size > 0 and intersection / union_size >= threshold:
                uf.union(i, j)

    # Group by root, return Fiber objects
    return [[fibers[i] for i in indices] for indices in uf.groups().values()]


def _extract_from_cluster(fibers: list[Fiber]) -> ExtractedPattern | None:
    """Extract a semantic pattern from a cluster of similar fibers.

    Finds common tags and entities, creates a CONCEPT neuron
    representing the generalized knowledge.
    """
    if not fibers:
        return None

    # Find common tags across all fibers
    common_tags = set(fibers[0].tags)
    for fiber in fibers[1:]:
        common_tags &= fiber.tags

    if not common_tags:
        return None

    # Find common entities by collecting all neuron IDs and
    # counting how many fibers each appears in
    neuron_counts: dict[str, int] = {}
    for fiber in fibers:
        for nid in fiber.neuron_ids:
            neuron_counts[nid] = neuron_counts.get(nid, 0) + 1

    # Entities appearing in majority of fibers
    majority_threshold = len(fibers) * 0.6
    common_entity_ids = [nid for nid, count in neuron_counts.items() if count >= majority_threshold]

    if not common_entity_ids:
        return None

    # Create CONCEPT neuron
    concept_content = f"Pattern: {', '.join(sorted(common_tags))}"
    concept_id = str(uuid4())
    concept_neuron = Neuron(
        id=concept_id,
        type=NeuronType.CONCEPT,
        content=concept_content,
        metadata={
            "source_fiber_count": len(fibers),
            "common_tags": sorted(common_tags),
            "extracted_at": utcnow().isoformat(),
        },
        created_at=utcnow(),
    )

    # Create IS_A synapses from common entities to concept
    synapses = [
        Synapse.create(
            source_id=entity_id,
            target_id=concept_id,
            type=SynapseType.IS_A,
            weight=0.6,
        )
        for entity_id in common_entity_ids[:10]  # Cap at 10 synapses
    ]

    return ExtractedPattern(
        concept_neuron=concept_neuron,
        synapses=synapses,
        source_fiber_ids=[f.id for f in fibers],
        common_tags=common_tags,
        common_entities=common_entity_ids,
    )
