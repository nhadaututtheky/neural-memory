"""Abstraction-level constraint for spreading activation.

Neurons are assigned an abstraction level based on their type:
  Level 1 (concrete): TIME, SPATIAL, ENTITY, ACTION, STATE, SENSORY
  Level 2 (abstract): CONCEPT, INTENT
  Level 3 (meta):     HYPOTHESIS, PREDICTION, SCHEMA

Level 0 means unassigned — unconstrained by default.

The constraint gate in SpreadingActivation uses can_activate() to skip
neighbors whose level distance from the current node exceeds the configured
max_distance. Both nodes must have a non-zero level for the check to fire.

Dynamic abstraction:
  induce_abstraction(cluster) condenses N episodic neurons into one semantic
  (Level 2) neuron using term-frequency extraction across the cluster. CLS
  pattern — the cluster is NOT fully replaced; callers keep 1-2 exemplars.
"""

from __future__ import annotations

import re
from collections import Counter

from neural_memory.core.neuron import Neuron, NeuronType

# English stopwords — kept inline to avoid adding nltk/spacy as a hard dep.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "now",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "as",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")

# Canonical abstraction level per NeuronType.
# Level 0 = unassigned (not listed here; caller falls back to 0 via .get()).
DEFAULT_ABSTRACTION_LEVELS: dict[NeuronType, int] = {
    # Level 1 — concrete, grounded in experience
    NeuronType.TIME: 1,
    NeuronType.SPATIAL: 1,
    NeuronType.ENTITY: 1,
    NeuronType.ACTION: 1,
    NeuronType.STATE: 1,
    NeuronType.SENSORY: 1,
    # Level 2 — abstract / intentional
    NeuronType.CONCEPT: 2,
    NeuronType.INTENT: 2,
    # Level 3 — meta / epistemic
    NeuronType.HYPOTHESIS: 3,
    NeuronType.PREDICTION: 3,
    NeuronType.SCHEMA: 3,
}


def assign_abstraction_level(neuron: Neuron) -> Neuron:
    """Return a neuron with its abstraction level set from DEFAULT_ABSTRACTION_LEVELS.

    If the neuron already has a non-zero abstraction level in its metadata,
    it is returned unchanged (preserves explicit overrides).

    Args:
        neuron: The source neuron (never mutated).

    Returns:
        New Neuron with _abstraction_level set, or the same neuron if already set.
    """
    if neuron.abstraction_level != 0:
        return neuron

    level = DEFAULT_ABSTRACTION_LEVELS.get(neuron.type, 0)
    if level == 0:
        return neuron

    return neuron.with_abstraction_level(level)


def can_activate(source: Neuron, target: Neuron, max_distance: int = 1) -> bool:
    """Return True if spreading activation may cross from source to target.

    The constraint only fires when *both* neurons have a non-zero abstraction
    level. If either is 0 (unassigned), activation is always allowed.

    Args:
        source: The neuron currently being spread from.
        target: The candidate neighbor neuron.
        max_distance: Maximum allowed difference in abstraction levels (default 1).

    Returns:
        True if activation may flow from source to target.
    """
    src_level = source.abstraction_level
    dst_level = target.abstraction_level

    if src_level == 0 or dst_level == 0:
        return True

    return abs(src_level - dst_level) <= max_distance


def _extract_top_terms(contents: list[str], top_n: int = 5) -> list[str]:
    """Tokenize and return the top-N most frequent non-stopword terms."""
    counter: Counter[str] = Counter()
    for content in contents:
        for token in _TOKEN_RE.findall(content.lower()):
            if len(token) < 3 or token in _STOPWORDS:
                continue
            counter[token] += 1
    return [term for term, _ in counter.most_common(top_n)]


def _dominant_type(cluster: list[Neuron]) -> NeuronType:
    """Return the most common NeuronType in the cluster."""
    counter: Counter[NeuronType] = Counter(n.type for n in cluster)
    return counter.most_common(1)[0][0]


def _highest_priority_neuron(cluster: list[Neuron]) -> Neuron:
    """Return the neuron with the highest goal_priority (ties broken by recency)."""
    return max(cluster, key=lambda n: (n.goal_priority, n.created_at))


def induce_abstraction(cluster: list[Neuron]) -> Neuron:
    """Condense a cluster of episodic neurons into one abstract (Level 2) neuron.

    Uses simple term-frequency extraction (stopword-filtered) to identify the
    topic and representative terms. The abstract neuron's content follows
    the template:

        "[N] memories about [TOPIC]: [TERMS]. Key: [highest-priority content]"

    Callers (e.g. the MERGE consolidation strategy) are responsible for keeping
    1-2 episodic exemplars alive — the CLS pattern requires episodes to survive
    so semantic memories have grounding.

    Args:
        cluster: At least 2 neurons to abstract from. Never mutated.

    Returns:
        A new CONCEPT-type Neuron with abstraction_level=2.

    Raises:
        ValueError: If cluster is empty.
    """
    if not cluster:
        raise ValueError("Cannot induce abstraction from an empty cluster")

    contents = [n.content for n in cluster if n.content]
    top_terms = _extract_top_terms(contents, top_n=5)
    dominant = _dominant_type(cluster)
    exemplar = _highest_priority_neuron(cluster)

    topic = dominant.value.replace("_", " ")
    terms_str = ", ".join(top_terms) if top_terms else "mixed"
    key_snippet = (exemplar.content or "").strip()
    if len(key_snippet) > 140:
        key_snippet = key_snippet[:137] + "..."

    summary = f"{len(cluster)} memories about {topic}: {terms_str}. Key: {key_snippet}"

    abstract = Neuron.create(
        type=NeuronType.CONCEPT,
        content=summary,
        metadata={
            "_abstraction_source_ids": [n.id for n in cluster],
            "_abstraction_terms": top_terms,
            "_abstraction_exemplar_id": exemplar.id,
            "_abstraction_induced": True,
        },
    )
    return abstract.with_abstraction_level(2)
