"""Causal auto-inclusion for goal-directed recall.

After the retrieval pipeline selects top fibers, this module traces
CAUSED_BY/LEADS_TO synapses from the selected neurons to gather
supporting causal context. The causal supplement is appended to
the retrieval result as metadata, capped by a token budget.

Design:
- Post-selection supplement: runs AFTER fiber ranking, not before
- Reuses trace_causal_chain() from causal_traversal.py
- Bounded cost: max 2 hops per fiber, max 10 fibers traced
- Deduplicates across fibers (same causal neuron found via two paths)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neural_memory.engine.causal_traversal import CausalChain, trace_causal_chain

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FIBERS_TO_TRACE = 10
MAX_CAUSAL_SUPPLEMENT_CHARS = 2000  # ~500 tokens at 4 chars/token


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalContext:
    """Aggregated causal context from multiple fiber traces.

    Attributes:
        chains: All unique causal chains discovered
        neuron_ids: Set of all causal neuron IDs (for dedup)
        supplement_text: Formatted text for injection into context
    """

    chains: tuple[CausalChain, ...]
    neuron_ids: frozenset[str]
    supplement_text: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


async def gather_causal_context(
    storage: NeuralStorage,
    fiber_neuron_ids: list[list[str]],
    max_hops: int = 2,
    max_tokens_budget: int = 0,
    exclude_neuron_ids: set[str] | None = None,
) -> CausalContext:
    """Trace causal chains from selected fibers' neurons.

    For each fiber's neuron IDs, traces both causes and effects
    up to max_hops. Deduplicates across fibers and against
    exclude_neuron_ids (e.g. neurons already in main results).

    Args:
        storage: Storage backend (brain context must be set)
        fiber_neuron_ids: List of neuron ID lists, one per selected fiber
        max_hops: Maximum causal hops per trace (default 2)
        max_tokens_budget: Max chars for supplement text (0 = use default)
        exclude_neuron_ids: Neuron IDs already in results (skip in supplement)

    Returns:
        CausalContext with aggregated chains and formatted supplement
    """
    budget = max_tokens_budget if max_tokens_budget > 0 else MAX_CAUSAL_SUPPLEMENT_CHARS
    seen_neuron_ids: set[str] = set(exclude_neuron_ids) if exclude_neuron_ids else set()
    all_chains: list[CausalChain] = []

    # Flatten seed neuron IDs from top fibers (bounded)
    seed_ids: list[str] = []
    for neuron_ids in fiber_neuron_ids[:MAX_FIBERS_TO_TRACE]:
        for nid in neuron_ids[:3]:  # max 3 neurons per fiber
            if nid not in seen_neuron_ids:
                seed_ids.append(nid)
                seen_neuron_ids.add(nid)

    # Trace causes and effects for each seed
    for seed_id in seed_ids:
        for direction in ("causes", "effects"):
            try:
                chain = await trace_causal_chain(
                    storage,
                    seed_neuron_id=seed_id,
                    direction=direction,
                    max_depth=max_hops,
                    min_weight=0.1,
                )
                if chain.steps:
                    # Only keep chains with novel neurons
                    novel_steps = tuple(
                        s for s in chain.steps if s.neuron_id not in seen_neuron_ids
                    )
                    if novel_steps:
                        all_chains.append(chain)
                        for step in novel_steps:
                            seen_neuron_ids.add(step.neuron_id)
            except Exception:
                logger.debug(
                    "Causal trace failed for %s/%s (non-critical)",
                    seed_id,
                    direction,
                    exc_info=True,
                )

    # Format supplement text
    supplement = format_causal_supplement(all_chains, budget)

    return CausalContext(
        chains=tuple(all_chains),
        neuron_ids=frozenset(seen_neuron_ids),
        supplement_text=supplement,
    )


def format_causal_supplement(chains: list[CausalChain], max_chars: int = 0) -> str:
    """Format causal chains into a readable supplement string.

    Args:
        chains: List of CausalChain objects
        max_chars: Maximum output length (0 = use default)

    Returns:
        Formatted string describing the causal context
    """
    if not chains:
        return ""

    budget = max_chars if max_chars > 0 else MAX_CAUSAL_SUPPLEMENT_CHARS
    parts: list[str] = []
    used = 0
    seen_ids: set[str] = set()

    for chain in chains:
        if not chain.steps:
            continue

        label = "Caused by" if chain.direction == "causes" else "Led to"
        for step in chain.steps:
            if step.neuron_id in seen_ids:
                continue
            seen_ids.add(step.neuron_id)
            line = f"[{label}] {step.content}"
            if used + len(line) + 1 > budget:
                break
            parts.append(line)
            used += len(line) + 1

        if used >= budget:
            break

    return "\n".join(parts)
