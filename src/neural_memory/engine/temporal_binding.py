"""Session-level temporal binding — auto-link memories created close in time.

Complements TemporalLinkingStep (24h window, directional BEFORE/AFTER synapses)
with short-window session binding (default 5 min, CO_OCCURS synapses with
proximity-weighted strength).

Neuroscience basis: theta-phase coupling — events close in time are
automatically linked by the hippocampus, creating contextual associations
even when content has no semantic overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.pipeline import PipelineContext

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Maximum bindings per encoding to prevent synapse explosion
_MAX_BINDINGS = 3


@dataclass
class TemporalBindingStep:
    """Auto-link memories created within a short time window.

    Creates CO_OCCURS synapses between anchor neurons of fibers
    encoded within ``temporal_binding_window_seconds`` of each other.
    Weight is proportional to temporal proximity (closer = stronger).

    Must run AFTER BuildFiberStep (needs the fiber's timestamp).
    """

    @property
    def name(self) -> str:
        return "temporal_binding"

    async def execute(
        self,
        ctx: PipelineContext,
        storage: NeuralStorage,
        config: BrainConfig,
    ) -> PipelineContext:
        if not getattr(config, "temporal_binding_enabled", True):
            return ctx

        if ctx.anchor_neuron is None:
            return ctx

        raw = getattr(config, "temporal_binding_window_seconds", 300.0)
        window = float(raw) if isinstance(raw, (int, float)) else 300.0
        start = ctx.timestamp - timedelta(seconds=window)
        end = ctx.timestamp

        nearby_fibers = await storage.find_fibers(
            time_overlaps=(start, end),
            limit=_MAX_BINDINGS + 1,  # +1 for potential self-match
        )

        created = 0
        for fiber in nearby_fibers:
            if created >= _MAX_BINDINGS:
                break

            # Skip self
            if fiber.anchor_neuron_id == ctx.anchor_neuron.id:
                continue

            # Compute proximity weight: closer in time = stronger link
            if fiber.time_start is not None:
                gap = abs((ctx.timestamp - fiber.time_start).total_seconds())
            else:
                gap = window  # fallback: max distance

            proximity = 1.0 - (gap / window) if window > 0 else 0.0
            weight = max(0.1, proximity * 0.4)  # cap at 0.4, floor at 0.1

            synapse = Synapse.create(
                source_id=ctx.anchor_neuron.id,
                target_id=fiber.anchor_neuron_id,
                type=SynapseType.CO_OCCURS,
                weight=weight,
                metadata={
                    "temporal_binding": True,
                    "gap_seconds": round(gap, 1),
                },
            )
            try:
                await storage.add_synapse(synapse)
                ctx.synapses_created.append(synapse)
                created += 1
            except ValueError:
                logger.debug("Temporal binding synapse already exists, skipping")

        if created > 0:
            logger.debug("Temporal binding: created %d CO_OCCURS synapses", created)

        return ctx
