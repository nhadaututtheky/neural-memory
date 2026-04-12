"""Layer routing engine — route memories to correct brain layer.

Implements the Layered Consciousness vision (Vision Pillar 6):
- Session layer (ephemeral, in-memory) — handled by ephemeral flag
- Project layer (current brain) — context-dependent knowledge
- Global layer (_global brain) — cross-project preferences/instructions

Routing is automatic based on memory type, with optional explicit override.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryLayer(Enum):
    """Memory storage layer."""

    SESSION = "session"  # Ephemeral, in-memory only
    PROJECT = "project"  # Current project brain
    GLOBAL = "global"  # Cross-project _global brain


# Memory types that belong in the global layer
_GLOBAL_TYPES: frozenset[str] = frozenset({"preference", "instruction"})


@dataclass(frozen=True)
class LayerDecision:
    """Result of layer routing decision."""

    layer: MemoryLayer
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"layer": self.layer.value, "reason": self.reason}


def route_memory(
    memory_type: str,
    *,
    is_ephemeral: bool = False,
    explicit_layer: str | None = None,
    tags: set[str] | None = None,
) -> LayerDecision:
    """Determine which layer a memory should be saved to.

    Routing priority:
    1. Explicit `layer=` override (user/agent control)
    2. Ephemeral flag → SESSION layer
    3. Type-based routing: preference/instruction → GLOBAL
    4. Default → PROJECT

    Args:
        memory_type: Memory type string (e.g. "preference", "decision").
        is_ephemeral: Whether memory is ephemeral (auto-expires).
        explicit_layer: Explicit layer override ("session", "project", "global").
        tags: Memory tags (reserved for future tag-based routing).

    Returns:
        LayerDecision with target layer and routing reason.
    """
    # 1. Explicit override
    if explicit_layer:
        layer_str = explicit_layer.lower().strip()
        try:
            layer = MemoryLayer(layer_str)
            return LayerDecision(layer=layer, reason=f"explicit layer={layer_str}")
        except ValueError:
            logger.warning("Invalid explicit layer '%s', falling back to auto-routing", layer_str)

    # 2. Ephemeral → SESSION
    if is_ephemeral:
        return LayerDecision(layer=MemoryLayer.SESSION, reason="ephemeral memory")

    # 3. Type-based routing
    if memory_type in _GLOBAL_TYPES:
        return LayerDecision(
            layer=MemoryLayer.GLOBAL,
            reason=f"type '{memory_type}' routes to global",
        )

    # 4. Default → PROJECT
    return LayerDecision(layer=MemoryLayer.PROJECT, reason="default project routing")
