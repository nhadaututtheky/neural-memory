"""Auto-resolution for trivial memory conflicts.

Applies conservative rules to automatically resolve conflicts that
don't require human judgment. All auto-resolutions are auditable via
metadata on the CONTRADICTS synapse.

Rules (in priority order):
1. New memory confidence >= 0.8 AND existing is STALE/ANCIENT -> keep_new
2. Same session (within 1 hour) AND new content is more specific -> keep_new
3. Existing neuron already superseded 2+ times -> keep_new
4. Safety guard: both FRESH and high-confidence -> require manual
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neural_memory.safety.freshness import FreshnessLevel, evaluate_freshness
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.engine.conflict_detection import Conflict
    from neural_memory.storage.base import NeuralStorage


@dataclass(frozen=True)
class AutoResolution:
    """Result of attempting auto-resolution on a conflict.

    Attributes:
        conflict: The original conflict
        resolution: Resolution action ("keep_new", "keep_existing", or "")
        reason: Human-readable reason for the resolution
        auto_resolved: Whether it was auto-resolved (False = needs manual)
    """

    conflict: Conflict
    resolution: str
    reason: str
    auto_resolved: bool


async def try_auto_resolve(
    conflict: Conflict,
    storage: NeuralStorage,
    new_confidence: float = 0.5,
) -> AutoResolution:
    """Attempt to auto-resolve a conflict using conservative rules.

    Args:
        conflict: The detected conflict
        storage: Storage backend for querying existing neuron state
        new_confidence: Confidence of the new memory (default 0.5)

    Returns:
        AutoResolution with resolution details
    """
    existing_neuron = await storage.get_neuron(conflict.existing_neuron_id)
    if existing_neuron is None:
        return AutoResolution(
            conflict=conflict,
            resolution="keep_new",
            reason="existing neuron no longer exists",
            auto_resolved=True,
        )

    # Rule 0 (highest priority): Grounded neurons always win
    if existing_neuron.grounded:
        return AutoResolution(
            conflict=conflict,
            resolution="keep_existing",
            reason="existing neuron is grounded (canonical truth)",
            auto_resolved=True,
        )

    # Get creation time and freshness of existing neuron
    existing_created_at = existing_neuron.created_at
    now = utcnow()

    existing_freshness = evaluate_freshness(existing_created_at, now)

    # Safety guard: if both memories are FRESH and high-confidence, require manual
    if existing_freshness.level == FreshnessLevel.FRESH and new_confidence >= 0.7:
        existing_state = await storage.get_neuron_state(conflict.existing_neuron_id)
        existing_activation = existing_state.activation_level if existing_state else 0.5
        if existing_activation >= 0.5:
            return AutoResolution(
                conflict=conflict,
                resolution="",
                reason="both memories are fresh and high-confidence",
                auto_resolved=False,
            )

    # Rule 1: New is high-confidence AND existing is STALE/ANCIENT
    if new_confidence >= 0.8 and existing_freshness.level in (
        FreshnessLevel.STALE,
        FreshnessLevel.ANCIENT,
    ):
        return AutoResolution(
            conflict=conflict,
            resolution="keep_new",
            reason=f"new memory is high-confidence ({new_confidence:.2f}) "
            f"and existing is {existing_freshness.level.value} "
            f"({existing_freshness.age_days} days old)",
            auto_resolved=True,
        )

    # Rule 2: Same session (within 1 hour) AND new is more specific (longer)
    age_seconds = (now - existing_created_at).total_seconds()
    if age_seconds < 3600 and len(conflict.new_content) > len(conflict.existing_content):
        return AutoResolution(
            conflict=conflict,
            resolution="keep_new",
            reason="same session correction — new content is more specific",
            auto_resolved=True,
        )

    # Rule 3: Existing neuron already superseded 2+ times
    superseded_count = _count_superseded(existing_neuron.metadata)
    if superseded_count >= 2:
        return AutoResolution(
            conflict=conflict,
            resolution="keep_new",
            reason=f"existing neuron already superseded {superseded_count} times",
            auto_resolved=True,
        )

    # No rule matched — require manual resolution
    return AutoResolution(
        conflict=conflict,
        resolution="",
        reason="no auto-resolve rule matched",
        auto_resolved=False,
    )


def _count_superseded(metadata: dict[str, object]) -> int:
    """Count how many times a neuron has been disputed/superseded.

    Checks for _disputed_by chain in metadata.
    """
    count = 0
    if metadata.get("_disputed"):
        count += 1
    if metadata.get("_superseded"):
        count += 1
    # Check for historical dispute count if tracked
    hist = metadata.get("_dispute_count")
    if isinstance(hist, int):
        count = max(count, hist)
    return count
