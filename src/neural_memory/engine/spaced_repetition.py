"""Spaced repetition engine — Leitner box review scheduling.

Manages review queues, processes reviews (reinforcing/decaying via Hebbian
system), and auto-schedules high-priority memories.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.core.review_schedule import ReviewSchedule
from neural_memory.engine.lifecycle import ReinforcementManager

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Auto-schedule fibers with priority >= this threshold
AUTO_SCHEDULE_PRIORITY = 7


class SpacedRepetitionEngine:
    """Leitner-box spaced repetition for memory reinforcement."""

    def __init__(self, storage: NeuralStorage, config: BrainConfig) -> None:
        self._storage = storage
        self._config = config
        self._reinforcer = ReinforcementManager(
            reinforcement_delta=config.reinforcement_delta,
        )

    async def get_review_queue(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get fibers due for review with their context.

        Returns list of dicts with schedule info + fiber summary.
        """
        safe_limit = min(limit, 100)
        due = await self._storage.get_due_reviews(limit=safe_limit)
        items: list[dict[str, Any]] = []

        for schedule in due:
            fiber = await self._storage.get_fiber(schedule.fiber_id)
            summary = ""
            if fiber:
                summary = fiber.summary or ""
                if not summary and fiber.neuron_ids:
                    # Try to get content from anchor neuron
                    anchor = await self._storage.get_neuron(fiber.anchor_neuron_id)
                    if anchor:
                        summary = anchor.content[:200]

            items.append(
                {
                    "fiber_id": schedule.fiber_id,
                    "box": schedule.box,
                    "streak": schedule.streak,
                    "review_count": schedule.review_count,
                    "next_review": schedule.next_review.isoformat()
                    if schedule.next_review
                    else None,
                    "summary": summary,
                }
            )

        return items

    async def process_review(
        self,
        fiber_id: str,
        success: bool,
    ) -> dict[str, Any]:
        """Process a review result for a fiber.

        On success: advance box, reinforce neuron states via Hebbian system.
        On failure: reset to box 1, let natural decay handle it.

        Returns dict with new schedule state.
        """
        schedule = await self._storage.get_review_schedule(fiber_id)
        if schedule is None:
            return {"error": f"No review schedule for fiber {fiber_id}"}

        # Advance the schedule
        new_schedule = schedule.advance(success)
        await self._storage.add_review_schedule(new_schedule)

        # Reinforce neurons on successful recall
        reinforced = 0
        if success:
            fiber = await self._storage.get_fiber(fiber_id)
            if fiber:
                neuron_ids = list(fiber.neuron_ids)
                synapse_ids = list(fiber.synapse_ids)
                reinforced = await self._reinforcer.reinforce(
                    self._storage, neuron_ids, synapse_ids
                )
                # Also boost fiber conductivity
                conducted = fiber.conduct(reinforce=True)
                await self._storage.update_fiber(conducted)

        return {
            "fiber_id": fiber_id,
            "success": success,
            "previous_box": schedule.box,
            "new_box": new_schedule.box,
            "streak": new_schedule.streak,
            "next_review": new_schedule.next_review.isoformat()
            if new_schedule.next_review
            else None,
            "neurons_reinforced": reinforced,
        }

    async def auto_schedule_fiber(
        self,
        fiber_id: str,
        brain_id: str,
    ) -> ReviewSchedule | None:
        """Auto-schedule a fiber for review if not already scheduled.

        Returns the created schedule, or None if already scheduled.
        """
        existing = await self._storage.get_review_schedule(fiber_id)
        if existing is not None:
            return None

        schedule = ReviewSchedule.create(fiber_id=fiber_id, brain_id=brain_id)
        await self._storage.add_review_schedule(schedule)
        logger.debug("Auto-scheduled fiber %s for review", fiber_id)
        return schedule

    async def get_stats(self) -> dict[str, int]:
        """Get review statistics."""
        return await self._storage.get_review_stats()
