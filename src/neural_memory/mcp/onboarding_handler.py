"""Fresh-brain onboarding handler for MCP server.

Detects empty brains on the first tool call and injects structured
onboarding hints to guide the agent through initial setup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnboardingHint:
    """A single onboarding guidance step."""

    step: int
    title: str
    description: str
    example_tool: str
    example_args: dict[str, Any] = field(default_factory=dict)


ONBOARDING_STEPS: tuple[OnboardingHint, ...] = (
    OnboardingHint(
        step=1,
        title="Store your first memory",
        description="Save a project fact, decision, or preference to seed the brain.",
        example_tool="nmem_remember",
        example_args={"content": "Project uses Python 3.12 with FastAPI", "type": "fact"},
    ),
    OnboardingHint(
        step=2,
        title="Enable auto-capture",
        description="Let NeuralMemory automatically detect and save decisions, errors, and TODOs from conversation text.",
        example_tool="nmem_auto",
        example_args={"action": "process", "text": "<paste conversation text>"},
    ),
    OnboardingHint(
        step=3,
        title="Set up session tracking",
        description="Track your current task and feature for context-aware recall across sessions.",
        example_tool="nmem_session",
        example_args={"action": "set", "feature": "your-feature", "task": "your-task"},
    ),
    OnboardingHint(
        step=4,
        title="Index your codebase",
        description="Scan source files so recall can find relevant code locations.",
        example_tool="nmem_index",
        example_args={"action": "scan", "path": "./src"},
    ),
)


class OnboardingHandler:
    """Mixin: fresh-brain onboarding for MCP server.

    Tracks onboarding state per-instance (in-memory flag).
    The flag resets on MCP server restart, but since a brain
    that already has data will have neuron_count > 0, the check
    is inherently idempotent.
    """

    _onboarding_shown: bool = False

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _check_onboarding(self) -> dict[str, Any] | None:
        """Check if brain is fresh and return onboarding data if so.

        Returns None if onboarding has already been shown or brain has data.
        """
        if self._onboarding_shown:
            return None

        try:
            storage = await self.get_storage()
            brain_id = storage._current_brain_id  # type: ignore[attr-defined]
            stats: dict[str, int] = await storage.get_stats(brain_id)
        except Exception:
            logger.debug("Onboarding check: get_stats failed", exc_info=True)
            return None

        neuron_count = stats.get("neuron_count", 0)
        fiber_count = stats.get("fiber_count", 0)

        if neuron_count > 0 or fiber_count > 0:
            self._onboarding_shown = True
            return None

        # Mark as shown so we don't repeat
        self._onboarding_shown = True

        return {
            "onboarding": True,
            "message": "Welcome to NeuralMemory! This brain is empty. Here's how to get started:",
            "steps": [
                {
                    "step": h.step,
                    "title": h.title,
                    "description": h.description,
                    "example_tool": h.example_tool,
                    "example_args": h.example_args,
                }
                for h in ONBOARDING_STEPS
            ],
        }
