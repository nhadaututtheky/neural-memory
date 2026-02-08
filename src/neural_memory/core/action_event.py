"""Action event — lightweight hippocampal buffer for tool usage tracking.

Action events record tool calls and operations as lightweight DB rows
(not neurons) to avoid graph bloat. They are mined by the sequence
mining engine to detect habitual action sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass(frozen=True)
class ActionEvent:
    """A single action event recorded in the hippocampal buffer.

    Attributes:
        id: Unique identifier
        brain_id: Brain this event belongs to
        session_id: Optional session grouping
        action_type: Type of action (e.g., "remember", "recall", "context")
        action_context: Optional context string
        tags: Tags associated with this action
        fiber_id: Optional associated fiber
        created_at: When this action occurred
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    brain_id: str = ""
    session_id: str | None = None
    action_type: str = ""
    action_context: str = ""
    tags: tuple[str, ...] = ()
    fiber_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
