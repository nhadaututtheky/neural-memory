"""Project scoping for memory organization.

Projects allow grouping memories by context (sprint, feature, research topic)
with automatic time-based prioritization.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from neural_memory.utils.timeutils import utcnow


@dataclass(frozen=True)
class Project:
    """A project scope for organizing memories.

    Projects define a context boundary for memories, typically representing
    a sprint, feature, research topic, or any focused work period.
    """

    id: str
    name: str
    description: str = ""
    start_date: datetime = field(default_factory=utcnow)
    end_date: datetime | None = None  # None = ongoing
    tags: frozenset[str] = field(default_factory=frozenset)
    priority: float = 1.0  # Higher = more important
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)

    @classmethod
    def create(
        cls,
        name: str,
        description: str = "",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        duration_days: int | None = None,
        tags: set[str] | None = None,
        priority: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Project:
        """Create a new project.

        Args:
            name: Project name
            description: Optional description
            start_date: When project starts (default: now)
            end_date: When project ends (optional)
            duration_days: Alternative to end_date - set duration from start
            tags: Optional tags for categorization
            priority: Project priority (default: 1.0)
            metadata: Optional metadata

        Returns:
            New Project instance
        """
        now = utcnow()
        start = start_date or now

        # Calculate end_date from duration if provided
        end = end_date
        if end is None and duration_days is not None:
            end = start + timedelta(days=duration_days)

        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            start_date=start,
            end_date=end,
            tags=frozenset(tags) if tags else frozenset(),
            priority=priority,
            metadata=metadata or {},
            created_at=now,
        )

    @property
    def is_active(self) -> bool:
        """Check if project is currently active."""
        now = utcnow()
        if now < self.start_date:
            return False
        if self.end_date is not None and now > self.end_date:
            return False
        return True

    @property
    def is_ongoing(self) -> bool:
        """Check if project has no defined end date."""
        return self.end_date is None

    @property
    def days_remaining(self) -> int | None:
        """Get days remaining until project end, or None if ongoing."""
        if self.end_date is None:
            return None
        delta = self.end_date - utcnow()
        return max(0, delta.days)

    @property
    def duration_days(self) -> int | None:
        """Get total project duration in days, or None if ongoing."""
        if self.end_date is None:
            return None
        delta = self.end_date - self.start_date
        return delta.days

    def contains_date(self, date: datetime) -> bool:
        """Check if a date falls within project timeframe."""
        if date < self.start_date:
            return False
        if self.end_date is not None and date > self.end_date:
            return False
        return True

    def with_end_date(self, end_date: datetime) -> Project:
        """Create a copy with new end date."""
        return Project(
            id=self.id,
            name=self.name,
            description=self.description,
            start_date=self.start_date,
            end_date=end_date,
            tags=self.tags,
            priority=self.priority,
            metadata=self.metadata,
            created_at=self.created_at,
        )

    def with_extended_deadline(self, extra_days: int) -> Project:
        """Extend project deadline by given days."""
        if extra_days < 0:
            raise ValueError("extra_days must be non-negative")
        if self.end_date is None:
            raise ValueError("Cannot extend ongoing project - set end_date first")
        new_end = self.end_date + timedelta(days=extra_days)
        return self.with_end_date(new_end)

    def with_tags(self, tags: set[str]) -> Project:
        """Create a copy with new tags."""
        return Project(
            id=self.id,
            name=self.name,
            description=self.description,
            start_date=self.start_date,
            end_date=self.end_date,
            tags=frozenset(tags),
            priority=self.priority,
            metadata=self.metadata,
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "tags": list(self.tags),
            "priority": self.priority,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        """Create from dictionary."""
        try:
            start_date = datetime.fromisoformat(data["start_date"])
        except (ValueError, TypeError, KeyError):
            start_date = utcnow()
        try:
            created_at = datetime.fromisoformat(data["created_at"])
        except (ValueError, TypeError, KeyError):
            created_at = utcnow()
        end_date = None
        if data.get("end_date"):
            try:
                end_date = datetime.fromisoformat(data["end_date"])
            except (ValueError, TypeError):
                pass
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            start_date=start_date,
            end_date=end_date,
            tags=frozenset(data.get("tags", [])),
            priority=data.get("priority", 1.0),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )


@dataclass(frozen=True)
class MemoryScope:
    """Defines what memories to prioritize for retrieval.

    Used to filter and boost memories based on project, time window, or tags.
    """

    project_id: str | None = None
    time_window_days: int | None = 7  # Auto-prioritize recent
    tags: frozenset[str] | None = None
    min_relevance: float = 0.3  # Threshold for inclusion

    @classmethod
    def for_project(cls, project_id: str) -> MemoryScope:
        """Create scope for specific project."""
        return cls(project_id=project_id)

    @classmethod
    def recent(cls, days: int = 7) -> MemoryScope:
        """Create scope for recent memories."""
        return cls(time_window_days=days)

    @classmethod
    def with_tags(cls, tags: set[str]) -> MemoryScope:
        """Create scope for specific tags."""
        return cls(tags=frozenset(tags))

    def matches(
        self,
        project_id: str | None = None,
        created_at: datetime | None = None,
        tags: frozenset[str] | None = None,
    ) -> bool:
        """Check if memory attributes match this scope."""
        # Project filter
        if self.project_id is not None and project_id != self.project_id:
            return False

        # Time window filter
        if self.time_window_days is not None and created_at is not None:
            cutoff = utcnow() - timedelta(days=self.time_window_days)
            if created_at < cutoff:
                return False

        # Tags filter (must have at least one matching tag)
        if self.tags is not None and tags is not None:
            if not self.tags.intersection(tags):
                return False

        return True

    def relevance_boost(
        self,
        created_at: datetime | None = None,
        project_priority: float = 1.0,
    ) -> float:
        """Calculate relevance boost for a memory.

        Returns multiplier (1.0 = no boost, >1.0 = boosted).
        """
        boost = 1.0

        # Recency boost (exponential decay within window)
        if (
            self.time_window_days is not None
            and self.time_window_days > 0
            and created_at is not None
        ):
            days_ago = (utcnow() - created_at).days
            if days_ago <= self.time_window_days:
                # Linear decay: 1.5 at day 0, 1.0 at window edge
                recency_factor = 1.0 + 0.5 * (1 - days_ago / self.time_window_days)
                boost *= recency_factor

        # Project priority boost
        boost *= project_priority

        return boost
