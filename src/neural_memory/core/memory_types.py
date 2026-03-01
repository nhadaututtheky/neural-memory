"""Memory types and classification for Neural Memory.

Integrates MemoCore concepts: typed memories with priority, expiry, and provenance.
This enables smarter query routing and memory lifecycle management.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum
from functools import lru_cache
from typing import Any

from neural_memory.utils.timeutils import utcnow


class MemoryType(StrEnum):
    """Types of memories based on their nature and purpose."""

    # Core types from MemoCore
    FACT = "fact"  # Objective information: "Python 3.11 was released in Oct 2022"
    DECISION = "decision"  # Choices made: "We decided to use PostgreSQL"
    PREFERENCE = "preference"  # User preferences: "I prefer tabs over spaces"
    TODO = "todo"  # Action items: "Need to refactor auth module"
    INSIGHT = "insight"  # Learned patterns: "This codebase uses dependency injection"
    CONTEXT = "context"  # Situational info: "Working on project X for client Y"
    INSTRUCTION = "instruction"  # User guidelines: "Always use type hints"

    # Additional useful types
    ERROR = "error"  # Error patterns: "This API returns 429 on rate limit"
    WORKFLOW = "workflow"  # Process patterns: "Deploy flow: test -> stage -> prod"
    REFERENCE = "reference"  # External refs: "Docs at https://..."
    TOOL = "tool"  # Tool usage patterns: "Grep is effective for finding code patterns"


class Priority(IntEnum):
    """Memory priority levels (0-10 scale)."""

    LOWEST = 0
    LOW = 2
    NORMAL = 5
    HIGH = 7
    CRITICAL = 10

    @classmethod
    def from_int(cls, value: int) -> Priority:
        """Convert integer to nearest Priority level."""
        if value <= 1:
            return cls.LOWEST
        elif value <= 3:
            return cls.LOW
        elif value <= 6:
            return cls.NORMAL
        elif value <= 8:
            return cls.HIGH
        else:
            return cls.CRITICAL


class Confidence(StrEnum):
    """Confidence level in the memory's accuracy."""

    VERIFIED = "verified"  # Confirmed by user or external source
    HIGH = "high"  # Very likely accurate
    MEDIUM = "medium"  # Probably accurate
    LOW = "low"  # Uncertain, needs verification
    INFERRED = "inferred"  # AI-inferred, may be wrong


@dataclass(frozen=True)
class Provenance:
    """Tracks the origin and reliability of a memory.

    Attributes:
        source: Where this memory came from
        confidence: How reliable this memory is
        verified: Whether explicitly verified by user
        verified_at: When verification happened
        created_by: Who/what created this memory
        last_confirmed: When last confirmed as still valid
    """

    source: str  # "user_input", "ai_inference", "import", "observation"
    confidence: Confidence = Confidence.MEDIUM
    verified: bool = False
    verified_at: datetime | None = None
    created_by: str = "user"
    last_confirmed: datetime | None = None

    def verify(self) -> Provenance:
        """Create a new Provenance marked as verified."""
        return Provenance(
            source=self.source,
            confidence=Confidence.VERIFIED,
            verified=True,
            verified_at=utcnow(),
            created_by=self.created_by,
            last_confirmed=utcnow(),
        )

    def confirm(self) -> Provenance:
        """Create a new Provenance with updated confirmation time."""
        return Provenance(
            source=self.source,
            confidence=self.confidence,
            verified=self.verified,
            verified_at=self.verified_at,
            created_by=self.created_by,
            last_confirmed=utcnow(),
        )


@dataclass(frozen=True)
class TypedMemory:
    """A memory with type classification, priority, and lifecycle metadata.

    This wraps around fibers to add MemoCore-style memory management.

    Attributes:
        fiber_id: Reference to the underlying Fiber
        memory_type: Classification of this memory
        priority: Importance level (0-10)
        provenance: Origin and reliability info
        expires_at: Optional expiration timestamp
        project_id: Optional project scope
        tags: Additional categorization tags
        metadata: Extra type-specific data
        created_at: Creation timestamp
    """

    fiber_id: str
    memory_type: MemoryType
    priority: Priority = Priority.NORMAL
    provenance: Provenance = field(default_factory=lambda: Provenance(source="user_input"))
    expires_at: datetime | None = None
    project_id: str | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)

    @classmethod
    def create(
        cls,
        fiber_id: str,
        memory_type: MemoryType,
        priority: Priority | int = Priority.NORMAL,
        source: str = "user_input",
        confidence: Confidence = Confidence.MEDIUM,
        expires_in_days: int | None = None,
        project_id: str | None = None,
        tags: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TypedMemory:
        """Factory method to create a TypedMemory.

        Args:
            fiber_id: The underlying fiber ID
            memory_type: Type of memory
            priority: Priority level (int or Priority enum)
            source: Source of this memory
            confidence: Confidence level
            expires_in_days: Optional days until expiry
            project_id: Optional project scope
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            A new TypedMemory instance
        """
        if isinstance(priority, int):
            priority = Priority.from_int(priority)

        expires_at = None
        if expires_in_days is not None:
            expires_at = utcnow() + timedelta(days=expires_in_days)

        return cls(
            fiber_id=fiber_id,
            memory_type=memory_type,
            priority=priority,
            provenance=Provenance(source=source, confidence=confidence),
            expires_at=expires_at,
            project_id=project_id,
            tags=frozenset(tags) if tags else frozenset(),
            metadata=metadata or {},
            created_at=utcnow(),
        )

    @property
    def is_expired(self) -> bool:
        """Check if this memory has expired."""
        if self.expires_at is None:
            return False
        return utcnow() > self.expires_at

    @property
    def days_until_expiry(self) -> int | None:
        """Days until this memory expires, or None if no expiry."""
        if self.expires_at is None:
            return None
        delta = self.expires_at - utcnow()
        return max(0, delta.days)

    def with_priority(self, priority: Priority | int) -> TypedMemory:
        """Create a new TypedMemory with updated priority."""
        if isinstance(priority, int):
            priority = Priority.from_int(priority)
        return TypedMemory(
            fiber_id=self.fiber_id,
            memory_type=self.memory_type,
            priority=priority,
            provenance=self.provenance,
            expires_at=self.expires_at,
            project_id=self.project_id,
            tags=self.tags,
            metadata=self.metadata,
            created_at=self.created_at,
        )

    def verify(self) -> TypedMemory:
        """Create a new TypedMemory marked as verified."""
        return TypedMemory(
            fiber_id=self.fiber_id,
            memory_type=self.memory_type,
            priority=self.priority,
            provenance=self.provenance.verify(),
            expires_at=self.expires_at,
            project_id=self.project_id,
            tags=self.tags,
            metadata=self.metadata,
            created_at=self.created_at,
        )

    def extend_expiry(self, days: int) -> TypedMemory:
        """Create a new TypedMemory with extended expiry."""
        new_expiry = utcnow() + timedelta(days=days)
        return TypedMemory(
            fiber_id=self.fiber_id,
            memory_type=self.memory_type,
            priority=self.priority,
            provenance=self.provenance,
            expires_at=new_expiry,
            project_id=self.project_id,
            tags=self.tags,
            metadata=self.metadata,
            created_at=self.created_at,
        )


# Default expiry settings per memory type
DEFAULT_EXPIRY_DAYS: dict[MemoryType, int | None] = {
    MemoryType.FACT: None,  # Facts don't expire by default
    MemoryType.DECISION: 90,  # Decisions may become stale
    MemoryType.PREFERENCE: None,  # Preferences persist
    MemoryType.TODO: 30,  # TODOs should be acted on
    MemoryType.INSIGHT: 180,  # Insights may become outdated
    MemoryType.CONTEXT: 7,  # Context is usually short-term
    MemoryType.INSTRUCTION: None,  # Instructions persist
    MemoryType.ERROR: 30,  # Error patterns may get fixed
    MemoryType.WORKFLOW: 365,  # Workflows change slowly
    MemoryType.REFERENCE: None,  # References persist
    MemoryType.TOOL: 90,  # Tool patterns become stale as workflows change
}


# Default decay rates per memory type (per day, for Ebbinghaus curve).
# Lower = slower decay = longer retention.
DEFAULT_DECAY_RATES: dict[MemoryType, float] = {
    MemoryType.FACT: 0.02,  # Facts persist longest
    MemoryType.DECISION: 0.03,  # Decisions are fairly stable
    MemoryType.PREFERENCE: 0.03,  # Preferences are stable
    MemoryType.REFERENCE: 0.04,  # References persist
    MemoryType.INSIGHT: 0.05,  # Insights fade moderately
    MemoryType.INSTRUCTION: 0.05,  # Instructions are semi-permanent
    MemoryType.CONTEXT: 0.08,  # Context is short-term
    MemoryType.WORKFLOW: 0.08,  # Workflows change over time
    MemoryType.ERROR: 0.12,  # Errors become less relevant
    MemoryType.TODO: 0.15,  # TODOs should be acted on quickly
    MemoryType.TOOL: 0.06,  # Tool patterns reinforced by repeated use
}


def get_decay_rate(memory_type: str) -> float:
    """Get the default decay rate for a memory type.

    Args:
        memory_type: The memory type string (e.g. "fact", "todo").

    Returns:
        Decay rate per day. Defaults to 0.1 for unknown types.
    """
    try:
        mt = MemoryType(memory_type)
        return DEFAULT_DECAY_RATES.get(mt, 0.1)
    except ValueError:
        return 0.1


@lru_cache(maxsize=256)
def _word_boundary_pattern(keyword: str) -> re.Pattern[str]:
    """Compile and cache a word-boundary regex for the given keyword."""
    return re.compile(rf"\b{re.escape(keyword)}\b")


def _has_keyword(content_lower: str, keyword: str) -> bool:
    """Check if keyword appears as a whole word (not substring) in content."""
    # Multi-word keywords: simple containment is fine (e.g. "need to", "found that")
    if " " in keyword:
        return keyword in content_lower
    # Single-word: require word boundary to avoid false positives
    # (e.g. "add" should not match "address")
    return bool(_word_boundary_pattern(keyword).search(content_lower))


def suggest_memory_type(content: str) -> MemoryType:
    """Suggest a memory type based on content analysis.

    This is a simple heuristic. For production, use NLP.

    Args:
        content: The memory content to analyze

    Returns:
        Suggested MemoryType
    """
    content_lower = content.lower()

    # TODO patterns (highest priority - actionable)
    if any(
        _has_keyword(content_lower, kw)
        for kw in ["todo", "need to", "should", "must", "fix", "implement", "add"]
    ):
        return MemoryType.TODO

    # Decision patterns
    if any(
        _has_keyword(content_lower, kw)
        for kw in ["decided", "chose", "will use", "going with", "picked", "selected"]
    ):
        return MemoryType.DECISION

    # Instruction patterns (check BEFORE preference - "always use" vs "always")
    if any(
        _has_keyword(content_lower, kw)
        for kw in [
            "always use",
            "never use",
            "make sure",
            "remember to",
            "don't forget",
        ]
    ):
        return MemoryType.INSTRUCTION

    # Preference patterns
    if any(_has_keyword(content_lower, kw) for kw in ["prefer", "like", "favorite", "hate"]):
        return MemoryType.PREFERENCE

    # Insight patterns (check BEFORE error - "discovered" vs "issue")
    if any(
        _has_keyword(content_lower, kw)
        for kw in ["learned", "realized", "discovered", "found that", "turns out"]
    ):
        return MemoryType.INSIGHT

    # Error patterns
    if any(
        _has_keyword(content_lower, kw)
        for kw in ["error", "bug", "issue", "problem", "fail", "crash", "exception"]
    ):
        return MemoryType.ERROR

    # Workflow patterns
    if any(
        _has_keyword(content_lower, kw)
        for kw in ["workflow", "process", "step", "flow", "pipeline", "deploy"]
    ):
        return MemoryType.WORKFLOW

    # Reference patterns
    if any(_has_keyword(content_lower, kw) for kw in ["http", "https", "docs", "documentation"]):
        return MemoryType.REFERENCE

    # Default to fact
    return MemoryType.FACT
