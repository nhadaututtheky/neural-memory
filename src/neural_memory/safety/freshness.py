"""Memory freshness evaluation for Neural Memory.

Evaluates how fresh/stale memories are and provides warnings
for potentially outdated information.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from neural_memory.utils.timeutils import utcnow


class FreshnessLevel(StrEnum):
    """Freshness levels for memories."""

    FRESH = "fresh"  # < 7 days
    RECENT = "recent"  # 7-30 days
    AGING = "aging"  # 30-90 days
    STALE = "stale"  # 90-365 days
    ANCIENT = "ancient"  # > 365 days


@dataclass
class FreshnessResult:
    """Result of freshness evaluation."""

    level: FreshnessLevel
    age_days: int
    warning: str | None
    should_verify: bool
    score: float  # 0.0 (ancient) to 1.0 (fresh)


# Default thresholds in days
DEFAULT_THRESHOLDS = {
    FreshnessLevel.FRESH: 7,
    FreshnessLevel.RECENT: 30,
    FreshnessLevel.AGING: 90,
    FreshnessLevel.STALE: 365,
}


def evaluate_freshness(
    created_at: datetime,
    reference_time: datetime | None = None,
    thresholds: dict[FreshnessLevel, int] | None = None,
) -> FreshnessResult:
    """
    Evaluate the freshness of a memory.

    Args:
        created_at: When the memory was created
        reference_time: Reference time for comparison (default: now)
        thresholds: Custom thresholds in days

    Returns:
        FreshnessResult with level, warning, and score
    """
    if reference_time is None:
        reference_time = utcnow()

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Calculate age
    age = reference_time - created_at
    age_days = max(0, age.days)

    # Determine level
    if age_days < thresholds[FreshnessLevel.FRESH]:
        level = FreshnessLevel.FRESH
        score = 1.0
        warning = None
        should_verify = False
    elif age_days < thresholds[FreshnessLevel.RECENT]:
        level = FreshnessLevel.RECENT
        score = 0.8
        warning = None
        should_verify = False
    elif age_days < thresholds[FreshnessLevel.AGING]:
        level = FreshnessLevel.AGING
        score = 0.5
        warning = f"[~] This memory is {age_days} days old - information may have changed"
        should_verify = True
    elif age_days < thresholds[FreshnessLevel.STALE]:
        level = FreshnessLevel.STALE
        score = 0.3
        warning = f"[!] STALE: This memory is {age_days} days old - verify before using"
        should_verify = True
    else:
        level = FreshnessLevel.ANCIENT
        score = 0.1
        warning = f"[!!] ANCIENT: This memory is {age_days} days old ({age_days // 365} years) - likely outdated"
        should_verify = True

    return FreshnessResult(
        level=level,
        age_days=age_days,
        warning=warning,
        should_verify=should_verify,
        score=score,
    )


def get_freshness_warning(
    created_at: datetime,
    reference_time: datetime | None = None,
) -> str | None:
    """
    Get a warning message if memory is stale.

    Args:
        created_at: When the memory was created
        reference_time: Reference time for comparison

    Returns:
        Warning message or None if fresh
    """
    result = evaluate_freshness(created_at, reference_time)
    return result.warning


def format_age(age_days: int) -> str:
    """Format age in human-readable form."""
    if age_days == 0:
        return "today"
    elif age_days == 1:
        return "yesterday"
    elif age_days < 7:
        return f"{age_days} days ago"
    elif age_days < 30:
        weeks = age_days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    elif age_days < 365:
        months = age_days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    else:
        years = age_days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"


def get_freshness_indicator(level: FreshnessLevel, use_ascii: bool = True) -> str:
    """Get a visual indicator for freshness level.

    Args:
        level: Freshness level
        use_ascii: Use ASCII characters instead of emojis (for Windows compatibility)
    """
    if use_ascii:
        indicators = {
            FreshnessLevel.FRESH: "[+]",
            FreshnessLevel.RECENT: "[+]",
            FreshnessLevel.AGING: "[~]",
            FreshnessLevel.STALE: "[!]",
            FreshnessLevel.ANCIENT: "[!!]",
        }
    else:
        indicators = {
            FreshnessLevel.FRESH: "G",
            FreshnessLevel.RECENT: "G",
            FreshnessLevel.AGING: "Y",
            FreshnessLevel.STALE: "O",
            FreshnessLevel.ANCIENT: "R",
        }
    return indicators.get(level, "[ ]")


@dataclass
class MemoryFreshnessReport:
    """Report on memory freshness for a set of memories."""

    total: int
    fresh: int
    recent: int
    aging: int
    stale: int
    ancient: int
    average_age_days: float
    oldest_days: int
    newest_days: int

    def summary(self) -> str:
        """Get a summary string."""
        lines = [
            f"Total memories: {self.total}",
            f"  🟢 Fresh (<7d): {self.fresh}",
            f"  🟢 Recent (7-30d): {self.recent}",
            f"  🟡 Aging (30-90d): {self.aging}",
            f"  🟠 Stale (90-365d): {self.stale}",
            f"  🔴 Ancient (>365d): {self.ancient}",
            f"Average age: {self.average_age_days:.1f} days",
        ]
        return "\n".join(lines)


def analyze_freshness(
    created_dates: list[datetime],
    reference_time: datetime | None = None,
) -> MemoryFreshnessReport:
    """
    Analyze freshness of a collection of memories.

    Args:
        created_dates: List of creation timestamps
        reference_time: Reference time for comparison

    Returns:
        MemoryFreshnessReport with statistics
    """
    if not created_dates:
        return MemoryFreshnessReport(
            total=0,
            fresh=0,
            recent=0,
            aging=0,
            stale=0,
            ancient=0,
            average_age_days=0,
            oldest_days=0,
            newest_days=0,
        )

    results = [evaluate_freshness(dt, reference_time) for dt in created_dates]

    ages = [r.age_days for r in results]

    return MemoryFreshnessReport(
        total=len(results),
        fresh=sum(1 for r in results if r.level == FreshnessLevel.FRESH),
        recent=sum(1 for r in results if r.level == FreshnessLevel.RECENT),
        aging=sum(1 for r in results if r.level == FreshnessLevel.AGING),
        stale=sum(1 for r in results if r.level == FreshnessLevel.STALE),
        ancient=sum(1 for r in results if r.level == FreshnessLevel.ANCIENT),
        average_age_days=sum(ages) / len(ages),
        oldest_days=max(ages),
        newest_days=min(ages),
    )
