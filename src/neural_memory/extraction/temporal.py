"""Temporal extraction for Vietnamese and English time expressions."""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class TimeGranularity(StrEnum):
    """Granularity level of a time reference."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


@dataclass
class TimeHint:
    """
    A parsed time reference from text.

    Attributes:
        original: The original text that was matched
        absolute_start: Resolved start datetime
        absolute_end: Resolved end datetime
        granularity: How precise this time reference is
        is_fuzzy: Whether this is an approximate time
    """

    original: str
    absolute_start: datetime
    absolute_end: datetime
    granularity: TimeGranularity
    is_fuzzy: bool = True

    @property
    def midpoint(self) -> datetime:
        """Get the midpoint of this time range."""
        delta = (self.absolute_end - self.absolute_start) / 2
        return self.absolute_start + delta


# Type alias for time resolver functions
# Some resolvers take (ref) and some take (ref, match) — arity is checked at runtime
TimeResolver = Callable[..., tuple[datetime, datetime]]


def _start_of_day(dt: datetime) -> datetime:
    """Get start of day (00:00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(dt: datetime) -> datetime:
    """Get end of day (23:59:59)."""
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


class TemporalExtractor:
    """
    Multi-language temporal expression extractor.

    Supports Vietnamese and English time expressions.
    """

    # Vietnamese time patterns
    VI_PATTERNS: dict[str, TimeResolver] = {
        # Relative days
        r"hôm nay": lambda ref: (_start_of_day(ref), _end_of_day(ref)),
        r"hôm qua": lambda ref: (
            _start_of_day(ref - timedelta(days=1)),
            _end_of_day(ref - timedelta(days=1)),
        ),
        r"hôm kia": lambda ref: (
            _start_of_day(ref - timedelta(days=2)),
            _end_of_day(ref - timedelta(days=2)),
        ),
        r"ngày mai": lambda ref: (
            _start_of_day(ref + timedelta(days=1)),
            _end_of_day(ref + timedelta(days=1)),
        ),
        r"ngày kia": lambda ref: (
            _start_of_day(ref + timedelta(days=2)),
            _end_of_day(ref + timedelta(days=2)),
        ),
        # Parts of day (today)
        r"sáng nay": lambda ref: (
            ref.replace(hour=6, minute=0, second=0, microsecond=0),
            ref.replace(hour=12, minute=0, second=0, microsecond=0),
        ),
        r"trưa nay": lambda ref: (
            ref.replace(hour=11, minute=0, second=0, microsecond=0),
            ref.replace(hour=14, minute=0, second=0, microsecond=0),
        ),
        r"chiều nay": lambda ref: (
            ref.replace(hour=14, minute=0, second=0, microsecond=0),
            ref.replace(hour=18, minute=0, second=0, microsecond=0),
        ),
        r"tối nay": lambda ref: (
            ref.replace(hour=18, minute=0, second=0, microsecond=0),
            ref.replace(hour=22, minute=0, second=0, microsecond=0),
        ),
        r"đêm nay": lambda ref: (
            ref.replace(hour=22, minute=0, second=0, microsecond=0),
            (ref + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0),
        ),
        # Parts of day (yesterday)
        r"sáng qua|sáng hôm qua": lambda ref: (
            (ref - timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0),
            (ref - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0),
        ),
        r"chiều qua|chiều hôm qua": lambda ref: (
            (ref - timedelta(days=1)).replace(hour=14, minute=0, second=0, microsecond=0),
            (ref - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0),
        ),
        r"tối qua|tối hôm qua": lambda ref: (
            (ref - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0),
            (ref - timedelta(days=1)).replace(hour=22, minute=0, second=0, microsecond=0),
        ),
        # Relative weeks
        r"tuần này": lambda ref: (
            _start_of_day(ref - timedelta(days=ref.weekday())),
            _end_of_day(ref - timedelta(days=ref.weekday()) + timedelta(days=6)),
        ),
        r"tuần trước|tuần rồi": lambda ref: (
            _start_of_day(ref - timedelta(days=ref.weekday() + 7)),
            _end_of_day(ref - timedelta(days=ref.weekday() + 1)),
        ),
        r"tuần sau|tuần tới": lambda ref: (
            _start_of_day(ref - timedelta(days=ref.weekday()) + timedelta(days=7)),
            _end_of_day(ref - timedelta(days=ref.weekday()) + timedelta(days=13)),
        ),
        # Relative months
        r"tháng này": lambda ref: (
            ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            _end_of_day(
                (ref.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            ),
        ),
        r"tháng trước|tháng rồi": lambda ref: (
            (ref.replace(day=1) - timedelta(days=1)).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ),
            _end_of_day(ref.replace(day=1) - timedelta(days=1)),
        ),
        # Recent time
        r"mới|vừa|vừa xong|mới đây": lambda ref: (
            ref - timedelta(hours=2),
            ref,
        ),
        r"nãy|lúc nãy": lambda ref: (
            ref - timedelta(hours=4),
            ref - timedelta(minutes=30),
        ),
    }

    # English time patterns
    EN_PATTERNS: dict[str, TimeResolver] = {
        # Relative days
        r"today": lambda ref: (_start_of_day(ref), _end_of_day(ref)),
        r"yesterday": lambda ref: (
            _start_of_day(ref - timedelta(days=1)),
            _end_of_day(ref - timedelta(days=1)),
        ),
        r"day before yesterday": lambda ref: (
            _start_of_day(ref - timedelta(days=2)),
            _end_of_day(ref - timedelta(days=2)),
        ),
        r"tomorrow": lambda ref: (
            _start_of_day(ref + timedelta(days=1)),
            _end_of_day(ref + timedelta(days=1)),
        ),
        # Parts of day (today)
        r"this morning": lambda ref: (
            ref.replace(hour=6, minute=0, second=0, microsecond=0),
            ref.replace(hour=12, minute=0, second=0, microsecond=0),
        ),
        r"this afternoon": lambda ref: (
            ref.replace(hour=12, minute=0, second=0, microsecond=0),
            ref.replace(hour=18, minute=0, second=0, microsecond=0),
        ),
        r"this evening": lambda ref: (
            ref.replace(hour=18, minute=0, second=0, microsecond=0),
            ref.replace(hour=22, minute=0, second=0, microsecond=0),
        ),
        r"tonight": lambda ref: (
            ref.replace(hour=20, minute=0, second=0, microsecond=0),
            (ref + timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0),
        ),
        # Parts of day (yesterday)
        r"yesterday morning": lambda ref: (
            (ref - timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0),
            (ref - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0),
        ),
        r"yesterday afternoon": lambda ref: (
            (ref - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0),
            (ref - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0),
        ),
        r"yesterday evening|last night": lambda ref: (
            (ref - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0),
            (ref - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0),
        ),
        # Relative weeks
        r"this week": lambda ref: (
            _start_of_day(ref - timedelta(days=ref.weekday())),
            _end_of_day(ref - timedelta(days=ref.weekday()) + timedelta(days=6)),
        ),
        r"last week": lambda ref: (
            _start_of_day(ref - timedelta(days=ref.weekday() + 7)),
            _end_of_day(ref - timedelta(days=ref.weekday() + 1)),
        ),
        r"next week": lambda ref: (
            _start_of_day(ref - timedelta(days=ref.weekday()) + timedelta(days=7)),
            _end_of_day(ref - timedelta(days=ref.weekday()) + timedelta(days=13)),
        ),
        # Relative months
        r"this month": lambda ref: (
            ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            _end_of_day(
                (ref.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            ),
        ),
        r"last month": lambda ref: (
            (ref.replace(day=1) - timedelta(days=1)).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ),
            _end_of_day(ref.replace(day=1) - timedelta(days=1)),
        ),
        # Recent time
        r"just now|just|recently": lambda ref: (
            ref - timedelta(hours=2),
            ref,
        ),
        r"earlier|earlier today": lambda ref: (
            _start_of_day(ref),
            ref - timedelta(hours=1),
        ),
        # Ago patterns
        r"(\d+)\s*minutes?\s*ago": lambda ref, m: (
            ref - timedelta(minutes=int(m.group(1)) + 5),
            ref - timedelta(minutes=max(0, int(m.group(1)) - 5)),
        ),
        r"(\d+)\s*hours?\s*ago": lambda ref, m: (
            ref - timedelta(hours=int(m.group(1)) + 0.5),
            ref - timedelta(hours=max(0, int(m.group(1)) - 0.5)),
        ),
        r"(\d+)\s*days?\s*ago": lambda ref, m: (
            _start_of_day(ref - timedelta(days=int(m.group(1)))),
            _end_of_day(ref - timedelta(days=int(m.group(1)))),
        ),
    }

    # Vietnamese time patterns with numbers
    VI_NUMBERED_PATTERNS: list[
        tuple[str, Callable[[datetime, re.Match[str]], tuple[datetime, datetime]], TimeGranularity]
    ] = [
        # Hour patterns: "3 giờ", "15h", "3h chiều"
        (
            r"(\d{1,2})\s*(?:giờ|h|g)(?:\s*(sáng|chiều|tối))?",
            lambda ref, m: _resolve_vi_hour(ref, m),
            TimeGranularity.HOUR,
        ),
        # N days ago: "2 ngày trước"
        (
            r"(\d+)\s*ngày\s*(?:trước|qua)",
            lambda ref, m: (
                _start_of_day(ref - timedelta(days=int(m.group(1)))),
                _end_of_day(ref - timedelta(days=int(m.group(1)))),
            ),
            TimeGranularity.DAY,
        ),
        # N weeks ago: "2 tuần trước"
        (
            r"(\d+)\s*tuần\s*(?:trước|qua)",
            lambda ref, m: (
                _start_of_day(ref - timedelta(weeks=int(m.group(1)))),
                _end_of_day(ref - timedelta(weeks=int(m.group(1)) - 1)),
            ),
            TimeGranularity.WEEK,
        ),
    ]

    def __init__(self) -> None:
        """Initialize the extractor."""
        # Compile regex patterns and cache resolver arity
        self._vi_compiled = [
            (re.compile(p, re.IGNORECASE), r, len(inspect.signature(r).parameters))
            for p, r in self.VI_PATTERNS.items()
        ]
        self._en_compiled = [
            (re.compile(p, re.IGNORECASE), r, len(inspect.signature(r).parameters))
            for p, r in self.EN_PATTERNS.items()
        ]
        self._vi_numbered = [
            (re.compile(p, re.IGNORECASE), r, g) for p, r, g in self.VI_NUMBERED_PATTERNS
        ]

    def extract(
        self,
        text: str,
        reference_time: datetime | None = None,
        language: str = "auto",
    ) -> list[TimeHint]:
        """
        Extract time references from text.

        Args:
            text: The text to extract from
            reference_time: Reference point for relative times (default: now)
            language: "vi", "en", or "auto" for auto-detection

        Returns:
            List of TimeHint objects for found time references
        """
        if reference_time is None:
            reference_time = utcnow()

        results: list[TimeHint] = []

        # Determine which patterns to use
        if language == "auto":
            # Use both
            patterns = self._vi_compiled + self._en_compiled
            numbered_patterns = self._vi_numbered
        elif language == "vi":
            patterns = self._vi_compiled
            numbered_patterns = self._vi_numbered
        else:  # en
            patterns = self._en_compiled
            numbered_patterns = []

        # Try each pattern
        for pattern, resolver, arity in patterns:
            for match in pattern.finditer(text):
                try:
                    if arity == 2:
                        start, end = resolver(reference_time, match)
                    else:
                        start, end = resolver(reference_time)

                    results.append(
                        TimeHint(
                            original=match.group(0),
                            absolute_start=start,
                            absolute_end=end,
                            granularity=_infer_granularity(start, end),
                            is_fuzzy=True,
                        )
                    )
                except (ValueError, TypeError, OverflowError) as e:
                    logger.debug("Time pattern failed to resolve '%s': %s", match.group(0), e)
                    continue

        # Try numbered patterns
        for pattern, resolver, granularity in numbered_patterns:
            for match in pattern.finditer(text):
                try:
                    start, end = resolver(reference_time, match)
                    results.append(
                        TimeHint(
                            original=match.group(0),
                            absolute_start=start,
                            absolute_end=end,
                            granularity=granularity,
                            is_fuzzy=False,
                        )
                    )
                except (ValueError, TypeError, OverflowError) as e:
                    logger.debug("Numbered time pattern failed '%s': %s", match.group(0), e)
                    continue

        # Remove duplicates (same time range)
        seen: set[tuple[datetime, datetime]] = set()
        unique_results: list[TimeHint] = []
        for hint in results:
            key = (hint.absolute_start, hint.absolute_end)
            if key not in seen:
                seen.add(key)
                unique_results.append(hint)

        return unique_results


def _resolve_vi_hour(ref: datetime, match: re.Match[str]) -> tuple[datetime, datetime]:
    """Resolve Vietnamese hour pattern."""
    hour = int(match.group(1))
    if not (0 <= hour <= 23):
        raise ValueError(f"Hour out of range: {hour}")
    period = match.group(2)  # sáng, chiều, tối

    if period:
        period = period.lower()
        if (period == "chiều" and hour < 12) or (period == "tối" and hour < 12):
            hour += 12
        elif period == "sáng" and hour == 12:
            hour = 0

    # Assume today if future, yesterday if past
    result = ref.replace(hour=hour, minute=0, second=0, microsecond=0)
    if result > ref:
        # Future time today is fine
        pass
    else:
        # Past time could be today or yesterday
        pass

    return (
        result - timedelta(minutes=30),
        result + timedelta(minutes=30),
    )


def _infer_granularity(start: datetime, end: datetime) -> TimeGranularity:
    """Infer granularity from time range."""
    delta = end - start
    seconds = delta.total_seconds()

    if seconds <= 3600:  # 1 hour
        return TimeGranularity.MINUTE
    elif seconds <= 86400:  # 1 day
        return TimeGranularity.HOUR
    elif seconds <= 604800:  # 1 week
        return TimeGranularity.DAY
    elif seconds <= 2678400:  # ~1 month
        return TimeGranularity.WEEK
    elif seconds <= 31536000:  # 1 year
        return TimeGranularity.MONTH
    else:
        return TimeGranularity.YEAR
