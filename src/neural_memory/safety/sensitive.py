"""Sensitive content detection for Neural Memory.

Detects potentially sensitive information like:
- API keys and secrets
- Passwords and tokens
- Personal identifiable information (PII)
- Credit card numbers
- Private keys
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

logger = logging.getLogger(__name__)

# Maximum content length for sensitive detection (prevent ReDoS on huge input)
_MAX_CONTENT_LENGTH = 100_000


class SensitiveType(StrEnum):
    """Types of sensitive content."""

    API_KEY = "api_key"
    PASSWORD = "password"
    SECRET = "secret"
    TOKEN = "token"
    PRIVATE_KEY = "private_key"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    AWS_KEY = "aws_key"
    DATABASE_URL = "database_url"
    JWT = "jwt"
    GENERIC_SECRET = "generic_secret"


@dataclass(frozen=True)
class SensitivePattern:
    """A pattern for detecting sensitive content."""

    name: str
    pattern: str
    type: SensitiveType
    description: str
    severity: int = 1  # 1=low, 2=medium, 3=high


@dataclass(frozen=True)
class SensitiveMatch:
    """A match found in content."""

    pattern_name: str
    matched_text: str
    type: SensitiveType
    severity: int
    start: int
    end: int

    def redacted(self) -> str:
        """Get redacted version of matched text."""
        if len(self.matched_text) <= 8:
            return "*" * len(self.matched_text)
        return self.matched_text[:4] + "*" * (len(self.matched_text) - 8) + self.matched_text[-4:]


@lru_cache(maxsize=1)
def get_default_patterns() -> tuple[SensitivePattern, ...]:
    """Get default sensitive content patterns (cached after first call)."""
    return (
        # API Keys and Secrets
        SensitivePattern(
            name="Generic API Key",
            pattern=r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?",
            type=SensitiveType.API_KEY,
            description="Generic API key assignment",
            severity=3,
        ),
        SensitivePattern(
            name="Generic Secret",
            pattern=r"(?i)(secret|secret[_-]?key)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?",
            type=SensitiveType.SECRET,
            description="Generic secret assignment",
            severity=3,
        ),
        SensitivePattern(
            name="Generic Password",
            pattern=r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]?([^\s'\"]{4,256})['\"]?",
            type=SensitiveType.PASSWORD,
            description="Password assignment",
            severity=3,
        ),
        SensitivePattern(
            name="Generic Token",
            pattern=r"(?i)(token|auth[_-]?token|access[_-]?token|bearer)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-\.]{16,})['\"]?",
            type=SensitiveType.TOKEN,
            description="Auth token assignment",
            severity=3,
        ),
        # AWS
        SensitivePattern(
            name="AWS Access Key",
            pattern=r"(?i)aws[_-]?access[_-]?key[_-]?id\s*[=:]\s*['\"]?(AKIA[0-9A-Z]{16})['\"]?",
            type=SensitiveType.AWS_KEY,
            description="AWS Access Key ID",
            severity=3,
        ),
        SensitivePattern(
            name="AWS Secret Key",
            pattern=r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*['\"]?([a-zA-Z0-9/+=]{40})['\"]?",
            type=SensitiveType.AWS_KEY,
            description="AWS Secret Access Key",
            severity=3,
        ),
        # Database
        SensitivePattern(
            name="Database URL",
            pattern=r"(?i)(postgres|mysql|mongodb|redis)://[^\s]+:[^\s]+@[^\s]+",
            type=SensitiveType.DATABASE_URL,
            description="Database connection string with credentials",
            severity=3,
        ),
        # Private Keys
        SensitivePattern(
            name="Private Key",
            pattern=r"-----BEGIN\s+(RSA|DSA|EC|OPENSSH|PGP)?\s*PRIVATE KEY-----",
            type=SensitiveType.PRIVATE_KEY,
            description="Private key header",
            severity=3,
        ),
        # JWT
        SensitivePattern(
            name="JWT Token",
            pattern=r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
            type=SensitiveType.JWT,
            description="JSON Web Token",
            severity=2,
        ),
        # Credit Card (basic pattern)
        SensitivePattern(
            name="Credit Card",
            pattern=r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            type=SensitiveType.CREDIT_CARD,
            description="Credit card number",
            severity=3,
        ),
        # Social Security Number pattern
        SensitivePattern(
            name="SSN",
            pattern=r"\b\d{3}-\d{2}-\d{4}\b",
            type=SensitiveType.SSN,
            description="Social Security Number format",
            severity=3,
        ),
        # Long random strings (potential secrets)
        SensitivePattern(
            name="Long Base64 String",
            pattern=r"\b[A-Za-z0-9+/]{40,512}={0,2}\b",
            type=SensitiveType.GENERIC_SECRET,
            description="Long base64-encoded string (potential secret)",
            severity=1,
        ),
        # Hex strings (potential keys)
        SensitivePattern(
            name="Long Hex String",
            pattern=r"\b[a-fA-F0-9]{32,512}\b",
            type=SensitiveType.GENERIC_SECRET,
            description="Long hexadecimal string (potential key)",
            severity=1,
        ),
    )


# Pre-compiled regex cache: pattern string -> compiled regex
_compiled_cache: dict[str, re.Pattern[str]] = {}


def _get_compiled(pattern_str: str) -> re.Pattern[str]:
    """Get or compile a regex pattern, with caching."""
    compiled = _compiled_cache.get(pattern_str)
    if compiled is None:
        compiled = re.compile(pattern_str)
        _compiled_cache[pattern_str] = compiled
    return compiled


def check_sensitive_content(
    content: str,
    patterns: tuple[SensitivePattern, ...] | list[SensitivePattern] | None = None,
    min_severity: int = 1,
) -> list[SensitiveMatch]:
    """
    Check content for sensitive information.

    Args:
        content: Text to check
        patterns: Patterns to use (default: get_default_patterns())
        min_severity: Minimum severity level to report (1-3)

    Returns:
        List of sensitive matches found
    """
    if patterns is None:
        patterns = get_default_patterns()

    # Cap content length to prevent ReDoS on huge input
    content = content[:_MAX_CONTENT_LENGTH]

    matches: list[SensitiveMatch] = []

    for pattern in patterns:
        if pattern.severity < min_severity:
            continue

        try:
            regex = _get_compiled(pattern.pattern)
            for match in regex.finditer(content):
                matches.append(
                    SensitiveMatch(
                        pattern_name=pattern.name,
                        matched_text=match.group(0),
                        type=pattern.type,
                        severity=pattern.severity,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        except re.error as e:
            logger.warning("Invalid regex pattern '%s': %s", pattern.name, e)
            continue

    # Remove duplicates (same position)
    seen_positions: set[tuple[int, int]] = set()
    unique_matches: list[SensitiveMatch] = []
    for sensitive_match in matches:
        pos = (sensitive_match.start, sensitive_match.end)
        if pos not in seen_positions:
            seen_positions.add(pos)
            unique_matches.append(sensitive_match)

    # Merge overlapping spans (keep highest severity for merged spans)
    if unique_matches:
        unique_matches.sort(key=lambda m: m.start)
        merged: list[SensitiveMatch] = [unique_matches[0]]
        for span in unique_matches[1:]:
            if span.start <= merged[-1].end:
                # Overlapping - extend the end, keep higher severity match
                prev = merged[-1]
                new_end = max(prev.end, span.end)
                best = span if span.severity > prev.severity else prev
                # Use actual content slice for accurate audit trail
                merged_text = content[prev.start : new_end]
                merged[-1] = SensitiveMatch(
                    pattern_name=best.pattern_name,
                    matched_text=merged_text,
                    type=best.type,
                    severity=max(prev.severity, span.severity),
                    start=prev.start,
                    end=new_end,
                )
            else:
                merged.append(span)
        unique_matches = merged

    return sorted(unique_matches, key=lambda m: (-m.severity, m.start))


def filter_sensitive_content(
    content: str,
    patterns: list[SensitivePattern] | None = None,
    replacement: str = "[REDACTED]",
) -> tuple[str, list[SensitiveMatch]]:
    """
    Filter sensitive content by replacing matches.

    Args:
        content: Text to filter
        patterns: Patterns to use
        replacement: Replacement text

    Returns:
        Tuple of (filtered_content, matches_found)
    """
    matches = check_sensitive_content(content, patterns)

    if not matches:
        return content, []

    # Sort by position descending to replace from end
    sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

    filtered = content
    for match in sorted_matches:
        filtered = filtered[: match.start] + replacement + filtered[match.end :]

    return filtered, matches


def auto_redact_content(
    content: str,
    min_severity: int = 3,
    patterns: list[SensitivePattern] | None = None,
    replacement: str = "[REDACTED]",
) -> tuple[str, list[SensitiveMatch], str | None]:
    """Auto-redact sensitive content at or above the given severity.

    Unlike filter_sensitive_content which redacts ALL matches, this
    selectively redacts only matches at or above min_severity.
    Preserves a SHA-256 hash of the original content for dedup.

    Args:
        content: Text to check and redact
        min_severity: Minimum severity level to auto-redact (1-3)
        patterns: Patterns to use (default: get_default_patterns())
        replacement: Replacement text for redacted matches

    Returns:
        Tuple of (redacted_content, redacted_matches, original_content_hash).
        If no matches found, returns (content, [], None).
    """
    import hashlib

    all_matches = check_sensitive_content(content, patterns, min_severity=1)

    if not all_matches:
        return content, [], None

    # Only redact matches at or above the threshold
    to_redact = [m for m in all_matches if m.severity >= min_severity]

    if not to_redact:
        return content, [], None

    # Hash original content for dedup tracking
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Sort by position descending to replace from end (preserves earlier positions)
    sorted_redact = sorted(to_redact, key=lambda m: m.start, reverse=True)

    redacted = content
    for match in sorted_redact:
        redacted = redacted[: match.start] + replacement + redacted[match.end :]

    return redacted, to_redact, content_hash


def format_sensitive_warning(matches: list[SensitiveMatch], use_ascii: bool = False) -> str:
    """Format a warning message for sensitive content.

    Args:
        matches: List of sensitive matches
        use_ascii: Use ASCII characters instead of emojis (for Windows compatibility)
    """
    if not matches:
        return ""

    # Use ASCII or Unicode based on preference/platform
    if use_ascii:
        warn_icon = "[!]"
        high_icon = "[!!!]"
        medium_icon = "[!!]"
        low_icon = "[!]"
    else:
        warn_icon = "<!>"
        high_icon = "[HIGH]"
        medium_icon = "[MED]"
        low_icon = "[LOW]"

    lines = [f"{warn_icon} SENSITIVE CONTENT DETECTED:"]

    # Group by severity
    high = [m for m in matches if m.severity == 3]
    medium = [m for m in matches if m.severity == 2]
    low = [m for m in matches if m.severity == 1]

    if high:
        lines.append(f"\n  {high_icon} HIGH RISK:")
        for m in high:
            lines.append(f"     - {m.pattern_name}: {m.redacted()}")

    if medium:
        lines.append(f"\n  {medium_icon} MEDIUM RISK:")
        for m in medium:
            lines.append(f"     - {m.pattern_name}: {m.redacted()}")

    if low:
        lines.append(f"\n  {low_icon} LOW RISK:")
        for m in low[:3]:  # Limit low risk to 3
            lines.append(f"     - {m.pattern_name}")
        if len(low) > 3:
            lines.append(f"     ... and {len(low) - 3} more")

    lines.append("\n  Use --force to store anyway, or --redact to auto-redact.")

    return "\n".join(lines)
