"""Auto-save trigger detection for the eternal context system.

Detects events that should trigger an auto-save of context state:
- Workflow completion ("done", "xong", "pass test")
- Decision made (reuses DECISION_PATTERNS from auto_capture)
- Error fixed ("fixed by", "sua xong")
- User leaving ("bye", "tam nghi", "het gio")
- Message checkpoint (every N messages)
- Context capacity warning (token estimate > threshold)

All detection is regex-based and lightweight — no blocking I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class TriggerType(StrEnum):
    """Types of auto-save triggers."""

    WORKFLOW_END = "workflow_end"
    DECISION_MADE = "decision_made"
    ERROR_FIXED = "error_fixed"
    USER_LEAVING = "user_leaving"
    CHECKPOINT = "checkpoint"
    CONTEXT_WARNING = "context_warning"
    HEALTH_DEGRADATION = "health_degradation"


@dataclass(frozen=True)
class TriggerResult:
    """Result of checking auto-save triggers.

    Attributes:
        triggered: Whether any trigger fired.
        trigger_type: Which trigger fired (None if not triggered).
        message: Human-readable notification message.
        save_tiers: Which tiers to save (1=Critical, 2=Important, 3=Context).
    """

    triggered: bool
    trigger_type: TriggerType | None = None
    message: str = ""
    save_tiers: tuple[int, ...] = ()


# ──────────────────── Pattern lists ────────────────────

USER_LEAVING_PATTERNS: list[re.Pattern[str]] = [
    # English
    re.compile(r"\b(?:bye|goodbye|good\s*bye|see you|gotta go|signing off|logging off)\b", re.IGNORECASE),
    re.compile(r"\b(?:i'm done|that's all|that's it|wrapping up|call it a day)\b", re.IGNORECASE),
    re.compile(r"\b(?:heading out|leaving now|going offline)\b", re.IGNORECASE),
    # Vietnamese
    re.compile(r"(?:tôi đi|tạm nghỉ|hết giờ|bye|tạm biệt|nghỉ thôi|đi ngủ)", re.IGNORECASE),
    re.compile(r"(?:kết thúc|xong rồi đi|thôi nhé|hẹn gặp lại)", re.IGNORECASE),
]

MILESTONE_PATTERNS: list[re.Pattern[str]] = [
    # English
    re.compile(r"\b(?:done|finished|completed|shipped|deployed|merged|released)\b", re.IGNORECASE),
    re.compile(r"\b(?:all tests pass|tests passing|build succeeded|build passing)\b", re.IGNORECASE),
    re.compile(r"\b(?:pass test|tests? green|ci green|pipeline green)\b", re.IGNORECASE),
    re.compile(r"\b(?:feature complete|implementation complete|task complete)\b", re.IGNORECASE),
    re.compile(r"\b(?:lgtm|approved|ready to merge)\b", re.IGNORECASE),
    # Vietnamese
    re.compile(r"(?:xong|hoàn thành|đã xong|xong rồi|hoàn tất|đã hoàn thành)", re.IGNORECASE),
    re.compile(r"(?:pass test|chạy được|build xong|deploy xong)", re.IGNORECASE),
]

ERROR_FIXED_PATTERNS: list[re.Pattern[str]] = [
    # English
    re.compile(r"(?:fixed|resolved|solved|patched|corrected)\s+(?:it|the|this|that|by)\b", re.IGNORECASE),
    re.compile(r"\b(?:bug fix|hotfix|fix applied|issue resolved)\b", re.IGNORECASE),
    re.compile(r"\b(?:no longer|not anymore|works now|working now)\b", re.IGNORECASE),
    # Vietnamese
    re.compile(r"(?:sửa xong|fix xong|đã sửa|đã fix|hết lỗi|không lỗi nữa)", re.IGNORECASE),
    re.compile(r"(?:fix được rồi|chạy được rồi)", re.IGNORECASE),
]

# Reuse decision patterns from auto_capture (imported at check time)
DECISION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:we |I )(?:decided|chose|selected|picked|opted)(?: to)?[:\s]+.{5,}", re.IGNORECASE),
    re.compile(r"(?:the )?decision(?: is)?[:\s]+.{5,}", re.IGNORECASE),
    re.compile(r"(?:we're |I'm )going (?:to|with)[:\s]+.{5,}", re.IGNORECASE),
    re.compile(r"let's (?:go with|use|choose)[:\s]+.{5,}", re.IGNORECASE),
    # Vietnamese
    re.compile(r"(?:quyết định|chọn|dùng|chuyển sang)[:\s]+.{5,}", re.IGNORECASE),
]

# Minimum text length to avoid false positives
_MIN_TEXT_LENGTH = 10


def check_triggers(
    text: str,
    message_count: int = 0,
    token_estimate: int = 0,
    max_tokens: int = 128_000,
    checkpoint_interval: int = 15,
    warning_threshold: float = 0.8,
) -> TriggerResult:
    """Check all auto-save triggers against input text and counters.

    Args:
        text: Text to scan for trigger patterns.
        message_count: Current message count in session.
        token_estimate: Estimated tokens used so far.
        max_tokens: Maximum context window tokens.
        checkpoint_interval: Messages between checkpoints.
        warning_threshold: Fraction of max_tokens that triggers warning.

    Returns:
        TriggerResult with trigger info if any trigger fired.
    """
    # Priority order: context_warning > user_leaving > workflow_end > error_fixed > decision > checkpoint

    # 1. Context capacity warning (highest priority — data loss risk)
    if token_estimate > 0 and max_tokens > 0:
        usage = token_estimate / max_tokens
        if usage >= warning_threshold:
            pct = int(usage * 100)
            return TriggerResult(
                triggered=True,
                trigger_type=TriggerType.CONTEXT_WARNING,
                message=f"Context {pct}% full. Auto-saved backup.",
                save_tiers=(1, 2, 3),
            )

    # Skip pattern matching on very short text
    if len(text.strip()) < _MIN_TEXT_LENGTH:
        # Still check checkpoint
        if message_count > 0 and message_count % checkpoint_interval == 0:
            return TriggerResult(
                triggered=True,
                trigger_type=TriggerType.CHECKPOINT,
                message=f"Checkpoint at message {message_count}.",
                save_tiers=(2, 3),
            )
        return TriggerResult(triggered=False)

    text_lower = text.lower()

    # 2. User leaving
    if _matches_any(text_lower, USER_LEAVING_PATTERNS):
        return TriggerResult(
            triggered=True,
            trigger_type=TriggerType.USER_LEAVING,
            message="User leaving detected. Auto-saved session.",
            save_tiers=(1, 2, 3),
        )

    # 3. Workflow/milestone end
    if _matches_any(text_lower, MILESTONE_PATTERNS):
        return TriggerResult(
            triggered=True,
            trigger_type=TriggerType.WORKFLOW_END,
            message="Workflow complete. Auto-saved progress.",
            save_tiers=(1, 2),
        )

    # 4. Error fixed
    if _matches_any(text_lower, ERROR_FIXED_PATTERNS):
        return TriggerResult(
            triggered=True,
            trigger_type=TriggerType.ERROR_FIXED,
            message="Error fix detected. Auto-saved.",
            save_tiers=(2,),
        )

    # 5. Decision made
    if _matches_any(text_lower, DECISION_PATTERNS):
        return TriggerResult(
            triggered=True,
            trigger_type=TriggerType.DECISION_MADE,
            message="Decision detected. Auto-saved.",
            save_tiers=(1, 2),
        )

    # 6. Checkpoint (lowest priority)
    if message_count > 0 and message_count % checkpoint_interval == 0:
        return TriggerResult(
            triggered=True,
            trigger_type=TriggerType.CHECKPOINT,
            message=f"Checkpoint at message {message_count}.",
            save_tiers=(2, 3),
        )

    return TriggerResult(triggered=False)


def estimate_session_tokens(
    message_count: int,
    code_lines: int = 0,
    error_count: int = 0,
) -> int:
    """Estimate total tokens used in a session.

    Heuristic formula from design doc:
        tokens = messages * 150 + code_lines * 5 + errors * 300

    Args:
        message_count: Number of messages exchanged.
        code_lines: Lines of code discussed/generated.
        error_count: Number of errors encountered.

    Returns:
        Estimated token count.
    """
    return message_count * 150 + code_lines * 5 + error_count * 300


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    """Check if text matches any pre-compiled regex pattern."""
    return any(pattern.search(text) for pattern in patterns)
