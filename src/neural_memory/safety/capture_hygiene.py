"""Capture hygiene — strip harness/tool noise before memory detection.

Gate 0 in the capture pipeline (before input_firewall and pattern detection).
Keeps human intent and curated conclusions; rejects fully synthetic dumps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Max cleaned content length returned to callers
_MAX_CLEAN_CHARS = 20_000

# Block markers that wrap agent harness / tool output
_WRAPPER_BLOCK_RE = re.compile(
    r"```(?:json|xml|html|tool|output|result|console|bash|shell|log)?\s*\n"
    r"[\s\S]*?```",
    re.IGNORECASE,
)

_SYSTEM_REMINDER_RE = re.compile(
    r"(?:^|\n)\s*(?:system\s*reminder|system\s*prompt|internal\s*prompt|"
    r"<\s*system(?:-reminder)?\s*>[\s\S]*?<\s*/\s*system(?:-reminder)?\s*>|"
    r"\[SYSTEM\][\s\S]*?(?=\[(?:USER|HUMAN|ASSISTANT)\]|\Z))",
    re.IGNORECASE,
)

_TOOL_RESULT_RE = re.compile(
    r"(?:^|\n)\s*(?:tool\s*result|function\s*response|tool_output|"
    r"<\s*tool_result\s*>[\s\S]*?<\s*/\s*tool_result\s*>|"
    r"\[Tool\s*(?:Result|Output|Response)\][\s\S]*?(?=\n\[|\Z))",
    re.IGNORECASE,
)

_HARNESS_DUMP_RE = re.compile(
    r"(?:^|\n)\s*(?:file\s*dump|harness\s*output|raw\s*stdout|"
    r"command\s*output\s*:|"
    r"={3,}\s*(?:BEGIN|END)\s+(?:DUMP|OUTPUT|LOG)\s*={3,})",
    re.IGNORECASE,
)

_REPEATED_WRAPPER_RE = re.compile(
    r"(?:\[(?:USER|ASSISTANT|SYSTEM|HUMAN|BOT)\]\s*){3,}",
    re.IGNORECASE,
)

# Lines that are pure structural noise
_NOISE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"#{1,6}\s*(?:Relevant Memories|Related Information|Neural Memory)\b.*"
    r"|-\s*\[(?:concept|entity|decision|error|preference|insight|memory|fact|"
    r"workflow|instruction|pattern)\]\s.*"
    r"|\[NeuralMemory\s*[\u2014\u2013\-].*\]"
    r"|\[src=.*?conf=.*?\]"
    r"|tokens?\s*used\s*[:=]\s*\d+"
    r"|neurons?\s*activated\s*[:=]\s*\d+"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_MIN_HUMAN_CHARS = 20


@dataclass(frozen=True)
class CaptureDecision:
    """Result of capture hygiene cleaning.

    Attributes:
        content: Cleaned text (empty when rejected).
        accepted: Whether content is suitable for detection/persistence.
        reason: Machine-readable reason code.
        source: Capture source label (passive, explicit, hook, etc.).
    """

    content: str
    accepted: bool
    reason: str
    source: str


def clean_capture_input(raw: str, source: str = "passive") -> CaptureDecision:
    """Clean raw capture input and decide whether to accept it.

    Args:
        raw: Raw text from hooks, passive capture, or explicit remember.
        source: Origin label (``passive``, ``explicit``, ``hook``, ``tool``).

    Returns:
        ``CaptureDecision`` with cleaned content or a rejection reason.
    """
    if raw is None:
        return CaptureDecision(content="", accepted=False, reason="empty_input", source=source)

    text = str(raw)
    if not text.strip():
        return CaptureDecision(content="", accepted=False, reason="empty_input", source=source)

    original_len = len(text)

    # Strip fenced tool/output blocks
    text = _WRAPPER_BLOCK_RE.sub("\n", text)
    # Strip system reminders / internal prompts
    text = _SYSTEM_REMINDER_RE.sub("\n", text)
    # Strip tool result wrappers
    text = _TOOL_RESULT_RE.sub("\n", text)
    # Strip harness dump markers (keep surrounding human text)
    text = _HARNESS_DUMP_RE.sub("\n", text)
    # Collapse repeated role wrappers
    text = _REPEATED_WRAPPER_RE.sub("\n", text)
    # Drop Neural Memory self-echo lines
    text = _NOISE_LINE_RE.sub("", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) > _MAX_CLEAN_CHARS:
        text = text[:_MAX_CLEAN_CHARS].rstrip()

    if not text or len(text) < _MIN_HUMAN_CHARS:
        # Fully stripped → synthetic/noise only
        if original_len >= _MIN_HUMAN_CHARS:
            return CaptureDecision(
                content="",
                accepted=False,
                reason="synthetic_noise",
                source=source,
            )
        return CaptureDecision(content="", accepted=False, reason="too_short", source=source)

    # High ratio of stripping may still leave junk — check remaining signal
    remaining_ratio = len(text) / max(original_len, 1)
    if remaining_ratio < 0.05 and original_len > 500:
        return CaptureDecision(
            content="",
            accepted=False,
            reason="mostly_noise",
            source=source,
        )

    return CaptureDecision(
        content=text,
        accepted=True,
        reason="ok",
        source=source,
    )
