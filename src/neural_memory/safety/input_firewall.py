"""Input firewall for auto-capture pipeline.

Prevents garbage, oversized, or adversarial content from entering the
memory graph through hooks and passive capture. Acts as Gate 1 in the
memory quality pipeline.

Detects:
- Oversized content (> 10KB for auto-capture)
- High-entropy / binary content
- Chat platform control sequences (<ctrl*>, protocol metadata)
- Fake role injection (user/assistant/system tags in content)
- Raw JSON metadata blocks (sender_id, message_id patterns)
- Repetitive content (copy-paste loops)

Auto-capture path (hooks, passive capture) uses check_content() which blocks.
Explicit path (nmem_remember) uses sanitize_explicit_content() which strips
dangerous patterns but never blocks — the caller made a deliberate decision.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- Thresholds ---

# Max content length for auto-capture (explicit nmem_remember has its own 100K limit)
MAX_AUTO_CAPTURE_CHARS = 10_000

# Minimum meaningful content
MIN_CONTENT_CHARS = 30

# Repetition: if the most common 3-gram appears > this ratio, it's repetitive
_REPETITION_RATIO_THRESHOLD = 0.3

# Entropy: below this = very repetitive/uniform content
_MIN_ENTROPY_THRESHOLD = 1.5

# --- Control sequence / injection patterns ---

# Chat platform control sequences (Zalo, Telegram, etc.)
_CONTROL_SEQ_RE = re.compile(
    r"<ctrl\d+>"  # Zalo control tags
    r"|<\/?(?:user|assistant|system|human|bot)\b[^>]*>"  # Fake role tags
    r"|\x00|\x01|\x02|\x03|\x04|\x05|\x06|\x07"  # Binary control chars
    r"|\x0e|\x0f|\x10|\x11|\x12|\x13|\x14|\x15|\x16|\x17|\x18|\x19|\x1a|\x1b|\x1c|\x1d|\x1e|\x1f",
    re.IGNORECASE,
)

# Fake conversation metadata injection
_METADATA_INJECTION_RE = re.compile(
    r"\"(?:sender_id|message_id|sender|recipient|chat_id)\"\s*:"  # JSON chat metadata
    r"|\"(?:role|type)\"\s*:\s*\"(?:user|assistant|system|tool)\""  # Fake role in JSON
    r"|Conversation\s+info\s*\((?:untrusted\s+)?metadata\)"  # Explicit metadata block
    r"|Sender\s*\((?:untrusted\s+)?metadata\)",
    re.IGNORECASE,
)

# Base64-like blocks (>100 chars of base64 alphabet without spaces)
_BASE64_BLOCK_RE = re.compile(r"[A-Za-z0-9+/=]{100,}")

# NeuralMemory context noise — self-referential metadata that gets re-ingested
# and creates junk neurons like "[concept] json message id"
_NM_CONTEXT_NOISE_RE = re.compile(
    r"^#{1,3}\s*(?:Relevant Memories|Related Information|Relevant Context|Neural Memory)\b.*$"
    r"|^\[NeuralMemory\s*[\u2014\u2013\-].*\]$"  # em-dash, en-dash, hyphen
    r"|^-\s*\[(?:concept|entity|decision|error|preference|insight|memory|fact|workflow|instruction|pattern)\]\s"
    r"|^(?:Conversation info|Sender|Context)\s*\((?:untrusted\s+)?metadata\).*$",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass(frozen=True)
class FirewallResult:
    """Result of content firewall check.

    Attributes:
        blocked: Whether the content was blocked.
        reason: Human-readable reason for blocking (empty if not blocked).
        sanitized: Cleaned content (control sequences stripped). Only set if
                   content passed but needed sanitization.
    """

    blocked: bool
    reason: str = ""
    sanitized: str = ""


def check_content(text: str) -> FirewallResult:
    """Run all firewall checks on content destined for auto-capture.

    This is the main entry point. Call before analyze_text_for_memories()
    in hooks and passive capture paths.

    Args:
        text: Raw content to check.

    Returns:
        FirewallResult with blocked=True if content should be rejected,
        or sanitized content if it passed with modifications.
    """
    if not text or not isinstance(text, str):
        return FirewallResult(blocked=True, reason="empty or non-string content")

    # --- Size gate ---
    if len(text) > MAX_AUTO_CAPTURE_CHARS:
        logger.debug(
            "Input firewall: blocked oversized content (%d chars, max %d)",
            len(text),
            MAX_AUTO_CAPTURE_CHARS,
        )
        return FirewallResult(blocked=True, reason=f"content too large ({len(text)} chars)")

    if len(text.strip()) < MIN_CONTENT_CHARS:
        return FirewallResult(blocked=True, reason="content too short")

    # --- Control sequence detection ---
    control_matches = _CONTROL_SEQ_RE.findall(text)
    if len(control_matches) >= 2:
        logger.debug(
            "Input firewall: blocked content with %d control sequences",
            len(control_matches),
        )
        return FirewallResult(
            blocked=True,
            reason=f"contains {len(control_matches)} control sequences (possible platform artifact)",
        )

    # --- Metadata injection detection ---
    metadata_matches = _METADATA_INJECTION_RE.findall(text)
    if len(metadata_matches) >= 2:
        logger.debug(
            "Input firewall: blocked content with %d metadata injection patterns",
            len(metadata_matches),
        )
        return FirewallResult(
            blocked=True,
            reason="contains chat metadata patterns (possible prompt injection)",
        )

    # --- Base64 / binary block detection ---
    base64_blocks = _BASE64_BLOCK_RE.findall(text)
    base64_chars = sum(len(b) for b in base64_blocks)
    if base64_chars > len(text) * 0.3:
        return FirewallResult(
            blocked=True,
            reason="content is mostly base64/binary data",
        )

    # --- Repetition detection ---
    if _is_highly_repetitive(text):
        return FirewallResult(
            blocked=True,
            reason="content is highly repetitive (copy-paste loop)",
        )

    # --- Low entropy detection ---
    entropy = _char_entropy(text)
    if entropy < _MIN_ENTROPY_THRESHOLD and len(text) > 100:
        return FirewallResult(
            blocked=True,
            reason=f"content entropy too low ({entropy:.2f}, min {_MIN_ENTROPY_THRESHOLD})",
        )

    # --- Sanitize: strip NM context noise (self-referential re-ingest) ---
    sanitized = _NM_CONTEXT_NOISE_RE.sub("", text)

    # --- Sanitize: strip any remaining single control sequences ---
    sanitized = _CONTROL_SEQ_RE.sub("", sanitized)
    # Collapse multiple whitespace from removals
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()

    if sanitized != text:
        stripped_chars = len(text) - len(sanitized)
        logger.debug("Input firewall: sanitized %d chars (control + NM noise)", stripped_chars)

    # Re-check length after sanitization — if only noise remained, block
    if len(sanitized.strip()) < MIN_CONTENT_CHARS:
        return FirewallResult(blocked=True, reason="content too short after noise removal")

    return FirewallResult(blocked=False, sanitized=sanitized)


def strip_nm_context_noise(text: str) -> str:
    """Strip NeuralMemory context wrappers and metadata noise from text.

    Use this to clean text that may contain re-ingested NM output before
    feeding it back into auto-capture. Removes:
    - ## Relevant Memories / ## Neural Memory section headers
    - [NeuralMemory — ...] wrapper lines
    - Neuron-type bullet lines (- [concept] ..., - [error] ...)
    - Metadata labels (Conversation info, Sender)

    Args:
        text: Raw text potentially containing NM context noise.

    Returns:
        Cleaned text with NM noise removed.
    """
    if not text or not isinstance(text, str):
        return text

    cleaned = _NM_CONTEXT_NOISE_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def sanitize_explicit_content(text: str) -> str:
    """Sanitize content for explicit nmem_remember path.

    Unlike check_content() (which blocks), this strips dangerous patterns
    but does NOT block content — the caller made a deliberate decision to
    store it.

    Strips:
    - Binary control characters and chat-platform artifacts (<ctrl*>)
    - Fake role tags (<user>, <assistant>, etc.)
    - Chat metadata injection patterns (sender_id, message_id in JSON)
    - NeuralMemory context noise (re-ingested NM output headers/bullets)

    Safe for technical content: base64, JSON data structures, code snippets
    all pass through unchanged. Only adversarial/artifact patterns are removed.

    Args:
        text: Raw content from explicit remember call.

    Returns:
        Sanitized content with dangerous patterns stripped.
    """
    if not text or not isinstance(text, str):
        return text

    sanitized = text

    # Strip control sequences and fake role tags
    sanitized = _CONTROL_SEQ_RE.sub("", sanitized)

    # Strip chat metadata injection patterns (non-blocking: remove, don't reject)
    sanitized = _METADATA_INJECTION_RE.sub("", sanitized)

    # Strip NM context noise (prevents re-ingest loops)
    sanitized = _NM_CONTEXT_NOISE_RE.sub("", sanitized)

    # Collapse excessive whitespace from removals
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()

    if sanitized != text:
        stripped_chars = len(text) - len(sanitized)
        logger.debug(
            "Explicit path: sanitized %d chars (control + metadata + NM noise)",
            stripped_chars,
        )

    return sanitized


def _is_highly_repetitive(text: str) -> bool:
    """Check if text contains excessive repetition using 3-gram frequency."""
    if len(text) < 100:
        return False

    # Sample from text (avoid O(n²) on large inputs)
    sample = text[:5000]
    words = sample.lower().split()
    if len(words) < 10:
        return False

    # Count 3-grams
    trigrams: dict[str, int] = {}
    for i in range(len(words) - 2):
        gram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
        trigrams[gram] = trigrams.get(gram, 0) + 1

    if not trigrams:
        return False

    total = sum(trigrams.values())
    max_count = max(trigrams.values())

    return max_count / total > _REPETITION_RATIO_THRESHOLD


def _char_entropy(text: str) -> float:
    """Calculate Shannon entropy of character distribution."""
    if not text:
        return 0.0

    # Sample for performance
    sample = text[:5000]
    freq: dict[str, int] = {}
    for char in sample:
        freq[char] = freq.get(char, 0) + 1

    total = len(sample)
    entropy = 0.0
    for count in freq.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy
