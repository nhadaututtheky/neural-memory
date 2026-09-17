"""Auto-capture pattern detection for extracting memories from text."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from neural_memory.extraction.keywords import STOP_WORDS_VI
from neural_memory.utils.simhash import is_near_duplicate, simhash

logger = logging.getLogger(__name__)

# Minimum text length to avoid false positives on tiny inputs
_MIN_TEXT_LENGTH = 20

# Maximum text length for regex processing — prevents ReDoS on huge inputs
_MAX_REGEX_TEXT_LENGTH = 50_000

# Vietnamese diacritical characters — used for language detection
_VI_DIACRITICS = re.compile(r"[ăâđêôơưắằẳẵặấầẩẫậếềểễệốồổỗộớờởỡợứừửữự]")

# Confidence penalty for Vietnamese auto-captures (regex is less reliable)
_VI_CONFIDENCE_PENALTY = 0.55

# Minimum captured content length for Vietnamese patterns
_VI_MIN_CAPTURE_LEN = 25

# Maximum ratio of stop words allowed in a Vietnamese capture
_VI_MAX_STOP_WORD_RATIO = 0.6

# CJK scripts — used for language detection and pattern classification.
# Han covers Chinese (and Japanese kanji); kana and hangul are unambiguous
# markers for Japanese/Korean, which this extractor has no patterns for.
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")

# Minimum captured content length for Chinese patterns. CJK has no word
# boundaries and packs more meaning per character, so the Latin-script
# thresholds (5-15 chars) would silently drop valid Chinese captures.
_CJK_MIN_CAPTURE_LEN = 4

# Share of Han characters above which text is treated as predominantly Chinese
_CHINESE_DOMINANCE_RATIO = 0.3

# Minimum input length before detection runs. CJK packs far more meaning per
# character, so the Latin floor (20) would silently discard valid one-sentence
# Chinese input such as "待办：更新部署文档。" (10 characters).
_CJK_MIN_TEXT_LENGTH = 8

# One-time warning flag for pyvi in auto-capture
_PYVI_AC_WARNED = False

# Type prefixes used for deduplication
_TYPE_PREFIXES = ("decision: ", "error: ", "todo: ", "insight: ", "preference: ")

DECISION_PATTERNS = [
    # English — deliberate choice language
    r"(?:we |I )(?:decided|chose|selected|picked|opted)(?: to)?[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?decision(?: is| was)?[:\s]+(.+?)(?:\.|$)",
    r"(?:chose|picked|selected) (.+?) (?:over|instead of) (.+?)(?:\.|$)",
    r"(?:switched|moved|migrated) (?:from .+? to|to) (.+?)(?:\.|$)",
    # Vietnamese
    r"(?:quyết định|chọn) (.+?) (?:thay vì|thay cho) (.+?)(?:\.|$)",
    r"(?:quyết định|đã chọn)[:\s]+(.+?)(?:\.|$)",
    # Chinese — CJK has no word boundaries, so no \b and no trailing dot anchor;
    # capture runs to the next sentence break or end of line instead.
    r"(?:我们|我|团队)?(?:决定|确定|选定|选用|采用|改用|换成|切换到|迁移到)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:方案|决定)(?:是|为|定为)[：:，,、\s]*([^\n。；;！!？?]{4,})",
]

ERROR_PATTERNS = [
    # English
    r"error[:\s]+(.+?)(?:\.|$)",
    r"failed[:\s]+(.+?)(?:\.|$)",
    r"bug[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?issue (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"problem[:\s]+(.+?)(?:\.|$)",
    r"(?:fixed|resolved|solved)(?: (?:it|this))? by[:\s]+(.+?)(?:\.|$)",
    r"(?:workaround|hack)[:\s]+(.+?)(?:\.|$)",
    # Vietnamese
    r"(?:lỗi|bug|vấn đề) (?:là|do|ở)[:\s]+(.+?)(?:\.|$)",
    r"(?:sửa|fix) (?:được |xong )?(?:bằng cách|bởi)[:\s]+(.+?)(?:\.|$)",
    # Chinese — errors and their fixes
    r"(?:错误|报错|出错|异常|失败)(?:原因)?[：:，,是为\s]+([^\n。；;！!？?]{4,})",
    r"(?:踩坑|坑点|坑)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:已修复|修好了|已经修好|解决了|已解决|已排除)[：:，,、\s]+([^\n。；;！!？?]{4,})",
]

TODO_PATTERNS = [
    # English
    r"(?:TODO|FIXME|HACK|XXX)[:\s]+(.+?)(?:\.|$)",
    r"(?:we |I )?(?:need to|should|must|have to)[:\s]+(.{5,80}?)(?:\.|,| but | or | and |$)",
    r"(?:remember to|don\'t forget to)[:\s]+(.+?)(?:\.|$)",
    r"(?:later|next)[:\s]+(.+?)(?:\.|$)",
    # Vietnamese — require compound verb+action (avoid bare cần/phải/nên)
    r"(?:cần phải|bắt buộc phải|nhất định phải) (\S+ .{10,80}?)(?:\.|$)",
    r"(?:nhớ là|đừng quên) (\S+ .{10,80}?)(?:\.|$)",
    # Chinese — action items. Bare verbs such as 需要/必须 appear mid-sentence in
    # ordinary statements ("因为 API 需要并发写入" is not a task), so a TODO
    # trigger only counts at the start of a sentence.
    r"(?:^|[。；;！!？?\n])\s*(?:待办(?:事项)?|下一步|接下来|后续|需要|还要|必须|记得|别忘了|不要忘记)[：:，,、\s]*([^\n。；;！!？?]{4,})",
]

FACT_PATTERNS = [
    # English
    r"(?:the |a )?(?:answer|solution|fix) (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"(?:it |this )(?:works|worked) because[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?(?:key|important|note)[:\s]+(.+?)(?:\.|$)",
    r"(?:learned|discovered|found out)[:\s]+(.+?)(?:\.|$)",
    # Vietnamese
    r"(?:đáp án|giải pháp|cách fix) (?:là|:)[:\s]+(.+?)(?:\.|$)",
    # Chinese
    r"(?:答案|解决办法|解决方案|结论)(?:是|为)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:原因是|原因在于)[：:，,、\s]*([^\n。；;！!？?]{4,})",
]

PREFERENCE_PATTERNS = [
    # English — explicit preferences
    r"(?:I |we )(?:prefer|like|want|favor)[:\s]+(.+?)(?:\.|$)",
    r"(?:I |we )(?:don\'t like|dislike|hate|avoid)[:\s]+(.+?)(?:\.|$)",
    r"(?:always|never) (?:use|do|include|add|write)[:\s]+(.+?)(?:\.|$)",
    r"(?:don\'t |do not |never )(?:use|do|include|add|write)[:\s]+(.+?)(?:\.|$)",
    r"(?:please |pls )?(?:stop|quit|avoid) (?:using|doing|adding)[:\s]+(.+?)(?:\.|$)",
    # English — corrections
    r"(?:that\'s |it\'s |this is )(?:wrong|incorrect|not right)[,:\s]+(.+?)(?:\.|$)",
    r"(?:actually|no)[,:\s]+(?:it |that )?should (?:be|have)[:\s]+(.+?)(?:\.|$)",
    r"(?:change|update|fix|correct) (?:it |that |this )?(?:to|from .+? to)[:\s]+(.+?)(?:\.|$)",
    r"(?:instead of .+?)[,:\s]+(?:use|do|try)[:\s]+(.+?)(?:\.|$)",
    # Vietnamese — preferences (require subject + verb + object for specificity)
    r"(?:tôi |mình |em |anh )(?:thích|muốn|ưu tiên) (?:dùng |xài |viết |code )?(.{10,}?)(?:\.|$)",
    r"(?:tôi |mình |em |anh )(?:không thích|ghét|không muốn|tránh) (.{10,}?)(?:\.|$)",
    r"(?:luôn luôn|lúc nào cũng) (?:dùng|làm|viết|thêm)[:\s]+(.+?)(?:\.|$)",
    r"(?:đừng bao giờ|cấm|không được) (?:dùng|làm|viết|thêm)[:\s]+(.+?)(?:\.|$)",
    # Vietnamese — corrections (require specific correction content)
    r"(?:sai rồi|không đúng|chưa đúng)[,:\s]+(.{10,}?)(?:\.|$)",
    r"(?:phải là|nên là|đúng ra là)[:\s]+(.{10,}?)(?:\.|$)",
    r"(?:sửa|đổi|chuyển) (?:lại |thành )(.{10,}?)(?:\.|$)",
    # Chinese — preferences and corrections
    r"(?:我们|我)(?:更喜欢|偏好|喜欢|倾向于|希望|想要)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:我们|我)(?:不喜欢|讨厌|不希望|不想|避免)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:不要|别|禁止|避免)(?:使用|用|做|写|添加|加上)[：:，,、\s]*([^\n。；;！!？?]{3,})",
    r"(?:应该|必须|始终|永远)(?:使用|用|做|写|添加|加上)[：:，,、\s]*([^\n。；;！!？?]{3,})",
]

INSIGHT_PATTERNS = [
    # English - "aha moments"
    r"turns out[:\s]+(.+?)(?:\.|$)",
    r"the trick (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"(?:I |we )(?:realized|discovered|figured out|noticed)(?: that)?[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?root cause (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"(?:it |this )(?:turns out|actually means)[:\s]+(.+?)(?:\.|$)",
    r"(?:lesson learned|takeaway|key insight)[:\s]+(.+?)(?:\.|$)",
    r"(?:TIL|today I learned)[:\s]+(.+?)(?:\.|$)",
    # Vietnamese
    r"(?:hóa ra|thì ra|té ra)[:\s]+(.+?)(?:\.|$)",
    r"(?:bài học|điều quan trọng)[:\s]+(.+?)(?:\.|$)",
    r"(?:nguyên nhân|root cause) (?:là|do)[:\s]+(.+?)(?:\.|$)",
    r"(?:mới biết|mới phát hiện)[:\s]+(.+?)(?:\.|$)",
    # Chinese
    r"(?:原来|发现|意识到|注意到)(?:是|了)?[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:根因|根本原因)(?:是|在于)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:经验教训|吸取教训|要点|关键在于)[：:，,、\s]*([^\n。；;！!？?]{3,})",
]


def _is_vietnamese_text(text: str) -> bool:
    """Check if text contains Vietnamese diacritical characters."""
    return bool(_VI_DIACRITICS.search(text))


def _is_vietnamese_pattern(pattern: str) -> bool:
    """Check if a regex pattern targets Vietnamese text."""
    return bool(_VI_DIACRITICS.search(pattern))


def _is_cjk_pattern(pattern: str) -> bool:
    """Check if a regex pattern targets Chinese (Han) text."""
    return bool(_HAN_RE.search(pattern))


def _cjk_script(text: str) -> str | None:
    """Classify the dominant CJK script in ``text``.

    Returns ``"chinese"``, ``"japanese"``, ``"korean"``, or ``None`` when the
    text is not predominantly CJK. Kana and hangul are unambiguous language
    markers, so a single occurrence wins; Han characters are shared between
    Chinese and Japanese, so they must dominate to count as Chinese.
    """
    stripped = text.strip()
    if not stripped:
        return None
    if _KANA_RE.search(stripped):
        return "japanese"
    if _HANGUL_RE.search(stripped):
        return "korean"
    han_count = len(_HAN_RE.findall(stripped))
    if han_count and han_count / len(stripped) >= _CHINESE_DOMINANCE_RATIO:
        return "chinese"
    return None


def empty_capture_hint(text: str) -> str:
    """Explain an empty detection result for non-Latin text.

    Without a hint, users of languages the extractor cannot read see
    "No memorable content detected" and cannot tell an empty input from an
    engine that never understood their text — the failure is silent.
    """
    script = _cjk_script(text)
    if script == "japanese":
        return (
            "No patterns matched. Japanese text is not yet supported by the "
            "extractor (patterns cover English, Vietnamese, and Chinese)."
        )
    if script == "korean":
        return (
            "No patterns matched. Korean text is not yet supported by the "
            "extractor (patterns cover English, Vietnamese, and Chinese)."
        )
    if script == "chinese":
        return (
            "No patterns matched. Chinese is supported, but detection needs an "
            "explicit trigger word: 决定/采用/改用/切换到 (decision), "
            "待办/需要/下一步 (todo), 错误/失败/报错 (error), "
            "已修复/解决了 (fix), 原来/发现 (insight)."
        )
    return ""


def _vi_quality_gate(captured: str) -> bool:
    """Check if a Vietnamese capture has enough meaningful content.

    Rejects captures where most words are stop words (no real content),
    or where the text is just a fragment without actionable information.

    Returns True if the capture passes quality checks.
    """
    words = re.findall(r"[a-zA-ZÀ-ỹ]+", captured.lower())
    if len(words) < 3:
        return False

    stop_count = sum(1 for w in words if w in STOP_WORDS_VI)
    ratio = stop_count / len(words)
    if ratio > _VI_MAX_STOP_WORD_RATIO:
        return False

    return True


def _warn_pyvi_missing() -> None:
    """Log a one-time warning when Vietnamese text is detected but pyvi is not installed."""
    global _PYVI_AC_WARNED
    if _PYVI_AC_WARNED:
        return
    try:
        import warnings

        with warnings.catch_warnings():
            # pyvi emits NumPy VisibleDeprecationWarning (UserWarning subclass, NOT
            # DeprecationWarning) during pickle.load at module import. Narrow
            # category filters miss it — suppress everything in this scoped block.
            warnings.simplefilter("ignore")
            import pyvi  # probe-only import to trigger ImportError if missing

        del pyvi
    except ImportError:
        logger.warning(
            "Vietnamese text detected in auto-capture but pyvi is not installed — "
            "keyword tags will be low-quality bigrams. "
            "Install with: pip install pyvi"
        )
        _PYVI_AC_WARNED = True


def _detect_patterns(
    text: str,
    patterns: list[str],
    memory_type: str,
    confidence: float,
    priority: int,
    min_match_len: int,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Run a list of regex patterns and return detected memories."""
    detected: list[dict[str, Any]] = []
    for pattern in patterns:
        is_vi = _is_vietnamese_pattern(pattern)
        is_cjk = _is_cjk_pattern(pattern)
        if is_vi:
            effective_min_len = max(min_match_len, _VI_MIN_CAPTURE_LEN)
        elif is_cjk:
            # Chinese packs more meaning per character and has no word
            # boundaries to lean on, so the Latin thresholds (5-15 chars)
            # would silently drop valid captures such as "改用 SQLite".
            effective_min_len = min(min_match_len, _CJK_MIN_CAPTURE_LEN)
        else:
            effective_min_len = min_match_len

        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Handle tuple matches from patterns with multiple groups
            if isinstance(match, tuple):
                match = " ".join(part for part in match if part)
            captured = match.strip()
            if len(captured) < effective_min_len:
                continue

            # Vietnamese quality gate — reject fragments with no real content
            if is_vi and not _vi_quality_gate(captured):
                continue

            # Adjust confidence based on capture quality. The "too short" floor is
            # script-aware: one Han character carries roughly a word of meaning,
            # so the Latin 10-char threshold would wrongly penalise valid Chinese
            # captures such as "没装 tsc".
            short_capture_floor = _CJK_MIN_CAPTURE_LEN if is_cjk else 10
            adjusted_confidence = confidence
            if is_vi:
                adjusted_confidence *= _VI_CONFIDENCE_PENALTY
            if len(captured) > 200:
                adjusted_confidence *= 0.7  # Penalize truly excessive captures
            elif len(captured) < short_capture_floor:
                adjusted_confidence *= 0.3  # Penalize too-short captures

            # Trim at sentence boundary if over-captured
            if len(captured) > 100:
                for sep in (".", "!", "?", ";"):
                    idx = captured.find(sep, 50)
                    if idx > 0:
                        captured = captured[:idx]
                        break

            content = f"{prefix}{captured}" if prefix else captured
            detected.append(
                {
                    "type": memory_type,
                    "content": content,
                    "confidence": adjusted_confidence,
                    "priority": priority,
                }
            )
    return detected


def _dedup_key(content: str) -> str:
    """Create a deduplication key by stripping type prefix and hashing."""
    key = content.lower()
    for prefix in _TYPE_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    return hashlib.md5(key.encode()).hexdigest()


def analyze_text_for_memories(
    text: str,
    *,
    capture_decisions: bool = True,
    capture_errors: bool = True,
    capture_todos: bool = True,
    capture_facts: bool = True,
    capture_insights: bool = True,
    capture_preferences: bool = True,
) -> list[dict[str, Any]]:
    """Analyze text and detect potential memories.

    Returns list of detected memories with type, content, and confidence.
    """
    stripped = text.strip()
    min_text_length = (
        _CJK_MIN_TEXT_LENGTH if _cjk_script(stripped) == "chinese" else _MIN_TEXT_LENGTH
    )
    if len(stripped) < min_text_length:
        return []

    # Truncate to prevent ReDoS on very large inputs
    if len(text) > _MAX_REGEX_TEXT_LENGTH:
        text = text[:_MAX_REGEX_TEXT_LENGTH]

    detected: list[dict[str, Any]] = []
    text_lower = text.lower()

    # Warn once if Vietnamese text detected but pyvi not installed
    if _is_vietnamese_text(text_lower):
        _warn_pyvi_missing()

    if capture_decisions:
        detected.extend(
            _detect_patterns(text_lower, DECISION_PATTERNS, "decision", 0.8, 6, 10, "Decision: ")
        )

    if capture_errors:
        detected.extend(
            _detect_patterns(text_lower, ERROR_PATTERNS, "error", 0.85, 7, 10, "Error: ")
        )

    if capture_todos:
        detected.extend(_detect_patterns(text, TODO_PATTERNS, "todo", 0.75, 5, 5, "TODO: "))

    if capture_facts:
        detected.extend(_detect_patterns(text_lower, FACT_PATTERNS, "fact", 0.7, 5, 15))

    if capture_insights:
        detected.extend(
            _detect_patterns(text_lower, INSIGHT_PATTERNS, "insight", 0.8, 6, 15, "Insight: ")
        )

    if capture_preferences:
        detected.extend(
            _detect_patterns(
                text_lower, PREFERENCE_PATTERNS, "preference", 0.85, 7, 10, "Preference: "
            )
        )

    # Remove duplicates: exact MD5 match + SimHash near-duplicate
    seen_exact: set[str] = set()
    seen_hashes: list[int] = []
    unique_detected: list[dict[str, Any]] = []
    for item in detected:
        content_key = _dedup_key(item["content"])
        if content_key in seen_exact:
            continue

        # Check simhash near-duplicate against already-seen items
        item_hash = simhash(item["content"])
        if any(is_near_duplicate(item_hash, h) for h in seen_hashes):
            continue

        seen_exact.add(content_key)
        seen_hashes.append(item_hash)
        unique_detected.append(item)

    return unique_detected
