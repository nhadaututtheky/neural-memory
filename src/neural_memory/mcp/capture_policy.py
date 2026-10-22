"""Language-specific auto-capture policies.

English, Vietnamese, and Chinese each own their patterns and thresholds.
``analyze_text_for_memories`` still runs every registered policy, in
category order, so a mixed sentence keeps the same matches as the old
combined lists. A new language is one subclass plus ``register_policy``.

Japanese and Korean are detected, but they have no policy yet. Callers
fall back to the Latin input floor and report those scripts as unsupported.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from neural_memory.extraction.keywords import STOP_WORDS_VI

# Minimum text length to avoid false positives on tiny inputs.
_MIN_TEXT_LENGTH = 20

# Vietnamese diacritical characters — used for language detection.
_VI_DIACRITICS = re.compile(r"[ăâđêôơưắằẳẵặấầẩẫậếềểễệốồổỗộớờởỡợứừửữự]")

# Confidence penalty for Vietnamese auto-captures (regex is less reliable).
_VI_CONFIDENCE_PENALTY = 0.55

# Minimum captured content length for Vietnamese patterns.
_VI_MIN_CAPTURE_LEN = 25

# Maximum ratio of stop words allowed in a Vietnamese capture.
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

# Share of Han characters above which text is treated as predominantly Chinese.
_CHINESE_DOMINANCE_RATIO = 0.3

# Minimum input length before detection runs. CJK packs far more meaning per
# character, so the Latin floor (20) would silently discard valid one-sentence
# Chinese input such as a short todo line (10 characters).
_CJK_MIN_TEXT_LENGTH = 8

# Trailing sentence particles that carry no meaning once captured as a memory.
_CJK_TRAILING_PARTICLES = re.compile(r"[了吧呢啊嘛呀哦]+$")

# Latin "too short" confidence floor. One Han character is roughly a word,
# so Chinese uses ``_CJK_MIN_CAPTURE_LEN`` instead of this.
_LATIN_SHORT_CAPTURE_FLOOR = 10


def _is_vietnamese_text(text: str) -> bool:
    """Check if text contains Vietnamese diacritical characters."""
    return bool(_VI_DIACRITICS.search(text))


def _cjk_script(text: str) -> str | None:
    """Classify the dominant CJK script in ``text``.

    Returns ``"chinese"``, ``"japanese"``, ``"korean"``, or ``None`` when the
    text is not predominantly CJK. A few kana inside a Han-dominant sentence
    stay Chinese (a stray ``の`` must not flip the length floor). A real
    Japanese sentence, with more kana, stays Japanese even when kanji are common.
    """
    stripped = text.strip()
    if not stripped:
        return None
    han_count = len(_HAN_RE.findall(stripped))
    kana_count = len(_KANA_RE.findall(stripped))
    hangul_count = len(_HANGUL_RE.findall(stripped))
    han_dominant = bool(han_count) and han_count / len(stripped) >= _CHINESE_DOMINANCE_RATIO
    # A handful of kana in otherwise Chinese text is not Japanese.
    few_kana = kana_count <= max(2, han_count // 10)
    if han_dominant and few_kana and hangul_count <= han_count:
        return "chinese"
    if hangul_count and hangul_count >= han_count and hangul_count >= kana_count:
        return "korean"
    if kana_count:
        return "japanese"
    if han_dominant:
        return "chinese"
    return None


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


@dataclass(frozen=True)
class CaptureSpec:
    """One memory type's patterns and the thresholds shared by a policy."""

    memory_type: str
    patterns: tuple[str, ...]
    confidence: float
    priority: int
    min_match_len: int
    prefix: str
    lowercase: bool


@dataclass(frozen=True)
class PatternMatch:
    """One capture. ``as_dict`` is the public memory shape.

    The type prefix is already in ``content``. Quality checks see the span
    before that prefix, so a Vietnamese stop-word count is not diluted by
    ``Decision:``.
    """

    memory_type: str
    content: str
    confidence: float
    priority: int

    def as_dict(self) -> dict[str, Any]:
        """Return the dict ``analyze_text_for_memories`` has always returned."""
        return {
            "type": self.memory_type,
            "content": self.content,
            "confidence": self.confidence,
            "priority": self.priority,
        }


def _captured_text(raw: object) -> str | None:
    """Flatten one ``re.findall`` hit to the captured span."""
    if isinstance(raw, tuple):
        return " ".join(part for part in raw if isinstance(part, str) and part).strip()
    if isinstance(raw, str):
        return raw.strip()
    return None


def _apply_length_penalty(confidence: float, length: int, short_floor: int) -> float:
    """Penalize huge captures and captures under the script's short floor."""
    if length > 200:
        return confidence * 0.7
    if length < short_floor:
        return confidence * 0.3
    return confidence


def _trim_sentence(captured: str) -> str:
    """Cut an over-capture at the first sentence break after 50 characters."""
    if len(captured) <= 100:
        return captured
    for sep in (".", "!", "?", ";"):
        idx = captured.find(sep, 50)
        if idx > 0:
            return captured[:idx]
    return captured


class CapturePolicy(ABC):
    """Language-specific capture rules.

    Subclasses supply patterns. Scanning, the length penalty, and sentence
    trimming stay here, so a new language does not copy the loop.
    """

    language_code: str
    min_input_length: int
    min_capture_length: int

    @abstractmethod
    def specs(self) -> tuple[CaptureSpec, ...]:
        """Return this language's pattern groups, in detection order."""

    def quality_gate(self, content: str) -> float:
        """Return a confidence multiplier. ``0`` rejects the capture.

        ``content`` is the captured span before the type prefix.
        """
        del content
        return 1.0

    def effective_min_len(self, requested: int) -> int:
        """Return the minimum captured length for a pattern group."""
        return requested

    def short_capture_floor(self) -> int:
        """Return the length under which confidence is penalized."""
        return _LATIN_SHORT_CAPTURE_FLOOR

    def trim_captured(self, captured: str) -> str:
        """Drop script-specific trailing particles. Latin text is unchanged."""
        return captured

    def empty_hint(self, text: str) -> str:
        """Explain an empty result for this language. Default: no hint."""
        del text
        return ""

    def matches_for(self, text: str, memory_type: str) -> list[PatternMatch]:
        """Scan one memory type. Unknown types contribute nothing."""
        for spec in self.specs():
            if spec.memory_type == memory_type:
                return self._scan(text, spec)
        return []

    def detect_patterns(
        self,
        text: str,
        *,
        capture_decisions: bool = True,
        capture_errors: bool = True,
        capture_todos: bool = True,
        capture_facts: bool = True,
        capture_insights: bool = True,
        capture_preferences: bool = True,
    ) -> list[PatternMatch]:
        """Detect this language's memories. Other languages are not scanned."""
        enabled = {
            "decision": capture_decisions,
            "error": capture_errors,
            "todo": capture_todos,
            "fact": capture_facts,
            "insight": capture_insights,
            "preference": capture_preferences,
        }
        found: list[PatternMatch] = []
        for spec in self.specs():
            if not enabled.get(spec.memory_type, False):
                continue
            found.extend(self._scan(text, spec))
        return found

    def _scan(self, text: str, spec: CaptureSpec) -> list[PatternMatch]:
        """Apply this policy's thresholds to one pattern group."""
        source = text.lower() if spec.lowercase else text
        min_len = self.effective_min_len(spec.min_match_len)
        short_floor = self.short_capture_floor()
        found: list[PatternMatch] = []
        for pattern in spec.patterns:
            for raw in re.findall(pattern, source, re.IGNORECASE):
                captured = _captured_text(raw)
                if captured is None:
                    continue
                captured = self.trim_captured(captured)
                if len(captured) < min_len:
                    continue
                factor = self.quality_gate(captured)
                if factor <= 0:
                    continue
                adjusted = _apply_length_penalty(
                    spec.confidence * factor, len(captured), short_floor
                )
                trimmed = _trim_sentence(captured)
                content = f"{spec.prefix}{trimmed}" if spec.prefix else trimmed
                found.append(
                    PatternMatch(
                        memory_type=spec.memory_type,
                        content=content,
                        confidence=adjusted,
                        priority=spec.priority,
                    )
                )
        return found


# English — deliberate choice language.
_LATIN_DECISIONS: tuple[str, ...] = (
    r"(?:we |I )(?:decided|chose|selected|picked|opted)(?: to)?[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?decision(?: is| was)?[:\s]+(.+?)(?:\.|$)",
    r"(?:chose|picked|selected) (.+?) (?:over|instead of) (.+?)(?:\.|$)",
    r"(?:switched|moved|migrated) (?:from .+? to|to) (.+?)(?:\.|$)",
)
_LATIN_ERRORS: tuple[str, ...] = (
    r"error[:\s]+(.+?)(?:\.|$)",
    r"failed[:\s]+(.+?)(?:\.|$)",
    r"bug[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?issue (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"problem[:\s]+(.+?)(?:\.|$)",
    r"(?:fixed|resolved|solved)(?: (?:it|this))? by[:\s]+(.+?)(?:\.|$)",
    r"(?:workaround|hack)[:\s]+(.+?)(?:\.|$)",
)
_LATIN_TODOS: tuple[str, ...] = (
    r"(?:TODO|FIXME|HACK|XXX)[:\s]+(.+?)(?:\.|$)",
    r"(?:we |I )?(?:need to|should|must|have to)[:\s]+(.{5,80}?)(?:\.|,| but | or | and |$)",
    r"(?:remember to|don\'t forget to)[:\s]+(.+?)(?:\.|$)",
    r"(?:later|next)[:\s]+(.+?)(?:\.|$)",
)
_LATIN_FACTS: tuple[str, ...] = (
    r"(?:the |a )?(?:answer|solution|fix) (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"(?:it |this )(?:works|worked) because[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?(?:key|important|note)[:\s]+(.+?)(?:\.|$)",
    r"(?:learned|discovered|found out)[:\s]+(.+?)(?:\.|$)",
)
_LATIN_PREFERENCES: tuple[str, ...] = (
    r"(?:I |we )(?:prefer|like|want|favor)[:\s]+(.+?)(?:\.|$)",
    r"(?:I |we )(?:don\'t like|dislike|hate|avoid)[:\s]+(.+?)(?:\.|$)",
    r"(?:always|never) (?:use|do|include|add|write)[:\s]+(.+?)(?:\.|$)",
    r"(?:don\'t |do not |never )(?:use|do|include|add|write)[:\s]+(.+?)(?:\.|$)",
    r"(?:please |pls )?(?:stop|quit|avoid) (?:using|doing|adding)[:\s]+(.+?)(?:\.|$)",
    r"(?:that\'s |it\'s |this is )(?:wrong|incorrect|not right)[,:\s]+(.+?)(?:\.|$)",
    r"(?:actually|no)[,:\s]+(?:it |that )?should (?:be|have)[:\s]+(.+?)(?:\.|$)",
    r"(?:change|update|fix|correct) (?:it |that |this )?(?:to|from .+? to)[:\s]+(.+?)(?:\.|$)",
    r"(?:instead of .+?)[,:\s]+(?:use|do|try)[:\s]+(.+?)(?:\.|$)",
)
_LATIN_INSIGHTS: tuple[str, ...] = (
    r"turns out[:\s]+(.+?)(?:\.|$)",
    r"the trick (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"(?:I |we )(?:realized|discovered|figured out|noticed)(?: that)?[:\s]+(.+?)(?:\.|$)",
    r"(?:the )?root cause (?:is|was)[:\s]+(.+?)(?:\.|$)",
    r"(?:it |this )(?:turns out|actually means)[:\s]+(.+?)(?:\.|$)",
    r"(?:lesson learned|takeaway|key insight)[:\s]+(.+?)(?:\.|$)",
    r"(?:TIL|today I learned)[:\s]+(.+?)(?:\.|$)",
)


def _groups(
    decisions: tuple[str, ...],
    errors: tuple[str, ...],
    todos: tuple[str, ...],
    facts: tuple[str, ...],
    insights: tuple[str, ...],
    preferences: tuple[str, ...],
) -> tuple[CaptureSpec, ...]:
    """Build the six memory-type groups in the historical scan order."""
    return (
        CaptureSpec("decision", decisions, 0.8, 6, 10, "Decision: ", True),
        CaptureSpec("error", errors, 0.85, 7, 10, "Error: ", True),
        CaptureSpec("todo", todos, 0.75, 5, 5, "TODO: ", False),
        CaptureSpec("fact", facts, 0.7, 5, 15, "", True),
        CaptureSpec("insight", insights, 0.8, 6, 15, "Insight: ", True),
        CaptureSpec("preference", preferences, 0.85, 7, 10, "Preference: ", True),
    )


class LatinPolicy(CapturePolicy):
    """English and any text that does not match a more specific policy."""

    language_code = "en"
    min_input_length = _MIN_TEXT_LENGTH
    min_capture_length = 5

    def specs(self) -> tuple[CaptureSpec, ...]:
        return _LATIN_SPECS


_LATIN_SPECS = _groups(
    _LATIN_DECISIONS,
    _LATIN_ERRORS,
    _LATIN_TODOS,
    _LATIN_FACTS,
    _LATIN_INSIGHTS,
    _LATIN_PREFERENCES,
)


# Vietnamese — compound verbs only. Bare cần/phải/nên are too broad.
_VI_DECISIONS: tuple[str, ...] = (
    r"(?:quyết định|chọn) (.+?) (?:thay vì|thay cho) (.+?)(?:\.|$)",
    r"(?:quyết định|đã chọn)[:\s]+(.+?)(?:\.|$)",
)
_VI_ERRORS: tuple[str, ...] = (
    r"(?:lỗi|bug|vấn đề) (?:là|do|ở)[:\s]+(.+?)(?:\.|$)",
    r"(?:sửa|fix) (?:được |xong )?(?:bằng cách|bởi)[:\s]+(.+?)(?:\.|$)",
)
_VI_TODOS: tuple[str, ...] = (
    r"(?:cần phải|bắt buộc phải|nhất định phải) (\S+ .{10,80}?)(?:\.|$)",
    r"(?:nhớ là|đừng quên) (\S+ .{10,80}?)(?:\.|$)",
)
_VI_FACTS: tuple[str, ...] = (r"(?:đáp án|giải pháp|cách fix) (?:là|:)[:\s]+(.+?)(?:\.|$)",)
_VI_PREFERENCES: tuple[str, ...] = (
    r"(?:tôi |mình |em |anh )(?:thích|muốn|ưu tiên) (?:dùng |xài |viết |code )?(.{10,}?)(?:\.|$)",
    r"(?:tôi |mình |em |anh )(?:không thích|ghét|không muốn|tránh) (.{10,}?)(?:\.|$)",
    r"(?:luôn luôn|lúc nào cũng) (?:dùng|làm|viết|thêm)[:\s]+(.+?)(?:\.|$)",
    r"(?:đừng bao giờ|cấm|không được) (?:dùng|làm|viết|thêm)[:\s]+(.+?)(?:\.|$)",
    r"(?:sai rồi|không đúng|chưa đúng)[,:\s]+(.{10,}?)(?:\.|$)",
    r"(?:phải là|nên là|đúng ra là)[:\s]+(.{10,}?)(?:\.|$)",
    r"(?:sửa|đổi|chuyển) (?:lại |thành )(.{10,}?)(?:\.|$)",
)
_VI_INSIGHTS: tuple[str, ...] = (
    r"(?:hóa ra|thì ra|té ra)[:\s]+(.+?)(?:\.|$)",
    r"(?:bài học|điều quan trọng)[:\s]+(.+?)(?:\.|$)",
    r"(?:nguyên nhân|root cause) (?:là|do)[:\s]+(.+?)(?:\.|$)",
    r"(?:mới biết|mới phát hiện)[:\s]+(.+?)(?:\.|$)",
)


class VietnamesePolicy(CapturePolicy):
    """Vietnamese patterns plus the stop-word gate and confidence penalty."""

    language_code = "vi"
    min_input_length = _MIN_TEXT_LENGTH
    min_capture_length = _VI_MIN_CAPTURE_LEN

    def specs(self) -> tuple[CaptureSpec, ...]:
        return _VI_SPECS

    def effective_min_len(self, requested: int) -> int:
        return max(requested, self.min_capture_length)

    def quality_gate(self, content: str) -> float:
        if not _vi_quality_gate(content):
            return 0.0
        return _VI_CONFIDENCE_PENALTY


_VI_SPECS = _groups(
    _VI_DECISIONS,
    _VI_ERRORS,
    _VI_TODOS,
    _VI_FACTS,
    _VI_INSIGHTS,
    _VI_PREFERENCES,
)


# Chinese — an explicit marker is required. Descriptive verbs (采用/改用/
# 切换到/迁移到) and bare 需要/必须/记得/发现 are ordinary prose, not memories.
_ZH_DECISIONS: tuple[str, ...] = (
    r"(?:我们|团队|(?<![\u4e00-\u9fff])我)(?:决定|选定|敲定|拍板)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:方案|决策|结论|决定)(?:是|为|定为|就是)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:决定|决策|方案)[：:]([^\n。；;！!？?]{4,})",
)
_ZH_ERRORS: tuple[str, ...] = (
    r"(?:错误|报错|出错|异常|失败)(?:原因)?[：:，,\s]+([^\n。；;！!？?]{4,})",
    r"(?:踩坑|坑点)[：:，,、\s]+([^\n。；;！!？?]{4,})",
    r"(?:已修复|修好了|已经修好|解决了|已解决|已排除)[：:，,、\s]+([^\n。；;！!？?]{4,})",
)
_ZH_TODOS: tuple[str, ...] = (
    r"(?:^|[。；;！!？?\n])\s*(?:待办(?:事项)?|下一步|后续)[：:，,、\s]+([^\n。；;！!？?]{4,})",
    r"(?:^|[。；;！!？?\n])\s*接下来[：:]([^\n。；;！!？?]{4,})",
    r"(?:^|[。；;！!？?\n])\s*(?:记得(?:要|去)|别忘了|不要忘记)[：:，,、\s]*([^\n。；;！!？?]{4,})",
)
_ZH_FACTS: tuple[str, ...] = (
    r"(?:答案|解决办法|解决方案|结论)(?:是|为)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:原因是|原因在于)[：:，,、\s]*([^\n。；;！!？?]{4,})",
)
_ZH_PREFERENCES: tuple[str, ...] = (
    r"(?:我们|(?<![\u4e00-\u9fff])我)(?:更喜欢|偏好|喜欢|倾向于|希望|想要)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:我们|(?<![\u4e00-\u9fff])我)(?:不喜欢|讨厌|不希望|不想|避免)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:不要|(?<![\u4e00-\u9fff])别|禁止|避免)(?:使用|用|做|写|添加|加上)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:应该|必须|始终|永远)(?:使用|用|做|写|添加|加上)[：:，,、\s]*([^\n。；;！!？?]{4,})",
)
_ZH_INSIGHTS: tuple[str, ...] = (
    r"(?:原来如此|这才明白|才明白)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:根因|根本原因)(?:是|在于)[：:，,、\s]*([^\n。；;！!？?]{4,})",
    r"(?:经验教训|吸取教训|关键在于)[：:，,、\s]*([^\n。；;！!？?]{4,})",
)


class ChinesePolicy(CapturePolicy):
    """Chinese patterns. The input floor is 8 characters, not the Latin 20."""

    language_code = "zh"
    min_input_length = _CJK_MIN_TEXT_LENGTH
    min_capture_length = _CJK_MIN_CAPTURE_LEN

    def specs(self) -> tuple[CaptureSpec, ...]:
        return _ZH_SPECS

    def effective_min_len(self, requested: int) -> int:
        # Chinese packs more meaning per character, so the Latin thresholds
        # (5-15 chars) would silently drop valid captures such as "改用 SQLite".
        return min(requested, self.min_capture_length)

    def short_capture_floor(self) -> int:
        return self.min_capture_length

    def trim_captured(self, captured: str) -> str:
        return _CJK_TRAILING_PARTICLES.sub("", captured).strip()

    def empty_hint(self, text: str) -> str:
        del text
        return (
            "No patterns matched. Chinese is supported, but detection needs an "
            "explicit trigger word: 决定/选定 (decision), 待办/下一步 (todo), "
            "错误/失败/已修复 (error), 原来如此/根因 (insight)."
        )


_ZH_SPECS = _groups(
    _ZH_DECISIONS,
    _ZH_ERRORS,
    _ZH_TODOS,
    _ZH_FACTS,
    _ZH_INSIGHTS,
    _ZH_PREFERENCES,
)


POLICY_REGISTRY: dict[str, CapturePolicy] = {}


def register_policy(policy: CapturePolicy) -> None:
    """Register ``policy``. A second call with the same code replaces it."""
    if not policy.language_code:
        raise ValueError("language_code is required")
    POLICY_REGISTRY[policy.language_code] = policy


def unregister_policy(language_code: str) -> None:
    """Remove one language. The Latin fallback is what callers use next."""
    POLICY_REGISTRY.pop(language_code, None)


def iter_policies() -> tuple[CapturePolicy, ...]:
    """Return registered policies in registration order."""
    return tuple(POLICY_REGISTRY.values())


def detect_language(text: str) -> str:
    """Detect the primary script: ``en``, ``vi``, ``zh``, ``ja``, or ``ko``."""
    stripped = text.strip()
    script = _cjk_script(stripped)
    if script == "chinese":
        return "zh"
    if script == "japanese":
        return "ja"
    if script == "korean":
        return "ko"
    if _is_vietnamese_text(stripped):
        return "vi"
    return "en"


def get_policy(text: str) -> CapturePolicy:
    """Return the policy for ``text``.

    Japanese, Korean, and any unregistered code use the Latin policy for
    the input floor. They still have no capture patterns of their own.
    """
    policy = POLICY_REGISTRY.get(detect_language(text))
    if policy is not None:
        return policy
    fallback = POLICY_REGISTRY.get("en")
    if fallback is None:
        raise RuntimeError("Latin capture policy is not registered")
    return fallback


register_policy(LatinPolicy())
register_policy(VietnamesePolicy())
register_policy(ChinesePolicy())
