"""Keyword extraction from text with Vietnamese word segmentation support."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# English stop words
STOP_WORDS_EN: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "what",
        "which",
        "who",
        "whom",
    }
)

# Vietnamese stop words
STOP_WORDS_VI: frozenset[str] = frozenset(
    {
        "và",
        "của",
        "là",
        "có",
        "được",
        "cho",
        "với",
        "này",
        "trong",
        "để",
        "các",
        "những",
        "một",
        "đã",
        "tôi",
        "bạn",
        "anh",
        "chị",
        "em",
        "ở",
        "tại",
        "khi",
        "thì",
        "mà",
        "nếu",
        "vì",
        "cũng",
        "như",
        "từ",
        "đến",
        "lại",
        "ra",
        "vào",
        "lên",
        "xuống",
        "rồi",
        "sẽ",
        "đang",
        "vẫn",
        "còn",
        "chỉ",
        "rất",
        "quá",
        "làm",
        "gì",
        "sao",
        "nào",
        "đâu",
        "ai",
        "bao",
        "nhiêu",
    }
)

# Combined stop words for backward compatibility
STOP_WORDS: frozenset[str] = STOP_WORDS_EN | STOP_WORDS_VI

# Vietnamese diacritical character pattern (unique to Vietnamese, not French)
_VI_DIACRITICS = re.compile(r"[ăâđêôơưắằẳẵặấầẩẫậếềểễệốồổỗộớờởỡợứừửữự]")


def _detect_vietnamese(text: str) -> bool:
    """Detect if text contains Vietnamese based on diacritical characters."""
    return bool(_VI_DIACRITICS.search(text.lower()))


def _get_stop_words(language: str, text: str) -> frozenset[str]:
    """Get appropriate stop words for the detected language."""
    if language == "vi":
        return STOP_WORDS_VI
    if language == "en":
        return STOP_WORDS_EN
    # "auto" — use both
    return STOP_WORDS


def _tokenize_vietnamese(text: str) -> str | None:
    """Try to tokenize Vietnamese text using pyvi.

    Returns tokenized text with compound words joined by underscores,
    or None if pyvi is not available.
    """
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyvi")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
            from pyvi import ViTokenizer

            return ViTokenizer.tokenize(text)  # type: ignore[no-any-return]
    except ImportError:
        return None


@dataclass(frozen=True)
class WeightedKeyword:
    """A keyword with an importance weight.

    Attributes:
        text: The keyword text (unigram or bi-gram)
        weight: Importance weight (0.0 - 1.5), higher = more important
    """

    text: str
    weight: float


def extract_weighted_keywords(
    text: str,
    min_length: int = 2,
    language: str = "auto",
) -> list[WeightedKeyword]:
    """
    Extract weighted keywords with bi-gram support.

    Scoring factors:
    - Position: earlier words score higher (1.0 → 0.5 linear decay)
    - Bi-grams: adjacent non-stop-word pairs get averaged weight * 1.2 boost

    For Vietnamese text, uses pyvi word segmentation to detect compound words
    (e.g., "học sinh" → single keyword "học sinh" instead of separate "học", "sinh").

    Args:
        text: The text to extract from
        min_length: Minimum word length for unigrams
        language: Language hint ("vi", "en", or "auto")

    Returns:
        List of WeightedKeyword sorted by weight descending
    """
    is_vietnamese = language == "vi" or (language == "auto" and _detect_vietnamese(text))
    stop_words = _get_stop_words(language, text)

    # For Vietnamese: try pyvi tokenization to detect compound words
    tokenized_text = text
    if is_vietnamese:
        vi_tokenized = _tokenize_vietnamese(text)
        if vi_tokenized is not None:
            tokenized_text = vi_tokenized

    words = re.findall(r"\b[a-zA-ZÀ-ỹ]+(?:_[a-zA-ZÀ-ỹ]+)*\b", tokenized_text.lower())

    # Filter to content words with original position
    filtered: list[tuple[str, int]] = [
        (w, i)
        for i, w in enumerate(words)
        if len(w.replace("_", "")) >= min_length
        and w.replace("_", " ") not in stop_words
        and w not in stop_words
    ]

    if not filtered:
        return []

    total = len(filtered)
    weighted: dict[str, float] = {}

    # Unigrams with position decay (1.0 at start → 0.5 at end)
    for idx, (word, _orig_pos) in enumerate(filtered):
        position_weight = 1.0 - 0.5 * (idx / max(1, total - 1))
        # Store with underscores replaced by spaces for readability
        display_word = word.replace("_", " ")
        weighted[display_word] = max(weighted.get(display_word, 0.0), position_weight)

    # Bi-grams from adjacent non-stop words within 3 original word positions
    for i in range(len(filtered) - 1):
        w1, p1 = filtered[i]
        w2, p2 = filtered[i + 1]
        if p2 - p1 <= 3:
            dw1 = w1.replace("_", " ")
            dw2 = w2.replace("_", " ")
            bigram = f"{dw1} {dw2}"
            bigram_weight = (weighted.get(dw1, 0.5) + weighted.get(dw2, 0.5)) / 2 * 1.2
            weighted[bigram] = max(weighted.get(bigram, 0.0), bigram_weight)

    results = [WeightedKeyword(text=k, weight=v) for k, v in weighted.items()]
    results.sort(key=lambda x: x.weight, reverse=True)
    return results


def extract_keywords(
    text: str,
    min_length: int = 2,
    language: str = "auto",
) -> list[str]:
    """
    Extract keywords from text, sorted by importance.

    Backward-compatible wrapper around extract_weighted_keywords().
    Returns bi-grams before unigrams, ordered by weight.

    Args:
        text: The text to extract from
        min_length: Minimum word length
        language: Language hint ("vi", "en", or "auto")

    Returns:
        List of keyword strings
    """
    weighted = extract_weighted_keywords(text, min_length, language=language)
    return [kw.text for kw in weighted]
