"""Keyword extraction from text with Vietnamese word segmentation support."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from neural_memory.utils.cjk import is_cjk_char

logger = logging.getLogger(__name__)

# English stop words — standard
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

# Conversational English — common contractions without apostrophes,
# filler words, and profanity that keyword extraction should skip.
# These produce noisy bigrams ("like dont", "fucking fall") when
# left in the token stream.
_STOP_WORDS_CONVERSATIONAL_EN: frozenset[str] = frozenset(
    {
        # Contractions without apostrophes (casual typing)
        "dont",
        "doesnt",
        "didnt",
        "wont",
        "wouldnt",
        "couldnt",
        "shouldnt",
        "cant",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "hasnt",
        "havent",
        "hadnt",
        "im",
        "ive",
        "id",
        "youre",
        "youve",
        "youll",
        "youd",
        "hes",
        "shes",
        "its",
        "were",
        "theyre",
        "theyve",
        "theyll",
        "thats",
        "theres",
        "heres",
        "whats",
        "lets",
        # Common filler / hedging
        "like",
        "really",
        "actually",
        "basically",
        "literally",
        "honestly",
        "seriously",
        "obviously",
        "suppose",
        "guess",
        "thing",
        "things",
        "something",
        "anything",
        "everything",
        "nothing",
        "kinda",
        "sorta",
        "gonna",
        "gotta",
        "wanna",
        "dunno",
        "yeah",
        "nah",
        "yep",
        "nope",
        "hey",
        "oh",
        "ugh",
        "lol",
        "lmao",
        "idk",
        "tbh",
        "imo",
        "imho",
        # Profanity (no topical signal)
        "fucking",
        "fuck",
        "shit",
        "damn",
        "hell",
        "crap",
        # Common verbs that rarely carry topical signal
        "think",
        "know",
        "want",
        "make",
        "go",
        "going",
        "come",
        "take",
        "give",
        "tell",
        "say",
        "said",
        "get",
        "got",
        "went",
        "put",
        "look",
        "looking",
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
STOP_WORDS: frozenset[str] = STOP_WORDS_EN | STOP_WORDS_VI | _STOP_WORDS_CONVERSATIONAL_EN

# Han characters present anywhere in the text. Used as a cheap pre-check so
# Latin-only input never pays for a per-character scan.
_CJK_ANY = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f"
    "\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U00030000-\U0003134f]"
)

# Longest Han run still treated as one term. A run has no internal word
# boundaries, so a window cut out of it would be a guess, and a guess that lands
# on a common but unrelated word retrieves unrelated memories --
# `_build_fts_query` phrases the term as a single FTS phrase and the
# fiber-summary tier answers from its top hit. So the whole run is used or
# nothing is.
#
# Five rather than four because a run of five is still a term, not a sentence:
# 服务器迁移 and 角色管理页 are ordinary domain words. It is a boundary, not a
# solution: a six-character term still yields nothing, because an exact phrase
# that long stops appearing in stored summaries and any window cut out of one
# would be a guess. Segmentation is what would actually lift this bound.
_CJK_MAX_WHOLE_RUN = 5


def _cjk_runs(text: str) -> list[str]:
    """Return every run of consecutive Han characters, in reading order.

    No splitting inside a run: Latin word boundaries do not exist here, and a
    single Han character is not a reliable separator -- 会 belongs to
    社会/会议/会计, 用 to 用户, 问 to 问题,
    so cutting at characters that look like particles destroys the word.

    There is deliberately no utterance or function-word filter. Callers are
    agents, not people typing chat messages: a recall query is a search term
    (媒体投稿, 用户, 服务器迁移), not
    "帮我看一下媒体投稿的记录 吧". A filter for
    chat phrasing was written and removed -- it cost 300 lines and every rule
    it grew was rejected by the next review round with a fresh counterexample,
    while never once changing the answer for a real query.
    """
    runs: list[str] = []
    current: list[str] = []
    for ch in text:
        if is_cjk_char(ch):
            current.append(ch)
        else:
            if current:
                runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return runs


def _extract_cjk_keyword(text: str, min_length: int) -> str | None:
    """Pick a Chinese keyword for the spaced FTS index, or None.

    The longest run that fits the word-length window wins. Nothing is cut out
    of a run: a window of a sentence is as likely to be a verb as the topic
    (数据库连接超时 -> 数据), and a wrong term silently
    retrieves the wrong memories because `_build_fts_query` phrases it as one
    FTS phrase. Returning None leaves the caller where it was before this path
    existed, which is honest.

    Exactly one keyword is returned because `_build_fts_query` joins keywords
    with an implicit AND, so emitting several competing runs would narrow the
    result to text containing all of them.
    """
    if not _CJK_ANY.search(text):
        return None
    usable = [run for run in _cjk_runs(text) if min_length <= len(run) <= _CJK_MAX_WHOLE_RUN]
    if not usable:
        return None
    return max(usable, key=len)


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
        return STOP_WORDS_EN | _STOP_WORDS_CONVERSATIONAL_EN
    # "auto" — use all
    return STOP_WORDS


_PYVI_WARNED = False


def _tokenize_vietnamese(text: str) -> str | None:
    """Try to tokenize Vietnamese text using pyvi.

    Returns tokenized text with compound words joined by underscores,
    or None if pyvi is not available.
    """
    global _PYVI_WARNED
    try:
        import warnings

        with warnings.catch_warnings():
            # pyvi emits NumPy VisibleDeprecationWarning (UserWarning subclass, NOT
            # DeprecationWarning) during pickle.load at module import. Narrow
            # category filters miss it — suppress everything in this scoped block.
            warnings.simplefilter("ignore")
            from pyvi import ViTokenizer

        return ViTokenizer.tokenize(text)  # type: ignore[no-any-return]
    except ImportError:
        if not _PYVI_WARNED:
            logger.warning(
                "Vietnamese text detected but pyvi is not installed — "
                "keyword extraction will produce low-quality bigrams. "
                "Install with: pip install pyvi"
            )
            _PYVI_WARNED = True
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

    # ── Phase 1: Extract code identifiers BEFORE lowercasing ─────────
    # These are preserved as-is (case-sensitive) for precise code recall.
    code_identifiers: dict[str, float] = {}
    code_spans: list[tuple[int, int]] = []  # spans to skip in natural-language pass

    # PascalCase: ReflexPipeline, MemoryEncoder (also split into sub-words)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", tokenized_text):
        ident = m.group(1)
        code_identifiers[ident] = 1.2  # high weight for compound identifiers
        code_spans.append((m.start(), m.end()))
        # Also add split sub-words: ReflexPipeline -> reflex, pipeline
        parts = re.findall(r"[A-Z][a-z]+", ident)
        for part in parts:
            code_identifiers.setdefault(part.lower(), 0.7)

    # camelCase: processData, getKeywords
    for m in re.finditer(r"\b([a-z][a-z0-9]*(?:[A-Z][a-z0-9]+)+)\b", tokenized_text):
        ident = m.group(1)
        code_identifiers[ident] = 1.2
        code_spans.append((m.start(), m.end()))
        parts = re.findall(r"[a-z][a-z0-9]*|[A-Z][a-z0-9]+", ident)
        for part in parts:
            code_identifiers.setdefault(part.lower(), 0.7)

    # snake_case (2+ segments): extract_keywords, activate_trail
    for m in re.finditer(r"\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*){1,})\b", tokenized_text):
        ident = m.group(1)
        code_identifiers[ident] = 1.2
        code_spans.append((m.start(), m.end()))
        for part in ident.split("_"):
            if len(part) >= min_length:
                code_identifiers.setdefault(part, 0.7)

    # Dotted modules: neural_memory.engine, os.path
    for m in re.finditer(r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){1,})\b", tokenized_text):
        code_identifiers[m.group(1)] = 1.2
        code_spans.append((m.start(), m.end()))

    # Error types: ValueError, ConnectionRefusedError
    for m in re.finditer(r"\b([A-Z][a-zA-Z]*(?:Error|Exception))\b", tokenized_text):
        code_identifiers[m.group(1)] = 1.1
        code_spans.append((m.start(), m.end()))

    # ── Phase 2: Natural language keywords (existing logic) ────────
    # Han characters are absent from the Latin range below, so a Chinese query
    # tokenizes to nothing and recall answers "no relevant memories" for text
    # that is indexed and retrievable. Extract the Chinese term separately.
    cjk_keyword = _extract_cjk_keyword(text, min_length)

    words = re.findall(r"\b[a-zA-ZÀ-ỹ]+(?:_[a-zA-ZÀ-ỹ]+)*\b", tokenized_text.lower())

    # Filter to content words with original position
    filtered: list[tuple[str, int]] = [
        (w, i)
        for i, w in enumerate(words)
        if len(w.replace("_", "")) >= min_length
        and w.replace("_", " ") not in stop_words
        and w not in stop_words
    ]

    if not filtered and not code_identifiers and not cjk_keyword:
        return []

    total = max(len(filtered), 1)
    weighted: dict[str, float] = {}

    # Seed with code identifiers (high priority)
    weighted.update(code_identifiers)

    # The Chinese term is what the query is about; rank it with the leading
    # natural-language words.
    if cjk_keyword is not None:
        weighted[cjk_keyword] = max(weighted.get(cjk_keyword, 0.0), 1.0)

    # Unigrams with position decay (1.0 at start -> 0.5 at end)
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
