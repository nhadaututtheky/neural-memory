"""Query parser for decomposing queries into activation signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from neural_memory.extraction.entities import Entity, EntityExtractor, extract_keywords
from neural_memory.extraction.temporal import TemporalExtractor, TimeHint


class QueryIntent(StrEnum):
    """The intent/purpose of a query."""

    ASK_WHAT = "ask_what"  # What happened?
    ASK_WHERE = "ask_where"  # Where did it happen?
    ASK_WHEN = "ask_when"  # When did it happen?
    ASK_WHO = "ask_who"  # Who was involved?
    ASK_WHY = "ask_why"  # Why did it happen?
    ASK_HOW = "ask_how"  # How did it happen?
    ASK_FEELING = "ask_feeling"  # How did I feel?
    ASK_PATTERN = "ask_pattern"  # What's the pattern?
    CONFIRM = "confirm"  # Did X happen?
    COMPARE = "compare"  # Compare X and Y
    RECALL = "recall"  # General recall
    UNKNOWN = "unknown"


class Perspective(StrEnum):
    """The perspective/framing of the query."""

    RECALL = "recall"  # Remember something
    CONFIRM = "confirm"  # Verify something
    COMPARE = "compare"  # Compare things
    ANALYZE = "analyze"  # Analyze/understand
    SUMMARIZE = "summarize"  # Get summary


@dataclass
class Stimulus:
    """
    Decomposed query signals for activation.

    A Stimulus represents all the extracted signals from a query
    that will be used to activate relevant neurons.

    Attributes:
        time_hints: Extracted time references
        keywords: Important keywords from the query
        entities: Named entities found
        intent: What the query is asking for
        perspective: How the query frames the request
        raw_query: The original query text
        language: Detected or specified language
    """

    time_hints: list[TimeHint]
    keywords: list[str]
    entities: list[Entity]
    intent: QueryIntent
    perspective: Perspective
    raw_query: str
    language: str = "auto"

    @property
    def has_time_context(self) -> bool:
        """Check if query has temporal constraints."""
        return len(self.time_hints) > 0

    @property
    def has_entities(self) -> bool:
        """Check if query mentions specific entities."""
        return len(self.entities) > 0

    @property
    def anchor_count(self) -> int:
        """Count of potential anchor points for activation."""
        return len(self.time_hints) + len(self.entities) + len(self.keywords)


class QueryParser:
    """
    Parser for decomposing queries into activation signals.

    The parser extracts:
    - Temporal references (time hints)
    - Named entities (people, places, etc.)
    - Keywords (important words)
    - Intent (what the query is asking)
    - Perspective (how the query frames the request)
    """

    # Intent detection patterns
    INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
        QueryIntent.ASK_WHAT: [
            # English
            r"what",
            r"which",
            r"tell me about",
            # Vietnamese
            r"gì",
            r"cái gì",
            r"điều gì",
            r"chuyện gì",
            r"việc gì",
        ],
        QueryIntent.ASK_WHERE: [
            # English
            r"where",
            r"location",
            r"place",
            # Vietnamese
            r"ở đâu",
            r"đâu",
            r"chỗ nào",
            r"nơi nào",
        ],
        QueryIntent.ASK_WHEN: [
            # English
            r"when",
            r"what time",
            # Vietnamese
            r"khi nào",
            r"lúc nào",
            r"bao giờ",
            r"mấy giờ",
        ],
        QueryIntent.ASK_WHO: [
            # English
            r"who",
            r"whom",
            # Vietnamese
            r"ai",
            r"người nào",
            r"với ai",
        ],
        QueryIntent.ASK_WHY: [
            # English
            r"why",
            r"reason",
            r"cause",
            # Vietnamese
            r"tại sao",
            r"vì sao",
            r"lý do",
        ],
        QueryIntent.ASK_HOW: [
            # English
            r"how did",
            r"how was",
            r"how to",
            # Vietnamese
            r"như thế nào",
            r"làm sao",
            r"thế nào",
            r"ra sao",
        ],
        QueryIntent.ASK_FEELING: [
            # English
            r"how (?:did|do) (?:i|you) feel",
            r"feeling",
            r"emotion",
            # Vietnamese
            r"cảm thấy",
            r"cảm xúc",
            r"tâm trạng",
            r"vui không",
            r"buồn không",
        ],
        QueryIntent.ASK_PATTERN: [
            # English
            r"usually",
            r"typically",
            r"pattern",
            r"often",
            r"always",
            # Vietnamese
            r"thường",
            r"hay",
            r"luôn",
            r"mỗi khi",
        ],
        QueryIntent.CONFIRM: [
            # English
            r"did (?:i|we|you)",
            r"was there",
            r"have (?:i|we|you)",
            r"is it true",
            # Vietnamese
            r"có phải",
            r"đúng không",
            r"phải không",
        ],
        QueryIntent.COMPARE: [
            # English
            r"compare",
            r"difference",
            r"versus",
            r"vs",
            r"better",
            r"worse",
            # Vietnamese
            r"so sánh",
            r"khác nhau",
            r"giống nhau",
            r"hơn",
        ],
    }

    def __init__(
        self,
        temporal_extractor: TemporalExtractor | None = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        """
        Initialize the parser.

        Args:
            temporal_extractor: Custom temporal extractor (creates default if None)
            entity_extractor: Custom entity extractor (creates default if None)
        """
        self._temporal = temporal_extractor or TemporalExtractor()
        self._entity = entity_extractor or EntityExtractor()

        # Compile intent patterns
        import re

        self._intent_compiled: dict[QueryIntent, list[re.Pattern[str]]] = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            self._intent_compiled[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def parse(
        self,
        query: str,
        reference_time: datetime | None = None,
        language: str = "auto",
    ) -> Stimulus:
        """
        Parse a query into a Stimulus.

        Args:
            query: The query text
            reference_time: Reference time for temporal parsing
            language: "vi", "en", or "auto"

        Returns:
            Stimulus containing all extracted signals
        """
        if reference_time is None:
            reference_time = datetime.now()

        # Detect language if auto
        if language == "auto":
            language = self._detect_language(query)

        # Extract components
        time_hints = self._temporal.extract(query, reference_time, language)
        entities = self._entity.extract(query, language)
        keywords = extract_keywords(query)

        # Detect intent
        intent = self._detect_intent(query)

        # Detect perspective
        perspective = self._detect_perspective(query, intent)

        return Stimulus(
            time_hints=time_hints,
            keywords=keywords,
            entities=entities,
            intent=intent,
            perspective=perspective,
            raw_query=query,
            language=language,
        )

    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # Vietnamese-specific characters
        vi_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")

        text_lower = text.lower()
        vi_count = sum(1 for c in text_lower if c in vi_chars)

        # If significant Vietnamese characters, it's Vietnamese
        if vi_count > len(text) * 0.05:
            return "vi"

        # Check for Vietnamese words
        vi_words = {"của", "và", "là", "có", "được", "cho", "với", "này", "trong", "để", "các"}
        words = set(text_lower.split())
        if words & vi_words:
            return "vi"

        return "en"

    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect the query intent."""
        query_lower = query.lower()

        for intent, patterns in self._intent_compiled.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    return intent

        return QueryIntent.RECALL

    def _detect_perspective(
        self,
        query: str,
        intent: QueryIntent,
    ) -> Perspective:
        """Detect the query perspective."""
        query_lower = query.lower()

        # Check for confirmation patterns
        if intent == QueryIntent.CONFIRM:
            return Perspective.CONFIRM

        # Check for comparison patterns
        if intent == QueryIntent.COMPARE:
            return Perspective.COMPARE

        # Check for summary patterns
        summary_patterns = ["summary", "summarize", "tóm tắt", "overview", "tổng kết"]
        for pattern in summary_patterns:
            if pattern in query_lower:
                return Perspective.SUMMARIZE

        # Check for analysis patterns
        analysis_patterns = ["analyze", "understand", "explain", "phân tích", "giải thích"]
        for pattern in analysis_patterns:
            if pattern in query_lower:
                return Perspective.ANALYZE

        return Perspective.RECALL
