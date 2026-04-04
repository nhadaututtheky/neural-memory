"""Relation extraction from text — causal, comparative, and sequential patterns."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

from neural_memory.core.synapse import SynapseType

logger = logging.getLogger(__name__)


class RelationType(StrEnum):
    """Types of relations that can be extracted from text."""

    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True)
class RelationCandidate:
    """
    A relation extracted from text between two spans.

    Attributes:
        source_span: Text span identified as source of the relation
        target_span: Text span identified as target of the relation
        relation_type: Category of relation (causal, comparative, sequential)
        synapse_type: Resolved synapse type for the neural graph
        confidence: Extraction confidence (0.0 - 1.0)
        source_start: Start character position of source in original text
        source_end: End character position of source in original text
        target_start: Start character position of target in original text
        target_end: End character position of target in original text
    """

    source_span: str
    target_span: str
    relation_type: RelationType
    synapse_type: SynapseType
    confidence: float
    source_start: int
    source_end: int
    target_start: int
    target_end: int


# Type alias for compiled pattern tuples
_PatternEntry = tuple[re.Pattern[str], SynapseType, RelationType, float, bool]
# bool = whether groups are (source, target) or (target, source)


def _build_causal_patterns() -> list[_PatternEntry]:
    """Build compiled regex patterns for causal relations."""
    patterns: list[_PatternEntry] = []

    # English: "X because Y" → X is CAUSED_BY Y (source=X, target=Y)
    # "because" indicates the cause follows the marker
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+because\s+(.{5,80}?)(?:\.|;|,\s+(?:and|but)|$)",
                re.IGNORECASE,
            ),
            SynapseType.CAUSED_BY,
            RelationType.CAUSAL,
            0.80,
            False,  # groups are (source, target) — source CAUSED_BY target
        )
    )

    # "X caused by Y" → X CAUSED_BY Y
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+(?:caused\s+by|due\s+to)\s+(.{5,80}?)(?:\.|;|,\s+(?:and|but)|$)",
                re.IGNORECASE,
            ),
            SynapseType.CAUSED_BY,
            RelationType.CAUSAL,
            0.85,
            False,
        )
    )

    # "X as a result of Y" → X CAUSED_BY Y
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+as\s+a\s+result\s+of\s+(.{5,80}?)(?:\.|;|,\s+(?:and|but)|$)",
                re.IGNORECASE,
            ),
            SynapseType.CAUSED_BY,
            RelationType.CAUSAL,
            0.80,
            False,
        )
    )

    # "X therefore Y" → X LEADS_TO Y
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+(?:therefore|thus|hence|consequently)\s+(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.LEADS_TO,
            RelationType.CAUSAL,
            0.75,
            False,
        )
    )

    # "X so Y" / "X so that Y" → X LEADS_TO Y
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+so\s+(?:that\s+)?(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.LEADS_TO,
            RelationType.CAUSAL,
            0.65,
            False,
        )
    )

    # "X leads to Y" / "X results in Y" → X LEADS_TO Y
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+(?:leads?\s+to|results?\s+in|causes?)\s+(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.LEADS_TO,
            RelationType.CAUSAL,
            0.85,
            False,
        )
    )

    # Vietnamese: "X vì Y" → X CAUSED_BY Y
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+(?:vì|do|bởi\s+vì)\s+(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.CAUSED_BY,
            RelationType.CAUSAL,
            0.80,
            False,
        )
    )

    # Vietnamese: "X nên Y" / "X cho nên Y" → X LEADS_TO Y
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+(?:nên|cho\s+nên|vì\s+vậy|do\s+đó)\s+(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.LEADS_TO,
            RelationType.CAUSAL,
            0.80,
            False,
        )
    )

    return patterns


def _build_comparative_patterns() -> list[_PatternEntry]:
    """Build compiled regex patterns for comparative relations."""
    patterns: list[_PatternEntry] = []

    # "X better than Y" / "X worse than Y" / "X faster than Y"
    patterns.append(
        (
            re.compile(
                r"(.{3,60}?)\s+(?:better|worse|faster|slower|bigger|smaller|more\s+\w+|less\s+\w+)"
                r"\s+than\s+(.{3,60}?)(?:\.|;|,\s+(?:and|but)|$)",
                re.IGNORECASE,
            ),
            SynapseType.SIMILAR_TO,
            RelationType.COMPARATIVE,
            0.70,
            False,
        )
    )

    # Similarity pattern: similar to, comparable to, resembles
    patterns.append(
        (
            re.compile(
                r"(.{3,60}?)\s+(?:similar\s+to|comparable\s+to|resembles?)\s+(.{3,60}?)(?:\.|;|,\s+(?:and|but)|$)",
                re.IGNORECASE,
            ),
            SynapseType.SIMILAR_TO,
            RelationType.COMPARATIVE,
            0.75,
            False,
        )
    )

    # "X unlike Y" / "X different from Y" / "X contrary to Y"
    patterns.append(
        (
            re.compile(
                r"(.{3,60}?)\s+(?:unlike|different\s+from|contrary\s+to|opposed\s+to)"
                r"\s+(.{3,60}?)(?:\.|;|,\s+(?:and|but)|$)",
                re.IGNORECASE,
            ),
            SynapseType.CONTRADICTS,
            RelationType.COMPARATIVE,
            0.70,
            False,
        )
    )

    # Vietnamese: "X giống như Y" → SIMILAR_TO
    patterns.append(
        (
            re.compile(
                r"(.{3,60}?)\s+(?:giống\s+như|tương\s+tự|giống)\s+(.{3,60}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.SIMILAR_TO,
            RelationType.COMPARATIVE,
            0.75,
            False,
        )
    )

    # Vietnamese comparative pattern (hon = than)
    patterns.append(
        (
            re.compile(
                r"(.{3,60}?)\s+(?:\w+\s+hơn)\s+(.{3,60}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.SIMILAR_TO,
            RelationType.COMPARATIVE,
            0.65,
            False,
        )
    )

    # Vietnamese: "X khác với Y" → CONTRADICTS
    patterns.append(
        (
            re.compile(
                r"(.{3,60}?)\s+(?:khác\s+với|trái\s+ngược\s+với|ngược\s+lại\s+với)"
                r"\s+(.{3,60}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.CONTRADICTS,
            RelationType.COMPARATIVE,
            0.70,
            False,
        )
    )

    return patterns


def _build_sequential_patterns() -> list[_PatternEntry]:
    """Build compiled regex patterns for sequential relations."""
    patterns: list[_PatternEntry] = []

    # "X then Y" / "X and then Y" → X BEFORE Y
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+(?:and\s+)?then\s+(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.BEFORE,
            RelationType.SEQUENTIAL,
            0.70,
            False,  # source BEFORE target
        )
    )

    # "X afterwards Y" → X BEFORE Y
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+afterwards?\s+(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.BEFORE,
            RelationType.SEQUENTIAL,
            0.70,
            False,
        )
    )

    # "after X, Y" → X BEFORE Y (X happened first)
    patterns.append(
        (
            re.compile(
                r"after\s+(.{5,80}?)\s*[,;]\s*(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.BEFORE,
            RelationType.SEQUENTIAL,
            0.75,
            False,  # "after X, Y" means X came first, then Y → X BEFORE Y
        )
    )

    # "before X, Y" → Y BEFORE X (Y happened first, leads to X)
    patterns.append(
        (
            re.compile(
                r"before\s+(.{5,80}?)\s*[,;]\s*(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.BEFORE,
            RelationType.SEQUENTIAL,
            0.75,
            True,  # reversed: "before X, Y" → Y BEFORE X
        )
    )

    # "first X then Y" → X BEFORE Y
    patterns.append(
        (
            re.compile(
                r"first\s+(.{5,80}?)\s*[,;]?\s*then\s+(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.BEFORE,
            RelationType.SEQUENTIAL,
            0.85,
            False,
        )
    )

    # "X followed by Y" → X BEFORE Y
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+followed\s+by\s+(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.BEFORE,
            RelationType.SEQUENTIAL,
            0.80,
            False,
        )
    )

    # Vietnamese: "trước khi X, Y" → Y BEFORE X
    patterns.append(
        (
            re.compile(
                r"trước\s+khi\s+(.{5,80}?)\s*[,;]\s*(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.BEFORE,
            RelationType.SEQUENTIAL,
            0.75,
            True,  # reversed
        )
    )

    # Vietnamese: "sau khi X, Y" → X BEFORE Y
    patterns.append(
        (
            re.compile(
                r"sau\s+khi\s+(.{5,80}?)\s*[,;]\s*(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.BEFORE,
            RelationType.SEQUENTIAL,
            0.75,
            False,
        )
    )

    # Vietnamese: "X rồi Y" / "X sau đó Y" → X BEFORE Y
    patterns.append(
        (
            re.compile(
                r"(.{5,80}?)\s+(?:rồi|sau\s+đó)\s+(.{5,80}?)(?:\.|;|$)",
                re.IGNORECASE,
            ),
            SynapseType.BEFORE,
            RelationType.SEQUENTIAL,
            0.70,
            False,
        )
    )

    return patterns


def _build_code_patterns() -> list[_PatternEntry]:
    """Build compiled regex patterns for code-semantic relations."""
    patterns: list[_PatternEntry] = []

    # Pattern: "X imports Y", "X import Y", "from Y import X"
    patterns.append(
        (
            re.compile(
                r"(?:from\s+)(\S{2,60})\s+imports?\s+(\S{2,60})",
                re.IGNORECASE,
            ),
            SynapseType.IMPORTS,
            RelationType.CAUSAL,  # reuse closest category
            0.90,
            True,  # "from Y import X" → X imports Y (reversed)
        )
    )
    patterns.append(
        (
            re.compile(
                r"(\S{2,60})\s+imports?\s+(\S{2,60})",
                re.IGNORECASE,
            ),
            SynapseType.IMPORTS,
            RelationType.CAUSAL,
            0.90,
            False,
        )
    )

    # "X calls Y" / "X invokes Y" / "X delegates to Y"
    patterns.append(
        (
            re.compile(
                r"(\S{2,60})\s+(?:calls?|invokes?|delegates?\s+to)\s+(\S{2,60})",
                re.IGNORECASE,
            ),
            SynapseType.CALLS,
            RelationType.CAUSAL,
            0.85,
            False,
        )
    )

    # "X depends on Y" / "X requires Y" / "X needs Y"
    patterns.append(
        (
            re.compile(
                r"(\S{2,60})\s+(?:depends?\s+on|requires?|needs?)\s+(\S{2,60})",
                re.IGNORECASE,
            ),
            SynapseType.DEPENDS_ON,
            RelationType.CAUSAL,
            0.80,
            False,
        )
    )

    # "X inherits from Y" / "X extends Y" / "X subclass of Y"
    patterns.append(
        (
            re.compile(
                r"(\S{2,60})\s+(?:inherits?\s+(?:from)?|extends?|subclass\s+of)\s+(\S{2,60})",
                re.IGNORECASE,
            ),
            SynapseType.INHERITS,
            RelationType.CAUSAL,
            0.90,
            False,
        )
    )

    # "X implements Y" / "X conforms to Y"
    patterns.append(
        (
            re.compile(
                r"(\S{2,60})\s+(?:implements?|conforms?\s+to)\s+(\S{2,60})",
                re.IGNORECASE,
            ),
            SynapseType.IMPLEMENTS,
            RelationType.CAUSAL,
            0.90,
            False,
        )
    )

    # "X defined in Y" / "X lives in Y"
    patterns.append(
        (
            re.compile(
                r"(\S{2,60})\s+(?:defined\s+in|lives?\s+in|located\s+in)\s+(\S{2,60})",
                re.IGNORECASE,
            ),
            SynapseType.DEFINED_IN,
            RelationType.CAUSAL,
            0.85,
            False,
        )
    )

    # "X raises Y" / "X throws Y"
    patterns.append(
        (
            re.compile(
                r"(\S{2,60})\s+(?:raises?|throws?)\s+(\S{2,60})",
                re.IGNORECASE,
            ),
            SynapseType.RAISES,
            RelationType.CAUSAL,
            0.85,
            False,
        )
    )

    return patterns


class RelationExtractor:
    """
    Extract relations from text using regex pattern matching.

    Identifies causal, comparative, sequential, and code-semantic
    relationships between text spans. Each relation maps to a specific
    SynapseType for the neural graph.

    No LLM dependency — pure regex-based extraction.
    """

    def __init__(self) -> None:
        """Initialize with compiled regex patterns for all relation families."""
        self._causal_patterns = _build_causal_patterns()
        self._comparative_patterns = _build_comparative_patterns()
        self._sequential_patterns = _build_sequential_patterns()
        self._code_patterns = _build_code_patterns()

    def extract(self, text: str, language: str = "auto") -> list[RelationCandidate]:
        """
        Extract relations from text.

        Args:
            text: Source text to analyze
            language: Language hint ("vi", "en", or "auto")

        Returns:
            List of extracted relation candidates, deduplicated
        """
        if not text or len(text) < 10:
            return []

        candidates: list[RelationCandidate] = []

        candidates.extend(self._extract_family(text, self._causal_patterns))
        candidates.extend(self._extract_family(text, self._comparative_patterns))
        candidates.extend(self._extract_family(text, self._sequential_patterns))
        candidates.extend(self._extract_family(text, self._code_patterns))

        return self._deduplicate(candidates)

    def _extract_family(
        self,
        text: str,
        patterns: list[_PatternEntry],
    ) -> list[RelationCandidate]:
        """Extract relations using a specific pattern family."""
        candidates: list[RelationCandidate] = []

        for pattern, synapse_type, relation_type, confidence, reversed_groups in patterns:
            for match in pattern.finditer(text):
                group1 = match.group(1).strip()
                group2 = match.group(2).strip()

                if reversed_groups:
                    source_span = group2
                    target_span = group1
                else:
                    source_span = group1
                    target_span = group2

                # Filter short or empty spans
                if len(source_span) < 3 or len(target_span) < 3:
                    continue

                # Calculate span positions in original text
                if reversed_groups:
                    source_start = match.start(2)
                    source_end = match.end(2)
                    target_start = match.start(1)
                    target_end = match.end(1)
                else:
                    source_start = match.start(1)
                    source_end = match.end(1)
                    target_start = match.start(2)
                    target_end = match.end(2)

                candidates.append(
                    RelationCandidate(
                        source_span=source_span,
                        target_span=target_span,
                        relation_type=relation_type,
                        synapse_type=synapse_type,
                        confidence=confidence,
                        source_start=source_start,
                        source_end=source_end,
                        target_start=target_start,
                        target_end=target_end,
                    )
                )

        return candidates

    def _deduplicate(
        self,
        candidates: list[RelationCandidate],
    ) -> list[RelationCandidate]:
        """Deduplicate candidates by (source_span, target_span, synapse_type) key.

        When duplicates exist, keep the one with highest confidence.
        """
        seen: dict[str, RelationCandidate] = {}

        for candidate in candidates:
            key = (
                f"{candidate.source_span.lower()}:"
                f"{candidate.target_span.lower()}:"
                f"{candidate.synapse_type.value}"
            )
            existing = seen.get(key)
            if existing is None or candidate.confidence > existing.confidence:
                seen[key] = candidate

        return list(seen.values())
