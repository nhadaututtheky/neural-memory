"""Entity extraction from text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class EntityType(StrEnum):
    """Types of named entities."""

    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    EVENT = "event"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """
    A named entity extracted from text.

    Attributes:
        text: The original text of the entity
        type: The entity type
        start: Start character position in source text
        end: End character position in source text
        confidence: Extraction confidence (0.0 - 1.0)
    """

    text: str
    type: EntityType
    start: int
    end: int
    confidence: float = 1.0


class EntityExtractor:
    """
    Entity extractor using pattern matching.

    For production use, consider using spaCy or underthesea
    for better entity recognition. This provides basic
    rule-based extraction as a fallback.
    """

    # Common Vietnamese person name prefixes
    VI_PERSON_PREFIXES = {
        "anh",
        "chị",
        "em",
        "bạn",
        "cô",
        "chú",
        "bác",
        "ông",
        "bà",
        "thầy",
        "cô giáo",
        "mr",
        "mrs",
        "ms",
        "miss",
    }

    # Common location indicators
    LOCATION_INDICATORS = {
        # Vietnamese
        "ở",
        "tại",
        "đến",
        "từ",
        "quán",
        "cafe",
        "cà phê",
        "nhà hàng",
        "công ty",
        "văn phòng",
        # English
        "at",
        "in",
        "to",
        "from",
        "restaurant",
        "office",
        "building",
        "hotel",
        "shop",
        "store",
    }

    # Pattern for capitalized words (potential entities)
    CAPITALIZED_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")

    # Pattern for Vietnamese names (words after person prefixes)
    VI_NAME_PATTERN = re.compile(
        r"\b(?:anh|chị|em|bạn|cô|chú|bác|ông|bà)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)",
        re.IGNORECASE,
    )

    def __init__(self, use_nlp: bool = False) -> None:
        """
        Initialize the extractor.

        Args:
            use_nlp: If True, try to use spaCy/underthesea (not implemented yet)
        """
        self._use_nlp = use_nlp
        self._nlp_en = None
        self._nlp_vi = None

        if use_nlp:
            self._init_nlp()

    def _init_nlp(self) -> None:
        """Initialize NLP models if available."""
        # Try to load spaCy for English
        try:
            import spacy

            self._nlp_en = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            pass

        # Try to load underthesea for Vietnamese
        try:
            import underthesea

            self._nlp_vi = underthesea
        except ImportError:
            pass

    def extract(
        self,
        text: str,
        language: str = "auto",
    ) -> list[Entity]:
        """
        Extract entities from text.

        Args:
            text: The text to extract from
            language: "vi", "en", or "auto"

        Returns:
            List of Entity objects
        """
        entities: list[Entity] = []

        # Try NLP-based extraction first
        if self._use_nlp:
            nlp_entities = self._extract_with_nlp(text, language)
            if nlp_entities:
                return nlp_entities

        # Fall back to pattern-based extraction
        entities.extend(self._extract_vietnamese_names(text))
        entities.extend(self._extract_capitalized_words(text, entities))
        entities.extend(self._extract_locations(text, entities))

        # Remove duplicates
        seen: set[str] = set()
        unique: list[Entity] = []
        for entity in entities:
            key = f"{entity.text.lower()}:{entity.type}"
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique

    def _extract_with_nlp(
        self,
        text: str,
        language: str,
    ) -> list[Entity] | None:
        """Try to extract using NLP models."""
        if language in ("en", "auto") and self._nlp_en:
            doc = self._nlp_en(text)
            entities = []
            for ent in doc.ents:
                entity_type = self._map_spacy_type(ent.label_)
                if entity_type:
                    entities.append(
                        Entity(
                            text=ent.text,
                            type=entity_type,
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.9,
                        )
                    )
            if entities:
                return entities

        if language in ("vi", "auto") and self._nlp_vi:
            try:
                ner_results = self._nlp_vi.ner(text)
                entities = []
                for word, tag in ner_results:
                    if tag.startswith("B-") or tag.startswith("I-"):
                        entity_type = self._map_underthesea_type(tag[2:])
                        if entity_type:
                            # Find position in text
                            start = text.find(word)
                            if start >= 0:
                                entities.append(
                                    Entity(
                                        text=word,
                                        type=entity_type,
                                        start=start,
                                        end=start + len(word),
                                        confidence=0.85,
                                    )
                                )
                if entities:
                    return entities
            except Exception:
                pass

        return None

    def _map_spacy_type(self, label: str) -> EntityType | None:
        """Map spaCy NER label to EntityType."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "PER": EntityType.PERSON,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "FAC": EntityType.LOCATION,
            "ORG": EntityType.ORGANIZATION,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
        }
        return mapping.get(label)

    def _map_underthesea_type(self, label: str) -> EntityType | None:
        """Map underthesea NER label to EntityType."""
        mapping = {
            "PER": EntityType.PERSON,
            "LOC": EntityType.LOCATION,
            "ORG": EntityType.ORGANIZATION,
        }
        return mapping.get(label)

    def _extract_vietnamese_names(self, text: str) -> list[Entity]:
        """Extract Vietnamese person names."""
        entities = []

        for match in self.VI_NAME_PATTERN.finditer(text):
            name = match.group(1)
            entities.append(
                Entity(
                    text=name,
                    type=EntityType.PERSON,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.8,
                )
            )

        return entities

    def _extract_capitalized_words(
        self,
        text: str,
        existing: list[Entity],
    ) -> list[Entity]:
        """Extract capitalized words as potential entities."""
        entities = []
        existing_spans = {(e.start, e.end) for e in existing}

        for match in self.CAPITALIZED_PATTERN.finditer(text):
            # Skip if already extracted
            if (match.start(), match.end()) in existing_spans:
                continue

            word = match.group(1)

            # Skip common words
            if word.lower() in {"the", "a", "an", "i", "my", "we", "they"}:
                continue

            # Skip if at start of sentence (could be just capitalization)
            if match.start() == 0 or text[match.start() - 1] in ".!?\n":
                # Still include if it looks like a proper noun
                if len(word.split()) == 1 and len(word) < 4:
                    continue

            entities.append(
                Entity(
                    text=word,
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.5,
                )
            )

        return entities

    def _extract_locations(
        self,
        text: str,
        existing: list[Entity],
    ) -> list[Entity]:
        """Extract locations based on context indicators."""
        entities = []
        existing_texts = {e.text.lower() for e in existing}

        # Find words after location indicators
        for indicator in self.LOCATION_INDICATORS:
            pattern = re.compile(
                rf"\b{re.escape(indicator)}\s+([A-ZÀ-Ỹ][a-zà-ỹA-ZÀ-Ỹ\s]+?)(?:[,.]|\s+(?:với|và|to|with|for)|$)",
                re.IGNORECASE,
            )

            for match in pattern.finditer(text):
                location = match.group(1).strip()

                if location.lower() in existing_texts:
                    continue

                if len(location) < 2:
                    continue

                entities.append(
                    Entity(
                        text=location,
                        type=EntityType.LOCATION,
                        start=match.start(1),
                        end=match.start(1) + len(location),
                        confidence=0.7,
                    )
                )

        return entities


def extract_keywords(text: str, min_length: int = 2) -> list[str]:
    """
    Extract keywords from text (simple word extraction).

    This is a basic keyword extractor. For better results,
    consider using TF-IDF or other NLP techniques.

    Args:
        text: The text to extract from
        min_length: Minimum word length

    Returns:
        List of keywords
    """
    # Common stop words (English + Vietnamese)
    stop_words = {
        # English
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
        # Vietnamese
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

    # Tokenize (simple split)
    words = re.findall(r"\b[a-zA-ZÀ-ỹ]+\b", text.lower())

    # Filter
    keywords = [w for w in words if len(w) >= min_length and w not in stop_words]

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            unique.append(word)

    return unique
