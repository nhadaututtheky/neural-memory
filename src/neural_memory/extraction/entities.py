"""Entity extraction from text."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class EntityType(StrEnum):
    """Types of named entities."""

    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    EVENT = "event"
    CODE = "code"
    UNKNOWN = "unknown"


class EntitySubtype(StrEnum):
    """Domain-specific entity subtypes for vertical intelligence."""

    # Financial
    FINANCIAL_METRIC = "financial_metric"  # ROE, revenue, EBITDA, P/E
    CURRENCY_AMOUNT = "currency_amount"  # $25M, 500 triệu VND
    FISCAL_PERIOD = "fiscal_period"  # Q1 2024, FY2025, H1/2024

    # Legal
    REGULATION = "regulation"  # Điều 468 BLDS, Section 301 SOX
    CONTRACT_CLAUSE = "contract_clause"  # Clause 5.2, Khoản 3
    LEGAL_ENTITY = "legal_entity"  # CTCP, LLC, Ltd, GmbH

    # Technical
    API_ENDPOINT = "api_endpoint"  # GET /api/v1/users, POST /webhook
    CODE_SYMBOL = "code_symbol"  # function_name(), ClassName, module.attr
    VERSION = "version"  # v2.1.0, Python 3.11, React 19

    # Code-semantic (function/class/module/package/error distinction)
    FUNCTION_NAME = "function_name"  # extract_keywords(), process_data()
    CLASS_NAME = "class_name"  # ReflexPipeline, MemoryEncoder
    MODULE_NAME = "module_name"  # neural_memory.engine, os.path
    PACKAGE_NAME = "package_name"  # numpy, aiohttp, neural-memory
    ERROR_TYPE = "error_type"  # ValueError, KeyError, ConnectionRefusedError


@dataclass(frozen=True)
class Entity:
    """
    A named entity extracted from text.

    Attributes:
        text: The original text of the entity
        type: The entity type
        subtype: Optional domain-specific subtype
        start: Start character position in source text
        end: End character position in source text
        confidence: Extraction confidence (0.0 - 1.0)
        raw_value: Original value string for verbatim recall
        unit: Unit of measurement if applicable (percent, USD, VND)
    """

    text: str
    type: EntityType
    start: int
    end: int
    subtype: EntitySubtype | None = None
    confidence: float = 1.0
    raw_value: str = ""
    unit: str = ""


class EntityExtractor:
    """
    Entity extractor using pattern matching.

    For production use, consider using spaCy or underthesea
    for better entity recognition. This provides basic
    rule-based extraction as a fallback.
    """

    # Common Vietnamese person name prefixes
    VI_PERSON_PREFIXES: frozenset[str] = frozenset(
        {
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
    )

    # Common location indicators
    LOCATION_INDICATORS: frozenset[str] = frozenset(
        {
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
    )

    # Pre-compiled location patterns (avoid recompilation in hot loop)
    _LOCATION_PATTERNS: dict[str, re.Pattern[str]] = {
        indicator: re.compile(
            rf"\b{re.escape(indicator)}\s+([A-ZÀ-Ỹ][a-zà-ỹA-ZÀ-Ỹ\s]+?)(?:[,.]|\s+(?:với|và|to|with|for)|$)",
            re.IGNORECASE,
        )
        for indicator in LOCATION_INDICATORS
    }

    # Pattern for capitalized words (potential entities)
    CAPITALIZED_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")

    # Code entity patterns
    # PascalCase: ReflexPipeline, MemoryEncoder (2+ capitalized segments)
    PASCAL_CASE_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
    # snake_case with 2+ segments: extract_keywords, activate_trail
    SNAKE_CASE_PATTERN = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*){1,})\b")
    # snake_case or PascalCase followed by () — function call
    FUNCTION_CALL_PATTERN = re.compile(
        r"\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*|[A-Z][a-z]+(?:[A-Z][a-z]+)*)\s*\("
    )
    # File paths: src/neural_memory/server.py, config.toml
    FILE_PATH_PATTERN = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.\w+")
    # Dotted module: neural_memory.engine, os.path (2+ dot-separated segments)
    MODULE_PATH_PATTERN = re.compile(r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){1,})\b")
    # Error types: ValueError, KeyError, ConnectionRefusedError, *Exception
    ERROR_TYPE_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]*(?:Error|Exception))\b")
    # Package names in context: pip install X, import X, require('X')
    PACKAGE_CONTEXT_PATTERN = re.compile(
        r"(?:pip install|pip3 install|import|from|require\()\s+['\"]?([a-z][a-z0-9_-]+)",
        re.IGNORECASE,
    )

    # Pattern for Vietnamese names (words after person prefixes)
    VI_NAME_PATTERN = re.compile(
        r"\b(?:anh|chị|em|bạn|cô|chú|bác|ông|bà)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)",
        re.IGNORECASE,
    )

    # ── Domain extraction patterns ──────────────────────────────────

    # Financial metrics: ROE, EBITDA, P/E, EPS, revenue, etc.
    FINANCIAL_METRIC_PATTERN = re.compile(
        r"\b(ROE|ROA|ROI|EBITDA|EPS|P/E|P/B|NPM|GPM|CAGR|WACC|"
        r"IRR|NPV|ROIC|FCF|D/E|"
        r"doanh thu|lợi nhuận|chi phí|tổng tài sản|vốn chủ sở hữu|"
        r"revenue|profit|margin|earnings|net income|gross profit|"
        r"operating income|total assets|equity|debt)"
        r"\s*[=:≈]?\s*"
        r"([\d.,]+\s*%?|[\d.,]+\s*(?:tỷ|triệu|nghìn|billion|million|thousand|[BMKbmk])?\b)?",
        re.IGNORECASE,
    )

    # Currency amounts: $25M, 500 triệu VND, €1.2B, ¥100K
    CURRENCY_AMOUNT_PATTERN = re.compile(
        r"(?:"
        r"[\$€£¥₫]\s*[\d.,]+\s*(?:billion|million|thousand|[BMKbmk])?"  # $25M
        r"|[\d.,]+\s*(?:tỷ|triệu|nghìn)\s*(?:VND|VNĐ|đồng|đ)?"  # 500 triệu VND
        r"|[\d.,]+\s*(?:USD|EUR|GBP|JPY|VND|VNĐ)"  # 25000 USD
        r"|[\d.,]+\s*(?:billion|million)\s*(?:USD|EUR|GBP|VND)?"  # 1.2 billion USD
        r")",
        re.IGNORECASE,
    )

    # Fiscal periods: Q1 2024, FY2025, H1/2024, năm 2024
    FISCAL_PERIOD_PATTERN = re.compile(
        r"\b(?:"
        r"Q[1-4]\s*[/.]?\s*\d{4}"  # Q1 2024, Q3/2024
        r"|FY\s*\d{4}"  # FY2025
        r"|H[12]\s*[/.]?\s*\d{4}"  # H1/2024
        r"|(?:quý|năm tài chính|năm)\s+\d{4}"  # quý 2024, năm 2024
        r"|(?:fiscal\s+)?(?:year|quarter)\s+\d{4}"  # fiscal year 2024
        r")\b",
        re.IGNORECASE,
    )

    # Legal: Điều 468 BLDS, Section 301 SOX, Article 5, Khoản 3
    REGULATION_PATTERN = re.compile(
        r"\b(?:"
        r"(?:Điều|Khoản|Mục)\s+\d+(?:\.\d+)*\s*(?:[A-ZÀ-Ỹ][a-zà-ỹA-ZÀ-Ỹ\s]*?)?"  # Điều 468 BLDS
        r"|(?:Section|Article|Clause|Rule|Regulation)\s+\d+(?:\.\d+)*"
        r"(?:\s+(?:of\s+)?(?:the\s+)?[A-Z][A-Za-z\s]*?)?"  # Section 301 SOX
        r"|(?:Nghị định|Thông tư|Luật)\s+\d+[/-]?\d*[/-]?(?:[A-ZĐ]+)?"  # Nghị định 123/2024
        r")\b",
        re.IGNORECASE,
    )

    # Contract clauses: Clause 5.2, Khoản 3 Điều 12
    CONTRACT_CLAUSE_PATTERN = re.compile(
        r"\b(?:"
        r"(?:Clause|clause)\s+\d+(?:\.\d+)+"  # Clause 5.2.1
        r"|Khoản\s+\d+\s+Điều\s+\d+"  # Khoản 3 Điều 12
        r"|(?:Điểm|Point)\s+[a-z]\s+Khoản\s+\d+"  # Điểm a Khoản 3
        r")\b",
        re.IGNORECASE,
    )

    # Legal entities: CTCP, LLC, Ltd, GmbH, Corp
    LEGAL_ENTITY_PATTERN = re.compile(
        r"\b(?:"
        r"(?:CTCP|TNHH|Công ty\s+(?:TNHH|Cổ phần|CP))\s+[A-ZÀ-Ỹ][^\n,;]{2,40}"
        r"|[A-Z][A-Za-z\s]+\s+(?:LLC|Ltd|Inc|Corp|GmbH|AG|S\.A\.|PLC|Co\.|Pty)"
        r")\b",
    )

    # API endpoints: GET /api/v1/users, POST /webhook
    API_ENDPOINT_PATTERN = re.compile(
        r"\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+"
        r"(/[a-zA-Z0-9_/{}:.-]+)",
        re.IGNORECASE,
    )

    # Version: v2.1.0, Python 3.11, React 19, Node.js 20
    VERSION_PATTERN = re.compile(
        r"\b(?:"
        r"v\d+(?:\.\d+){1,3}(?:-[a-zA-Z0-9.]+)?"  # v2.1.0, v3.0.0-beta.1
        r"|(?:Python|Node\.?js?|React|Vue|Angular|Java|Go|Rust|Ruby|PHP|Swift|Kotlin|TypeScript|TS)"
        r"\s+\d+(?:\.\d+){0,2}"  # Python 3.11
        r")\b",
        re.IGNORECASE,
    )

    # Stack trace: Python traceback lines — File "path", line N, in func
    TRACEBACK_FRAME_PATTERN = re.compile(
        r'File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+(\w+)',
    )
    # Stack trace: final error line — ErrorType: message
    TRACEBACK_ERROR_PATTERN = re.compile(
        r"^([A-Z][a-zA-Z]*(?:Error|Exception)):\s*(.+)$",
        re.MULTILINE,
    )

    def __init__(self, use_nlp: bool = False) -> None:
        """
        Initialize the extractor.

        Args:
            use_nlp: If True, try to use spaCy/underthesea (not implemented yet)
        """
        self._use_nlp = use_nlp
        self._nlp_en: Any = None
        self._nlp_vi: Any = None

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
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
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
        entities.extend(self._extract_domain_entities(text))
        entities.extend(self._extract_code_entities(text, entities))
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
                # Use cumulative offset to handle duplicate words
                offset = 0
                for word, tag in ner_results:
                    if tag.startswith(("B-", "I-")):
                        entity_type = self._map_underthesea_type(tag[2:])
                        if entity_type:
                            # Find position in text from current offset
                            start = text.find(word, offset)
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
                                offset = start + len(word)
                if entities:
                    return entities
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug("Vietnamese NER failed: %s", e)

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

    def _extract_code_entities(
        self,
        text: str,
        existing: list[Entity],
    ) -> list[Entity]:
        """Extract code identifiers with semantic subtypes."""
        entities: list[Entity] = []
        existing_spans = {(e.start, e.end) for e in existing}
        existing_texts = {e.text.lower() for e in existing}

        # Build set of known function call names for subtype inference
        function_names: set[str] = set()
        for match in self.FUNCTION_CALL_PATTERN.finditer(text):
            function_names.add(match.group(1))

        # ── Stack trace extraction (highest priority) ──────────────
        # Extract BEFORE general patterns so traceback entities get richer metadata.
        traceback_texts: set[str] = set()

        # Traceback frames: File "path", line N, in func_name
        for match in self.TRACEBACK_FRAME_PATTERN.finditer(text):
            func_name = match.group(3)
            if func_name != "<module>" and func_name.lower() not in existing_texts:
                traceback_texts.add(func_name.lower())
                entities.append(
                    Entity(
                        text=func_name,
                        type=EntityType.CODE,
                        start=match.start(3),
                        end=match.end(3),
                        subtype=EntitySubtype.FUNCTION_NAME,
                        confidence=0.9,
                        raw_value=f"{match.group(1)}:{match.group(2)}",
                    )
                )

        # Traceback error lines: ValueError: message
        for match in self.TRACEBACK_ERROR_PATTERN.finditer(text):
            error_name = match.group(1)
            if (
                error_name.lower() not in existing_texts
                and error_name.lower() not in traceback_texts
            ):
                traceback_texts.add(error_name.lower())
                entities.append(
                    Entity(
                        text=error_name,
                        type=EntityType.CODE,
                        start=match.start(1),
                        end=match.end(1),
                        subtype=EntitySubtype.ERROR_TYPE,
                        confidence=0.95,
                        raw_value=match.group(2).strip(),
                    )
                )

        # Merge traceback texts into existing_texts to prevent duplicate extraction
        existing_texts = existing_texts | traceback_texts

        # ── General code patterns ─────────────────────────────────

        # Error types (ValueError, KeyError, etc.) — extract before PascalCase
        error_texts: set[str] = set()
        for match in self.ERROR_TYPE_PATTERN.finditer(text):
            if (match.start(), match.end()) in existing_spans:
                continue
            name = match.group(1)
            if name.lower() in existing_texts:
                continue
            error_texts.add(name)
            entities.append(
                Entity(
                    text=name,
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.ERROR_TYPE,
                    confidence=0.9,
                )
            )

        # PascalCase (e.g., ReflexPipeline, MemoryEncoder)
        for match in self.PASCAL_CASE_PATTERN.finditer(text):
            if (match.start(), match.end()) in existing_spans:
                continue
            name = match.group(1)
            if name.lower() in existing_texts or name in error_texts:
                continue
            subtype = EntitySubtype.CLASS_NAME
            if name in function_names:
                subtype = EntitySubtype.FUNCTION_NAME
            entities.append(
                Entity(
                    text=name,
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    subtype=subtype,
                    confidence=0.85,
                )
            )

        # Dotted module paths (e.g., neural_memory.engine, os.path)
        for match in self.MODULE_PATH_PATTERN.finditer(text):
            if (match.start(), match.end()) in existing_spans:
                continue
            if match.group(1).lower() in existing_texts:
                continue
            entities.append(
                Entity(
                    text=match.group(1),
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.MODULE_NAME,
                    confidence=0.85,
                )
            )

        # snake_case (e.g., extract_keywords, activate_trail)
        for match in self.SNAKE_CASE_PATTERN.finditer(text):
            if (match.start(), match.end()) in existing_spans:
                continue
            word = match.group(1)
            if word.lower() in existing_texts:
                continue
            # Skip common non-code snake_case (e.g., stop words joined)
            if len(word) < 5:
                continue
            subtype = (
                EntitySubtype.FUNCTION_NAME if word in function_names else EntitySubtype.CODE_SYMBOL
            )
            entities.append(
                Entity(
                    text=word,
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    subtype=subtype,
                    confidence=0.8,
                )
            )

        # Package names in context (pip install X, import X)
        for match in self.PACKAGE_CONTEXT_PATTERN.finditer(text):
            pkg = match.group(1)
            if pkg.lower() in existing_texts:
                continue
            entities.append(
                Entity(
                    text=pkg,
                    type=EntityType.CODE,
                    start=match.start(1),
                    end=match.end(1),
                    subtype=EntitySubtype.PACKAGE_NAME,
                    confidence=0.85,
                )
            )

        # File paths (e.g., src/neural_memory/server.py)
        for match in self.FILE_PATH_PATTERN.finditer(text):
            if (match.start(), match.end()) in existing_spans:
                continue
            entities.append(
                Entity(
                    text=match.group(0),
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.MODULE_NAME,
                    confidence=0.9,
                )
            )

        return entities

    def _extract_domain_entities(self, text: str) -> list[Entity]:
        """Extract financial, legal, and technical domain entities."""
        entities: list[Entity] = []

        # Financial metrics (ROE = 12.8%, revenue = 500 tỷ)
        for match in self.FINANCIAL_METRIC_PATTERN.finditer(text):
            raw_val = match.group(2) or ""
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.FINANCIAL_METRIC,
                    confidence=0.85,
                    raw_value=raw_val.strip(),
                    unit=_detect_unit(raw_val) if raw_val else "",
                )
            )

        # Currency amounts ($25M, 500 triệu VND)
        for match in self.CURRENCY_AMOUNT_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.CURRENCY_AMOUNT,
                    confidence=0.9,
                    raw_value=match.group(0).strip(),
                    unit=_detect_currency(match.group(0)),
                )
            )

        # Fiscal periods (Q1 2024, FY2025)
        for match in self.FISCAL_PERIOD_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.FISCAL_PERIOD,
                    confidence=0.9,
                    raw_value=match.group(0).strip(),
                )
            )

        # Regulations (Điều 468 BLDS, Section 301)
        for match in self.REGULATION_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.REGULATION,
                    confidence=0.85,
                    raw_value=match.group(0).strip(),
                )
            )

        # Contract clauses (Clause 5.2, Khoản 3 Điều 12)
        for match in self.CONTRACT_CLAUSE_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.CONTRACT_CLAUSE,
                    confidence=0.85,
                    raw_value=match.group(0).strip(),
                )
            )

        # Legal entities (CTCP ABC, XYZ LLC)
        for match in self.LEGAL_ENTITY_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.ORGANIZATION,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.LEGAL_ENTITY,
                    confidence=0.8,
                    raw_value=match.group(0).strip(),
                )
            )

        # API endpoints (GET /api/v1/users)
        for match in self.API_ENDPOINT_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.API_ENDPOINT,
                    confidence=0.9,
                    raw_value=match.group(1),
                )
            )

        # Versions (v2.1.0, Python 3.11)
        for match in self.VERSION_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.VERSION,
                    confidence=0.9,
                    raw_value=match.group(0).strip(),
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

        # Find words after location indicators (pre-compiled patterns)
        for pattern in self._LOCATION_PATTERNS.values():
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


# ── Module-level helpers for domain extraction ─────────────────────


def _detect_unit(value: str) -> str:
    """Detect unit from a financial value string."""
    v = value.strip().lower()
    if "%" in v:
        return "percent"
    if any(w in v for w in ("tỷ", "billion", "b")):
        return "billion"
    if any(w in v for w in ("triệu", "million", "m")):
        return "million"
    if any(w in v for w in ("nghìn", "thousand", "k")):
        return "thousand"
    return ""


def _detect_currency(text: str) -> str:
    """Detect currency from a currency amount string."""
    t = text.strip()
    if t.startswith("$") or "USD" in t.upper():
        return "USD"
    if t.startswith("€") or "EUR" in t.upper():
        return "EUR"
    if t.startswith("£") or "GBP" in t.upper():
        return "GBP"
    if t.startswith("¥") or "JPY" in t.upper():
        return "JPY"
    if t.startswith("₫") or any(w in t.upper() for w in ("VND", "VNĐ", "ĐỒNG")):
        return "VND"
    if any(w in t.lower() for w in ("tỷ", "triệu", "nghìn")):
        return "VND"
    return ""
