"""Tests for Brain Quality C1+C2 — Domain entity extraction + structured data encoding.

Covers:
- EntitySubtype enum definition
- Financial metric extraction (Vietnamese + English)
- Currency amount detection (multi-currency)
- Fiscal period extraction
- Legal regulation patterns (Vietnamese + English)
- Legal entity patterns
- API endpoint extraction
- Version detection
- Subtype metadata on Entity dataclass
- Domain synapse types (HAS_VALUE, MEASURED_AT, etc.)
"""

from __future__ import annotations

import pytest

from neural_memory.core.synapse import SynapseType
from neural_memory.extraction.entities import (
    Entity,
    EntityExtractor,
    EntitySubtype,
    EntityType,
    _detect_currency,
    _detect_unit,
)

# ── EntitySubtype enum ─────────────────────────────────────────────


class TestEntitySubtype:
    def test_financial_subtypes(self) -> None:
        assert EntitySubtype.FINANCIAL_METRIC == "financial_metric"
        assert EntitySubtype.CURRENCY_AMOUNT == "currency_amount"
        assert EntitySubtype.FISCAL_PERIOD == "fiscal_period"

    def test_legal_subtypes(self) -> None:
        assert EntitySubtype.REGULATION == "regulation"
        assert EntitySubtype.CONTRACT_CLAUSE == "contract_clause"
        assert EntitySubtype.LEGAL_ENTITY == "legal_entity"

    def test_technical_subtypes(self) -> None:
        assert EntitySubtype.API_ENDPOINT == "api_endpoint"
        assert EntitySubtype.CODE_SYMBOL == "code_symbol"
        assert EntitySubtype.VERSION == "version"

    def test_all_subtypes_are_strings(self) -> None:
        for subtype in EntitySubtype:
            assert isinstance(subtype, str)


# ── Domain Synapse Types ───────────────────────────────────────────


class TestDomainSynapseTypes:
    def test_domain_synapse_types_exist(self) -> None:
        assert SynapseType.HAS_VALUE == "has_value"
        assert SynapseType.MEASURED_AT == "measured_at"
        assert SynapseType.REGULATES == "regulates"
        assert SynapseType.IN_ROW == "in_row"
        assert SynapseType.IN_COLUMN == "in_column"


# ── Entity dataclass ──────────────────────────────────────────────


class TestEntityDataclass:
    def test_entity_with_subtype(self) -> None:
        entity = Entity(
            text="ROE = 12.8%",
            type=EntityType.UNKNOWN,
            start=0,
            end=11,
            subtype=EntitySubtype.FINANCIAL_METRIC,
            raw_value="12.8%",
            unit="percent",
        )
        assert entity.subtype == EntitySubtype.FINANCIAL_METRIC
        assert entity.raw_value == "12.8%"
        assert entity.unit == "percent"

    def test_entity_without_subtype(self) -> None:
        entity = Entity(text="Alice", type=EntityType.PERSON, start=0, end=5)
        assert entity.subtype is None
        assert entity.raw_value == ""
        assert entity.unit == ""

    def test_entity_frozen(self) -> None:
        entity = Entity(text="test", type=EntityType.UNKNOWN, start=0, end=4)
        with pytest.raises(AttributeError):
            entity.subtype = EntitySubtype.VERSION  # type: ignore[misc]


# ── Financial Extraction ──────────────────────────────────────────


class TestFinancialExtraction:
    def setup_method(self) -> None:
        self.extractor = EntityExtractor()

    def test_roe_metric(self) -> None:
        entities = self.extractor.extract("ROE = 12.8% cho quý này")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.FINANCIAL_METRIC in subtypes

    def test_ebitda(self) -> None:
        entities = self.extractor.extract("EBITDA reached $500M")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.FINANCIAL_METRIC in subtypes

    def test_vietnamese_financial(self) -> None:
        entities = self.extractor.extract("Doanh thu = 500 tỷ VND")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert any(
            s in (EntitySubtype.FINANCIAL_METRIC, EntitySubtype.CURRENCY_AMOUNT) for s in subtypes
        )

    def test_currency_usd(self) -> None:
        entities = self.extractor.extract("Total investment: $25M")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.CURRENCY_AMOUNT in subtypes

    def test_currency_vnd(self) -> None:
        entities = self.extractor.extract("Chi phí: 500 triệu VND")
        subtypes = [e.subtype for e in entities if e.subtype]
        # Should match either financial_metric or currency_amount
        assert any(
            s in (EntitySubtype.FINANCIAL_METRIC, EntitySubtype.CURRENCY_AMOUNT) for s in subtypes
        )

    def test_fiscal_period_quarter(self) -> None:
        entities = self.extractor.extract("Results for Q3 2024 are strong")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.FISCAL_PERIOD in subtypes

    def test_fiscal_period_fy(self) -> None:
        entities = self.extractor.extract("FY2025 projections indicate growth")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.FISCAL_PERIOD in subtypes


# ── Legal Extraction ──────────────────────────────────────────────


class TestLegalExtraction:
    def setup_method(self) -> None:
        self.extractor = EntityExtractor()

    def test_vietnamese_regulation(self) -> None:
        entities = self.extractor.extract("Theo Điều 468 BLDS")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.REGULATION in subtypes

    def test_english_section(self) -> None:
        entities = self.extractor.extract("Under Section 301 of the SOX Act")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.REGULATION in subtypes

    def test_contract_clause(self) -> None:
        entities = self.extractor.extract("See Clause 5.2.1 for details")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.CONTRACT_CLAUSE in subtypes

    def test_legal_entity_vietnamese(self) -> None:
        entities = self.extractor.extract("CTCP Vinamilk đã công bố")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.LEGAL_ENTITY in subtypes


# ── Technical Extraction ──────────────────────────────────────────


class TestTechnicalExtraction:
    def setup_method(self) -> None:
        self.extractor = EntityExtractor()

    def test_api_endpoint(self) -> None:
        entities = self.extractor.extract("Call GET /api/v1/users for data")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.API_ENDPOINT in subtypes

    def test_api_endpoint_post(self) -> None:
        entities = self.extractor.extract("POST /webhook/events")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.API_ENDPOINT in subtypes

    def test_version_semver(self) -> None:
        entities = self.extractor.extract("Updated to v2.1.0")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.VERSION in subtypes

    def test_version_python(self) -> None:
        entities = self.extractor.extract("Requires Python 3.11")
        subtypes = [e.subtype for e in entities if e.subtype]
        assert EntitySubtype.VERSION in subtypes


# ── Helper functions ──────────────────────────────────────────────


class TestHelpers:
    def test_detect_unit_percent(self) -> None:
        assert _detect_unit("12.8%") == "percent"

    def test_detect_unit_billion(self) -> None:
        assert _detect_unit("500 tỷ") == "billion"

    def test_detect_unit_million(self) -> None:
        assert _detect_unit("25M") == "million"
        assert _detect_unit("100 triệu") == "million"

    def test_detect_unit_empty(self) -> None:
        assert _detect_unit("42") == ""

    def test_detect_currency_usd(self) -> None:
        assert _detect_currency("$25M") == "USD"
        assert _detect_currency("100 USD") == "USD"

    def test_detect_currency_vnd(self) -> None:
        assert _detect_currency("500 triệu VND") == "VND"
        assert _detect_currency("₫100") == "VND"

    def test_detect_currency_eur(self) -> None:
        assert _detect_currency("€1.2B") == "EUR"

    def test_detect_currency_empty(self) -> None:
        assert _detect_currency("42") == ""


# ── Backward compatibility ────────────────────────────────────────


class TestBackwardCompatibility:
    def test_existing_person_extraction_unchanged(self) -> None:
        extractor = EntityExtractor()
        entities = extractor.extract("Anh Nguyễn Văn A đã gọi")
        person_entities = [e for e in entities if e.type == EntityType.PERSON]
        assert len(person_entities) >= 1

    def test_existing_code_extraction_unchanged(self) -> None:
        extractor = EntityExtractor()
        entities = extractor.extract("The ReflexPipeline class handles retrieval")
        code_entities = [e for e in entities if e.type == EntityType.CODE]
        assert len(code_entities) >= 1

    def test_entity_without_subtype_works(self) -> None:
        """Pre-existing entities without subtypes must still work."""
        entity = Entity(text="Alice", type=EntityType.PERSON, start=0, end=5)
        assert entity.subtype is None
        assert entity.confidence == 1.0
