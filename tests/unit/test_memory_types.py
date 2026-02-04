"""Tests for memory types module."""

from __future__ import annotations

from datetime import datetime, timedelta

from neural_memory.core.memory_types import (
    DEFAULT_EXPIRY_DAYS,
    Confidence,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
    suggest_memory_type,
)


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_all_memory_types_exist(self) -> None:
        """Test all expected memory types are defined."""
        expected_types = [
            "fact",
            "decision",
            "preference",
            "todo",
            "insight",
            "context",
            "instruction",
            "error",
            "workflow",
            "reference",
        ]
        for t in expected_types:
            assert MemoryType(t) is not None

    def test_memory_type_from_string(self) -> None:
        """Test creating MemoryType from string."""
        assert MemoryType("fact") == MemoryType.FACT
        assert MemoryType("decision") == MemoryType.DECISION
        assert MemoryType("todo") == MemoryType.TODO


class TestPriority:
    """Tests for Priority enum."""

    def test_priority_values(self) -> None:
        """Test priority values are correct."""
        assert Priority.LOWEST == 0
        assert Priority.LOW == 2
        assert Priority.NORMAL == 5
        assert Priority.HIGH == 7
        assert Priority.CRITICAL == 10

    def test_from_int_low(self) -> None:
        """Test converting low integers to Priority."""
        assert Priority.from_int(0) == Priority.LOWEST
        assert Priority.from_int(1) == Priority.LOWEST
        assert Priority.from_int(2) == Priority.LOW
        assert Priority.from_int(3) == Priority.LOW

    def test_from_int_normal(self) -> None:
        """Test converting middle integers to Priority."""
        assert Priority.from_int(4) == Priority.NORMAL
        assert Priority.from_int(5) == Priority.NORMAL
        assert Priority.from_int(6) == Priority.NORMAL

    def test_from_int_high(self) -> None:
        """Test converting high integers to Priority."""
        assert Priority.from_int(7) == Priority.HIGH
        assert Priority.from_int(8) == Priority.HIGH
        assert Priority.from_int(9) == Priority.CRITICAL
        assert Priority.from_int(10) == Priority.CRITICAL


class TestConfidence:
    """Tests for Confidence enum."""

    def test_confidence_levels_exist(self) -> None:
        """Test all confidence levels are defined."""
        expected = ["verified", "high", "medium", "low", "inferred"]
        for c in expected:
            assert Confidence(c) is not None


class TestProvenance:
    """Tests for Provenance dataclass."""

    def test_create_provenance(self) -> None:
        """Test creating a Provenance."""
        prov = Provenance(source="user_input")
        assert prov.source == "user_input"
        assert prov.confidence == Confidence.MEDIUM
        assert prov.verified is False
        assert prov.verified_at is None

    def test_verify_provenance(self) -> None:
        """Test verifying a Provenance."""
        prov = Provenance(source="ai_inference", confidence=Confidence.LOW)
        verified = prov.verify()

        assert verified.verified is True
        assert verified.confidence == Confidence.VERIFIED
        assert verified.verified_at is not None
        assert verified.source == "ai_inference"  # Preserved

    def test_confirm_provenance(self) -> None:
        """Test confirming a Provenance."""
        prov = Provenance(source="user_input")
        confirmed = prov.confirm()

        assert confirmed.last_confirmed is not None
        assert confirmed.source == prov.source
        assert confirmed.confidence == prov.confidence


class TestTypedMemory:
    """Tests for TypedMemory dataclass."""

    def test_create_typed_memory(self) -> None:
        """Test creating a TypedMemory."""
        mem = TypedMemory.create(
            fiber_id="fiber-123",
            memory_type=MemoryType.FACT,
        )
        assert mem.fiber_id == "fiber-123"
        assert mem.memory_type == MemoryType.FACT
        assert mem.priority == Priority.NORMAL
        assert mem.expires_at is None
        assert mem.is_expired is False

    def test_create_with_expiry(self) -> None:
        """Test creating TypedMemory with expiry."""
        mem = TypedMemory.create(
            fiber_id="fiber-123",
            memory_type=MemoryType.TODO,
            expires_in_days=7,
        )
        assert mem.expires_at is not None
        # Allow for slight timing variance
        assert mem.days_until_expiry in (6, 7)
        assert mem.is_expired is False

    def test_create_with_priority_int(self) -> None:
        """Test creating TypedMemory with integer priority."""
        mem = TypedMemory.create(
            fiber_id="fiber-123",
            memory_type=MemoryType.DECISION,
            priority=8,
        )
        assert mem.priority == Priority.HIGH

    def test_create_with_priority_enum(self) -> None:
        """Test creating TypedMemory with Priority enum."""
        mem = TypedMemory.create(
            fiber_id="fiber-123",
            memory_type=MemoryType.DECISION,
            priority=Priority.CRITICAL,
        )
        assert mem.priority == Priority.CRITICAL

    def test_is_expired(self) -> None:
        """Test expiry detection."""
        mem = TypedMemory(
            fiber_id="fiber-123",
            memory_type=MemoryType.CONTEXT,
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        assert mem.is_expired is True

    def test_not_expired(self) -> None:
        """Test non-expired memory."""
        mem = TypedMemory(
            fiber_id="fiber-123",
            memory_type=MemoryType.CONTEXT,
            expires_at=datetime.utcnow() + timedelta(days=7),
        )
        assert mem.is_expired is False

    def test_no_expiry_never_expires(self) -> None:
        """Test memory without expiry never expires."""
        mem = TypedMemory(
            fiber_id="fiber-123",
            memory_type=MemoryType.FACT,
            expires_at=None,
        )
        assert mem.is_expired is False
        assert mem.days_until_expiry is None

    def test_with_priority(self) -> None:
        """Test updating priority creates new instance."""
        mem = TypedMemory.create(
            fiber_id="fiber-123",
            memory_type=MemoryType.TODO,
            priority=Priority.LOW,
        )
        updated = mem.with_priority(Priority.CRITICAL)

        assert updated.priority == Priority.CRITICAL
        assert mem.priority == Priority.LOW  # Original unchanged

    def test_verify(self) -> None:
        """Test verifying a TypedMemory."""
        mem = TypedMemory.create(
            fiber_id="fiber-123",
            memory_type=MemoryType.FACT,
        )
        verified = mem.verify()

        assert verified.provenance.verified is True
        assert verified.provenance.confidence == Confidence.VERIFIED

    def test_extend_expiry(self) -> None:
        """Test extending expiry."""
        mem = TypedMemory.create(
            fiber_id="fiber-123",
            memory_type=MemoryType.TODO,
            expires_in_days=7,
        )
        extended = mem.extend_expiry(30)

        # Allow for slight timing variance
        assert extended.days_until_expiry in (29, 30)

    def test_create_with_tags(self) -> None:
        """Test creating with tags."""
        mem = TypedMemory.create(
            fiber_id="fiber-123",
            memory_type=MemoryType.DECISION,
            tags={"architecture", "database"},
        )
        assert "architecture" in mem.tags
        assert "database" in mem.tags

    def test_create_with_project(self) -> None:
        """Test creating with project scope."""
        mem = TypedMemory.create(
            fiber_id="fiber-123",
            memory_type=MemoryType.CONTEXT,
            project_id="project-abc",
        )
        assert mem.project_id == "project-abc"


class TestDefaultExpiry:
    """Tests for default expiry settings."""

    def test_facts_dont_expire(self) -> None:
        """Test facts have no default expiry."""
        assert DEFAULT_EXPIRY_DAYS[MemoryType.FACT] is None

    def test_todos_expire_in_30_days(self) -> None:
        """Test TODOs expire in 30 days."""
        assert DEFAULT_EXPIRY_DAYS[MemoryType.TODO] == 30

    def test_context_expires_in_7_days(self) -> None:
        """Test context expires in 7 days."""
        assert DEFAULT_EXPIRY_DAYS[MemoryType.CONTEXT] == 7

    def test_decisions_expire_in_90_days(self) -> None:
        """Test decisions expire in 90 days."""
        assert DEFAULT_EXPIRY_DAYS[MemoryType.DECISION] == 90


class TestSuggestMemoryType:
    """Tests for memory type suggestion."""

    def test_suggests_todo(self) -> None:
        """Test suggesting TODO type."""
        assert suggest_memory_type("Need to refactor the auth module") == MemoryType.TODO
        assert suggest_memory_type("TODO: fix the bug") == MemoryType.TODO
        assert suggest_memory_type("Should implement caching") == MemoryType.TODO

    def test_suggests_decision(self) -> None:
        """Test suggesting DECISION type."""
        assert suggest_memory_type("We decided to use PostgreSQL") == MemoryType.DECISION
        assert suggest_memory_type("Chose React over Vue") == MemoryType.DECISION
        assert suggest_memory_type("Going with microservices") == MemoryType.DECISION

    def test_suggests_preference(self) -> None:
        """Test suggesting PREFERENCE type."""
        assert suggest_memory_type("I prefer tabs over spaces") == MemoryType.PREFERENCE
        assert suggest_memory_type("I like dark themes") == MemoryType.PREFERENCE
        assert suggest_memory_type("I hate verbose code") == MemoryType.PREFERENCE

    def test_suggests_error(self) -> None:
        """Test suggesting ERROR type."""
        assert suggest_memory_type("API returns 429 error on rate limit") == MemoryType.ERROR
        assert suggest_memory_type("Bug in auth module causes crash") == MemoryType.ERROR
        assert suggest_memory_type("Exception when parsing JSON") == MemoryType.ERROR

    def test_suggests_workflow(self) -> None:
        """Test suggesting WORKFLOW type."""
        assert suggest_memory_type("Deploy flow: test -> stage -> prod") == MemoryType.WORKFLOW
        assert suggest_memory_type("CI/CD pipeline runs on push") == MemoryType.WORKFLOW
        assert suggest_memory_type("Review process: create PR, get approval, merge") == MemoryType.WORKFLOW

    def test_suggests_instruction(self) -> None:
        """Test suggesting INSTRUCTION type."""
        assert suggest_memory_type("Always use type hints in Python") == MemoryType.INSTRUCTION
        assert suggest_memory_type("Never use eval() in production") == MemoryType.INSTRUCTION
        assert suggest_memory_type("Make sure to run tests before commit") == MemoryType.INSTRUCTION

    def test_suggests_reference(self) -> None:
        """Test suggesting REFERENCE type."""
        assert suggest_memory_type("Documentation at https://docs.example.com") == MemoryType.REFERENCE
        assert suggest_memory_type("See docs for more info") == MemoryType.REFERENCE

    def test_suggests_insight(self) -> None:
        """Test suggesting INSIGHT type."""
        assert suggest_memory_type("Learned that caching improves perf 10x") == MemoryType.INSIGHT
        assert suggest_memory_type("Discovered the issue was in the parser") == MemoryType.INSIGHT
        assert suggest_memory_type("Turns out the API needs auth headers") == MemoryType.INSIGHT

    def test_defaults_to_fact(self) -> None:
        """Test defaulting to FACT for general content."""
        assert suggest_memory_type("Python 3.11 was released in October 2022") == MemoryType.FACT
        assert suggest_memory_type("The project uses FastAPI") == MemoryType.FACT
        assert suggest_memory_type("Meeting with Alice about API design") == MemoryType.FACT
