"""Tests for MCP response compactor."""

from __future__ import annotations

import copy

from neural_memory.mcp.response_compactor import (
    STRIPPABLE_KEYS,
    apply_token_budget,
    compact_response,
    needs_auto_compact,
    should_compact,
)
from neural_memory.unified_config import ResponseConfig


def _default_config(**overrides: object) -> ResponseConfig:
    """Create a ResponseConfig with optional overrides."""
    kwargs: dict = {
        "compact_mode": True,
        "max_list_items": 3,
        "strip_hints": True,
        "content_preview_length": 120,
    }
    kwargs.update(overrides)
    return ResponseConfig(**kwargs)


# ---------- compact_response ----------


class TestCompactResponseStripKeys:
    """Test that strippable metadata keys are removed."""

    def test_strips_all_metadata_keys(self) -> None:
        result = {key: f"value_{key}" for key in STRIPPABLE_KEYS}
        result["answer"] = "keep me"
        result["confidence"] = 0.85

        compacted = compact_response(result, _default_config())

        for key in STRIPPABLE_KEYS:
            assert key not in compacted, f"{key} should be stripped"
        assert compacted["answer"] == "keep me"
        assert compacted["confidence"] == 0.85

    def test_preserves_all_keys_when_strip_hints_disabled(self) -> None:
        result = {"maintenance_hint": "hint", "answer": "data"}
        compacted = compact_response(result, _default_config(strip_hints=False))

        assert "maintenance_hint" in compacted
        assert compacted["answer"] == "data"

    def test_handles_missing_strippable_keys(self) -> None:
        result = {"answer": "only core data"}
        compacted = compact_response(result, _default_config())

        assert compacted == {"answer": "only core data"}


class TestCompactResponseListTruncation:
    """Test that long lists are truncated with metadata."""

    def test_truncates_long_lists(self) -> None:
        result = {"warnings": [f"warn_{i}" for i in range(10)]}
        compacted = compact_response(result, _default_config(max_list_items=3))

        assert len(compacted["warnings"]) == 3
        assert compacted["_warnings_truncated"] is True
        assert compacted["_warnings_total"] == 10

    def test_preserves_short_lists(self) -> None:
        result = {"warnings": ["a", "b"]}
        compacted = compact_response(result, _default_config(max_list_items=3))

        assert compacted["warnings"] == ["a", "b"]
        assert "_warnings_truncated" not in compacted

    def test_replaces_fibers_matched_with_count(self) -> None:
        result = {"fibers_matched": ["f1", "f2", "f3"]}
        compacted = compact_response(result, _default_config())

        assert "fibers_matched" not in compacted
        assert compacted["fibers_matched_count"] == 3


class TestCompactResponseImmutability:
    """Test that input dict is never mutated."""

    def test_input_not_mutated(self) -> None:
        original = {
            "answer": "test",
            "maintenance_hint": "hint",
            "warnings": [1, 2, 3, 4, 5],
        }
        frozen = copy.deepcopy(original)

        compact_response(original, _default_config(max_list_items=2))

        assert original == frozen, "Input dict was mutated"

    def test_nested_dict_not_mutated(self) -> None:
        original = {"nested": {"maintenance_hint": "strip", "data": "keep"}}
        frozen = copy.deepcopy(original)

        compact_response(original, _default_config())

        assert original == frozen, "Nested dict was mutated"


class TestCompactResponseNestedDicts:
    """Test recursive handling of nested dicts."""

    def test_strips_keys_in_nested_dicts(self) -> None:
        result = {
            "data": {
                "answer": "keep",
                "maintenance_hint": "strip",
                "update_hint": "strip",
            }
        }
        compacted = compact_response(result, _default_config())

        assert compacted["data"]["answer"] == "keep"
        assert "maintenance_hint" not in compacted["data"]
        assert "update_hint" not in compacted["data"]

    def test_handles_non_dict_input(self) -> None:
        assert compact_response("string", _default_config()) == "string"  # type: ignore[arg-type]
        assert compact_response(42, _default_config()) == 42  # type: ignore[arg-type]


class TestCompactResponseEdgeCases:
    """Test edge cases."""

    def test_empty_dict(self) -> None:
        assert compact_response({}, _default_config()) == {}

    def test_only_strippable_keys(self) -> None:
        result = {"maintenance_hint": "a", "update_hint": "b"}
        assert compact_response(result, _default_config()) == {}

    def test_list_exactly_at_limit(self) -> None:
        result = {"items": [1, 2, 3]}
        compacted = compact_response(result, _default_config(max_list_items=3))
        assert compacted["items"] == [1, 2, 3]
        assert "_items_truncated" not in compacted


# ---------- should_compact ----------


class TestShouldCompact:
    """Test compact mode decision logic."""

    def test_per_call_true_overrides_config_false(self) -> None:
        args: dict[str, object] = {"compact": True, "query": "test"}
        assert should_compact(tool_args=args, config=ResponseConfig(compact_mode=False))
        assert "compact" not in args, "compact key should be popped"

    def test_per_call_false_overrides_config_true(self) -> None:
        args: dict[str, object] = {"compact": False}
        assert not should_compact(tool_args=args, config=ResponseConfig(compact_mode=True))

    def test_no_per_call_uses_config(self) -> None:
        args: dict[str, object] = {"query": "test"}
        assert should_compact(tool_args=args, config=ResponseConfig(compact_mode=True))
        assert not should_compact(tool_args=args, config=ResponseConfig(compact_mode=False))

    def test_pops_compact_from_args(self) -> None:
        args: dict[str, object] = {"compact": True, "query": "test"}
        should_compact(tool_args=args, config=ResponseConfig())
        assert "compact" not in args


# ---------- ResponseConfig ----------


class TestResponseConfig:
    """Test ResponseConfig creation and from_dict."""

    def test_defaults(self) -> None:
        config = ResponseConfig()
        assert config.compact_mode is False
        assert config.max_list_items == 10
        assert config.strip_hints is True
        assert config.content_preview_length == 120

    def test_from_dict(self) -> None:
        config = ResponseConfig.from_dict(
            {"compact_mode": True, "max_list_items": 5, "content_preview_length": 80}
        )
        assert config.compact_mode is True
        assert config.max_list_items == 5
        assert config.content_preview_length == 80
        assert config.strip_hints is True  # default

    def test_from_empty_dict(self) -> None:
        config = ResponseConfig.from_dict({})
        assert config.compact_mode is False
        assert config.max_list_items == 10


# ---------- Phase 2: Per-Tool Compaction ----------


class TestContentPreview:
    """Test content/body/description truncation in list items."""

    def test_truncates_long_content_in_list_items(self) -> None:
        result = {
            "memories": [
                {"fiber_id": "f1", "content": "A" * 200, "type": "fact"},
                {"fiber_id": "f2", "content": "short", "type": "fact"},
            ]
        }
        compacted = compact_response(result, _default_config(content_preview_length=50))

        assert compacted["memories"][0]["content"] == "A" * 50 + "..."
        assert compacted["memories"][0]["_content_truncated"] is True
        assert compacted["memories"][1]["content"] == "short"
        assert "_content_truncated" not in compacted["memories"][1]

    def test_truncates_body_and_description_keys(self) -> None:
        result = {
            "items": [
                {"body": "B" * 200, "description": "D" * 200},
            ]
        }
        compacted = compact_response(result, _default_config(content_preview_length=30))

        assert compacted["items"][0]["body"] == "B" * 30 + "..."
        assert compacted["items"][0]["_body_truncated"] is True
        assert compacted["items"][0]["description"] == "D" * 30 + "..."
        assert compacted["items"][0]["_description_truncated"] is True

    def test_preserves_non_string_content(self) -> None:
        result = {"items": [{"content": 42, "data": [1, 2]}]}
        compacted = compact_response(result, _default_config(content_preview_length=10))

        assert compacted["items"][0]["content"] == 42
        assert compacted["items"][0]["data"] == [1, 2]

    def test_content_preview_with_list_truncation(self) -> None:
        """Both list truncation and content preview applied together."""
        result = {"memories": [{"content": "X" * 200, "id": i} for i in range(10)]}
        compacted = compact_response(
            result, _default_config(max_list_items=2, content_preview_length=50)
        )

        assert len(compacted["memories"]) == 2
        assert compacted["_memories_truncated"] is True
        assert compacted["_memories_total"] == 10
        assert compacted["memories"][0]["content"] == "X" * 50 + "..."

    def test_list_item_immutability(self) -> None:
        original_item = {"content": "A" * 200, "id": "f1"}
        result = {"memories": [original_item]}
        frozen = copy.deepcopy(original_item)

        compact_response(result, _default_config(content_preview_length=50))

        assert original_item == frozen, "List item was mutated"


class TestCountReplaceKeys:
    """Test count-replace for conflicts, expiry_warnings, etc."""

    def test_replaces_conflicts_with_count(self) -> None:
        result = {"conflicts": [{"id": 1}, {"id": 2}], "answer": "data"}
        compacted = compact_response(result, _default_config())

        assert "conflicts" not in compacted
        assert compacted["conflicts_count"] == 2
        assert compacted["answer"] == "data"

    def test_replaces_expiry_warnings_with_count(self) -> None:
        result = {"expiry_warnings": ["w1", "w2", "w3"]}
        compacted = compact_response(result, _default_config())

        assert "expiry_warnings" not in compacted
        assert compacted["expiry_warnings_count"] == 3

    def test_empty_count_replace_list(self) -> None:
        result = {"conflicts": []}
        compacted = compact_response(result, _default_config())

        assert compacted["conflicts_count"] == 0


class TestLongStringTruncation:
    """Test truncation of long string fields (markdown, etc.)."""

    def test_truncates_long_markdown(self) -> None:
        result = {"markdown": "M" * 1000, "title": "keep"}
        compacted = compact_response(result, _default_config())

        assert compacted["markdown"] == "M" * 500 + "...(truncated)"
        assert compacted["_markdown_truncated"] is True
        assert compacted["title"] == "keep"

    def test_preserves_short_markdown(self) -> None:
        result = {"markdown": "short text"}
        compacted = compact_response(result, _default_config())

        assert compacted["markdown"] == "short text"
        assert "_markdown_truncated" not in compacted


class TestSimulatedToolResponses:
    """Test with realistic MCP tool response shapes."""

    def test_recall_associative_compact(self) -> None:
        """Simulate nmem_recall associative response."""
        result = {
            "answer": "The project uses PostgreSQL for storage.",
            "confidence": 0.85,
            "neurons_activated": 12,
            "fibers_matched": ["f1", "f2", "f3", "f4", "f5"],
            "depth_used": 2,
            "tokens_used": 450,
            "score_breakdown": {"fts": 0.6, "embedding": 0.3},
            "conflicts": [{"id": "c1", "type": "factual"}],
            "expiry_warnings": [],
            "related_queries": ["postgres setup", "db config"],
            "maintenance_hint": "Consider consolidation",
            "update_hint": "v5.0 available",
            "onboarding": None,
            "pending_alerts": [{"type": "stale"}],
            "session_topics": "postgresql",
        }
        compacted = compact_response(result, _default_config())

        # Core data preserved
        assert compacted["answer"] == "The project uses PostgreSQL for storage."
        assert compacted["confidence"] == 0.85
        assert compacted["neurons_activated"] == 12
        assert compacted["depth_used"] == 2
        assert compacted["tokens_used"] == 450

        # Count-replaced
        assert compacted["fibers_matched_count"] == 5
        assert compacted["conflicts_count"] == 1
        assert compacted["expiry_warnings_count"] == 0

        # Stripped
        for key in [
            "score_breakdown",
            "related_queries",
            "maintenance_hint",
            "update_hint",
            "onboarding",
            "pending_alerts",
            "session_topics",
        ]:
            assert key not in compacted

    def test_recall_exact_compact(self) -> None:
        """Simulate nmem_recall exact mode with long content."""
        result = {
            "mode": "exact",
            "memories": [
                {
                    "fiber_id": f"f{i}",
                    "content": f"Memory content that is quite long " * 10,
                    "memory_type": "fact",
                    "priority": 5,
                    "tags": ["test"],
                    "created_at": "2026-03-17",
                }
                for i in range(20)
            ],
            "confidence": 0.9,
            "neurons_activated": 20,
            "fibers_matched": [f"f{i}" for i in range(20)],
            "depth_used": 1,
        }
        compacted = compact_response(
            result, _default_config(max_list_items=5, content_preview_length=80)
        )

        assert compacted["mode"] == "exact"
        assert len(compacted["memories"]) == 5
        assert compacted["_memories_truncated"] is True
        assert compacted["_memories_total"] == 20
        assert compacted["fibers_matched_count"] == 20

        # Content previewed
        mem = compacted["memories"][0]
        assert len(mem["content"]) == 83  # 80 + "..."
        assert mem["content"].endswith("...")
        assert mem["_content_truncated"] is True
        # Non-content fields preserved
        assert mem["memory_type"] == "fact"
        assert mem["tags"] == ["test"]

    def test_health_compact(self) -> None:
        """Simulate nmem_health response."""
        result = {
            "grade": "B",
            "purity_score": 82.0,
            "warnings": [f"warn_{i}" for i in range(8)],
            "recommendations": [f"rec_{i}" for i in range(6)],
            "top_penalties": [f"pen_{i}" for i in range(5)],
            "roadmap": {"steps": [f"step_{i}" for i in range(10)]},
        }
        compacted = compact_response(result, _default_config(max_list_items=3))

        assert compacted["grade"] == "B"
        assert compacted["purity_score"] == 82.0
        assert len(compacted["warnings"]) == 3
        assert compacted["_warnings_total"] == 8
        assert len(compacted["recommendations"]) == 3
        assert len(compacted["top_penalties"]) == 3
        # roadmap is in STRIPPABLE_KEYS
        assert "roadmap" not in compacted

    def test_narrative_compact(self) -> None:
        """Simulate nmem_narrative response with long markdown."""
        result = {
            "action": "timeline",
            "title": "Project Timeline",
            "markdown": "# Timeline\n" + "- Event details here\n" * 100,
        }
        compacted = compact_response(result, _default_config())

        assert compacted["title"] == "Project Timeline"
        assert compacted["action"] == "timeline"
        assert compacted["_markdown_truncated"] is True
        assert len(compacted["markdown"]) < 520  # 500 + "...(truncated)"


# ---------- Phase 3: Auto-Compact + Token Budget ----------


class TestNeedsAutoCompact:
    """Test auto-compact threshold detection."""

    def test_triggers_when_list_exceeds_threshold(self) -> None:
        result = {"memories": [{"id": i} for i in range(25)]}
        assert needs_auto_compact(result, threshold=20) is True

    def test_does_not_trigger_below_threshold(self) -> None:
        result = {"memories": [{"id": i} for i in range(15)]}
        assert needs_auto_compact(result, threshold=20) is False

    def test_does_not_trigger_at_exact_threshold(self) -> None:
        result = {"memories": [{"id": i} for i in range(20)]}
        assert needs_auto_compact(result, threshold=20) is False

    def test_disabled_when_threshold_zero(self) -> None:
        result = {"memories": [{"id": i} for i in range(100)]}
        assert needs_auto_compact(result, threshold=0) is False

    def test_checks_all_list_fields(self) -> None:
        result = {"small": [1, 2], "big": list(range(50))}
        assert needs_auto_compact(result, threshold=20) is True

    def test_ignores_non_list_values(self) -> None:
        result = {"answer": "long" * 100, "count": 999}
        assert needs_auto_compact(result, threshold=5) is False


class TestApplyTokenBudget:
    """Test progressive token budget enforcement."""

    def test_no_op_when_within_budget(self) -> None:
        result = {"answer": "short", "confidence": 0.9}
        budgeted = apply_token_budget(result, budget=500)
        assert budgeted == result
        assert "_token_budget_applied" not in budgeted

    def test_strips_metadata_at_level1(self) -> None:
        result = {
            "answer": "ok",
            "maintenance_hint": "x" * 200,
            "update_hint": "y" * 200,
            "roadmap": {"steps": ["a" * 100] * 10},
        }
        budgeted = apply_token_budget(result, budget=50)
        assert "_token_budget_applied" in budgeted
        assert "maintenance_hint" not in budgeted
        assert "update_hint" not in budgeted
        assert "roadmap" not in budgeted

    def test_truncates_lists_at_level2(self) -> None:
        result = {
            "memories": [{"id": i, "content": f"mem_{i}"} for i in range(50)],
        }
        budgeted = apply_token_budget(result, budget=200)
        assert "_token_budget_applied" in budgeted
        assert len(budgeted["memories"]) <= 5

    def test_truncates_strings_at_level3(self) -> None:
        result = {
            "answer": "A" * 500,
            "memories": [{"content": "B" * 500}],
        }
        budgeted = apply_token_budget(result, budget=100)
        assert "_token_budget_applied" in budgeted
        assert len(budgeted["answer"]) <= 84  # 80 + "..."

    def test_preserves_immutability(self) -> None:
        original = {"answer": "A" * 500, "maintenance_hint": "big" * 100}
        frozen = copy.deepcopy(original)
        apply_token_budget(original, budget=50)
        assert original == frozen


class TestResponseConfigAutoCompact:
    """Test auto_compact_threshold config field."""

    def test_default_threshold(self) -> None:
        config = ResponseConfig()
        assert config.auto_compact_threshold == 20

    def test_from_dict_with_threshold(self) -> None:
        config = ResponseConfig.from_dict({"auto_compact_threshold": 50})
        assert config.auto_compact_threshold == 50
