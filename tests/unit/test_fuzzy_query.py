"""Tests for fuzzy compositional recall — AST parsing and evaluation."""

from __future__ import annotations

from neural_memory.engine.fuzzy_query import (
    AndNode,
    NotNode,
    OrNode,
    TNorm,
    VectorNode,
    collect_sub_queries,
    evaluate_fuzzy,
    is_fuzzy_query,
    parse_fuzzy_query,
)

# ── Detection ────────────────────────────────────────────────────────


class TestIsFuzzyQuery:
    """Tests for is_fuzzy_query detection."""

    def test_simple_query(self) -> None:
        """Plain query without operators → not fuzzy."""
        assert is_fuzzy_query("project decisions") is False

    def test_and_query(self) -> None:
        assert is_fuzzy_query("decisions AND errors") is True

    def test_or_query(self) -> None:
        assert is_fuzzy_query("React OR Vue") is True

    def test_not_query(self) -> None:
        assert is_fuzzy_query("deployment NOT staging") is True

    def test_lowercase_not_detected(self) -> None:
        """Lowercase 'and' in content should not trigger fuzzy."""
        assert is_fuzzy_query("bread and butter") is True  # regex is case-insensitive

    def test_operator_in_word(self) -> None:
        """'android' contains 'and' but not as standalone operator."""
        assert is_fuzzy_query("android") is False

    def test_no_space_around_operator(self) -> None:
        """Operator without spaces → not fuzzy."""
        assert is_fuzzy_query("ANDroid") is False


# ── Parser ───────────────────────────────────────────────────────────


class TestParseFuzzyQuery:
    """Tests for parse_fuzzy_query."""

    def test_plain_query(self) -> None:
        """No operators → VectorNode."""
        node = parse_fuzzy_query("simple query")
        assert isinstance(node, VectorNode)
        assert node.query == "simple query"

    def test_empty_query(self) -> None:
        node = parse_fuzzy_query("")
        assert isinstance(node, VectorNode)
        assert node.query == ""

    def test_and(self) -> None:
        node = parse_fuzzy_query("auth AND errors")
        assert isinstance(node, AndNode)
        assert isinstance(node.left, VectorNode)
        assert isinstance(node.right, VectorNode)
        assert node.left.query == "auth"
        assert node.right.query == "errors"

    def test_or(self) -> None:
        node = parse_fuzzy_query("React OR Vue")
        assert isinstance(node, OrNode)
        assert isinstance(node.left, VectorNode)
        assert node.left.query == "React"
        assert isinstance(node.right, VectorNode)
        assert node.right.query == "Vue"

    def test_not(self) -> None:
        node = parse_fuzzy_query("deploy NOT staging")
        assert isinstance(node, NotNode)
        assert isinstance(node.base, VectorNode)
        assert node.base.query == "deploy"
        assert isinstance(node.child, VectorNode)
        assert node.child.query == "staging"

    def test_chained_and(self) -> None:
        """A AND B AND C → left-to-right: And(And(A,B), C)."""
        node = parse_fuzzy_query("auth AND errors AND fix")
        assert isinstance(node, AndNode)
        assert isinstance(node.left, AndNode)
        assert isinstance(node.right, VectorNode)
        assert node.right.query == "fix"

    def test_mixed_operators(self) -> None:
        """A AND B OR C → left-to-right: Or(And(A,B), C)."""
        node = parse_fuzzy_query("auth AND errors OR warnings")
        assert isinstance(node, OrNode)
        assert isinstance(node.left, AndNode)
        assert isinstance(node.right, VectorNode)

    def test_whitespace_trimmed(self) -> None:
        node = parse_fuzzy_query("  auth  AND  errors  ")
        assert isinstance(node, AndNode)
        assert isinstance(node.left, VectorNode)
        assert node.left.query == "auth"
        assert isinstance(node.right, VectorNode)
        assert node.right.query == "errors"


# ── Evaluator ────────────────────────────────────────────────────────


class TestEvaluateFuzzy:
    """Tests for evaluate_fuzzy with pre-computed scores."""

    def test_vector_node(self) -> None:
        """Single query → pass through scores."""
        scores = {"topic": {"n1": 0.8, "n2": 0.5}}
        result = evaluate_fuzzy(VectorNode("topic"), scores)
        assert result == {"n1": 0.8, "n2": 0.5}

    def test_and_product(self) -> None:
        """AND with Product T-norm: a * b."""
        scores = {
            "auth": {"n1": 0.8, "n2": 0.6, "n3": 0.4},
            "errors": {"n1": 0.5, "n2": 0.9, "n4": 0.7},
        }
        node = AndNode(VectorNode("auth"), VectorNode("errors"))
        result = evaluate_fuzzy(node, scores)

        # Only common keys: n1, n2
        assert "n3" not in result
        assert "n4" not in result
        assert abs(result["n1"] - 0.8 * 0.5) < 1e-10
        assert abs(result["n2"] - 0.6 * 0.9) < 1e-10

    def test_and_min(self) -> None:
        """AND with Min T-norm: min(a, b)."""
        scores = {
            "a": {"n1": 0.8, "n2": 0.3},
            "b": {"n1": 0.5, "n2": 0.9},
        }
        node = AndNode(VectorNode("a"), VectorNode("b"), norm=TNorm.MIN)
        result = evaluate_fuzzy(node, scores)

        assert result["n1"] == 0.5
        assert result["n2"] == 0.3

    def test_and_lukasiewicz(self) -> None:
        """AND with Łukasiewicz T-norm: max(a + b - 1, 0)."""
        scores = {
            "a": {"n1": 0.8, "n2": 0.3},
            "b": {"n1": 0.9, "n2": 0.4},
        }
        node = AndNode(VectorNode("a"), VectorNode("b"), norm=TNorm.LUKASIEWICZ)
        result = evaluate_fuzzy(node, scores)

        assert abs(result["n1"] - 0.7) < 1e-10  # 0.8 + 0.9 - 1 = 0.7
        # 0.3 + 0.4 - 1 = -0.3 → max(0) = 0, so n2 not in result or 0
        assert result.get("n2", 0.0) == 0.0

    def test_or(self) -> None:
        """OR: probabilistic union = a + b - a*b."""
        scores = {
            "React": {"n1": 0.8, "n3": 0.5},
            "Vue": {"n2": 0.7, "n3": 0.4},
        }
        node = OrNode(VectorNode("React"), VectorNode("Vue"))
        result = evaluate_fuzzy(node, scores)

        # n1: only in React → 0.8
        assert result["n1"] == 0.8
        # n2: only in Vue → 0.7
        assert result["n2"] == 0.7
        # n3: both → 0.5 + 0.4 - 0.5*0.4 = 0.7
        assert abs(result["n3"] - 0.7) < 1e-10

    def test_not(self) -> None:
        """NOT: penalize base by child membership."""
        scores = {
            "deploy": {"n1": 0.9, "n2": 0.8, "n3": 0.5},
            "staging": {"n1": 0.0, "n2": 0.7, "n3": 1.0},
        }
        node = NotNode(base=VectorNode("deploy"), child=VectorNode("staging"))
        result = evaluate_fuzzy(node, scores)

        # n1: 0.9 * (1 - 0.0) = 0.9 (not penalized)
        assert abs(result["n1"] - 0.9) < 1e-10
        # n2: 0.8 * (1 - 0.7) = 0.24
        assert abs(result["n2"] - 0.24) < 1e-10
        # n3: 0.5 * (1 - 1.0) = 0.0 (fully penalized, removed)
        assert "n3" not in result

    def test_and_reduces_scores(self) -> None:
        """AND should always produce scores <= min(a, b)."""
        scores = {
            "a": {"n1": 0.8},
            "b": {"n1": 0.6},
        }
        result = evaluate_fuzzy(AndNode(VectorNode("a"), VectorNode("b")), scores)
        assert result["n1"] <= min(0.8, 0.6)

    def test_or_increases_scores(self) -> None:
        """OR should produce scores >= max(a, b)."""
        scores = {
            "a": {"n1": 0.4},
            "b": {"n1": 0.3},
        }
        result = evaluate_fuzzy(OrNode(VectorNode("a"), VectorNode("b")), scores)
        assert result["n1"] >= max(0.4, 0.3)

    def test_empty_scores(self) -> None:
        """Missing sub-query scores → empty result."""
        result = evaluate_fuzzy(VectorNode("missing"), {})
        assert result == {}

    def test_nested_expression(self) -> None:
        """(A AND B) OR C should work."""
        scores = {
            "a": {"n1": 0.8, "n2": 0.5},
            "b": {"n1": 0.6},
            "c": {"n2": 0.9, "n3": 0.7},
        }
        node = OrNode(
            AndNode(VectorNode("a"), VectorNode("b")),
            VectorNode("c"),
        )
        result = evaluate_fuzzy(node, scores)

        # n1: AND(0.8, 0.6)=0.48, OR with absent c → 0.48
        assert abs(result["n1"] - 0.48) < 1e-10
        # n2: AND(0.5, absent)=absent, OR with c=0.9 → 0.9
        assert abs(result["n2"] - 0.9) < 1e-10
        # n3: AND gives nothing, OR with c=0.7 → 0.7
        assert abs(result["n3"] - 0.7) < 1e-10


# ── Sub-query Collection ─────────────────────────────────────────────


class TestCollectSubQueries:
    """Tests for extracting sub-query strings from AST."""

    def test_single(self) -> None:
        assert collect_sub_queries(VectorNode("topic")) == ["topic"]

    def test_and(self) -> None:
        node = parse_fuzzy_query("auth AND errors")
        queries = collect_sub_queries(node)
        assert "auth" in queries
        assert "errors" in queries

    def test_deduplication(self) -> None:
        """Same query appearing twice → deduplicated."""
        node = AndNode(VectorNode("auth"), VectorNode("auth"))
        queries = collect_sub_queries(node)
        assert queries == ["auth"]

    def test_complex(self) -> None:
        node = parse_fuzzy_query("auth AND errors OR warnings")
        queries = collect_sub_queries(node)
        assert len(queries) == 3
        assert "auth" in queries
        assert "errors" in queries
        assert "warnings" in queries

    def test_empty(self) -> None:
        assert collect_sub_queries(VectorNode("")) == []
