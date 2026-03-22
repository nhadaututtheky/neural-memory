"""Fuzzy compositional recall — soft boolean logic for memory queries.

Supports AND, OR, NOT operators to compose multiple recall sub-queries
with T-norm based membership scores. Each sub-query runs independently,
then results are combined using fuzzy logic:

    "auth decisions AND error patterns"  → intersection (Product T-norm)
    "React OR Vue"                       → union (probabilistic OR)
    "deployment NOT staging"             → complement

Inspired by HyperspaceDB's FuzzyQuery AST with T-norms.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)

# Operators must be uppercase and surrounded by whitespace + at least 2 chars on each side
_FUZZY_PATTERN = re.compile(
    r"\s+(AND|OR|NOT)\s+",
    re.IGNORECASE,
)


class TNorm(StrEnum):
    """T-norm for combining fuzzy membership scores."""

    PRODUCT = "product"  # a * b (default, most natural)
    MIN = "min"  # min(a, b) (Gödel)
    LUKASIEWICZ = "lukasiewicz"  # max(a + b - 1, 0)


# ── AST Nodes ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VectorNode:
    """Leaf node: a single query string."""

    query: str


@dataclass(frozen=True)
class AndNode:
    """Intersection: memories must score high on both sub-queries."""

    left: FuzzyNode
    right: FuzzyNode
    norm: TNorm = TNorm.PRODUCT


@dataclass(frozen=True)
class OrNode:
    """Union: memories scoring high on either sub-query."""

    left: FuzzyNode
    right: FuzzyNode


@dataclass(frozen=True)
class NotNode:
    """Complement: penalize memories matching the sub-query."""

    child: FuzzyNode
    base: FuzzyNode


FuzzyNode = VectorNode | AndNode | OrNode | NotNode


# ── Parser ───────────────────────────────────────────────────────────


def is_fuzzy_query(query: str) -> bool:
    """Check if a query contains fuzzy operators (AND/OR/NOT).

    Only matches uppercase operators surrounded by whitespace with
    at least 2 characters on each side.
    """
    return bool(_FUZZY_PATTERN.search(query))


def parse_fuzzy_query(query: str) -> FuzzyNode:
    """Parse a query string into a FuzzyNode AST.

    Supports: "A AND B", "A OR B", "A NOT B"
    Operators must be uppercase. Left-to-right precedence (no grouping).

    Args:
        query: Query string, potentially with AND/OR/NOT operators.

    Returns:
        FuzzyNode AST. If no operators, returns a single VectorNode.
    """
    query = query.strip()

    if not query:
        return VectorNode(query="")

    # Split on operators, keeping the operator tokens
    tokens = _FUZZY_PATTERN.split(query)

    if len(tokens) <= 1:
        return VectorNode(query=query)

    # Build AST left-to-right
    node: FuzzyNode = VectorNode(query=tokens[0].strip())

    i = 1
    while i < len(tokens) - 1:
        op = tokens[i].upper()
        right_query = tokens[i + 1].strip()

        if not right_query:
            i += 2
            continue

        right = VectorNode(query=right_query)

        if op == "AND":
            node = AndNode(left=node, right=right)
        elif op == "OR":
            node = OrNode(left=node, right=right)
        elif op == "NOT":
            node = NotNode(base=node, child=right)

        i += 2

    return node


# ── Evaluator ────────────────────────────────────────────────────────


def evaluate_fuzzy(
    node: FuzzyNode,
    score_fn: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Evaluate a FuzzyNode AST against pre-computed sub-query scores.

    Args:
        node: The fuzzy query AST.
        score_fn: Map of {sub_query: {neuron_id: score}} where scores
                  are in [0, 1] (membership degree).

    Returns:
        Combined {neuron_id: score} after applying fuzzy operations.
    """
    if isinstance(node, VectorNode):
        return dict(score_fn.get(node.query, {}))

    if isinstance(node, AndNode):
        left_scores = evaluate_fuzzy(node.left, score_fn)
        right_scores = evaluate_fuzzy(node.right, score_fn)
        return _apply_and(left_scores, right_scores, node.norm)

    if isinstance(node, OrNode):
        left_scores = evaluate_fuzzy(node.left, score_fn)
        right_scores = evaluate_fuzzy(node.right, score_fn)
        return _apply_or(left_scores, right_scores)

    if isinstance(node, NotNode):
        base_scores = evaluate_fuzzy(node.base, score_fn)
        child_scores = evaluate_fuzzy(node.child, score_fn)
        return _apply_not(base_scores, child_scores)

    return {}


def _apply_and(
    left: dict[str, float],
    right: dict[str, float],
    norm: TNorm,
) -> dict[str, float]:
    """Fuzzy AND: only neurons present in both, combined by T-norm."""
    common_keys = set(left.keys()) & set(right.keys())
    result: dict[str, float] = {}

    for key in common_keys:
        a, b = left[key], right[key]
        if norm == TNorm.PRODUCT:
            result[key] = a * b
        elif norm == TNorm.MIN:
            result[key] = min(a, b)
        elif norm == TNorm.LUKASIEWICZ:
            result[key] = max(a + b - 1.0, 0.0)

    return result


def _apply_or(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    """Fuzzy OR: probabilistic union = a + b - a*b."""
    all_keys = set(left.keys()) | set(right.keys())
    result: dict[str, float] = {}

    for key in all_keys:
        a = left.get(key, 0.0)
        b = right.get(key, 0.0)
        result[key] = a + b - a * b  # Probabilistic OR

    return result


def _apply_not(
    base: dict[str, float],
    child: dict[str, float],
) -> dict[str, float]:
    """Fuzzy NOT: penalize base scores by child membership."""
    result: dict[str, float] = {}

    for key, base_score in base.items():
        child_score = child.get(key, 0.0)
        penalized = base_score * (1.0 - child_score)
        if penalized > 0.0:
            result[key] = penalized

    return result


def collect_sub_queries(node: FuzzyNode) -> list[str]:
    """Extract all unique sub-query strings from a FuzzyNode AST.

    Args:
        node: The fuzzy query AST.

    Returns:
        List of unique query strings (leaf VectorNode values).
    """
    queries: list[str] = []
    _collect(node, queries)
    return list(dict.fromkeys(queries))  # Dedupe preserving order


def _collect(node: FuzzyNode, acc: list[str]) -> None:
    if isinstance(node, VectorNode):
        if node.query:
            acc.append(node.query)
    elif isinstance(node, (AndNode, OrNode)):
        _collect(node.left, acc)
        _collect(node.right, acc)
    elif isinstance(node, NotNode):
        _collect(node.base, acc)
        _collect(node.child, acc)
