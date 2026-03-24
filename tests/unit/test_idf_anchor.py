"""Tests for Phase C — IDF-Weighted Anchor Selection.

Covers:
- IDF score computation
- Anchor limit mapping
- Keyword limits with mixed rare/common terms
- Edge cases: cold start, empty input, unknown keywords
"""

from __future__ import annotations

from neural_memory.engine.idf_anchor import (
    compute_anchor_limit,
    compute_idf_scores,
    compute_keyword_limits,
)

# ── IDF score computation ────────────────────────────────────────


class TestComputeIdfScores:
    def test_rare_term_high_score(self) -> None:
        scores = compute_idf_scores({"rare": 1}, total_docs=1000)
        assert scores["rare"] > 0.8

    def test_common_term_low_score(self) -> None:
        scores = compute_idf_scores({"common": 500}, total_docs=1000)
        assert scores["common"] < 0.3

    def test_very_common_near_zero(self) -> None:
        scores = compute_idf_scores({"ubiquitous": 999}, total_docs=1000)
        assert scores["ubiquitous"] < 0.1

    def test_zero_df_max_score(self) -> None:
        scores = compute_idf_scores({"unknown": 0}, total_docs=1000)
        assert scores["unknown"] == 1.0

    def test_cold_start_all_max(self) -> None:
        scores = compute_idf_scores({"a": 0, "b": 0}, total_docs=0)
        assert all(s == 1.0 for s in scores.values())

    def test_scores_normalized_zero_one(self) -> None:
        scores = compute_idf_scores({"rare": 1, "mid": 50, "common": 500}, total_docs=1000)
        for score in scores.values():
            assert 0.0 <= score <= 1.0


# ── Anchor limit mapping ────────────────────────────────────────


class TestComputeAnchorLimit:
    def test_high_idf_max_limit(self) -> None:
        # ceil(1.0 * (5-1+1)) = 5, clamped to max_limit=5
        assert compute_anchor_limit(1.0) == 5

    def test_max_limit_reachable(self) -> None:
        assert compute_anchor_limit(1.0, max_limit=5) == 5

    def test_low_idf_min_limit(self) -> None:
        # ceil(0.1 * 5) = 1
        assert compute_anchor_limit(0.1) == 1

    def test_mid_idf(self) -> None:
        # ceil(0.5 * 5) = 3
        assert compute_anchor_limit(0.5) == 3

    def test_custom_min_max(self) -> None:
        # ceil(0.05 * (10-2+1)) = ceil(0.45) = 1, clamped to min_limit=2
        assert compute_anchor_limit(0.05, min_limit=2, max_limit=10) == 2


# ── Keyword limits ───────────────────────────────────────────────


class TestComputeKeywordLimits:
    def test_mixed_keywords(self) -> None:
        limits = compute_keyword_limits(
            keywords=["rare", "common"],
            keyword_df={"rare": 1, "common": 500},
            total_docs=1000,
        )
        assert limits["rare"] > limits["common"]

    def test_unknown_keyword_gets_max(self) -> None:
        limits = compute_keyword_limits(
            keywords=["never_seen"],
            keyword_df={},
            total_docs=1000,
            max_limit=5,
        )
        assert limits["never_seen"] == 5

    def test_empty_keywords(self) -> None:
        limits = compute_keyword_limits(
            keywords=[],
            keyword_df={},
            total_docs=100,
        )
        assert limits == {}

    def test_cold_start_all_max(self) -> None:
        limits = compute_keyword_limits(
            keywords=["a", "b", "c"],
            keyword_df={"a": 0, "b": 0, "c": 0},
            total_docs=0,
            max_limit=5,
        )
        assert all(v == 5 for v in limits.values())

    def test_case_insensitive_df_lookup(self) -> None:
        """Keywords are lowercased before DF lookup."""
        limits = compute_keyword_limits(
            keywords=["API"],
            keyword_df={"api": 50},
            total_docs=1000,
        )
        assert "API" in limits
        assert limits["API"] >= 1
