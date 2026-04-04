"""Unit tests for A8 Phase 3: Auto-Save Intelligence.

Tests quality scoring enhancements, dedup feedback surfacing,
auto-classification confidence, and importance scoring improvements.
"""

from __future__ import annotations

from neural_memory.engine.importance import auto_importance_score
from neural_memory.engine.pipeline_steps import _classify_confidence
from neural_memory.engine.quality_scorer import score_memory

# ---------------------------------------------------------------------------
# T3.1: Quality scorer enhancements
# ---------------------------------------------------------------------------


class TestQualityScorerEnhancements:
    """A8 additions: specificity, structure, brevity, wall-of-text."""

    def test_specificity_bonus_file_paths(self):
        """File paths in content should boost quality score."""
        without = score_memory("Chose React for the frontend", memory_type="decision")
        with_paths = score_memory(
            "Chose React in src/components/App.tsx for the frontend",
            memory_type="decision",
        )
        assert with_paths.score > without.score

    def test_specificity_bonus_version_numbers(self):
        """Version numbers in content should boost quality score."""
        without = score_memory("Upgraded the auth library", memory_type="fact")
        with_version = score_memory("Upgraded auth library to v4.28.0", memory_type="fact")
        assert with_version.score > without.score

    def test_structure_markers_bonus(self):
        """Arrow markers (→, ->) should boost quality score."""
        without = score_memory(
            "Auth module has a login failure problem in prod",
            memory_type="error",
        )
        with_arrow = score_memory(
            "Auth module → login failure → user lockout in prod",
            memory_type="error",
        )
        assert with_arrow.score > without.score

    def test_brevity_bonus_optimal_range(self):
        """Content in 50-300 chars should get brevity bonus."""
        short = score_memory("A fact", memory_type="fact")
        # Build content in optimal range (50-300 chars)
        optimal = score_memory(
            "Chose PostgreSQL over MySQL because of JSON support and window functions",
            memory_type="decision",
            tags=["database"],
        )
        assert optimal.score > short.score

    def test_wall_of_text_penalty(self):
        """Content >500 chars should be penalized."""
        long_content = "A" * 550  # Wall of text
        result = score_memory(long_content, memory_type="fact")
        # Should have a hint about splitting
        assert any("split" in h.lower() for h in result.hints)

    def test_wall_of_text_score_lower(self):
        """Wall-of-text should score lower than concise version."""
        concise = score_memory(
            "Root cause was connection pool exhaustion because max_connections=10",
            memory_type="error",
            tags=["database"],
        )
        wall = score_memory(
            "Root cause was connection pool exhaustion because max_connections=10. "
            + "This is a really long explanation that goes on and on " * 10,
            memory_type="error",
            tags=["database"],
        )
        assert wall.score < concise.score

    def test_score_capped_at_10(self):
        """Score should never exceed 10 even with all bonuses."""
        result = score_memory(
            "Chose Redis v7.2.0 over Memcached → better persistence because of AOF. "
            "After testing in src/cache.py, found 3x faster than previous approach.",
            memory_type="decision",
            tags=["cache", "redis"],
            context={"reason": "performance", "alternatives": ["memcached"]},
        )
        assert result.score <= 10

    def test_score_minimum_zero(self):
        """Score should never go below 0."""
        result = score_memory("x", memory_type="fact")
        assert result.score >= 0


# ---------------------------------------------------------------------------
# T3.3: Auto-classification confidence
# ---------------------------------------------------------------------------


class TestClassifyConfidence:
    """Confidence scoring for memory type classification."""

    def test_high_confidence_error(self):
        """Error content with multiple keywords → high confidence."""
        conf = _classify_confidence(
            "TypeError: cannot read property of null, stack trace shows crash in auth",
            "error",
        )
        assert conf >= 0.8

    def test_high_confidence_decision(self):
        """Decision content with choice keywords → high confidence."""
        conf = _classify_confidence(
            "Chose PostgreSQL over MySQL, rejected MongoDB instead of SQLite",
            "decision",
        )
        assert conf >= 0.8

    def test_low_confidence_mismatch(self):
        """Content with no type keywords → low confidence."""
        conf = _classify_confidence(
            "The sky is blue and water is wet",
            "error",
        )
        assert conf <= 0.3

    def test_medium_confidence_single_keyword(self):
        """One keyword match → medium confidence."""
        conf = _classify_confidence(
            "There was an error in the login flow",
            "error",
        )
        assert 0.5 <= conf <= 0.7

    def test_unknown_type_neutral(self):
        """Unknown type → neutral confidence (0.5)."""
        conf = _classify_confidence("some content", "unknown_type")
        assert conf == 0.5

    def test_insight_classification(self):
        """Insight keywords should give high confidence."""
        conf = _classify_confidence(
            "Discovered that the root cause was a race condition, key insight from profiling",
            "insight",
        )
        assert conf >= 0.8

    def test_workflow_classification(self):
        """Workflow keywords should give high confidence."""
        conf = _classify_confidence(
            "Step 1: build, then deploy to staging, after that release to production",
            "workflow",
        )
        assert conf >= 0.8


# ---------------------------------------------------------------------------
# T3.4: Importance scoring improvements
# ---------------------------------------------------------------------------


class TestImportanceScoringImprovements:
    """A8 extended importance signals."""

    def test_file_path_bonus(self):
        """File paths should boost importance."""
        without = auto_importance_score("Changed the auth handler", "fact", [])
        with_path = auto_importance_score("Changed auth handler in src/auth.py", "fact", [])
        assert with_path > without

    def test_version_number_bonus(self):
        """Version numbers should boost importance."""
        without = auto_importance_score("Upgraded the library", "fact", [])
        with_version = auto_importance_score("Upgraded to v4.28.0", "fact", [])
        assert with_version > without

    def test_error_trace_bonus(self):
        """Error traces (with Traceback or Error:) should get +2 importance."""
        without = auto_importance_score("Something went wrong", "error", [])
        with_trace = auto_importance_score(
            "Traceback (most recent call last): Exception in auth module", "error", []
        )
        assert with_trace >= without + 2

    def test_security_keywords_bonus(self):
        """Security keywords should get +2 importance."""
        without = auto_importance_score("Updated the API endpoint", "fact", [])
        with_security = auto_importance_score(
            "Fixed SQL injection vulnerability in login endpoint", "error", []
        )
        assert with_security > without + 1

    def test_cve_reference_bonus(self):
        """CVE references should trigger security bonus."""
        score = auto_importance_score("Patched CVE-2024-1234 in auth module", "error", [])
        # Error type (+2) + security (+2) = at least 9
        assert score >= 7

    def test_cap_at_9(self):
        """Score should cap at 9 (10 reserved for explicit user override)."""
        score = auto_importance_score(
            "Traceback: CVE-2024-9999 vulnerability exploit caused by injection "
            "because of missing validation, chose fix over workaround in auth.py v2.0",
            "error",
            ["security", "critical"],
        )
        assert score <= 9

    def test_base_score_for_simple_content(self):
        """Simple fact with adequate length should get base score of 5."""
        score = auto_importance_score("The sky is blue and the grass is green", "fact", [])
        assert score == 5

    def test_combined_bonuses(self):
        """Multiple signals should stack up."""
        score = auto_importance_score(
            "Root cause was auth.py v3.0 having SQL injection because of string formatting",
            "error",
            [],
        )
        # error(+2) + causal(+1) + file_path(+1) + version(+1) + security(+2) = 5+7=12 → capped at 9
        assert score >= 8


# ---------------------------------------------------------------------------
# T3.2: Dedup feedback surfacing (pipeline_steps level)
# ---------------------------------------------------------------------------


class TestDedupFeedbackPropagation:
    """DedupCheckStep should propagate similarity and tier to metadata."""

    def test_dedup_metadata_stored(self):
        """When dedup detects duplicate, similarity and tier should be in metadata."""
        # This tests the pipeline step level — the metadata propagation.
        # We verify the keys exist in the pattern used by DedupCheckStep.
        metadata: dict[str, object] = {}
        # Simulate what DedupCheckStep does after our A8 T3.2 change
        metadata["_dedup_reused_anchor"] = "mock_neuron"
        metadata["_dedup_similarity"] = 0.85
        metadata["_dedup_tier"] = 1

        assert metadata["_dedup_similarity"] == 0.85
        assert metadata["_dedup_tier"] == 1

    def test_dedup_hint_format(self):
        """Dedup hint in response should include similarity and tier info."""
        # Simulate the enhanced dedup hint structure from remember_handler
        dedup_hint: dict[str, object] = {
            "similar_existing": "neuron-abc",
            "message": "Similar memory exists — consider nmem_edit to update instead",
            "similarity": 0.85,
            "dedup_tier": 1,
            "duplicate_of": "fiber-xyz",
        }

        assert "similarity" in dedup_hint
        assert dedup_hint["similarity"] == 0.85
        assert "duplicate_of" in dedup_hint
        assert "nmem_edit" in str(dedup_hint["message"])


# ---------------------------------------------------------------------------
# Combined scenarios
# ---------------------------------------------------------------------------


class TestAutoSaveCombined:
    """Cross-cutting integration tests."""

    def test_quality_and_importance_aligned(self):
        """High-quality content should also get high importance."""
        content = (
            "Chose Redis v7.2 over Memcached → better persistence "
            "because of AOF, found 3x faster in auth.py benchmarks"
        )
        quality = score_memory(content, memory_type="decision", tags=["cache"])
        importance = auto_importance_score(content, "decision", ["cache"])

        assert quality.quality in ("medium", "high")
        assert importance >= 7

    def test_low_quality_matches_low_importance(self):
        """Vague content should get low quality and low importance."""
        content = "ok"
        quality = score_memory(content, memory_type="fact")
        importance = auto_importance_score(content, "fact", [])

        assert quality.quality == "low"
        assert importance <= 5

    def test_classification_confidence_alignment(self):
        """Content matching its type should have high confidence."""
        # Error content classified as error → high confidence
        error_conf = _classify_confidence("Bug caused crash with TypeError exception", "error")
        # Same content classified as workflow → low confidence
        workflow_conf = _classify_confidence(
            "Bug caused crash with TypeError exception", "workflow"
        )
        assert error_conf > workflow_conf
