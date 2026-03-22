"""Tests for semantic drift detection engine and storage."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.engine.clustering import UnionFind
from neural_memory.engine.drift_detection import (
    JACCARD_MERGE_THRESHOLD,
    MIN_COOCCURRENCE_COUNT,
    DriftReport,
    TagCluster,
    compute_jaccard,
    detect_activation_drift,
    detect_clusters,
    detect_temporal_drift,
    run_drift_detection,
    wasserstein_1,
)
from neural_memory.utils.timeutils import utcnow

# ── compute_jaccard ─────────────────────────────────────────────────


class TestComputeJaccard:
    """Tests for Jaccard similarity computation."""

    def test_perfect_overlap(self) -> None:
        # 10 fibers each, co-occur in 10 → J = 10/(10+10-10) = 1.0
        result = compute_jaccard("a", "b", {"a": 10, "b": 10}, 10)
        assert result == 1.0

    def test_no_overlap(self) -> None:
        result = compute_jaccard("a", "b", {"a": 10, "b": 10}, 0)
        assert result == 0.0

    def test_partial_overlap(self) -> None:
        # 10 fibers each, co-occur in 5 → J = 5/(10+10-5) = 5/15 ≈ 0.333
        result = compute_jaccard("a", "b", {"a": 10, "b": 10}, 5)
        assert abs(result - 1 / 3) < 0.01

    def test_missing_tag_a(self) -> None:
        result = compute_jaccard("x", "b", {"b": 10}, 5)
        assert result == 0.0

    def test_missing_tag_b(self) -> None:
        result = compute_jaccard("a", "x", {"a": 10}, 5)
        assert result == 0.0

    def test_asymmetric_counts(self) -> None:
        # a=20, b=5, co-occur=5 → J = 5/(20+5-5) = 5/20 = 0.25
        result = compute_jaccard("a", "b", {"a": 20, "b": 5}, 5)
        assert result == 0.25

    def test_zero_union(self) -> None:
        result = compute_jaccard("a", "b", {"a": 0, "b": 0}, 0)
        assert result == 0.0

    def test_high_jaccard(self) -> None:
        # a=10, b=10, co-occur=8 → J = 8/(10+10-8) = 8/12 ≈ 0.667
        result = compute_jaccard("a", "b", {"a": 10, "b": 10}, 8)
        assert abs(result - 8 / 12) < 0.001


# ── detect_clusters ─────────────────────────────────────────────────


class TestDetectClusters:
    """Tests for Union-Find cluster detection."""

    def test_empty_cooccurrences(self) -> None:
        assert detect_clusters([], {}) == []

    def test_single_pair_below_threshold(self) -> None:
        # co-occurrence count below MIN_COOCCURRENCE_COUNT
        cooccurrences = [("a", "b", 1)]
        counts = {"a": 10, "b": 10}
        assert detect_clusters(cooccurrences, counts) == []

    def test_single_pair_low_jaccard(self) -> None:
        # count >= MIN but Jaccard below REVIEW threshold
        cooccurrences = [("a", "b", MIN_COOCCURRENCE_COUNT)]
        counts = {"a": 100, "b": 100}  # J = 3/197 ≈ 0.015 — way below 0.3
        assert detect_clusters(cooccurrences, counts) == []

    def test_merge_suggestion(self) -> None:
        # High Jaccard (>= 0.7) → merge suggestion
        cooccurrences = [("react", "reactjs", 10)]
        counts = {"react": 12, "reactjs": 11}
        # J = 10/(12+11-10) = 10/13 ≈ 0.769
        reports = detect_clusters(cooccurrences, counts)
        assert len(reports) == 1
        assert reports[0].suggestion == "merge"
        assert reports[0].cluster.confidence >= JACCARD_MERGE_THRESHOLD

    def test_alias_suggestion(self) -> None:
        # Medium Jaccard (0.4-0.7) → alias
        # J = 6/(8+8-6) = 6/10 = 0.6
        cooccurrences = [("auth", "authentication", 6)]
        counts = {"auth": 8, "authentication": 8}
        reports = detect_clusters(cooccurrences, counts)
        assert len(reports) == 1
        assert reports[0].suggestion == "alias"

    def test_canonical_is_most_used(self) -> None:
        cooccurrences = [("js", "javascript", 10)]
        counts = {"js": 5, "javascript": 20}
        reports = detect_clusters(cooccurrences, counts)
        assert len(reports) == 1
        assert reports[0].cluster.canonical == "javascript"

    def test_multiple_clusters(self) -> None:
        cooccurrences = [
            ("react", "reactjs", 10),
            ("vue", "vuejs", 8),
        ]
        counts = {"react": 12, "reactjs": 11, "vue": 10, "vuejs": 9}
        reports = detect_clusters(cooccurrences, counts)
        assert len(reports) == 2

    def test_transitive_union(self) -> None:
        # a-b and b-c should be in same cluster if both above alias threshold
        cooccurrences = [
            ("a", "b", 8),
            ("b", "c", 8),
        ]
        counts = {"a": 10, "b": 10, "c": 10}
        # J(a,b) = 8/12 ≈ 0.667, J(b,c) = 8/12 ≈ 0.667 — both above alias threshold
        reports = detect_clusters(cooccurrences, counts)
        assert len(reports) == 1
        assert len(reports[0].cluster.members) == 3

    def test_tag_below_min_fibers_excluded(self) -> None:
        cooccurrences = [("rare", "common", 5)]
        counts = {"rare": 1, "common": 10}  # rare < MIN_TAG_FIBERS
        reports = detect_clusters(cooccurrences, counts)
        assert len(reports) == 0

    def test_cluster_id_is_stable(self) -> None:
        cooccurrences = [("x", "y", 10)]
        counts = {"x": 12, "y": 12}
        reports1 = detect_clusters(cooccurrences, counts)
        reports2 = detect_clusters(cooccurrences, counts)
        assert reports1[0].cluster_id == reports2[0].cluster_id

    def test_cluster_sorted_by_confidence(self) -> None:
        cooccurrences = [
            ("low_a", "low_b", 5),
            ("high_a", "high_b", 10),
        ]
        counts = {"low_a": 10, "low_b": 10, "high_a": 11, "high_b": 11}
        reports = detect_clusters(cooccurrences, counts)
        if len(reports) >= 2:
            assert reports[0].cluster.confidence >= reports[1].cluster.confidence


# ── TagCluster / DriftReport ────────────────────────────────────────


class TestDataModels:
    """Tests for frozen data models."""

    def test_tag_cluster_frozen(self) -> None:
        tc = TagCluster(
            canonical="react",
            members=frozenset({"react", "reactjs"}),
            confidence=0.8,
        )
        with pytest.raises(AttributeError):
            tc.canonical = "vue"  # type: ignore[misc]

    def test_drift_report_frozen(self) -> None:
        tc = TagCluster(canonical="a", members=frozenset({"a", "b"}), confidence=0.5)
        dr = DriftReport(cluster=tc, suggestion="merge", cluster_id="abc123")
        assert dr.suggestion == "merge"
        with pytest.raises(AttributeError):
            dr.suggestion = "alias"  # type: ignore[misc]

    def test_tag_cluster_evidence(self) -> None:
        tc = TagCluster(
            canonical="react",
            members=frozenset({"react", "reactjs"}),
            confidence=0.8,
            evidence="Tags co-occur frequently",
        )
        assert "co-occur" in tc.evidence


# ── detect_temporal_drift ───────────────────────────────────────────


class TestTemporalDrift:
    """Tests for cross-session terminology drift detection."""

    @pytest.mark.asyncio
    async def test_insufficient_history(self) -> None:
        storage = MagicMock()
        storage.get_session_summaries = AsyncMock(
            return_value=[
                {"topics": ["a"]},
                {"topics": ["b"]},
            ]
        )
        result = await detect_temporal_drift(storage)
        assert result == []

    @pytest.mark.asyncio
    async def test_detects_terminology_shift(self) -> None:
        # Recent sessions use "kubernetes", early sessions used "k8s"
        summaries = [
            # Recent (first in DESC order)
            {"topics": ["kubernetes", "docker"]},
            {"topics": ["kubernetes", "ci"]},
            # Early
            {"topics": ["k8s", "docker"]},
            {"topics": ["k8s", "ci"]},
        ]
        storage = MagicMock()
        storage.get_session_summaries = AsyncMock(return_value=summaries)
        result = await detect_temporal_drift(storage)
        # Should detect k8s → kubernetes shift
        assert len(result) >= 1
        terms = {(d["old_term"], d["new_term"]) for d in result}
        assert ("k8s", "kubernetes") in terms

    @pytest.mark.asyncio
    async def test_no_drift_when_terms_persist(self) -> None:
        summaries = [
            {"topics": ["react", "typescript"]},
            {"topics": ["react", "typescript"]},
            {"topics": ["react", "typescript"]},
            {"topics": ["react", "typescript"]},
        ]
        storage = MagicMock()
        storage.get_session_summaries = AsyncMock(return_value=summaries)
        result = await detect_temporal_drift(storage)
        assert result == []

    @pytest.mark.asyncio
    async def test_handles_storage_error(self) -> None:
        storage = MagicMock()
        storage.get_session_summaries = AsyncMock(side_effect=RuntimeError("boom"))
        result = await detect_temporal_drift(storage)
        assert result == []


# ── run_drift_detection ─────────────────────────────────────────────


class TestRunDriftDetection:
    """Tests for the orchestrator function."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self) -> None:
        storage = MagicMock()
        storage.get_tag_cooccurrence = AsyncMock(
            return_value=[
                ("react", "reactjs", 10),
            ]
        )
        storage.get_tag_fiber_counts = AsyncMock(
            return_value={
                "react": 12,
                "reactjs": 11,
            }
        )
        storage.save_drift_cluster = AsyncMock()
        storage.get_session_summaries = AsyncMock(return_value=[])

        result = await run_drift_detection(storage)
        assert "clusters" in result
        assert "temporal_drifts" in result
        assert "summary" in result
        assert result["summary"]["total_clusters"] >= 1

    @pytest.mark.asyncio
    async def test_persists_clusters(self) -> None:
        storage = MagicMock()
        storage.get_tag_cooccurrence = AsyncMock(
            return_value=[
                ("a", "b", 10),
            ]
        )
        storage.get_tag_fiber_counts = AsyncMock(return_value={"a": 12, "b": 12})
        storage.save_drift_cluster = AsyncMock()
        storage.get_session_summaries = AsyncMock(return_value=[])

        await run_drift_detection(storage)
        storage.save_drift_cluster.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_missing_storage_methods(self) -> None:
        storage = MagicMock()
        storage.get_tag_cooccurrence = AsyncMock(side_effect=AttributeError)
        storage.get_tag_fiber_counts = AsyncMock(side_effect=AttributeError)
        storage.get_session_summaries = AsyncMock(side_effect=AttributeError)

        result = await run_drift_detection(storage)
        assert result["summary"]["total_clusters"] == 0

    @pytest.mark.asyncio
    async def test_empty_brain(self) -> None:
        storage = MagicMock()
        storage.get_tag_cooccurrence = AsyncMock(return_value=[])
        storage.get_tag_fiber_counts = AsyncMock(return_value={})
        storage.get_session_summaries = AsyncMock(return_value=[])

        result = await run_drift_detection(storage)
        assert result["clusters"] == []
        assert result["summary"]["total_clusters"] == 0


# ── SQLiteDriftMixin ────────────────────────────────────────────────


class TestSQLiteDriftMixin:
    """Tests for the drift storage mixin via SQLiteStorage."""

    @pytest.fixture
    async def storage(self, tmp_path):
        from neural_memory.storage.sqlite_store import SQLiteStorage

        db_path = tmp_path / "test_drift.db"
        store = SQLiteStorage(db_path)
        await store.initialize()
        store.set_brain("test-brain")
        # Create brain row with all required columns
        conn = store._ensure_conn()
        now = utcnow().isoformat()
        await conn.execute(
            "INSERT OR IGNORE INTO brains (id, name, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("test-brain", "Test Brain", "{}", now, now),
        )
        await conn.commit()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_record_and_get_cooccurrence(self, storage) -> None:
        await storage.record_tag_cooccurrence({"react", "typescript", "frontend"})
        pairs = await storage.get_tag_cooccurrence(min_count=1)
        assert len(pairs) == 3  # 3 choose 2 = 3 pairs

    @pytest.mark.asyncio
    async def test_cooccurrence_count_increments(self, storage) -> None:
        await storage.record_tag_cooccurrence({"a", "b"})
        await storage.record_tag_cooccurrence({"a", "b"})
        pairs = await storage.get_tag_cooccurrence(min_count=1)
        assert len(pairs) == 1
        assert pairs[0][2] == 2  # count = 2

    @pytest.mark.asyncio
    async def test_cooccurrence_canonical_order(self, storage) -> None:
        await storage.record_tag_cooccurrence({"z", "a"})
        pairs = await storage.get_tag_cooccurrence(min_count=1)
        assert pairs[0][0] == "a"  # tag_a < tag_b
        assert pairs[0][1] == "z"

    @pytest.mark.asyncio
    async def test_single_tag_no_cooccurrence(self, storage) -> None:
        await storage.record_tag_cooccurrence({"only_one"})
        pairs = await storage.get_tag_cooccurrence(min_count=1)
        assert len(pairs) == 0

    @pytest.mark.asyncio
    async def test_empty_tags_no_cooccurrence(self, storage) -> None:
        await storage.record_tag_cooccurrence(set())
        pairs = await storage.get_tag_cooccurrence(min_count=1)
        assert len(pairs) == 0

    @pytest.mark.asyncio
    async def test_save_and_get_drift_cluster(self, storage) -> None:
        await storage.save_drift_cluster(
            cluster_id="c1",
            canonical="react",
            members=["react", "reactjs"],
            confidence=0.85,
            status="detected",
        )
        clusters = await storage.get_drift_clusters()
        assert len(clusters) == 1
        assert clusters[0]["canonical"] == "react"
        assert clusters[0]["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_get_clusters_filter_by_status(self, storage) -> None:
        await storage.save_drift_cluster("c1", "a", ["a", "b"], 0.8, "detected")
        await storage.save_drift_cluster("c2", "x", ["x", "y"], 0.6, "merged")
        detected = await storage.get_drift_clusters(status="detected")
        merged = await storage.get_drift_clusters(status="merged")
        assert len(detected) == 1
        assert len(merged) == 1

    @pytest.mark.asyncio
    async def test_resolve_drift_cluster(self, storage) -> None:
        await storage.save_drift_cluster("c1", "a", ["a", "b"], 0.8, "detected")
        result = await storage.resolve_drift_cluster("c1", "merged")
        assert result is True
        clusters = await storage.get_drift_clusters(status="merged")
        assert len(clusters) == 1
        assert clusters[0]["resolved_at"] is not None

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_cluster(self, storage) -> None:
        result = await storage.resolve_drift_cluster("nonexistent", "merged")
        assert result is False

    @pytest.mark.asyncio
    async def test_upsert_drift_cluster(self, storage) -> None:
        await storage.save_drift_cluster("c1", "a", ["a", "b"], 0.5, "detected")
        await storage.save_drift_cluster("c1", "a", ["a", "b", "c"], 0.9, "detected")
        clusters = await storage.get_drift_clusters()
        assert len(clusters) == 1
        assert clusters[0]["confidence"] == 0.9
        assert (
            "c" in json.loads(clusters[0]["members"])
            if isinstance(clusters[0]["members"], str)
            else "c" in clusters[0]["members"]
        )

    @pytest.mark.asyncio
    async def test_get_tag_fiber_counts(self, storage) -> None:
        conn = storage._ensure_conn()
        now = utcnow().isoformat()
        # Insert fibers with tags
        await conn.execute(
            "INSERT INTO fibers (id, brain_id, anchor_neuron_id, neuron_ids, synapse_ids, summary, auto_tags, agent_tags, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "f1",
                "test-brain",
                "n1",
                "[]",
                "[]",
                "test",
                json.dumps(["react", "typescript"]),
                json.dumps([]),
                now,
            ),
        )
        await conn.execute(
            "INSERT INTO fibers (id, brain_id, anchor_neuron_id, neuron_ids, synapse_ids, summary, auto_tags, agent_tags, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "f2",
                "test-brain",
                "n2",
                "[]",
                "[]",
                "test2",
                json.dumps(["react", "python"]),
                json.dumps(["api"]),
                now,
            ),
        )
        await conn.commit()

        counts = await storage.get_tag_fiber_counts()
        assert counts["react"] == 2
        assert counts["typescript"] == 1
        assert counts["python"] == 1
        assert counts["api"] == 1


# ── DriftHandler (MCP tool) ────────────────────────────────────────


class TestDriftHandler:
    """Tests for the MCP drift handler."""

    @pytest.mark.asyncio
    async def test_invalid_action(self) -> None:
        from neural_memory.mcp.drift_handler import DriftHandler

        handler = DriftHandler()
        handler.get_storage = AsyncMock()
        result = await handler._drift({"action": "bogus"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_no_brain(self) -> None:
        from neural_memory.mcp.drift_handler import DriftHandler

        handler = DriftHandler()
        mock_storage = MagicMock()
        mock_storage.brain_id = None
        handler.get_storage = AsyncMock(return_value=mock_storage)
        result = await handler._drift({"action": "detect"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_detect_clean(self) -> None:
        from neural_memory.mcp.drift_handler import DriftHandler

        handler = DriftHandler()
        mock_storage = MagicMock()
        mock_storage.brain_id = "test"
        mock_storage._current_brain_id = "test"
        handler.get_storage = AsyncMock(return_value=mock_storage)

        with patch("neural_memory.engine.drift_detection.run_drift_detection") as mock_run:
            mock_run.return_value = {
                "clusters": [],
                "temporal_drifts": [],
                "summary": {"total_clusters": 0},
            }
            result = await handler._drift({"action": "detect"})
        assert result["status"] == "clean"

    @pytest.mark.asyncio
    async def test_list_clusters(self) -> None:
        from neural_memory.mcp.drift_handler import DriftHandler

        handler = DriftHandler()
        mock_storage = MagicMock()
        mock_storage.brain_id = "test"
        mock_storage._current_brain_id = "test"
        mock_storage.get_drift_clusters = AsyncMock(
            return_value=[
                {"id": "c1", "canonical": "react", "members": ["react", "reactjs"]},
            ]
        )
        handler.get_storage = AsyncMock(return_value=mock_storage)

        result = await handler._drift({"action": "list"})
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_merge_requires_cluster_id(self) -> None:
        from neural_memory.mcp.drift_handler import DriftHandler

        handler = DriftHandler()
        mock_storage = MagicMock()
        mock_storage.brain_id = "test"
        mock_storage._current_brain_id = "test"
        handler.get_storage = AsyncMock(return_value=mock_storage)

        result = await handler._drift({"action": "merge"})
        assert "error" in result
        assert "cluster_id" in result["error"]

    @pytest.mark.asyncio
    async def test_dismiss_cluster(self) -> None:
        from neural_memory.mcp.drift_handler import DriftHandler

        handler = DriftHandler()
        mock_storage = MagicMock()
        mock_storage.brain_id = "test"
        mock_storage._current_brain_id = "test"
        mock_storage.resolve_drift_cluster = AsyncMock(return_value=True)
        handler.get_storage = AsyncMock(return_value=mock_storage)

        result = await handler._drift({"action": "dismiss", "cluster_id": "c1"})
        assert result["status"] == "resolved"
        assert result["resolution"] == "dismissed"


# ── UnionFind ───────────────────────────────────────────────────────


class TestUnionFind:
    """Tests for the Union-Find data structure."""

    def test_basic_union(self) -> None:
        uf = UnionFind(5)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)
        assert uf.find(2) != uf.find(0)

    def test_transitive_union(self) -> None:
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_groups(self) -> None:
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        groups = uf.groups()
        assert len(groups) == 3  # {0,1}, {2,3}, {4}

    def test_single_element_groups(self) -> None:
        uf = UnionFind(3)
        groups = uf.groups()
        assert len(groups) == 3


# ── Wasserstein-1 Distance ─────────────────────────────────────────


class TestWasserstein1:
    """Tests for W1 (Earth Mover's) distance computation."""

    def test_identical_distributions(self) -> None:
        """Identical distributions → W1 = 0."""
        assert wasserstein_1([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_empty_distributions(self) -> None:
        """Empty distributions → W1 = 0."""
        assert wasserstein_1([], []) == 0.0
        assert wasserstein_1([1.0], []) == 0.0
        assert wasserstein_1([], [1.0]) == 0.0

    def test_opposite_distributions(self) -> None:
        """Maximally different distributions → high W1."""
        w1 = wasserstein_1([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        assert w1 > 0.3  # Should be substantial

    def test_gradual_shift(self) -> None:
        """Gradual shift → proportional W1."""
        small = wasserstein_1([1.0, 2.0, 3.0], [1.5, 2.0, 2.5])
        large = wasserstein_1([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert small < large

    def test_symmetric(self) -> None:
        """W1 should be symmetric."""
        a = [1.0, 2.0, 3.0]
        b = [3.0, 1.0, 2.0]
        assert abs(wasserstein_1(a, b) - wasserstein_1(b, a)) < 1e-10

    def test_different_lengths(self) -> None:
        """Handles distributions of different lengths."""
        w1 = wasserstein_1([1.0, 2.0], [1.0, 2.0, 3.0])
        assert w1 >= 0.0  # Should work without error

    def test_all_zeros(self) -> None:
        """All-zero distribution → W1 = 0."""
        assert wasserstein_1([0.0, 0.0], [0.0, 0.0]) == 0.0
        assert wasserstein_1([0.0], [1.0]) == 0.0  # One is zero-sum


class TestActivationDrift:
    """Tests for activation drift detection using W1."""

    def test_no_drift(self) -> None:
        """Identical activations → stable."""
        report = detect_activation_drift(
            {"concept": [0.5, 0.6, 0.7]},
            {"concept": [0.5, 0.6, 0.7]},
        )
        assert report.overall_drift == 0.0
        assert len(report.drifting_types) == 0
        assert report.results[0].status == "stable"

    def test_drift_detected(self) -> None:
        """Differently shaped activations → non-zero drift."""
        # Concentrated at start vs concentrated at end
        report = detect_activation_drift(
            {"concept": [0.9, 0.1, 0.0, 0.0]},
            {"concept": [0.0, 0.0, 0.1, 0.9]},
        )
        assert report.overall_drift > 0.0

    def test_multiple_types(self) -> None:
        """Different types should be analyzed independently."""
        report = detect_activation_drift(
            {"concept": [0.5, 0.5], "entity": [0.1, 0.1]},
            {"concept": [0.5, 0.5], "entity": [0.9, 0.9]},
        )
        assert len(report.results) == 2
        # concept should be stable, entity may drift
        concept = next(r for r in report.results if r.neuron_type == "concept")
        assert concept.status == "stable"

    def test_missing_type_in_one_period(self) -> None:
        """Type present in only one period → included with empty other."""
        report = detect_activation_drift(
            {"concept": [0.5, 0.5]},
            {"concept": [0.5, 0.5], "new_type": [0.8, 0.8]},
        )
        assert len(report.results) == 2
        new_type = next(r for r in report.results if r.neuron_type == "new_type")
        assert new_type.period_a_count == 0
        assert new_type.period_b_count == 2

    def test_empty_input(self) -> None:
        """Empty input → zero drift."""
        report = detect_activation_drift({}, {})
        assert report.overall_drift == 0.0
        assert len(report.results) == 0

    def test_counts_reported(self) -> None:
        """Period counts should be correctly reported."""
        report = detect_activation_drift(
            {"concept": [0.1, 0.2, 0.3]},
            {"concept": [0.4, 0.5]},
        )
        assert report.results[0].period_a_count == 3
        assert report.results[0].period_b_count == 2
