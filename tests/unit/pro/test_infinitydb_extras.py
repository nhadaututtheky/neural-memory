"""Tests for InfinityDB cognitive, alerts, reviews, and extras mixin."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from neural_memory.core.alert import Alert, AlertStatus, AlertType
from neural_memory.core.review_schedule import ReviewSchedule
from neural_memory.pro.storage_adapter import InfinityDBStorage
from neural_memory.utils.timeutils import utcnow


@pytest.fixture
async def storage(tmp_path: Path) -> InfinityDBStorage:
    s = InfinityDBStorage(base_dir=str(tmp_path), brain_id="test")
    await s.open()
    yield s
    await s.close()


class TestCognitive:
    async def test_upsert_get(self, storage: InfinityDBStorage) -> None:
        await storage.upsert_cognitive_state("n1", confidence=0.8, status="active")
        got = await storage.get_cognitive_state("n1")
        assert got is not None
        assert got["confidence"] == 0.8

    async def test_list_filter_status(self, storage: InfinityDBStorage) -> None:
        await storage.upsert_cognitive_state("n1", status="active")
        await storage.upsert_cognitive_state("n2", status="resolved_correct")
        active = await storage.list_cognitive_states(status="active")
        assert len(active) == 1

    async def test_update_evidence(self, storage: InfinityDBStorage) -> None:
        await storage.upsert_cognitive_state("n1", confidence=0.5)
        await storage.update_cognitive_evidence(
            "n1",
            confidence=0.9,
            evidence_for_count=5,
            evidence_against_count=1,
            status="active",
        )
        got = await storage.get_cognitive_state("n1")
        assert got["confidence"] == 0.9
        assert got["evidence_for_count"] == 5

    async def test_list_predictions(self, storage: InfinityDBStorage) -> None:
        await storage.upsert_cognitive_state("n1", predicted_at="2026-01-01")
        await storage.upsert_cognitive_state("n2")
        preds = await storage.list_predictions()
        assert len(preds) == 1

    async def test_calibration_stats(self, storage: InfinityDBStorage) -> None:
        await storage.upsert_cognitive_state("n1", predicted_at="2026-01-01", status="active")
        await storage.upsert_cognitive_state(
            "n2", predicted_at="2026-01-01", status="resolved_correct"
        )
        stats = await storage.get_calibration_stats()
        assert stats["total"] == 2
        assert stats["active"] == 1
        assert stats["resolved_correct"] == 1

    async def test_schema_history(self, storage: InfinityDBStorage) -> None:
        await storage.upsert_cognitive_state("n1", parent_schema_id=None)
        await storage.upsert_cognitive_state("n2", parent_schema_id="n1")
        chain = await storage.get_schema_history("n2")
        assert len(chain) == 2


class TestAlerts:
    def _make_alert(self, aid: str = "", **kw) -> Alert:
        return Alert(
            id=aid,
            brain_id="test",
            alert_type=kw.pop("alert_type", AlertType.STALE_FIBERS),
            **kw,
        )

    async def test_record_get(self, storage: InfinityDBStorage) -> None:
        alert = self._make_alert(message="Test alert")
        aid = await storage.record_alert(alert)
        assert aid
        active = await storage.get_active_alerts()
        assert len(active) == 1

    async def test_duplicate_suppression(self, storage: InfinityDBStorage) -> None:
        a1 = self._make_alert(alert_type=AlertType.STALE_FIBERS)
        a2 = self._make_alert(alert_type=AlertType.STALE_FIBERS)
        aid1 = await storage.record_alert(a1)
        aid2 = await storage.record_alert(a2)
        assert aid1 != ""
        assert aid2 == ""

    async def test_count_pending(self, storage: InfinityDBStorage) -> None:
        await storage.record_alert(self._make_alert(alert_type=AlertType.STALE_FIBERS))
        await storage.record_alert(self._make_alert(alert_type=AlertType.EXPIRED_MEMORIES))
        assert await storage.count_pending_alerts() == 2

    async def test_mark_seen(self, storage: InfinityDBStorage) -> None:
        aid = await storage.record_alert(self._make_alert())
        count = await storage.mark_alerts_seen([aid])
        assert count == 1
        alerts = await storage.get_active_alerts()
        assert alerts[0].status == AlertStatus.SEEN

    async def test_mark_acknowledged(self, storage: InfinityDBStorage) -> None:
        aid = await storage.record_alert(self._make_alert())
        assert await storage.mark_alert_acknowledged(aid)
        alerts = await storage.get_active_alerts()
        assert alerts[0].status == AlertStatus.ACKNOWLEDGED

    async def test_resolve_by_type(self, storage: InfinityDBStorage) -> None:
        await storage.record_alert(self._make_alert(alert_type=AlertType.STALE_FIBERS))
        count = await storage.resolve_alerts_by_type(["stale_fibers"])
        assert count == 1
        active = await storage.get_active_alerts()
        assert len(active) == 0


class TestReviews:
    def _make_schedule(self, fid: str = "f1", **kw) -> ReviewSchedule:
        return ReviewSchedule(fiber_id=fid, brain_id="test", **kw)

    async def test_add_get(self, storage: InfinityDBStorage) -> None:
        s = self._make_schedule()
        fid = await storage.add_review_schedule(s)
        assert fid == "f1"
        got = await storage.get_review_schedule("f1")
        assert got is not None

    async def test_due_reviews(self, storage: InfinityDBStorage) -> None:
        past = utcnow() - timedelta(hours=1)
        future = utcnow() + timedelta(days=7)
        await storage.add_review_schedule(self._make_schedule("f1", next_review=past))
        await storage.add_review_schedule(self._make_schedule("f2", next_review=future))
        due = await storage.get_due_reviews()
        assert len(due) == 1
        assert due[0].fiber_id == "f1"

    async def test_delete(self, storage: InfinityDBStorage) -> None:
        await storage.add_review_schedule(self._make_schedule())
        assert await storage.delete_review_schedule("f1")
        assert not await storage.delete_review_schedule("f1")

    async def test_stats(self, storage: InfinityDBStorage) -> None:
        past = utcnow() - timedelta(hours=1)
        await storage.add_review_schedule(self._make_schedule("f1", next_review=past, box=1))
        await storage.add_review_schedule(self._make_schedule("f2", box=3))
        stats = await storage.get_review_stats()
        assert stats["total"] == 2
        assert stats["due"] == 1
        assert stats["box_1"] == 1
        assert stats["box_3"] == 1


class TestCoActivation:
    async def test_record_and_get(self, storage: InfinityDBStorage) -> None:
        eid = await storage.record_co_activation("a", "b", 0.8)
        assert eid
        counts = await storage.get_co_activation_counts()
        assert len(counts) == 1
        a, b, cnt, avg = counts[0]
        assert cnt == 1
        assert avg == 0.8

    async def test_self_loop_skipped(self, storage: InfinityDBStorage) -> None:
        eid = await storage.record_co_activation("a", "a", 0.5)
        assert eid == ""

    async def test_canonical_order(self, storage: InfinityDBStorage) -> None:
        await storage.record_co_activation("z", "a", 0.5)
        counts = await storage.get_co_activation_counts()
        assert counts[0][0] == "a"
        assert counts[0][1] == "z"

    async def test_prune(self, storage: InfinityDBStorage) -> None:
        await storage.record_co_activation("a", "b", 0.5)
        pruned = await storage.prune_co_activations(utcnow() + timedelta(hours=1))
        assert pruned == 1


class TestActions:
    async def test_record_and_get(self, storage: InfinityDBStorage) -> None:
        eid = await storage.record_action("remember", session_id="s1")
        assert eid
        actions = await storage.get_action_sequences(session_id="s1")
        assert len(actions) == 1

    async def test_prune(self, storage: InfinityDBStorage) -> None:
        await storage.record_action("recall")
        pruned = await storage.prune_action_events(utcnow() + timedelta(hours=1))
        assert pruned == 1


class TestCompressionBackups:
    async def test_save_get_delete(self, storage: InfinityDBStorage) -> None:
        await storage.save_compression_backup("f1", "original text", 1, 100, 50)
        got = await storage.get_compression_backup("f1")
        assert got is not None
        assert got["original_content"] == "original text"
        assert await storage.delete_compression_backup("f1")
        assert await storage.get_compression_backup("f1") is None

    async def test_stats(self, storage: InfinityDBStorage) -> None:
        await storage.save_compression_backup("f1", "text", 1, 100, 50)
        await storage.save_compression_backup("f2", "text", 2, 200, 80)
        stats = await storage.get_compression_stats()
        assert stats["total_backups"] == 2
        assert stats["total_tokens_saved"] == 170


class TestNeuronSnapshots:
    async def test_save_get_delete(self, storage: InfinityDBStorage) -> None:
        await storage.save_neuron_snapshot("n1", "test", "original", "2026-01-01", 3)
        got = await storage.get_neuron_snapshot("n1")
        assert got is not None
        assert got["original_content"] == "original"
        assert await storage.delete_neuron_snapshot("n1")
        assert await storage.get_neuron_snapshot("n1") is None


class TestKnowledgeGaps:
    async def test_add_list_resolve(self, storage: InfinityDBStorage) -> None:
        gid = await storage.add_knowledge_gap(topic="quantum", detection_source="recall")
        assert gid
        gaps = await storage.list_knowledge_gaps()
        assert len(gaps) == 1
        assert await storage.resolve_knowledge_gap(gid)
        gaps = await storage.list_knowledge_gaps()
        assert len(gaps) == 0
        gaps = await storage.list_knowledge_gaps(include_resolved=True)
        assert len(gaps) == 1

    async def test_get_gap(self, storage: InfinityDBStorage) -> None:
        gid = await storage.add_knowledge_gap(topic="AI", detection_source="test")
        got = await storage.get_knowledge_gap(gid)
        assert got is not None
        assert got["topic"] == "AI"


class TestSources:
    async def test_add_get_list_delete(self, storage: InfinityDBStorage) -> None:
        sid = await storage.add_source({"id": "s1", "name": "Test"})
        got = await storage.get_source(sid)
        assert got is not None
        sources = await storage.list_sources()
        assert len(sources) == 1
        assert await storage.delete_source(sid)
        assert await storage.get_source(sid) is None


class TestHotIndex:
    async def test_refresh_and_get(self, storage: InfinityDBStorage) -> None:
        items = [{"id": "n1", "score": 0.9}, {"id": "n2", "score": 0.8}]
        count = await storage.refresh_hot_index(items)
        assert count == 2
        hot = await storage.get_hot_index(limit=1)
        assert len(hot) == 1

    async def test_empty(self, storage: InfinityDBStorage) -> None:
        hot = await storage.get_hot_index()
        assert hot == []


class TestGetBrainId:
    async def test_get_brain_id(self, storage: InfinityDBStorage) -> None:
        assert storage._get_brain_id() == "test"

    async def test_get_brain_id_not_set(self, tmp_path: Path) -> None:
        s = InfinityDBStorage(base_dir=str(tmp_path), brain_id="")
        s._current_brain_id = None
        with pytest.raises(ValueError, match="Brain ID not set"):
            s._get_brain_id()
