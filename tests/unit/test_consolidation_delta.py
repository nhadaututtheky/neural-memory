"""Tests for consolidation_delta.py — before/after health comparison."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from neural_memory.engine.consolidation_delta import (
    ConsolidationDelta,
    HealthSnapshot,
    _snapshot_from_report,
)


def _make_snapshot(**overrides: object) -> HealthSnapshot:
    defaults = {
        "purity_score": 50.0,
        "grade": "C",
        "connectivity": 0.4,
        "diversity": 0.3,
        "freshness": 0.5,
        "consolidation_ratio": 0.1,
        "orphan_rate": 0.2,
        "activation_efficiency": 0.3,
        "recall_confidence": 0.4,
        "neuron_count": 100,
        "synapse_count": 200,
        "fiber_count": 50,
    }
    defaults.update(overrides)
    return HealthSnapshot(**defaults)  # type: ignore[arg-type]


def _make_report() -> MagicMock:
    report = MagicMock()
    report.summary.return_value = "Pruned 5 synapses."
    return report


class TestHealthSnapshot:
    def test_is_frozen(self) -> None:
        snap = _make_snapshot()
        with pytest.raises(AttributeError):
            snap.purity_score = 99.0  # type: ignore[misc]

    def test_values_stored(self) -> None:
        snap = _make_snapshot(purity_score=75.0, grade="B")
        assert snap.purity_score == 75.0
        assert snap.grade == "B"


class TestConsolidationDelta:
    def test_purity_delta_positive(self) -> None:
        before = _make_snapshot(purity_score=50.0)
        after = _make_snapshot(purity_score=55.5)
        delta = ConsolidationDelta(
            before=before,
            after=after,
            report=_make_report(),
            computed_at=datetime(2026, 1, 1),
        )
        assert delta.purity_delta == 5.5

    def test_purity_delta_negative(self) -> None:
        before = _make_snapshot(purity_score=60.0)
        after = _make_snapshot(purity_score=58.0)
        delta = ConsolidationDelta(
            before=before,
            after=after,
            report=_make_report(),
            computed_at=datetime(2026, 1, 1),
        )
        assert delta.purity_delta == -2.0

    def test_connectivity_delta(self) -> None:
        before = _make_snapshot(connectivity=0.4)
        after = _make_snapshot(connectivity=0.52)
        delta = ConsolidationDelta(
            before=before,
            after=after,
            report=_make_report(),
            computed_at=datetime(2026, 1, 1),
        )
        assert delta.connectivity_delta == 0.12

    def test_orphan_rate_delta(self) -> None:
        before = _make_snapshot(orphan_rate=0.2)
        after = _make_snapshot(orphan_rate=0.1)
        delta = ConsolidationDelta(
            before=before,
            after=after,
            report=_make_report(),
            computed_at=datetime(2026, 1, 1),
        )
        assert delta.orphan_rate_delta == -0.1

    def test_summary_includes_report(self) -> None:
        before = _make_snapshot(purity_score=50.0)
        after = _make_snapshot(purity_score=55.0)
        delta = ConsolidationDelta(
            before=before,
            after=after,
            report=_make_report(),
            computed_at=datetime(2026, 1, 1),
        )
        text = delta.summary()
        assert "Pruned 5 synapses." in text
        assert "Health Delta:" in text
        assert "Purity" in text

    def test_summary_shows_grade_change(self) -> None:
        before = _make_snapshot(purity_score=50.0, grade="C")
        after = _make_snapshot(purity_score=76.0, grade="B")
        delta = ConsolidationDelta(
            before=before,
            after=after,
            report=_make_report(),
            computed_at=datetime(2026, 1, 1),
        )
        text = delta.summary()
        assert "Grade: C -> B" in text

    def test_to_dict_structure(self) -> None:
        before = _make_snapshot(purity_score=50.0)
        after = _make_snapshot(purity_score=55.0)
        delta = ConsolidationDelta(
            before=before,
            after=after,
            report=_make_report(),
            computed_at=datetime(2026, 1, 1),
        )
        d = delta.to_dict()
        assert "before" in d
        assert "after" in d
        assert "delta" in d
        assert "grade_changed" in d
        assert d["before"]["purity_score"] == 50.0
        assert d["after"]["purity_score"] == 55.0
        assert d["delta"]["purity"] == 5.0

    def test_is_frozen(self) -> None:
        before = _make_snapshot()
        after = _make_snapshot()
        delta = ConsolidationDelta(
            before=before,
            after=after,
            report=_make_report(),
            computed_at=datetime(2026, 1, 1),
        )
        with pytest.raises(AttributeError):
            delta.before = _make_snapshot()  # type: ignore[misc]


class TestSnapshotFromReport:
    def test_extracts_all_fields(self) -> None:
        from neural_memory.engine.diagnostics import BrainHealthReport

        report = BrainHealthReport(
            purity_score=65.0,
            grade="C",
            connectivity=0.5,
            diversity=0.4,
            freshness=0.6,
            consolidation_ratio=0.2,
            orphan_rate=0.15,
            activation_efficiency=0.3,
            recall_confidence=0.5,
            neuron_count=100,
            synapse_count=200,
            fiber_count=50,
            warnings=(),
            recommendations=(),
        )
        snap = _snapshot_from_report(report)
        assert snap.purity_score == 65.0
        assert snap.grade == "C"
        assert snap.connectivity == 0.5
        assert snap.neuron_count == 100
