"""Tests for SQLiteCalibrationMixin (v2.16.0)."""

from __future__ import annotations

import pathlib

import pytest_asyncio

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.storage.sqlite_store import SQLiteStorage


@pytest_asyncio.fixture
async def store(tmp_path: object) -> SQLiteStorage:
    """SQLite storage with calibration table ready."""
    db_path = pathlib.Path(str(tmp_path)) / "test_calibration.db"
    storage = SQLiteStorage(db_path)
    await storage.initialize()
    brain = Brain.create(name="cal-test", config=BrainConfig(), owner_id="test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


class TestCalibrationOps:
    async def test_save_and_get_calibration(self, store: SQLiteStorage) -> None:
        """Round-trip: save a record and retrieve it."""
        await store.save_calibration_record(
            gate="focused_result",
            predicted_sufficient=True,
            actual_confidence=0.85,
            actual_fibers=3,
            query_intent="direct",
            metrics_json={"entropy": 1.2},
        )

        records = await store.get_recent_calibration(gate="focused_result")
        assert len(records) == 1
        assert records[0]["gate"] == "focused_result"
        assert records[0]["predicted_sufficient"] == 1
        assert records[0]["actual_confidence"] == 0.85
        assert records[0]["actual_fibers"] == 3

    async def test_get_all_gates(self, store: SQLiteStorage) -> None:
        """Get records across all gates."""
        await store.save_calibration_record(
            gate="focused_result",
            predicted_sufficient=True,
            actual_confidence=0.8,
            actual_fibers=2,
        )
        await store.save_calibration_record(
            gate="no_anchors",
            predicted_sufficient=False,
            actual_confidence=0.0,
            actual_fibers=0,
        )

        records = await store.get_recent_calibration()
        assert len(records) == 2

    async def test_empty_calibration(self, store: SQLiteStorage) -> None:
        """No records returns empty list."""
        records = await store.get_recent_calibration()
        assert records == []
