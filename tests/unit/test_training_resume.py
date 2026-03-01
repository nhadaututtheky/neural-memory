"""Tests for training file tracking and resume."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_memory.storage.sqlite_training_files import compute_file_hash


class TestComputeFileHash:
    """Test file hashing utility."""

    def test_consistent_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("# Hello World", encoding="utf-8")

        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("Content A", encoding="utf-8")
        f2.write_text("Content B", encoding="utf-8")

        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("Same content", encoding="utf-8")
        f2.write_text("Same content", encoding="utf-8")

        assert compute_file_hash(f1) == compute_file_hash(f2)


class TestTrainingFilesStorage:
    """Test training file CRUD operations."""

    async def test_upsert_and_lookup(self, tmp_path: Path) -> None:
        from neural_memory.core.brain import Brain
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Insert a training file record
        record_id = await storage.upsert_training_file(
            file_hash="abc123",
            file_path="/docs/test.md",
            file_size=1024,
            chunks_total=10,
            chunks_completed=10,
            status="completed",
            domain_tag="react",
        )
        assert record_id

        # Look it up
        record = await storage.get_training_file_by_hash("abc123")
        assert record is not None
        assert record["status"] == "completed"
        assert record["file_path"] == "/docs/test.md"

        # Not found
        not_found = await storage.get_training_file_by_hash("nonexistent")
        assert not_found is None

        await storage.close()

    async def test_upsert_updates_existing(self, tmp_path: Path) -> None:
        from neural_memory.core.brain import Brain
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # First insert
        record_id = await storage.upsert_training_file(
            file_hash="abc123",
            file_path="/docs/test.md",
            file_size=1024,
            chunks_total=5,
            chunks_completed=3,
            status="in_progress",
        )

        # Upsert with same hash → should update
        record_id2 = await storage.upsert_training_file(
            file_hash="abc123",
            file_path="/docs/test.md",
            file_size=1024,
            chunks_total=5,
            chunks_completed=5,
            status="completed",
        )

        assert record_id == record_id2

        record = await storage.get_training_file_by_hash("abc123")
        assert record is not None
        assert record["status"] == "completed"
        assert record["chunks_completed"] == 5

        await storage.close()

    async def test_training_stats(self, tmp_path: Path) -> None:
        from neural_memory.core.brain import Brain
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Add some records
        await storage.upsert_training_file(
            file_hash="h1", file_path="a.md", file_size=100,
            chunks_total=5, chunks_completed=5, status="completed",
        )
        await storage.upsert_training_file(
            file_hash="h2", file_path="b.md", file_size=200,
            chunks_total=3, chunks_completed=1, status="in_progress",
        )

        stats = await storage.get_training_stats()
        assert stats["total_files"] == 2
        assert stats["completed"] == 1
        assert stats["in_progress"] == 1
        assert stats["total_chunks"] == 6

        await storage.close()
