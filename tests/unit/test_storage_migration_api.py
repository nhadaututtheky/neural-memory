"""Tests for storage management dashboard API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from neural_memory.server.routes.dashboard_api import (
    MigrationJobStatus,
    _migration_jobs,
    router,
)


@pytest.fixture()
def app() -> FastAPI:
    """Create a test FastAPI app with dashboard router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_jobs() -> None:
    """Clear migration jobs between tests."""
    _migration_jobs.clear()


def _mock_config(
    *,
    storage_backend: str = "sqlite",
    current_brain: str = "default",
    data_dir: str = "/tmp/neuralmemory",
    is_pro: bool = False,
) -> MagicMock:
    cfg = MagicMock()
    cfg.storage_backend = storage_backend
    cfg.current_brain = current_brain
    cfg.data_dir = data_dir
    cfg.is_pro.return_value = is_pro
    cfg.save = MagicMock()
    cfg.postgres = MagicMock(host="", database="", port=5432, user="", password="")
    return cfg


# Patch targets: imports inside functions use the source module
_PATCH_GET_CONFIG = "neural_memory.unified_config.get_config"
_PATCH_SET_CONFIG = "neural_memory.unified_config.set_config"
_PATCH_HAS_PRO = "neural_memory.plugins.has_pro"


class TestStorageStatus:
    """Tests for GET /api/dashboard/storage/status."""

    def test_returns_sqlite_status(self, client: TestClient) -> None:
        mock_stat = MagicMock()
        mock_stat.st_size = 2 * 1024 * 1024  # 2MB

        with (
            patch(
                _PATCH_GET_CONFIG,
                return_value=_mock_config(storage_backend="sqlite"),
            ),
            patch(_PATCH_HAS_PRO, return_value=True),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat", return_value=mock_stat),
        ):
            resp = client.get("/api/dashboard/storage/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["current_backend"] == "sqlite"
            assert data["pro_installed"] is True
            assert data["sqlite_exists"] is True
            assert data["sqlite_size_bytes"] == 2 * 1024 * 1024

    def test_returns_no_migration_job(self, client: TestClient) -> None:
        with (
            patch(_PATCH_GET_CONFIG, return_value=_mock_config()),
            patch(_PATCH_HAS_PRO, return_value=False),
            patch("pathlib.Path.exists", return_value=False),
        ):
            resp = client.get("/api/dashboard/storage/status")
            assert resp.status_code == 200
            assert resp.json()["migration_job"] is None

    def test_returns_active_migration_job(self, client: TestClient) -> None:
        _migration_jobs["active"] = MigrationJobStatus(
            job_id="active",
            state="running",
            direction="to_infinitydb",
            brain="default",
            started_at="2026-03-26T00:00:00Z",
        )
        with (
            patch(_PATCH_GET_CONFIG, return_value=_mock_config()),
            patch(_PATCH_HAS_PRO, return_value=True),
            patch("pathlib.Path.exists", return_value=False),
        ):
            resp = client.get("/api/dashboard/storage/status")
            assert resp.status_code == 200
            job = resp.json()["migration_job"]
            assert job is not None
            assert job["job_id"] == "active"
            assert job["state"] == "running"


class TestStartMigration:
    """Tests for POST /api/dashboard/storage/migrate."""

    def test_rejects_invalid_direction(self, client: TestClient) -> None:
        with (
            patch(_PATCH_GET_CONFIG, return_value=_mock_config()),
            patch(_PATCH_HAS_PRO, return_value=True),
        ):
            resp = client.post(
                "/api/dashboard/storage/migrate",
                json={"direction": "to_redis"},
            )
            assert resp.status_code == 422

    def test_rejects_no_pro_installed(self, client: TestClient) -> None:
        with (
            patch(_PATCH_GET_CONFIG, return_value=_mock_config()),
            patch(_PATCH_HAS_PRO, return_value=False),
        ):
            resp = client.post(
                "/api/dashboard/storage/migrate",
                json={"direction": "to_infinitydb"},
            )
            assert resp.status_code == 403
            assert "not installed" in resp.json()["detail"]

    def test_rejects_no_pro_license(self, client: TestClient) -> None:
        with (
            patch(_PATCH_GET_CONFIG, return_value=_mock_config(is_pro=False)),
            patch(_PATCH_HAS_PRO, return_value=True),
        ):
            resp = client.post(
                "/api/dashboard/storage/migrate",
                json={"direction": "to_infinitydb"},
            )
            assert resp.status_code == 403
            assert "license" in resp.json()["detail"].lower()

    def test_rejects_duplicate_running_job(self, client: TestClient) -> None:
        _migration_jobs["existing"] = MigrationJobStatus(
            job_id="existing",
            state="running",
            direction="to_infinitydb",
            brain="default",
            started_at="2026-03-26T00:00:00Z",
        )

        with (
            patch(_PATCH_GET_CONFIG, return_value=_mock_config(is_pro=True)),
            patch(_PATCH_HAS_PRO, return_value=True),
            patch("pathlib.Path.exists", return_value=True),
        ):
            resp = client.post(
                "/api/dashboard/storage/migrate",
                json={"direction": "to_infinitydb"},
            )
            assert resp.status_code == 409
            assert "already running" in resp.json()["detail"]

    def test_allows_completed_job_same_brain(self, client: TestClient) -> None:
        """A completed job should NOT block new migration for same brain."""
        _migration_jobs["old"] = MigrationJobStatus(
            job_id="old",
            state="done",
            direction="to_infinitydb",
            brain="default",
            started_at="2026-03-26T00:00:00Z",
            finished_at="2026-03-26T00:01:00Z",
        )

        mock_stat = MagicMock()
        mock_stat.st_size = 1024 * 1024  # 1MB

        with (
            patch(_PATCH_GET_CONFIG, return_value=_mock_config(is_pro=True)),
            patch(_PATCH_HAS_PRO, return_value=True),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat", return_value=mock_stat),
            patch(
                "shutil.disk_usage",
                return_value=MagicMock(free=1024 * 1024 * 1024),
            ),
        ):
            resp = client.post(
                "/api/dashboard/storage/migrate",
                json={"direction": "to_infinitydb"},
            )
            assert resp.status_code == 200
            assert "job_id" in resp.json()

    def test_returns_disk_warning_when_low_space(self, client: TestClient) -> None:
        """Should return disk_warning when free space is below estimate."""
        mock_stat = MagicMock()
        mock_stat.st_size = 100 * 1024 * 1024  # 100MB source

        with (
            patch(_PATCH_GET_CONFIG, return_value=_mock_config(is_pro=True)),
            patch(_PATCH_HAS_PRO, return_value=True),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat", return_value=mock_stat),
            patch(
                "shutil.disk_usage",
                return_value=MagicMock(free=10 * 1024 * 1024),  # Only 10MB free
            ),
        ):
            resp = client.post(
                "/api/dashboard/storage/migrate",
                json={"direction": "to_infinitydb"},
            )
            assert resp.status_code == 200
            assert "disk_warning" in resp.json()


class TestMigrationProgress:
    """Tests for GET /api/dashboard/storage/migrate/{job_id}."""

    def test_returns_job(self, client: TestClient) -> None:
        _migration_jobs["test-job"] = MigrationJobStatus(
            job_id="test-job",
            state="running",
            direction="to_infinitydb",
            brain="default",
            neurons_total=100,
            neurons_done=50,
            started_at="2026-03-26T00:00:00Z",
        )
        resp = client.get("/api/dashboard/storage/migrate/test-job")
        assert resp.status_code == 200
        data = resp.json()
        assert data["neurons_total"] == 100
        assert data["neurons_done"] == 50
        assert data["state"] == "running"

    def test_returns_404_for_unknown_job(self, client: TestClient) -> None:
        resp = client.get("/api/dashboard/storage/migrate/nonexistent")
        assert resp.status_code == 404


class TestSetBackend:
    """Tests for POST /api/dashboard/storage/backend."""

    def test_rejects_invalid_backend(self, client: TestClient) -> None:
        resp = client.post(
            "/api/dashboard/storage/backend",
            json={"backend": "redis"},
        )
        assert resp.status_code == 422

    def test_rejects_infinitydb_without_pro(self, client: TestClient) -> None:
        with (
            patch(_PATCH_GET_CONFIG, return_value=_mock_config()),
            patch(_PATCH_SET_CONFIG),
            patch(_PATCH_HAS_PRO, return_value=False),
        ):
            resp = client.post(
                "/api/dashboard/storage/backend",
                json={"backend": "infinitydb"},
            )
            assert resp.status_code == 403

    def test_rejects_infinitydb_without_data(self, client: TestClient) -> None:
        with (
            patch(_PATCH_GET_CONFIG, return_value=_mock_config()),
            patch(_PATCH_SET_CONFIG),
            patch(_PATCH_HAS_PRO, return_value=True),
            patch("pathlib.Path.exists", return_value=False),
        ):
            resp = client.post(
                "/api/dashboard/storage/backend",
                json={"backend": "infinitydb"},
            )
            assert resp.status_code == 400
            assert "not found" in resp.json()["detail"].lower()

    def test_unchanged_returns_unchanged(self, client: TestClient) -> None:
        with (
            patch(_PATCH_GET_CONFIG, return_value=_mock_config(storage_backend="sqlite")),
            patch(_PATCH_SET_CONFIG),
        ):
            resp = client.post(
                "/api/dashboard/storage/backend",
                json={"backend": "sqlite"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "unchanged"

    def test_switch_to_infinitydb_succeeds(self, client: TestClient) -> None:
        from dataclasses import dataclass

        @dataclass
        class FakeConfig:
            storage_backend: str = "sqlite"
            current_brain: str = "default"
            data_dir: str = "/tmp/neuralmemory"

            def is_pro(self) -> bool:
                return True

            def save(self) -> None:
                pass

        cfg = FakeConfig()
        with (
            patch(_PATCH_GET_CONFIG, return_value=cfg),
            patch(_PATCH_SET_CONFIG),
            patch(_PATCH_HAS_PRO, return_value=True),
            patch("pathlib.Path.exists", return_value=True),
            patch("neural_memory.unified_config._storage_cache", {}),
        ):
            resp = client.post(
                "/api/dashboard/storage/backend",
                json={"backend": "infinitydb"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "switched"
            assert resp.json()["backend"] == "infinitydb"


class TestMigrationJobStatus:
    """Tests for MigrationJobStatus model."""

    def test_model_defaults(self) -> None:
        job = MigrationJobStatus(
            job_id="test",
            state="running",
            direction="to_infinitydb",
            brain="default",
            started_at="2026-03-26T00:00:00Z",
        )
        assert job.neurons_total == 0
        assert job.neurons_done == 0
        assert job.error is None
        assert job.finished_at is None

    def test_model_with_progress(self) -> None:
        job = MigrationJobStatus(
            job_id="test",
            state="done",
            direction="to_infinitydb",
            brain="default",
            neurons_total=100,
            neurons_done=100,
            synapses_total=500,
            synapses_done=500,
            fibers_total=50,
            fibers_done=50,
            started_at="2026-03-26T00:00:00Z",
            finished_at="2026-03-26T00:01:00Z",
        )
        assert job.state == "done"
        assert job.neurons_done == job.neurons_total

    def test_model_with_error(self) -> None:
        job = MigrationJobStatus(
            job_id="test",
            state="error",
            direction="to_infinitydb",
            brain="default",
            error="Verification failed",
            started_at="2026-03-26T00:00:00Z",
            finished_at="2026-03-26T00:01:00Z",
        )
        assert job.state == "error"
        assert "Verification" in (job.error or "")
