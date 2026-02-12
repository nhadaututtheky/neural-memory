"""Tests for dashboard API routes — timeline, fibers, fiber diagram endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from neural_memory.server.routes.dashboard_api import router


@dataclass
class FakeNeuron:
    """Minimal neuron for testing."""

    id: str
    content: str
    type: Any  # StrEnum-like with .value
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None


@dataclass
class FakeFiber:
    """Minimal fiber for testing."""

    id: str
    summary: str = ""
    neuron_ids: list[str] = field(default_factory=list)


@dataclass
class FakeSynapse:
    """Minimal synapse for testing."""

    id: str
    source_id: str
    target_id: str
    type: Any
    weight: float = 1.0
    direction: Any = None


class FakeType:
    """Mimics a StrEnum with .value."""

    def __init__(self, value: str) -> None:
        self.value = value


class FakeDirection:
    """Mimics SynapseDirection."""

    def __init__(self, value: str) -> None:
        self.value = value


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def mock_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.find_neurons = AsyncMock(return_value=[])
    storage.get_fibers = AsyncMock(return_value=[])
    storage.get_all_synapses = AsyncMock(return_value=[])
    storage.get_neurons_batch = AsyncMock(return_value={})
    return storage


@pytest.fixture()
def client(mock_storage: AsyncMock) -> TestClient:
    app = _make_app()
    app.dependency_overrides = {}

    from neural_memory.server.routes.dashboard_api import get_storage

    app.dependency_overrides[get_storage] = lambda: mock_storage
    return TestClient(app)


class TestTimelineEndpoint:
    def test_empty_timeline(self, client: TestClient) -> None:
        resp = client.get("/api/dashboard/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entries"] == []
        assert data["total"] == 0

    def test_timeline_with_neurons(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        neurons = [
            FakeNeuron(
                id="n1",
                content="Test memory",
                type=FakeType("concept"),
                metadata={"_created_at": "2026-02-10T10:00:00"},
            ),
            FakeNeuron(
                id="n2",
                content="Another memory",
                type=FakeType("entity"),
                metadata={"_created_at": "2026-02-11T12:00:00"},
            ),
        ]
        mock_storage.find_neurons.return_value = neurons

        resp = client.get("/api/dashboard/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["entries"]) == 2
        # Should be sorted descending by created_at
        assert data["entries"][0]["id"] == "n2"
        assert data["entries"][1]["id"] == "n1"

    def test_timeline_respects_limit(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        neurons = [
            FakeNeuron(
                id=f"n{i}",
                content=f"Memory {i}",
                type=FakeType("concept"),
                metadata={"_created_at": f"2026-02-{10+i:02d}T10:00:00"},
            )
            for i in range(5)
        ]
        mock_storage.find_neurons.return_value = neurons

        resp = client.get("/api/dashboard/timeline?limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["entries"]) == 2

    def test_timeline_type_filter_in_response(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        neurons = [
            FakeNeuron(
                id="n1",
                content="Concept memory",
                type=FakeType("concept"),
                metadata={"_created_at": "2026-02-10T10:00:00"},
            ),
        ]
        mock_storage.find_neurons.return_value = neurons

        resp = client.get("/api/dashboard/timeline")
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert entries[0]["neuron_type"] == "concept"


class TestFibersEndpoint:
    def test_empty_fibers(self, client: TestClient) -> None:
        resp = client.get("/api/dashboard/fibers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["fibers"] == []

    def test_fibers_list(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        fibers = [
            FakeFiber(id="f1", summary="Test fiber", neuron_ids=["n1", "n2"]),
            FakeFiber(id="f2", summary="Another fiber", neuron_ids=["n3"]),
        ]
        mock_storage.get_fibers.return_value = fibers

        resp = client.get("/api/dashboard/fibers")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["fibers"]) == 2
        assert data["fibers"][0]["id"] == "f1"
        assert data["fibers"][0]["neuron_count"] == 2
        assert data["fibers"][1]["neuron_count"] == 1

    def test_fibers_limit(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        fibers = [
            FakeFiber(id=f"f{i}", summary=f"Fiber {i}") for i in range(10)
        ]
        mock_storage.get_fibers.return_value = fibers

        resp = client.get("/api/dashboard/fibers?limit=5")
        assert resp.status_code == 200
        # The endpoint passes limit to storage.get_fibers


class TestFiberDiagramEndpoint:
    def test_fiber_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/dashboard/fiber/nonexistent/diagram")
        assert resp.status_code == 404

    def test_fiber_diagram_success(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        fiber = FakeFiber(id="f1", summary="Test", neuron_ids=["n1", "n2"])
        mock_storage.get_fibers.return_value = [fiber]

        n1 = FakeNeuron(id="n1", content="Node 1", type=FakeType("concept"))
        n2 = FakeNeuron(id="n2", content="Node 2", type=FakeType("entity"))
        mock_storage.get_neurons_batch.return_value = {"n1": n1, "n2": n2}

        syn = FakeSynapse(
            id="s1",
            source_id="n1",
            target_id="n2",
            type=FakeType("temporal"),
            weight=0.8,
            direction=FakeDirection("forward"),
        )
        mock_storage.get_all_synapses.return_value = [syn]

        resp = client.get("/api/dashboard/fiber/f1/diagram")
        assert resp.status_code == 200
        data = resp.json()
        assert data["fiber_id"] == "f1"
        assert len(data["neurons"]) == 2
        assert len(data["synapses"]) == 1
        assert data["synapses"][0]["source_id"] == "n1"
        assert data["synapses"][0]["target_id"] == "n2"

    def test_fiber_diagram_no_synapses(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        fiber = FakeFiber(id="f1", summary="Test", neuron_ids=["n1"])
        mock_storage.get_fibers.return_value = [fiber]

        n1 = FakeNeuron(id="n1", content="Alone", type=FakeType("concept"))
        mock_storage.get_neurons_batch.return_value = {"n1": n1}
        mock_storage.get_all_synapses.return_value = []

        resp = client.get("/api/dashboard/fiber/f1/diagram")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["neurons"]) == 1
        assert len(data["synapses"]) == 0

    def test_fiber_diagram_filters_external_synapses(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Synapses referencing external neurons should be excluded."""
        fiber = FakeFiber(id="f1", summary="Test", neuron_ids=["n1"])
        mock_storage.get_fibers.return_value = [fiber]

        n1 = FakeNeuron(id="n1", content="Inside", type=FakeType("concept"))
        mock_storage.get_neurons_batch.return_value = {"n1": n1}

        syn = FakeSynapse(
            id="s1",
            source_id="n1",
            target_id="n_external",  # not in fiber
            type=FakeType("related"),
            direction=FakeDirection("forward"),
        )
        mock_storage.get_all_synapses.return_value = [syn]

        resp = client.get("/api/dashboard/fiber/f1/diagram")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["synapses"]) == 0  # External synapse filtered
