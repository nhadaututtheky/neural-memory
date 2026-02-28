"""End-to-end API tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from neural_memory.server.app import create_app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create test client with lifespan context."""
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    """Tests for health and root endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_root_endpoint(self, client: TestClient) -> None:
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestBrainEndpoints:
    """Tests for brain management endpoints."""

    def test_create_brain(self, client: TestClient) -> None:
        """Test creating a new brain."""
        response = client.post(
            "/brain/create",
            json={"name": "test_brain", "owner_id": "user1"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_brain"
        assert data["owner_id"] == "user1"
        assert "id" in data

    def test_get_brain(self, client: TestClient) -> None:
        """Test getting brain details."""
        # Create brain first
        create_response = client.post(
            "/brain/create",
            json={"name": "get_test"},
        )
        brain_id = create_response.json()["id"]

        # Get brain
        response = client.get(f"/brain/{brain_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == brain_id
        assert data["name"] == "get_test"

    def test_get_nonexistent_brain(self, client: TestClient) -> None:
        """Test getting a nonexistent brain returns 404."""
        response = client.get("/brain/nonexistent-id")

        assert response.status_code == 404

    def test_get_brain_stats(self, client: TestClient) -> None:
        """Test getting brain statistics."""
        # Create brain
        create_response = client.post(
            "/brain/create",
            json={"name": "stats_test"},
        )
        brain_id = create_response.json()["id"]

        # Get stats
        response = client.get(f"/brain/{brain_id}/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["brain_id"] == brain_id
        assert "neuron_count" in data
        assert "synapse_count" in data
        assert "fiber_count" in data

    def test_delete_brain(self, client: TestClient) -> None:
        """Test deleting a brain."""
        # Create brain
        create_response = client.post(
            "/brain/create",
            json={"name": "delete_test"},
        )
        brain_id = create_response.json()["id"]

        # Delete brain
        response = client.delete(f"/brain/{brain_id}")

        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # Verify deleted
        get_response = client.get(f"/brain/{brain_id}")
        assert get_response.status_code == 404


class TestMemoryEndpoints:
    """Tests for memory encoding and querying endpoints."""

    @pytest.fixture
    def brain_id(self, client: TestClient) -> str:
        """Create a brain and return its ID."""
        response = client.post(
            "/brain/create",
            json={"name": "memory_test"},
        )
        return response.json()["id"]

    def test_encode_memory(self, client: TestClient, brain_id: str) -> None:
        """Test encoding a new memory."""
        response = client.post(
            "/memory/encode",
            json={"content": "Met Alice at the coffee shop"},
            headers={"X-Brain-ID": brain_id},
        )

        assert response.status_code == 200
        data = response.json()
        assert "fiber_id" in data
        assert data["neurons_created"] > 0

    def test_encode_memory_without_brain(self, client: TestClient) -> None:
        """Test encoding without brain ID returns error."""
        response = client.post(
            "/memory/encode",
            json={"content": "Test memory"},
        )

        assert response.status_code == 422  # Missing header

    def test_query_memory(self, client: TestClient, brain_id: str) -> None:
        """Test querying memories."""
        # Encode a memory first
        client.post(
            "/memory/encode",
            json={"content": "Alice suggested rate limiting"},
            headers={"X-Brain-ID": brain_id},
        )

        # Query
        response = client.post(
            "/memory/query",
            json={"query": "What did Alice suggest?"},
            headers={"X-Brain-ID": brain_id},
        )

        assert response.status_code == 200
        data = response.json()
        assert "confidence" in data
        assert "context" in data
        assert "latency_ms" in data

    def test_query_with_depth(self, client: TestClient, brain_id: str) -> None:
        """Test querying with specific depth level."""
        response = client.post(
            "/memory/query",
            json={"query": "What happened?", "depth": 0},
            headers={"X-Brain-ID": brain_id},
        )

        assert response.status_code == 200
        assert response.json()["depth_used"] == 0

    def test_query_with_subgraph(self, client: TestClient, brain_id: str) -> None:
        """Test querying with subgraph included."""
        # Encode memory
        client.post(
            "/memory/encode",
            json={"content": "Important meeting"},
            headers={"X-Brain-ID": brain_id},
        )

        response = client.post(
            "/memory/query",
            json={"query": "meeting", "include_subgraph": True},
            headers={"X-Brain-ID": brain_id},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["subgraph"] is not None
        assert "neuron_ids" in data["subgraph"]

    def test_list_neurons(self, client: TestClient, brain_id: str) -> None:
        """Test listing neurons."""
        # Encode memory to create neurons
        client.post(
            "/memory/encode",
            json={"content": "Test content"},
            headers={"X-Brain-ID": brain_id},
        )

        response = client.get(
            "/memory/neurons",
            headers={"X-Brain-ID": brain_id},
        )

        assert response.status_code == 200
        data = response.json()
        assert "neurons" in data
        assert "count" in data


class TestNeuronCRUD:
    """Tests for neuron CRUD endpoints (SharedStorage support)."""

    @pytest.fixture
    def brain_id(self, client: TestClient) -> str:
        """Create a brain and return its ID."""
        response = client.post(
            "/brain/create",
            json={"name": "neuron_crud_test"},
        )
        return response.json()["id"]

    def test_create_neuron(self, client: TestClient, brain_id: str) -> None:
        """Test creating a neuron directly."""
        response = client.post(
            "/memory/neurons",
            json={"type": "concept", "content": "Test concept"},
            headers={"X-Brain-ID": brain_id},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "concept"
        assert data["content"] == "Test concept"
        assert "id" in data

    def test_get_neuron(self, client: TestClient, brain_id: str) -> None:
        """Test getting a neuron by ID."""
        create_resp = client.post(
            "/memory/neurons",
            json={"type": "entity", "content": "Alice"},
            headers={"X-Brain-ID": brain_id},
        )
        neuron_id = create_resp.json()["id"]

        response = client.get(
            f"/memory/neurons/{neuron_id}",
            headers={"X-Brain-ID": brain_id},
        )

        assert response.status_code == 200
        assert response.json()["content"] == "Alice"

    def test_get_nonexistent_neuron(self, client: TestClient, brain_id: str) -> None:
        """Test getting a nonexistent neuron returns 404."""
        response = client.get(
            "/memory/neurons/nonexistent-id",
            headers={"X-Brain-ID": brain_id},
        )
        assert response.status_code == 404

    def test_update_neuron(self, client: TestClient, brain_id: str) -> None:
        """Test updating a neuron."""
        create_resp = client.post(
            "/memory/neurons",
            json={"type": "concept", "content": "Old content"},
            headers={"X-Brain-ID": brain_id},
        )
        neuron_id = create_resp.json()["id"]

        response = client.put(
            f"/memory/neurons/{neuron_id}",
            json={"content": "New content"},
            headers={"X-Brain-ID": brain_id},
        )

        assert response.status_code == 200
        assert response.json()["content"] == "New content"

    def test_delete_neuron(self, client: TestClient, brain_id: str) -> None:
        """Test deleting a neuron."""
        create_resp = client.post(
            "/memory/neurons",
            json={"type": "concept", "content": "To delete"},
            headers={"X-Brain-ID": brain_id},
        )
        neuron_id = create_resp.json()["id"]

        response = client.delete(
            f"/memory/neurons/{neuron_id}",
            headers={"X-Brain-ID": brain_id},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # Verify deleted
        get_resp = client.get(
            f"/memory/neurons/{neuron_id}",
            headers={"X-Brain-ID": brain_id},
        )
        assert get_resp.status_code == 404

    def test_create_neuron_invalid_type(self, client: TestClient, brain_id: str) -> None:
        """Test creating a neuron with invalid type returns 400."""
        response = client.post(
            "/memory/neurons",
            json={"type": "invalid_type", "content": "Test"},
            headers={"X-Brain-ID": brain_id},
        )
        assert response.status_code == 400


class TestSynapseCRUD:
    """Tests for synapse CRUD endpoints (SharedStorage support)."""

    @pytest.fixture
    def brain_with_neurons(self, client: TestClient) -> dict[str, str]:
        """Create a brain with two neurons."""
        brain_resp = client.post(
            "/brain/create",
            json={"name": "synapse_crud_test"},
        )
        brain_id = brain_resp.json()["id"]
        headers = {"X-Brain-ID": brain_id}

        n1 = client.post(
            "/memory/neurons",
            json={"type": "entity", "content": "Alice"},
            headers=headers,
        ).json()["id"]

        n2 = client.post(
            "/memory/neurons",
            json={"type": "concept", "content": "FastAPI"},
            headers=headers,
        ).json()["id"]

        return {"brain_id": brain_id, "neuron1": n1, "neuron2": n2}

    def test_create_synapse(self, client: TestClient, brain_with_neurons: dict[str, str]) -> None:
        """Test creating a synapse between neurons."""
        ids = brain_with_neurons
        response = client.post(
            "/memory/synapses",
            json={
                "source_id": ids["neuron1"],
                "target_id": ids["neuron2"],
                "type": "related_to",
                "weight": 0.8,
            },
            headers={"X-Brain-ID": ids["brain_id"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["weight"] == 0.8

    def test_get_synapse(self, client: TestClient, brain_with_neurons: dict[str, str]) -> None:
        """Test getting a synapse by ID."""
        ids = brain_with_neurons
        headers = {"X-Brain-ID": ids["brain_id"]}

        create_resp = client.post(
            "/memory/synapses",
            json={
                "source_id": ids["neuron1"],
                "target_id": ids["neuron2"],
                "type": "related_to",
            },
            headers=headers,
        )
        synapse_id = create_resp.json()["id"]

        response = client.get(f"/memory/synapses/{synapse_id}", headers=headers)

        assert response.status_code == 200
        assert response.json()["source_id"] == ids["neuron1"]

    def test_list_synapses(self, client: TestClient, brain_with_neurons: dict[str, str]) -> None:
        """Test listing synapses."""
        ids = brain_with_neurons
        headers = {"X-Brain-ID": ids["brain_id"]}

        client.post(
            "/memory/synapses",
            json={
                "source_id": ids["neuron1"],
                "target_id": ids["neuron2"],
                "type": "related_to",
            },
            headers=headers,
        )

        response = client.get(
            "/memory/synapses",
            params={"source_id": ids["neuron1"]},
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1

    def test_delete_synapse(self, client: TestClient, brain_with_neurons: dict[str, str]) -> None:
        """Test deleting a synapse."""
        ids = brain_with_neurons
        headers = {"X-Brain-ID": ids["brain_id"]}

        create_resp = client.post(
            "/memory/synapses",
            json={
                "source_id": ids["neuron1"],
                "target_id": ids["neuron2"],
                "type": "related_to",
            },
            headers=headers,
        )
        synapse_id = create_resp.json()["id"]

        response = client.delete(f"/memory/synapses/{synapse_id}", headers=headers)

        assert response.status_code == 200
        assert response.json()["status"] == "deleted"


class TestExportImport:
    """Tests for brain export/import functionality."""

    def test_export_brain(self, client: TestClient) -> None:
        """Test exporting a brain."""
        # Create brain with some data
        create_response = client.post(
            "/brain/create",
            json={"name": "export_test"},
        )
        brain_id = create_response.json()["id"]

        # Add a memory
        client.post(
            "/memory/encode",
            json={"content": "Memory to export"},
            headers={"X-Brain-ID": brain_id},
        )

        # Export
        response = client.get(f"/brain/{brain_id}/export")

        assert response.status_code == 200
        data = response.json()
        assert data["brain_id"] == brain_id
        assert "neurons" in data
        assert "synapses" in data
        assert "fibers" in data
        assert "version" in data

    def test_import_brain(self, client: TestClient) -> None:
        """Test importing a brain from snapshot."""
        # Create and export a brain
        create_response = client.post(
            "/brain/create",
            json={"name": "import_source"},
        )
        brain_id = create_response.json()["id"]

        client.post(
            "/memory/encode",
            json={"content": "Memory to import"},
            headers={"X-Brain-ID": brain_id},
        )

        export_response = client.get(f"/brain/{brain_id}/export")
        snapshot = export_response.json()

        # Import to new brain
        import_response = client.post(
            "/brain/new_brain/import",
            json=snapshot,
        )

        assert import_response.status_code == 200
        data = import_response.json()
        assert data["neuron_count"] > 0
