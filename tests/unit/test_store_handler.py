"""Tests for MCP store handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.mcp.store_handler import StoreHandler
from neural_memory.storage.memory_store import InMemoryStorage


class MockStoreServer(StoreHandler):
    """Mock server for testing StoreHandler mixin."""

    def __init__(self, storage: InMemoryStorage, config: MagicMock) -> None:
        self._storage = storage
        self.config = config

    async def get_storage(self) -> InMemoryStorage:
        return self._storage


@pytest.fixture
def brain_config() -> BrainConfig:
    return BrainConfig()


@pytest.fixture
def brain(brain_config: BrainConfig) -> Brain:
    return Brain.create(name="test", config=brain_config)


@pytest.fixture
async def storage(brain: Brain) -> InMemoryStorage:
    store = InMemoryStorage()
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


@pytest.fixture
def server(storage: InMemoryStorage) -> MockStoreServer:
    config = MagicMock()
    return MockStoreServer(storage, config)


class TestStoreHandler:
    """Tests for StoreHandler mixin."""

    async def test_unknown_action(self, server: MockStoreServer) -> None:
        """Unknown action returns error."""
        result = await server._store({"action": "invalid"})
        assert "error" in result

    async def test_default_action_is_browse(self, server: MockStoreServer) -> None:
        """No action defaults to browse."""
        with patch("neural_memory.mcp.store_handler._registry") as mock_registry:
            mock_registry.fetch_index = AsyncMock(return_value=[])
            mock_registry.filter_index = MagicMock(return_value=[])
            result = await server._store({})
            assert "brains" in result
            assert result["total"] == 0

    async def test_browse_empty_registry(self, server: MockStoreServer) -> None:
        """Browse returns empty list when registry is empty."""
        with patch("neural_memory.mcp.store_handler._registry") as mock_registry:
            mock_registry.fetch_index = AsyncMock(return_value=[])
            mock_registry.filter_index = MagicMock(return_value=[])
            result = await server._store({"action": "browse"})
            assert result["brains"] == []
            assert result["total"] == 0
            assert "message" in result

    async def test_browse_with_results(self, server: MockStoreServer) -> None:
        """Browse returns formatted results."""
        manifests = [
            {
                "name": "python-tips",
                "display_name": "Python Tips",
                "description": "Great tips for Python developers",
                "author": "testuser",
                "category": "programming",
                "stats": {"neuron_count": 100},
                "size_tier": "micro",
                "rating_avg": 4.5,
                "tags": ["python", "tips"],
            }
        ]
        with patch("neural_memory.mcp.store_handler._registry") as mock_registry:
            mock_registry.fetch_index = AsyncMock(return_value=manifests)
            mock_registry.filter_index = MagicMock(return_value=manifests)
            result = await server._store({"action": "browse"})
            assert result["total"] == 1
            brain = result["brains"][0]
            assert brain["name"] == "python-tips"
            assert brain["display_name"] == "Python Tips"
            assert brain["neurons"] == 100

    async def test_browse_limit_capped_at_50(self, server: MockStoreServer) -> None:
        """Browse limit is capped at 50."""
        with patch("neural_memory.mcp.store_handler._registry") as mock_registry:
            mock_registry.fetch_index = AsyncMock(return_value=[])
            mock_registry.filter_index = MagicMock(return_value=[])
            await server._store({"action": "browse", "limit": 999})
            # Check that filter_index was called with limit=50
            call_kwargs = mock_registry.filter_index.call_args
            assert call_kwargs[1]["limit"] == 50 or call_kwargs.kwargs.get("limit") == 50

    async def test_preview_requires_brain_name(self, server: MockStoreServer) -> None:
        """Preview without brain_name returns error."""
        result = await server._store({"action": "preview"})
        assert "error" in result

    async def test_preview_not_found(self, server: MockStoreServer) -> None:
        """Preview of nonexistent brain returns error."""
        with patch("neural_memory.mcp.store_handler._registry") as mock_registry:
            mock_registry.fetch_brain = AsyncMock(return_value=None)
            result = await server._store({"action": "preview", "brain_name": "nonexistent"})
            assert "error" in result

    async def test_preview_success(self, server: MockStoreServer) -> None:
        """Preview returns brain details."""
        package_data = {
            "nmem_brain_package": "1.0",
            "manifest": {
                "display_name": "Test Brain",
                "description": "A test brain",
                "author": "tester",
                "version": "1.0.0",
                "license": "CC-BY-4.0",
                "stats": {"neuron_count": 5},
            },
            "snapshot": {
                "neurons": [{"type": "fact", "content": "Sample neuron"}],
            },
        }
        with (
            patch("neural_memory.mcp.store_handler._registry") as mock_registry,
            patch("neural_memory.mcp.store_handler.preview_brain_package") as mock_preview,
        ):
            mock_registry.fetch_brain = AsyncMock(return_value=package_data)
            mock_preview.return_value = {
                "manifest": package_data["manifest"],
                "scan_result": {"safe": True, "risk_level": "low", "finding_count": 0},
                "neuron_type_breakdown": {"fact": 1},
                "top_tags": [],
                "sample_neurons": [{"type": "fact", "content": "Sample neuron"}],
            }
            result = await server._store({"action": "preview", "brain_name": "test-brain"})
            assert result["name"] == "Test Brain"
            assert result["scan"]["safe"] is True

    async def test_import_requires_brain_name(self, server: MockStoreServer) -> None:
        """Import without brain_name returns error."""
        result = await server._store({"action": "import"})
        assert "error" in result

    async def test_import_not_found(self, server: MockStoreServer) -> None:
        """Import of nonexistent brain returns error."""
        with patch("neural_memory.mcp.store_handler._registry") as mock_registry:
            mock_registry.fetch_brain = AsyncMock(return_value=None)
            result = await server._store({"action": "import", "brain_name": "nonexistent"})
            assert "error" in result

    async def test_import_invalid_package(self, server: MockStoreServer) -> None:
        """Import of invalid package returns error."""
        with (
            patch("neural_memory.mcp.store_handler._registry") as mock_registry,
            patch("neural_memory.mcp.store_handler.validate_brain_package") as mock_validate,
        ):
            mock_registry.fetch_brain = AsyncMock(return_value={"manifest": {}})
            mock_validate.return_value = (False, ["Missing manifest field"])
            result = await server._store({"action": "import", "brain_name": "bad-brain"})
            assert "error" in result
            assert "Invalid" in result["error"]

    async def test_import_high_risk_blocked(self, server: MockStoreServer) -> None:
        """Import is blocked if security scan finds high risk."""
        scan_result = MagicMock()
        scan_result.risk_level = "high"
        scan_result.safe = False
        scan_result.findings = [MagicMock(description="Prompt injection detected")]

        with (
            patch("neural_memory.mcp.store_handler._registry") as mock_registry,
            patch("neural_memory.mcp.store_handler.validate_brain_package") as mock_validate,
            patch("neural_memory.mcp.store_handler.scan_brain_package") as mock_scan,
        ):
            mock_registry.fetch_brain = AsyncMock(return_value={"manifest": {}})
            mock_validate.return_value = (True, [])
            mock_scan.return_value = scan_result
            result = await server._store({"action": "import", "brain_name": "risky-brain"})
            assert "error" in result
            assert "security" in result["error"].lower() or "blocked" in result["error"].lower()

    async def test_import_success(self, server: MockStoreServer) -> None:
        """Successful import returns brain details."""
        scan_result = MagicMock()
        scan_result.risk_level = "low"
        scan_result.safe = True
        scan_result.findings = []

        package_data = {
            "manifest": {"name": "test-brain", "display_name": "Test Brain"},
            "snapshot": {
                "neurons": [{"type": "fact", "content": "hello"}],
                "synapses": [],
                "fibers": [],
                "config": {},
                "metadata": {},
                "version": "1",
            },
        }

        with (
            patch("neural_memory.mcp.store_handler._registry") as mock_registry,
            patch("neural_memory.mcp.store_handler.validate_brain_package") as mock_validate,
            patch("neural_memory.mcp.store_handler.scan_brain_package") as mock_scan,
        ):
            mock_registry.fetch_brain = AsyncMock(return_value=package_data)
            mock_validate.return_value = (True, [])
            mock_scan.return_value = scan_result
            # Mock import_brain on storage to avoid needing full neuron IDs
            server._storage.import_brain = AsyncMock(return_value="new-brain-id")  # type: ignore[method-assign]
            result = await server._store({"action": "import", "brain_name": "test-brain"})
            assert result["status"] == "imported"
            assert result["brain_name"] == "Test Brain"
            assert result["neurons_imported"] == 1

    async def test_export_requires_display_name(self, server: MockStoreServer) -> None:
        """Export without display_name returns error."""
        result = await server._store({"action": "export"})
        assert "error" in result

    async def test_export_success(self, server: MockStoreServer) -> None:
        """Successful export returns manifest info."""
        with patch("neural_memory.mcp.store_handler.create_brain_package") as mock_create:
            mock_create.return_value = {
                "manifest": {
                    "name": "my-brain",
                    "stats": {"neuron_count": 10, "synapse_count": 5, "fiber_count": 2},
                    "size_bytes": 5000,
                    "size_tier": "micro",
                    "content_hash": "sha256:abc",
                },
            }
            result = await server._store(
                {
                    "action": "export",
                    "display_name": "My Brain",
                    "description": "A test brain",
                    "author": "tester",
                }
            )
            assert result["status"] == "exported"
            assert result["display_name"] == "My Brain"
            assert "hint" in result

    async def test_export_with_output_path(self, server: MockStoreServer, tmp_path: object) -> None:
        """Export with output_path saves file to disk."""
        import tempfile
        from pathlib import Path

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("neural_memory.mcp.store_handler.create_brain_package") as mock_create,
        ):
            mock_create.return_value = {
                "manifest": {
                    "name": "my-brain",
                    "stats": {"neuron_count": 10},
                    "size_bytes": 5000,
                    "size_tier": "micro",
                },
            }
            out_file = str(Path(tmpdir) / "test.brain")
            result = await server._store(
                {
                    "action": "export",
                    "display_name": "My Brain",
                    "output_path": out_file,
                }
            )
            assert result["status"] == "exported"
            assert result["path"] == str(Path(out_file).resolve())
            assert Path(out_file).exists()

    async def test_exception_handling(self, server: MockStoreServer) -> None:
        """Exceptions in handlers are caught and logged."""
        with patch("neural_memory.mcp.store_handler._registry") as mock_registry:
            mock_registry.fetch_index = AsyncMock(side_effect=RuntimeError("Network error"))
            result = await server._store({"action": "browse"})
            assert "error" in result
