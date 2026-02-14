"""Tests for storage/shared_store.py — HTTP-based remote storage client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.storage.shared_store import SharedStorage
from neural_memory.storage.shared_store_collections import SharedStorageError

# ─────────── Init / properties ───────────


class TestSharedStorageInit:
    """Tests for SharedStorage constructor and properties."""

    def test_basic_init(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        assert storage.server_url == "http://localhost:8000"
        assert storage.brain_id == "brain-1"
        assert storage.is_connected is False

    def test_trailing_slash_stripped(self) -> None:
        storage = SharedStorage("http://localhost:8000/", "brain-1")
        assert storage.server_url == "http://localhost:8000"

    def test_set_brain(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        storage.set_brain("brain-2")
        assert storage.brain_id == "brain-2"

    def test_is_connected_false_when_no_session(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        storage._connected = True
        storage._session = None
        assert storage.is_connected is False


# ─────────── connect / disconnect ───────────


class TestSharedStorageConnection:
    """Tests for connect() and disconnect()."""

    @pytest.mark.asyncio
    async def test_connect_creates_session(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = AsyncMock()
            await storage.connect()

        assert storage.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_with_api_key(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1", api_key="secret-key")

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value = AsyncMock()
            await storage.connect()
            call_kwargs = mock_cls.call_args
            headers = call_kwargs.kwargs.get("headers", {})
            assert headers["Authorization"] == "Bearer secret-key"

    @pytest.mark.asyncio
    async def test_connect_noop_if_session_exists(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        storage._session = AsyncMock()
        storage._connected = True

        await storage.connect()
        # Should not create a new session

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        mock_session = AsyncMock()
        storage._session = mock_session
        storage._connected = True

        await storage.disconnect()

        mock_session.close.assert_awaited_once()
        assert storage._session is None
        assert storage.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_noop_when_no_session(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        await storage.disconnect()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        with patch("aiohttp.ClientSession") as mock_cls:
            mock_session = AsyncMock()
            mock_cls.return_value = mock_session

            async with SharedStorage("http://localhost:8000", "brain-1") as storage:
                assert storage.is_connected

            mock_session.close.assert_awaited_once()


# ─────────── _get_headers ───────────


class TestSharedStorageHeaders:
    """Tests for _get_headers()."""

    def test_headers_without_api_key(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        headers = storage._get_headers()
        assert headers["X-Brain-ID"] == "brain-1"
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_headers_with_api_key(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1", api_key="key-123")
        headers = storage._get_headers()
        assert headers["Authorization"] == "Bearer key-123"


# ─────────── _request ───────────


class TestSharedStorageRequest:
    """Tests for _request() base method."""

    @pytest.mark.asyncio
    async def test_request_success(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ok": True})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.request = MagicMock(return_value=mock_response)
        storage._session = mock_session
        storage._connected = True

        result = await storage._request("GET", "/test")
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_request_auto_connects(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ok": True})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_session = AsyncMock()
            mock_session.request = MagicMock(return_value=mock_response)
            mock_cls.return_value = mock_session
            result = await storage._request("GET", "/test")

        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_request_raises_on_4xx(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")

        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not found")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.request = MagicMock(return_value=mock_response)
        storage._session = mock_session
        storage._connected = True

        with pytest.raises(SharedStorageError) as exc_info:
            await storage._request("GET", "/test")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_request_raises_on_5xx(self) -> None:
        storage = SharedStorage("http://localhost:8000", "brain-1")

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.request = MagicMock(return_value=mock_response)
        storage._session = mock_session
        storage._connected = True

        with pytest.raises(SharedStorageError) as exc_info:
            await storage._request("GET", "/test")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_request_wraps_client_error(self) -> None:
        import aiohttp

        storage = SharedStorage("http://localhost:8000", "brain-1")

        mock_session = AsyncMock()
        mock_session.request = MagicMock(side_effect=aiohttp.ClientError("connection failed"))
        storage._session = mock_session
        storage._connected = True

        with pytest.raises(SharedStorageError, match="Failed to connect"):
            await storage._request("GET", "/test")


# ─────────── Neuron operations ───────────


class TestSharedStorageNeurons:
    """Tests for neuron CRUD operations."""

    def _make_storage(self) -> SharedStorage:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        storage._request = AsyncMock()  # type: ignore[method-assign]
        return storage

    @pytest.mark.asyncio
    async def test_add_neuron(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {"id": "n-1"}

        from neural_memory.core.neuron import Neuron, NeuronType
        from neural_memory.utils.timeutils import utcnow

        neuron = Neuron(id="n-1", type=NeuronType.CONCEPT, content="test", created_at=utcnow())
        result = await storage.add_neuron(neuron)

        assert result == "n-1"
        storage._request.assert_awaited_once()
        call_args = storage._request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "/memory/neurons"

    @pytest.mark.asyncio
    async def test_get_neuron_found(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {
            "id": "n-1",
            "type": "concept",
            "content": "test",
            "created_at": "2026-01-01T00:00:00",
        }

        result = await storage.get_neuron("n-1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_neuron_not_found(self) -> None:
        storage = self._make_storage()
        storage._request.side_effect = SharedStorageError("Not found", status_code=404)

        result = await storage.get_neuron("n-missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_neuron_reraises_non_404(self) -> None:
        storage = self._make_storage()
        storage._request.side_effect = SharedStorageError("Error", status_code=500)

        with pytest.raises(SharedStorageError):
            await storage.get_neuron("n-1")

    @pytest.mark.asyncio
    async def test_find_neurons(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {
            "neurons": [
                {
                    "id": "n-1",
                    "type": "concept",
                    "content": "a",
                    "created_at": "2026-01-01T00:00:00",
                },
            ]
        }

        result = await storage.find_neurons(limit=10)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_find_neurons_with_filters(self) -> None:
        from neural_memory.core.neuron import NeuronType

        storage = self._make_storage()
        storage._request.return_value = {"neurons": []}

        await storage.find_neurons(
            type=NeuronType.CONCEPT,
            content_contains="test",
            limit=5,
        )

        call_kwargs = storage._request.call_args.kwargs
        params = call_kwargs["params"]
        assert params["type"] == "concept"
        assert params["content_contains"] == "test"

    @pytest.mark.asyncio
    async def test_update_neuron(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {}

        from neural_memory.core.neuron import Neuron, NeuronType
        from neural_memory.utils.timeutils import utcnow

        neuron = Neuron(id="n-1", type=NeuronType.CONCEPT, content="updated", created_at=utcnow())
        await storage.update_neuron(neuron)

        call_args = storage._request.call_args
        assert call_args[0][0] == "PUT"
        assert "/n-1" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_delete_neuron_success(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {}

        result = await storage.delete_neuron("n-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_neuron_not_found(self) -> None:
        storage = self._make_storage()
        storage._request.side_effect = SharedStorageError("Not found", status_code=404)

        result = await storage.delete_neuron("n-missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_suggest_neurons(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {"suggestions": [{"content": "test"}]}

        result = await storage.suggest_neurons("te")
        assert len(result) == 1


# ─────────── Neuron state ───────────


class TestSharedStorageNeuronState:
    """Tests for neuron state operations."""

    def _make_storage(self) -> SharedStorage:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        storage._request = AsyncMock()  # type: ignore[method-assign]
        return storage

    @pytest.mark.asyncio
    async def test_get_neuron_state_found(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {
            "neuron_id": "n-1",
            "activation_level": 0.5,
            "access_frequency": 10,
        }

        result = await storage.get_neuron_state("n-1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_neuron_state_not_found(self) -> None:
        storage = self._make_storage()
        storage._request.side_effect = SharedStorageError("Not found", status_code=404)

        result = await storage.get_neuron_state("n-missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_neuron_state(self) -> None:
        from neural_memory.core.neuron import NeuronState

        storage = self._make_storage()
        storage._request.return_value = {}

        state = NeuronState(neuron_id="n-1")
        await storage.update_neuron_state(state)

        call_args = storage._request.call_args
        assert call_args[0][0] == "PUT"
        assert "/state" in call_args[0][1]


# ─────────── Synapse operations ───────────


class TestSharedStorageSynapses:
    """Tests for synapse CRUD operations."""

    def _make_storage(self) -> SharedStorage:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        storage._request = AsyncMock()  # type: ignore[method-assign]
        return storage

    @pytest.mark.asyncio
    async def test_add_synapse(self) -> None:
        from neural_memory.core.synapse import Direction, Synapse, SynapseType
        from neural_memory.utils.timeutils import utcnow

        storage = self._make_storage()
        storage._request.return_value = {"id": "s-1"}

        synapse = Synapse(
            id="s-1",
            source_id="n-1",
            target_id="n-2",
            type=SynapseType.CAUSED_BY,
            weight=0.5,
            direction=Direction.UNIDIRECTIONAL,
            created_at=utcnow(),
        )
        result = await storage.add_synapse(synapse)
        assert result == "s-1"

    @pytest.mark.asyncio
    async def test_get_synapse_found(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {
            "id": "s-1",
            "source_id": "n-1",
            "target_id": "n-2",
            "type": "caused_by",
            "weight": 0.5,
            "direction": "uni",
            "created_at": "2026-01-01T00:00:00",
        }

        result = await storage.get_synapse("s-1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_synapse_not_found(self) -> None:
        storage = self._make_storage()
        storage._request.side_effect = SharedStorageError("Not found", status_code=404)

        result = await storage.get_synapse("s-missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_synapses(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {
            "synapses": [
                {
                    "id": "s-1",
                    "source_id": "n-1",
                    "target_id": "n-2",
                    "type": "caused_by",
                    "weight": 0.5,
                    "direction": "uni",
                    "created_at": "2026-01-01T00:00:00",
                },
            ]
        }

        result = await storage.get_synapses(source_id="n-1")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_delete_synapse_success(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {}

        assert await storage.delete_synapse("s-1") is True

    @pytest.mark.asyncio
    async def test_delete_synapse_not_found(self) -> None:
        storage = self._make_storage()
        storage._request.side_effect = SharedStorageError("Not found", status_code=404)

        assert await storage.delete_synapse("s-missing") is False


# ─────────── Graph traversal ───────────


class TestSharedStorageGraph:
    """Tests for graph traversal operations."""

    def _make_storage(self) -> SharedStorage:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        storage._request = AsyncMock()  # type: ignore[method-assign]
        return storage

    @pytest.mark.asyncio
    async def test_get_neighbors(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {
            "neighbors": [
                {
                    "neuron": {
                        "id": "n-2",
                        "type": "concept",
                        "content": "b",
                        "created_at": "2026-01-01T00:00:00",
                    },
                    "synapse": {
                        "id": "s-1",
                        "source_id": "n-1",
                        "target_id": "n-2",
                        "type": "caused_by",
                        "weight": 0.5,
                        "direction": "uni",
                        "created_at": "2026-01-01T00:00:00",
                    },
                },
            ]
        }

        result = await storage.get_neighbors("n-1")
        assert len(result) == 1
        neuron, synapse = result[0]
        assert neuron.id == "n-2"

    @pytest.mark.asyncio
    async def test_get_neighbors_empty(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {"neighbors": []}

        result = await storage.get_neighbors("n-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_path_found(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {
            "path": [
                {
                    "neuron": {
                        "id": "n-2",
                        "type": "concept",
                        "content": "b",
                        "created_at": "2026-01-01T00:00:00",
                    },
                    "synapse": {
                        "id": "s-1",
                        "source_id": "n-1",
                        "target_id": "n-2",
                        "type": "caused_by",
                        "weight": 0.5,
                        "direction": "uni",
                        "created_at": "2026-01-01T00:00:00",
                    },
                },
            ]
        }

        result = await storage.get_path("n-1", "n-2")
        assert result is not None
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_path_not_found(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {"path": []}

        result = await storage.get_path("n-1", "n-99")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_path_404(self) -> None:
        storage = self._make_storage()
        storage._request.side_effect = SharedStorageError("Not found", status_code=404)

        result = await storage.get_path("n-missing", "n-2")
        assert result is None


# ─────────── Statistics & cleanup ───────────


class TestSharedStorageStats:
    """Tests for statistics and cleanup operations."""

    def _make_storage(self) -> SharedStorage:
        storage = SharedStorage("http://localhost:8000", "brain-1")
        storage._request = AsyncMock()  # type: ignore[method-assign]
        return storage

    @pytest.mark.asyncio
    async def test_get_stats(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {
            "neuron_count": 10,
            "synapse_count": 20,
            "fiber_count": 5,
        }

        result = await storage.get_stats("brain-1")
        assert result["neuron_count"] == 10
        assert result["synapse_count"] == 20
        assert result["fiber_count"] == 5

    @pytest.mark.asyncio
    async def test_get_enhanced_stats(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {
            "neuron_count": 10,
            "extra_metric": 42,
        }

        result = await storage.get_enhanced_stats("brain-1")
        assert result["extra_metric"] == 42

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        storage = self._make_storage()
        storage._request.return_value = {}

        await storage.clear("brain-1")

        storage._request.assert_awaited_once_with("DELETE", "/brain/brain-1")


# ─────────── SharedStorageError ───────────


class TestSharedStorageError:
    """Tests for SharedStorageError exception."""

    def test_error_with_status_code(self) -> None:
        err = SharedStorageError("Not found", status_code=404)
        assert str(err) == "Not found"
        assert err.status_code == 404

    def test_error_without_status_code(self) -> None:
        err = SharedStorageError("Connection failed")
        assert err.status_code is None
