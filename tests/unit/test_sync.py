"""Tests for sync components."""

import asyncio
from datetime import datetime, timedelta

import pytest

from neural_memory.server.routes.sync import (
    ConnectedClient,
    SyncEvent,
    SyncEventType,
    SyncManager,
)
from neural_memory.sync.client import SyncClient, SyncClientState


class TestSyncEventType:
    """Tests for SyncEventType enum."""

    def test_event_types(self) -> None:
        """Test event type values."""
        assert SyncEventType.CONNECTED == "connected"
        assert SyncEventType.NEURON_CREATED == "neuron_created"
        assert SyncEventType.MEMORY_ENCODED == "memory_encoded"
        assert SyncEventType.ERROR == "error"


class TestSyncEvent:
    """Tests for SyncEvent dataclass."""

    def test_create_event(self) -> None:
        """Test creating a sync event."""
        event = SyncEvent(
            type=SyncEventType.NEURON_CREATED,
            brain_id="brain-123",
            data={"id": "neuron-1", "content": "test"},
        )

        assert event.type == SyncEventType.NEURON_CREATED
        assert event.brain_id == "brain-123"
        assert event.data["id"] == "neuron-1"
        assert event.timestamp is not None

    def test_event_to_dict(self) -> None:
        """Test converting event to dict."""
        timestamp = datetime(2026, 2, 5, 12, 0, 0)
        event = SyncEvent(
            type=SyncEventType.NEURON_CREATED,
            brain_id="brain-123",
            timestamp=timestamp,
            data={"id": "neuron-1"},
            source_client_id="client-1",
        )

        data = event.to_dict()

        assert data["type"] == "neuron_created"
        assert data["brain_id"] == "brain-123"
        assert data["timestamp"] == "2026-02-05T12:00:00"
        assert data["data"]["id"] == "neuron-1"
        assert data["source_client_id"] == "client-1"

    def test_event_to_json(self) -> None:
        """Test converting event to JSON."""
        event = SyncEvent(
            type=SyncEventType.CONNECTED,
            brain_id="*",
            data={"client_id": "test"},
        )

        json_str = event.to_json()

        assert '"type": "connected"' in json_str
        assert '"brain_id": "*"' in json_str

    def test_event_from_dict(self) -> None:
        """Test creating event from dict."""
        data = {
            "type": "memory_encoded",
            "brain_id": "brain-123",
            "timestamp": "2026-02-05T12:00:00",
            "data": {"fiber_id": "fiber-1"},
            "source_client_id": "client-1",
        }

        event = SyncEvent.from_dict(data)

        assert event.type == SyncEventType.MEMORY_ENCODED
        assert event.brain_id == "brain-123"
        assert event.timestamp == datetime(2026, 2, 5, 12, 0, 0)
        assert event.data["fiber_id"] == "fiber-1"
        assert event.source_client_id == "client-1"


class TestSyncManager:
    """Tests for SyncManager."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        SyncManager.reset()

    def test_singleton(self) -> None:
        """Test that SyncManager is a singleton."""
        manager1 = SyncManager.instance()
        manager2 = SyncManager.instance()

        assert manager1 is manager2

    def test_reset(self) -> None:
        """Test resetting the singleton."""
        manager1 = SyncManager.instance()
        SyncManager.reset()
        manager2 = SyncManager.instance()

        assert manager1 is not manager2

    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe(self) -> None:
        """Test subscribing and unsubscribing clients."""

        class MockWebSocket:
            async def send_text(self, data: str) -> None:
                pass

        manager = SyncManager.instance()
        ws = MockWebSocket()

        # Connect client
        client = await manager.connect("client-1", ws)  # type: ignore
        assert client.client_id == "client-1"

        # Subscribe to brain
        result = await manager.subscribe("client-1", "brain-123")
        assert result is True
        assert "brain-123" in client.brain_ids

        # Unsubscribe from brain
        result = await manager.unsubscribe("client-1", "brain-123")
        assert result is True
        assert "brain-123" not in client.brain_ids

        # Disconnect client
        await manager.disconnect("client-1")

    @pytest.mark.asyncio
    async def test_broadcast(self) -> None:
        """Test broadcasting events to subscribed clients."""

        class MockWebSocket:
            def __init__(self):
                self.messages = []

            async def send_text(self, data: str) -> None:
                self.messages.append(data)

        manager = SyncManager.instance()
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        # Connect and subscribe clients
        await manager.connect("client-1", ws1)  # type: ignore
        await manager.connect("client-2", ws2)  # type: ignore
        await manager.subscribe("client-1", "brain-123")
        await manager.subscribe("client-2", "brain-123")

        # Broadcast event
        event = SyncEvent(
            type=SyncEventType.NEURON_CREATED,
            brain_id="brain-123",
            data={"id": "neuron-1"},
            source_client_id="client-1",
        )

        sent = await manager.broadcast(event, exclude_client="client-1")

        # Should only be sent to client-2
        assert sent == 1
        assert len(ws2.messages) == 1
        assert len(ws1.messages) == 0  # Excluded

        await manager.disconnect("client-1")
        await manager.disconnect("client-2")

    @pytest.mark.asyncio
    async def test_get_recent_events(self) -> None:
        """Test getting recent events."""
        manager = SyncManager.instance()

        # Broadcast some events (need a client to store history)

        class MockWebSocket:
            async def send_text(self, data: str) -> None:
                pass

        await manager.connect("client-1", MockWebSocket())  # type: ignore
        await manager.subscribe("client-1", "brain-123")

        # Broadcast events
        for i in range(5):
            event = SyncEvent(
                type=SyncEventType.NEURON_CREATED,
                brain_id="brain-123",
                data={"index": i},
            )
            await manager.broadcast(event)

        # Get history
        events = await manager.get_recent_events("brain-123")

        assert len(events) == 5
        assert events[0].data["index"] == 0
        assert events[4].data["index"] == 4

        await manager.disconnect("client-1")

    @pytest.mark.asyncio
    async def test_get_stats(self) -> None:
        """Test getting stats."""

        class MockWebSocket:
            async def send_text(self, data: str) -> None:
                pass

        manager = SyncManager.instance()

        stats = manager.get_stats()
        assert stats["connected_clients"] == 0

        await manager.connect("client-1", MockWebSocket())  # type: ignore
        await manager.subscribe("client-1", "brain-123")

        stats = manager.get_stats()
        assert stats["connected_clients"] == 1
        assert stats["brain_subscriptions"]["brain-123"] == 1

        await manager.disconnect("client-1")


class TestSyncClient:
    """Tests for SyncClient."""

    def test_create_client(self) -> None:
        """Test creating a sync client."""
        client = SyncClient("http://localhost:8000")

        assert client.client_id.startswith("client-")
        assert client.state == SyncClientState.DISCONNECTED
        assert client.is_connected is False

    def test_create_client_with_options(self) -> None:
        """Test creating client with options."""
        client = SyncClient(
            "http://localhost:8000",
            client_id="my-client",
            auto_reconnect=False,
            reconnect_delay=2.0,
            max_reconnect_attempts=5,
        )

        assert client.client_id == "my-client"
        assert client._auto_reconnect is False
        assert client._reconnect_delay == 2.0
        assert client._max_reconnect_attempts == 5

    def test_url_normalization(self) -> None:
        """Test that URLs are normalized correctly."""
        # HTTP becomes WS
        client1 = SyncClient("http://localhost:8000")
        assert client1._server_url == "ws://localhost:8000/sync/ws"

        # HTTPS becomes WSS
        client2 = SyncClient("https://server.com")
        assert client2._server_url == "wss://server.com/sync/ws"

        # Already has path
        client3 = SyncClient("ws://localhost:8000/sync/ws")
        assert client3._server_url == "ws://localhost:8000/sync/ws"

    def test_register_handler(self) -> None:
        """Test registering event handlers."""
        client = SyncClient("http://localhost:8000")

        async def handler(event):
            pass

        client.on("neuron_created", handler)

        assert "neuron_created" in client._handlers
        assert handler in client._handlers["neuron_created"]

    def test_unregister_handler(self) -> None:
        """Test unregistering event handlers."""
        client = SyncClient("http://localhost:8000")

        async def handler(event):
            pass

        client.on("neuron_created", handler)
        client.off("neuron_created", handler)

        assert len(client._handlers.get("neuron_created", [])) == 0

    def test_unregister_all_handlers(self) -> None:
        """Test unregistering all handlers for an event type."""
        client = SyncClient("http://localhost:8000")

        async def handler1(event):
            pass

        async def handler2(event):
            pass

        client.on("neuron_created", handler1)
        client.on("neuron_created", handler2)
        client.off("neuron_created")

        assert "neuron_created" not in client._handlers

    def test_subscribed_brains(self) -> None:
        """Test subscribed brains property."""
        client = SyncClient("http://localhost:8000")

        # Add brain to subscribed set (normally done via subscribe())
        client._subscribed_brains.add("brain-1")
        client._subscribed_brains.add("brain-2")

        brains = client.subscribed_brains

        assert isinstance(brains, frozenset)
        assert "brain-1" in brains
        assert "brain-2" in brains
