"""Tests for sync/client.py — WebSocket synchronization client."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.sync.client import SyncClient, SyncClientState, SyncEvent

# ─────────── SyncEvent tests ───────────


class TestSyncEventFromDict:
    """Tests for SyncEvent.from_dict()."""

    def test_valid_event(self) -> None:
        data = {
            "type": "neuron_created",
            "brain_id": "brain-1",
            "timestamp": "2026-01-15T10:00:00",
            "data": {"id": "n-1"},
            "source_client_id": "client-x",
        }
        event = SyncEvent.from_dict(data)
        assert event is not None
        assert event.type == "neuron_created"
        assert event.brain_id == "brain-1"
        assert event.timestamp == datetime(2026, 1, 15, 10, 0, 0)
        assert event.data == {"id": "n-1"}
        assert event.source_client_id == "client-x"

    def test_missing_type_returns_none(self) -> None:
        data = {"brain_id": "brain-1"}
        assert SyncEvent.from_dict(data) is None

    def test_missing_brain_id_returns_none(self) -> None:
        data = {"type": "neuron_created"}
        assert SyncEvent.from_dict(data) is None

    def test_empty_dict_returns_none(self) -> None:
        assert SyncEvent.from_dict({}) is None

    def test_missing_timestamp_uses_utcnow(self) -> None:
        data = {"type": "test", "brain_id": "b1"}
        event = SyncEvent.from_dict(data)
        assert event is not None
        assert isinstance(event.timestamp, datetime)

    def test_missing_data_defaults_empty_dict(self) -> None:
        data = {"type": "test", "brain_id": "b1"}
        event = SyncEvent.from_dict(data)
        assert event is not None
        assert event.data == {}

    def test_missing_source_client_id_defaults_none(self) -> None:
        data = {"type": "test", "brain_id": "b1"}
        event = SyncEvent.from_dict(data)
        assert event is not None
        assert event.source_client_id is None


# ─────────── SyncClient init / properties ───────────


class TestSyncClientInit:
    """Tests for SyncClient constructor and URL handling."""

    def test_invalid_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid WebSocket URL"):
            SyncClient("ftp://localhost:8000")

    def test_ws_url_stays_unchanged(self) -> None:
        client = SyncClient("ws://localhost:8000/sync/ws")
        assert client._server_url == "ws://localhost:8000/sync/ws"

    def test_wss_url_stays_unchanged(self) -> None:
        client = SyncClient("wss://secure.example.com/sync/ws")
        assert client._server_url == "wss://secure.example.com/sync/ws"

    def test_auto_appends_sync_ws_path(self) -> None:
        client = SyncClient("ws://localhost:8000")
        assert client._server_url.endswith("/sync/ws")

    def test_trailing_slash_handled(self) -> None:
        client = SyncClient("ws://localhost:8000/")
        assert client._server_url == "ws://localhost:8000/sync/ws"

    def test_custom_client_id(self) -> None:
        client = SyncClient("ws://localhost:8000", client_id="custom-id")
        assert client.client_id == "custom-id"

    def test_auto_generated_client_id(self) -> None:
        client = SyncClient("ws://localhost:8000")
        assert client.client_id.startswith("client-")

    def test_initial_state_disconnected(self) -> None:
        client = SyncClient("ws://localhost:8000")
        assert client.state == SyncClientState.DISCONNECTED
        assert client.is_connected is False
        assert client.subscribed_brains == frozenset()


# ─────────── connect / disconnect ───────────


class TestSyncClientConnect:
    """Tests for connect() and disconnect()."""

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        client = SyncClient("ws://localhost:8000", client_id="test-1")

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(
            return_value=MagicMock(
                type=__import__("aiohttp").WSMsgType.TEXT,
                data=json.dumps({"type": "connected"}),
            )
        )

        mock_session = AsyncMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await client.connect()

        assert client.state == SyncClientState.CONNECTED
        assert client.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_already_connected_is_noop(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED

        await client.connect()  # Should not raise or do anything
        assert client.state == SyncClientState.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_failure_raises_connection_error(self) -> None:
        client = SyncClient("ws://localhost:8000")

        mock_session = AsyncMock()
        mock_session.ws_connect = AsyncMock(side_effect=OSError("Connection refused"))

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await client.connect()

        assert client.state == SyncClientState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_resubscribes_brains(self) -> None:
        client = SyncClient("ws://localhost:8000", client_id="test-1")
        client._subscribed_brains = {"brain-a", "brain-b"}

        sent_messages: list[dict[str, str]] = []

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(
            return_value=MagicMock(
                type=__import__("aiohttp").WSMsgType.TEXT,
                data=json.dumps({"type": "connected"}),
            )
        )

        async def capture_send(data: str) -> None:
            sent_messages.append(json.loads(data))

        mock_ws.send_str = AsyncMock(side_effect=capture_send)

        mock_session = AsyncMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await client.connect()

        # First message is the connect, then 2 subscribe messages
        subscribe_msgs = [m for m in sent_messages if m.get("action") == "subscribe"]
        assert len(subscribe_msgs) == 2

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED
        client._ws = AsyncMock()
        client._session = AsyncMock()

        await client.disconnect()

        assert client.state == SyncClientState.DISCONNECTED
        assert client._ws is None
        assert client._session is None

    @pytest.mark.asyncio
    async def test_disconnect_cancels_receive_task(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED
        client._ws = AsyncMock()
        client._session = AsyncMock()

        # Create a real task that just sleeps forever
        async def _forever() -> None:
            await asyncio.sleep(9999)

        task = asyncio.create_task(_forever())
        client._receive_task = task

        await client.disconnect()

        assert task.cancelled()
        assert client._receive_task is None


# ─────────── context manager ───────────


class TestSyncClientContextManager:
    """Tests for __aenter__ / __aexit__."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(
            return_value=MagicMock(
                type=__import__("aiohttp").WSMsgType.TEXT,
                data=json.dumps({"type": "connected"}),
            )
        )

        mock_session = AsyncMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            async with SyncClient("ws://localhost:8000") as client:
                assert client.is_connected

        # After exiting context, should be disconnected
        assert client.state == SyncClientState.DISCONNECTED


# ─────────── subscribe / unsubscribe ───────────


class TestSyncClientSubscribe:
    """Tests for subscribe() and unsubscribe()."""

    @pytest.mark.asyncio
    async def test_subscribe_when_disconnected(self) -> None:
        client = SyncClient("ws://localhost:8000")
        result = await client.subscribe("brain-1")
        assert result is True
        assert "brain-1" in client.subscribed_brains

    @pytest.mark.asyncio
    async def test_subscribe_when_connected(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(
            return_value=MagicMock(
                type=__import__("aiohttp").WSMsgType.TEXT,
                data=json.dumps({"type": "subscribed"}),
            )
        )
        client._ws = mock_ws

        result = await client.subscribe("brain-1")
        assert result is True
        assert "brain-1" in client.subscribed_brains

    @pytest.mark.asyncio
    async def test_subscribe_confirmation_failure(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(
            return_value=MagicMock(
                type=__import__("aiohttp").WSMsgType.TEXT,
                data=json.dumps({"type": "error"}),
            )
        )
        client._ws = mock_ws

        result = await client.subscribe("brain-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_unsubscribe_when_disconnected(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._subscribed_brains.add("brain-1")
        result = await client.unsubscribe("brain-1")
        assert result is True
        assert "brain-1" not in client.subscribed_brains

    @pytest.mark.asyncio
    async def test_unsubscribe_when_connected(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED
        client._ws = AsyncMock()
        client._subscribed_brains.add("brain-1")

        result = await client.unsubscribe("brain-1")
        assert result is True
        assert "brain-1" not in client.subscribed_brains


# ─────────── push_event ───────────


class TestSyncClientPushEvent:
    """Tests for push_event()."""

    @pytest.mark.asyncio
    async def test_push_when_disconnected_raises(self) -> None:
        client = SyncClient("ws://localhost:8000")
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.push_event("test", "brain-1")

    @pytest.mark.asyncio
    async def test_push_event_sends_message(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED

        sent: list[str] = []
        mock_ws = AsyncMock()
        mock_ws.send_str = AsyncMock(side_effect=lambda d: sent.append(d))
        client._ws = mock_ws

        await client.push_event("neuron_created", "brain-1", {"id": "n-1"})

        assert len(sent) == 1
        msg = json.loads(sent[0])
        assert msg["action"] == "event"
        assert msg["event"]["type"] == "neuron_created"
        assert msg["event"]["brain_id"] == "brain-1"
        assert msg["event"]["data"]["id"] == "n-1"


# ─────────── get_history ───────────


class TestSyncClientGetHistory:
    """Tests for get_history()."""

    @pytest.mark.asyncio
    async def test_get_history_when_disconnected_raises(self) -> None:
        client = SyncClient("ws://localhost:8000")
        with pytest.raises(ConnectionError, match="Not connected"):
            await client.get_history("brain-1")

    @pytest.mark.asyncio
    async def test_get_history_returns_events(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(
            return_value=MagicMock(
                type=__import__("aiohttp").WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "type": "history",
                        "events": [
                            {
                                "type": "neuron_created",
                                "brain_id": "brain-1",
                                "timestamp": "2026-01-01T00:00:00",
                                "data": {"id": "n-1"},
                            },
                        ],
                    }
                ),
            )
        )
        client._ws = mock_ws

        events = await client.get_history("brain-1")
        assert len(events) == 1
        assert events[0].type == "neuron_created"

    @pytest.mark.asyncio
    async def test_get_history_with_since_param(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED

        sent: list[str] = []
        mock_ws = AsyncMock()
        mock_ws.send_str = AsyncMock(side_effect=lambda d: sent.append(d))
        mock_ws.receive = AsyncMock(
            return_value=MagicMock(
                type=__import__("aiohttp").WSMsgType.TEXT,
                data=json.dumps({"type": "history", "events": []}),
            )
        )
        client._ws = mock_ws

        since = datetime(2026, 1, 1)
        await client.get_history("brain-1", since=since)

        msg = json.loads(sent[0])
        assert msg["since"] == "2026-01-01T00:00:00"

    @pytest.mark.asyncio
    async def test_get_history_empty_response(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(
            return_value=MagicMock(
                type=__import__("aiohttp").WSMsgType.TEXT,
                data=json.dumps({"type": "history", "events": []}),
            )
        )
        client._ws = mock_ws

        events = await client.get_history("brain-1")
        assert events == []

    @pytest.mark.asyncio
    async def test_get_history_filters_invalid_events(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(
            return_value=MagicMock(
                type=__import__("aiohttp").WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "type": "history",
                        "events": [
                            {"type": "valid", "brain_id": "b1"},
                            {"no_type": True},  # Invalid — no type
                        ],
                    }
                ),
            )
        )
        client._ws = mock_ws

        events = await client.get_history("brain-1")
        assert len(events) == 1


# ─────────── _handle_message ───────────


class TestSyncClientHandleMessage:
    """Tests for _handle_message()."""

    @pytest.mark.asyncio
    async def test_dispatches_to_typed_handler(self) -> None:
        client = SyncClient("ws://localhost:8000", client_id="me")
        received: list[SyncEvent] = []

        async def handler(event: SyncEvent) -> None:
            received.append(event)

        client.on("neuron_created", handler)

        await client._handle_message(
            {
                "type": "neuron_created",
                "brain_id": "b1",
                "source_client_id": "other",
            }
        )

        assert len(received) == 1
        assert received[0].type == "neuron_created"

    @pytest.mark.asyncio
    async def test_wildcard_handler(self) -> None:
        client = SyncClient("ws://localhost:8000", client_id="me")
        received: list[SyncEvent] = []

        async def handler(event: SyncEvent) -> None:
            received.append(event)

        client.on("*", handler)

        await client._handle_message(
            {
                "type": "anything",
                "brain_id": "b1",
                "source_client_id": "other",
            }
        )

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_skips_self_originated_events(self) -> None:
        client = SyncClient("ws://localhost:8000", client_id="me")
        received: list[SyncEvent] = []

        async def handler(event: SyncEvent) -> None:
            received.append(event)

        client.on("neuron_created", handler)

        await client._handle_message(
            {
                "type": "neuron_created",
                "brain_id": "b1",
                "source_client_id": "me",  # Same as client_id
            }
        )

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_missing_type_is_ignored(self) -> None:
        client = SyncClient("ws://localhost:8000")
        # Should not raise
        await client._handle_message({"brain_id": "b1"})

    @pytest.mark.asyncio
    async def test_handler_exception_is_caught(self) -> None:
        client = SyncClient("ws://localhost:8000", client_id="me")

        async def bad_handler(event: SyncEvent) -> None:
            raise ValueError("boom")

        client.on("test", bad_handler)

        # Should not raise — exception is caught and logged
        await client._handle_message(
            {
                "type": "test",
                "brain_id": "b1",
                "source_client_id": "other",
            }
        )

    @pytest.mark.asyncio
    async def test_sync_handler_called(self) -> None:
        """Test non-coroutine handler is called."""
        client = SyncClient("ws://localhost:8000", client_id="me")
        received: list[SyncEvent] = []

        def sync_handler(event: SyncEvent) -> None:
            received.append(event)

        client.on("test", sync_handler)  # type: ignore[arg-type]

        await client._handle_message(
            {
                "type": "test",
                "brain_id": "b1",
                "source_client_id": "other",
            }
        )

        assert len(received) == 1


# ─────────── _try_reconnect ───────────


class TestSyncClientReconnect:
    """Tests for _try_reconnect()."""

    @pytest.mark.asyncio
    async def test_reconnect_increments_attempts(self) -> None:
        client = SyncClient("ws://localhost:8000", reconnect_delay=0.01)
        client._reconnect_attempts = 0

        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = ConnectionError("fail")
            await client._try_reconnect()

        assert client._reconnect_attempts == 1
        assert client.state == SyncClientState.RECONNECTING

    @pytest.mark.asyncio
    async def test_reconnect_stops_at_max_attempts(self) -> None:
        client = SyncClient(
            "ws://localhost:8000",
            max_reconnect_attempts=3,
            reconnect_delay=0.01,
        )
        client._reconnect_attempts = 3

        await client._try_reconnect()

        assert client._running is False

    @pytest.mark.asyncio
    async def test_reconnect_exponential_backoff(self) -> None:
        client = SyncClient("ws://localhost:8000", reconnect_delay=0.01)
        client._reconnect_attempts = 2

        with (
            patch.object(client, "connect", new_callable=AsyncMock),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            await client._try_reconnect()

        # attempt 3: delay = 0.01 * 2^(3-1) = 0.04
        called_delay = mock_sleep.call_args[0][0]
        assert 0.03 < called_delay < 0.05

    @pytest.mark.asyncio
    async def test_reconnect_caps_at_60_seconds(self) -> None:
        client = SyncClient("ws://localhost:8000", reconnect_delay=10.0)
        client._reconnect_attempts = 9  # 10.0 * 2^9 = 5120, capped at 60

        with (
            patch.object(client, "connect", new_callable=AsyncMock),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            await client._try_reconnect()

        called_delay = mock_sleep.call_args[0][0]
        assert called_delay == 60.0


# ─────────── run_forever ───────────


class TestSyncClientRunForever:
    """Tests for run_forever()."""

    @pytest.mark.asyncio
    async def test_run_forever_breaks_on_cancelled(self) -> None:
        client = SyncClient("ws://localhost:8000")
        client._state = SyncClientState.CONNECTED

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(side_effect=asyncio.CancelledError)
        client._ws = mock_ws

        await client.run_forever()
        # Should exit cleanly

    @pytest.mark.asyncio
    async def test_run_forever_handles_close_message(self) -> None:
        client = SyncClient("ws://localhost:8000", auto_reconnect=False)
        client._state = SyncClientState.CONNECTED

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(
            return_value=MagicMock(type=__import__("aiohttp").WSMsgType.CLOSED)
        )
        client._ws = mock_ws

        await client.run_forever()
        assert client.state == SyncClientState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_run_forever_breaks_when_not_connected_no_reconnect(self) -> None:
        client = SyncClient("ws://localhost:8000", auto_reconnect=False)
        # Already disconnected, no reconnect
        await client.run_forever()
        # Should exit immediately
