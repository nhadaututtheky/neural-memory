"""WebSocket client for real-time brain synchronization."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import aiohttp

from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class SyncClientState(StrEnum):
    """Client connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


@dataclass
class SyncEvent:
    """A synchronization event received from server."""

    type: str
    brain_id: str
    timestamp: datetime
    data: dict[str, Any]
    source_client_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncEvent | None:
        """Create from dictionary. Returns None if required fields missing."""
        event_type = data.get("type")
        brain_id = data.get("brain_id")
        if not event_type or not brain_id:
            return None
        return cls(
            type=event_type,
            brain_id=brain_id,
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else utcnow(),
            data=data.get("data", {}),
            source_client_id=data.get("source_client_id"),
        )


EventHandler = Callable[[SyncEvent], None]
AsyncEventHandler = Callable[[SyncEvent], Any]


class SyncClient:
    """
    WebSocket client for real-time brain synchronization.

    Enables receiving real-time updates when other agents modify
    shared brains, and pushing local changes to the server.

    Usage:
        async with SyncClient("ws://localhost:8000") as client:
            # Subscribe to brain updates
            await client.subscribe("brain-123")

            # Register event handlers
            client.on("neuron_created", handle_neuron_created)
            client.on("memory_encoded", handle_memory_encoded)

            # Push local changes
            await client.push_event("neuron_created", "brain-123", {"id": "..."})

            # Keep running
            await client.run_forever()
    """

    def __init__(
        self,
        server_url: str,
        *,
        client_id: str | None = None,
        auto_reconnect: bool = True,
        reconnect_delay: float = 1.0,
        max_reconnect_attempts: int = 10,
    ) -> None:
        """
        Initialize sync client.

        Args:
            server_url: WebSocket URL (e.g., "ws://localhost:8000/sync/ws")
            client_id: Optional client identifier (auto-generated if not provided)
            auto_reconnect: Whether to auto-reconnect on disconnect
            reconnect_delay: Base delay between reconnect attempts (exponential backoff)
            max_reconnect_attempts: Maximum reconnect attempts (0 = unlimited)
        """
        # Normalize WebSocket URL
        if server_url.startswith("http://"):
            server_url = server_url.replace("http://", "ws://", 1)
        elif server_url.startswith("https://"):
            server_url = server_url.replace("https://", "wss://", 1)

        # Validate URL scheme
        if not server_url.startswith(("ws://", "wss://")):
            raise ValueError("Invalid WebSocket URL scheme: must start with ws:// or wss://")

        if not server_url.endswith("/sync/ws"):
            server_url = server_url.rstrip("/") + "/sync/ws"

        self._server_url = server_url
        self._client_id = client_id or f"client-{uuid.uuid4().hex[:8]}"
        self._auto_reconnect = auto_reconnect
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_attempts = max_reconnect_attempts

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._state = SyncClientState.DISCONNECTED
        self._subscribed_brains: set[str] = set()
        self._handlers: dict[str, list[AsyncEventHandler]] = {}
        self._reconnect_attempts = 0
        self._running = False
        self._receive_task: asyncio.Task[None] | None = None

    @property
    def client_id(self) -> str:
        """Get client ID."""
        return self._client_id

    @property
    def state(self) -> SyncClientState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._state == SyncClientState.CONNECTED

    @property
    def subscribed_brains(self) -> frozenset[str]:
        """Get set of subscribed brain IDs."""
        return frozenset(self._subscribed_brains)

    async def connect(self) -> None:
        """Connect to the sync server."""
        if self._state == SyncClientState.CONNECTED:
            return

        self._state = SyncClientState.CONNECTING

        if self._session is None:
            self._session = aiohttp.ClientSession()

        try:
            self._ws = await self._session.ws_connect(self._server_url)

            # Send connect message
            await self._send({"action": "connect", "client_id": self._client_id})

            # Wait for connected confirmation
            response = await self._ws.receive()
            if response.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(response.data)
                if data.get("type") == "connected":
                    self._state = SyncClientState.CONNECTED
                    self._reconnect_attempts = 0

                    # Re-subscribe to brains
                    for brain_id in self._subscribed_brains:
                        await self._send({"action": "subscribe", "brain_id": brain_id})

        except Exception as e:
            self._state = SyncClientState.DISCONNECTED
            raise ConnectionError(f"Failed to connect to sync server: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from the sync server."""
        self._running = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._session:
            await self._session.close()
            self._session = None

        self._state = SyncClientState.DISCONNECTED

    async def __aenter__(self) -> SyncClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def subscribe(self, brain_id: str) -> bool:
        """
        Subscribe to updates for a brain.

        Args:
            brain_id: The brain ID to subscribe to

        Returns:
            True if subscription succeeded
        """
        self._subscribed_brains.add(brain_id)

        if not self.is_connected:
            return True  # Will subscribe when connected

        await self._send({"action": "subscribe", "brain_id": brain_id})

        # Wait for confirmation
        if self._ws:
            response = await self._ws.receive()
            if response.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(response.data)
                result: bool = data.get("type") == "subscribed"
                return result

        return False

    async def unsubscribe(self, brain_id: str) -> bool:
        """
        Unsubscribe from updates for a brain.

        Args:
            brain_id: The brain ID to unsubscribe from

        Returns:
            True if unsubscription succeeded
        """
        self._subscribed_brains.discard(brain_id)

        if not self.is_connected:
            return True

        await self._send({"action": "unsubscribe", "brain_id": brain_id})
        return True

    def on(self, event_type: str, handler: AsyncEventHandler) -> None:
        """
        Register an event handler.

        Args:
            event_type: The event type to handle (e.g., "neuron_created")
            handler: Async function to call when event is received
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def off(self, event_type: str, handler: AsyncEventHandler | None = None) -> None:
        """
        Unregister an event handler.

        Args:
            event_type: The event type
            handler: Specific handler to remove (all if None)
        """
        if event_type not in self._handlers:
            return

        if handler is None:
            del self._handlers[event_type]
        else:
            self._handlers[event_type] = [h for h in self._handlers[event_type] if h != handler]

    async def push_event(
        self,
        event_type: str,
        brain_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Push an event to the server for broadcasting.

        Args:
            event_type: Type of event (e.g., "neuron_created")
            brain_id: The brain this event relates to
            data: Event data
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to sync server")

        event = {
            "type": event_type,
            "brain_id": brain_id,
            "timestamp": utcnow().isoformat(),
            "data": data or {},
        }

        await self._send({"action": "event", "event": event})

    async def get_history(
        self,
        brain_id: str,
        since: datetime | None = None,
    ) -> list[SyncEvent]:
        """
        Get recent event history for a brain.

        Args:
            brain_id: The brain ID
            since: Only get events after this time

        Returns:
            List of recent events
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to sync server")

        message: dict[str, Any] = {"action": "get_history", "brain_id": brain_id}
        if since:
            message["since"] = since.isoformat()

        await self._send(message)

        # Wait for response
        if self._ws:
            response = await self._ws.receive()
            if response.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(response.data)
                if data.get("type") == "history":
                    return [
                        evt
                        for e in data.get("events", [])
                        if (evt := SyncEvent.from_dict(e)) is not None
                    ]

        return []

    async def run_forever(self) -> None:
        """
        Run the client, receiving and dispatching events.

        This method blocks until disconnect() is called or
        max reconnect attempts is exceeded.
        """
        self._running = True

        while self._running:
            if not self.is_connected:
                if self._auto_reconnect:
                    await self._try_reconnect()
                else:
                    break

            if not self._ws:
                await asyncio.sleep(0.1)
                continue

            try:
                message = await self._ws.receive()

                if message.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(message.data)
                    await self._handle_message(data)

                elif message.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    self._state = SyncClientState.DISCONNECTED
                    if self._auto_reconnect:
                        continue
                    else:
                        break

            except asyncio.CancelledError:
                break
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "WebSocket receive error, state → DISCONNECTED", exc_info=True
                )
                self._state = SyncClientState.DISCONNECTED
                if self._auto_reconnect:
                    continue
                else:
                    raise

    async def _send(self, data: dict[str, Any]) -> None:
        """Send a message to the server."""
        if self._ws:
            await self._ws.send_str(json.dumps(data))

    async def _handle_message(self, data: dict[str, Any]) -> None:
        """Handle an incoming message."""
        event_type = data.get("type")
        if not event_type:
            return

        # Convert to SyncEvent
        event = SyncEvent.from_dict(data)
        if event is None:
            return

        # Skip events we originated
        if event.source_client_id == self._client_id:
            return

        # Dispatch to handlers (copy to avoid mutating _handlers)
        handlers = [*self._handlers.get(event_type, []), *self._handlers.get("*", [])]

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning("Sync event handler error for '%s': %s", event_type, e)

    async def _try_reconnect(self) -> None:
        """Attempt to reconnect to the server."""
        if self._max_reconnect_attempts > 0:
            if self._reconnect_attempts >= self._max_reconnect_attempts:
                self._running = False
                return

        self._state = SyncClientState.RECONNECTING
        self._reconnect_attempts += 1

        # Exponential backoff
        delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))
        delay = min(delay, 60.0)  # Cap at 60 seconds

        await asyncio.sleep(delay)

        try:
            await self.connect()
        except (ConnectionError, OSError) as e:
            logger.debug("Reconnect attempt %d failed: %s", self._reconnect_attempts, e)
