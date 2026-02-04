"""WebSocket routes for real-time brain synchronization."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/sync", tags=["sync"])


class SyncEventType(StrEnum):
    """Types of sync events."""

    # Connection events
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"

    # Data events
    NEURON_CREATED = "neuron_created"
    NEURON_UPDATED = "neuron_updated"
    NEURON_DELETED = "neuron_deleted"

    SYNAPSE_CREATED = "synapse_created"
    SYNAPSE_UPDATED = "synapse_updated"
    SYNAPSE_DELETED = "synapse_deleted"

    FIBER_CREATED = "fiber_created"
    FIBER_UPDATED = "fiber_updated"
    FIBER_DELETED = "fiber_deleted"

    # Memory events
    MEMORY_ENCODED = "memory_encoded"
    MEMORY_QUERIED = "memory_queried"

    # Sync events
    FULL_SYNC = "full_sync"
    PARTIAL_SYNC = "partial_sync"

    # Error events
    ERROR = "error"


@dataclass
class SyncEvent:
    """A synchronization event."""

    type: SyncEventType
    brain_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: dict[str, Any] = field(default_factory=dict)
    source_client_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "brain_id": self.brain_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source_client_id": self.source_client_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncEvent:
        """Create from dictionary."""
        return cls(
            type=SyncEventType(data["type"]),
            brain_id=data["brain_id"],
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.utcnow(),
            data=data.get("data", {}),
            source_client_id=data.get("source_client_id"),
        )


@dataclass
class ConnectedClient:
    """A connected WebSocket client."""

    client_id: str
    websocket: WebSocket
    brain_ids: set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.utcnow)

    async def send_event(self, event: SyncEvent) -> bool:
        """Send event to client. Returns False if connection closed."""
        try:
            await self.websocket.send_text(event.to_json())
            return True
        except Exception:
            return False


class SyncManager:
    """
    Manages WebSocket connections and event broadcasting.

    Singleton pattern - use SyncManager.instance() to get the shared instance.
    """

    _instance: SyncManager | None = None

    def __init__(self) -> None:
        self._clients: dict[str, ConnectedClient] = {}
        self._brain_subscriptions: dict[str, set[str]] = {}  # brain_id -> client_ids
        self._event_history: dict[str, list[SyncEvent]] = {}  # brain_id -> recent events
        self._max_history = 100
        self._lock = asyncio.Lock()

    @classmethod
    def instance(cls) -> SyncManager:
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    async def connect(self, client_id: str, websocket: WebSocket) -> ConnectedClient:
        """Register a new WebSocket client."""
        async with self._lock:
            client = ConnectedClient(client_id=client_id, websocket=websocket)
            self._clients[client_id] = client
            return client

    async def disconnect(self, client_id: str) -> None:
        """Unregister a WebSocket client."""
        async with self._lock:
            if client_id in self._clients:
                client = self._clients[client_id]
                # Unsubscribe from all brains
                for brain_id in client.brain_ids:
                    if brain_id in self._brain_subscriptions:
                        self._brain_subscriptions[brain_id].discard(client_id)
                del self._clients[client_id]

    async def subscribe(self, client_id: str, brain_id: str) -> bool:
        """Subscribe a client to a brain's events."""
        async with self._lock:
            if client_id not in self._clients:
                return False

            client = self._clients[client_id]
            client.brain_ids.add(brain_id)

            if brain_id not in self._brain_subscriptions:
                self._brain_subscriptions[brain_id] = set()
            self._brain_subscriptions[brain_id].add(client_id)

            return True

    async def unsubscribe(self, client_id: str, brain_id: str) -> bool:
        """Unsubscribe a client from a brain's events."""
        async with self._lock:
            if client_id not in self._clients:
                return False

            client = self._clients[client_id]
            client.brain_ids.discard(brain_id)

            if brain_id in self._brain_subscriptions:
                self._brain_subscriptions[brain_id].discard(client_id)

            return True

    async def broadcast(
        self,
        event: SyncEvent,
        exclude_client: str | None = None,
    ) -> int:
        """
        Broadcast event to all clients subscribed to the brain.

        Args:
            event: The event to broadcast
            exclude_client: Don't send to this client (usually the source)

        Returns:
            Number of clients that received the event
        """
        # Store in history
        async with self._lock:
            if event.brain_id not in self._event_history:
                self._event_history[event.brain_id] = []

            history = self._event_history[event.brain_id]
            history.append(event)
            if len(history) > self._max_history:
                self._event_history[event.brain_id] = history[-self._max_history :]

        # Get subscribed clients
        async with self._lock:
            client_ids = self._brain_subscriptions.get(event.brain_id, set()).copy()

        # Send to all subscribed clients
        sent_count = 0
        disconnected = []

        for client_id in client_ids:
            if client_id == exclude_client:
                continue

            async with self._lock:
                client = self._clients.get(client_id)

            if client:
                success = await client.send_event(event)
                if success:
                    sent_count += 1
                else:
                    disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)

        return sent_count

    async def get_recent_events(
        self,
        brain_id: str,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[SyncEvent]:
        """Get recent events for a brain."""
        async with self._lock:
            history = self._event_history.get(brain_id, [])

            if since:
                history = [e for e in history if e.timestamp > since]

            return history[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get sync manager statistics."""
        return {
            "connected_clients": len(self._clients),
            "brain_subscriptions": {
                brain_id: len(clients)
                for brain_id, clients in self._brain_subscriptions.items()
            },
            "event_history_size": sum(
                len(events) for events in self._event_history.values()
            ),
        }


# Global sync manager instance
_sync_manager: SyncManager | None = None


def get_sync_manager() -> SyncManager:
    """Get the global sync manager instance."""
    return SyncManager.instance()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time brain synchronization.

    Protocol:
        1. Client connects and sends: {"action": "connect", "client_id": "..."}
        2. Server responds with: {"type": "connected", ...}
        3. Client subscribes to brain: {"action": "subscribe", "brain_id": "..."}
        4. Server sends events as they occur
        5. Client can send changes: {"action": "event", "event": {...}}
    """
    await websocket.accept()
    sync_manager = get_sync_manager()

    client_id: str | None = None
    client: ConnectedClient | None = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            action = message.get("action")

            if action == "connect":
                client_id = message.get("client_id", f"client-{id(websocket)}")
                client = await sync_manager.connect(client_id, websocket)
                await websocket.send_text(
                    SyncEvent(
                        type=SyncEventType.CONNECTED,
                        brain_id="*",
                        data={"client_id": client_id},
                    ).to_json()
                )

            elif action == "subscribe" and client_id:
                brain_id = message.get("brain_id")
                if brain_id:
                    success = await sync_manager.subscribe(client_id, brain_id)
                    event_type = (
                        SyncEventType.SUBSCRIBED if success else SyncEventType.ERROR
                    )
                    await websocket.send_text(
                        SyncEvent(
                            type=event_type,
                            brain_id=brain_id,
                            data={
                                "success": success,
                                "client_id": client_id,
                            },
                        ).to_json()
                    )

            elif action == "unsubscribe" and client_id:
                brain_id = message.get("brain_id")
                if brain_id:
                    success = await sync_manager.unsubscribe(client_id, brain_id)
                    await websocket.send_text(
                        SyncEvent(
                            type=SyncEventType.UNSUBSCRIBED,
                            brain_id=brain_id,
                            data={"success": success},
                        ).to_json()
                    )

            elif action == "event" and client_id:
                # Client is pushing an event
                event_data = message.get("event", {})
                event = SyncEvent.from_dict(event_data)
                event = SyncEvent(
                    type=event.type,
                    brain_id=event.brain_id,
                    timestamp=event.timestamp,
                    data=event.data,
                    source_client_id=client_id,
                )
                # Broadcast to other clients
                await sync_manager.broadcast(event, exclude_client=client_id)

            elif action == "get_history" and client_id:
                brain_id = message.get("brain_id")
                since = message.get("since")
                if brain_id:
                    since_dt = datetime.fromisoformat(since) if since else None
                    events = await sync_manager.get_recent_events(brain_id, since_dt)
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "history",
                                "brain_id": brain_id,
                                "events": [e.to_dict() for e in events],
                            }
                        )
                    )

            elif action == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(
                SyncEvent(
                    type=SyncEventType.ERROR,
                    brain_id="*",
                    data={"error": str(e)},
                ).to_json()
            )
        except Exception:
            pass
    finally:
        if client_id:
            await sync_manager.disconnect(client_id)


@router.get("/stats")
async def get_sync_stats() -> dict[str, Any]:
    """Get sync manager statistics."""
    sync_manager = get_sync_manager()
    return sync_manager.get_stats()
