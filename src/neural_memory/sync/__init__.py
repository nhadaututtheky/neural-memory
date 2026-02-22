"""Real-time and incremental synchronization for NeuralMemory."""

from neural_memory.sync.client import SyncClient, SyncClientState
from neural_memory.sync.device import DeviceInfo, get_device_id, get_device_info, get_device_name
from neural_memory.sync.protocol import (
    ConflictStrategy,
    SyncChange,
    SyncConflict,
    SyncRequest,
    SyncResponse,
    SyncStatus,
)
from neural_memory.sync.sync_engine import SyncEngine

__all__ = [
    "SyncClient",
    "SyncClientState",
    "DeviceInfo",
    "get_device_id",
    "get_device_info",
    "get_device_name",
    "ConflictStrategy",
    "SyncChange",
    "SyncConflict",
    "SyncRequest",
    "SyncResponse",
    "SyncStatus",
    "SyncEngine",
]
