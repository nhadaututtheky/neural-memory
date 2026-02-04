"""API routes for NeuralMemory server."""

from neural_memory.server.routes.brain import router as brain_router
from neural_memory.server.routes.memory import router as memory_router
from neural_memory.server.routes.sync import router as sync_router

__all__ = ["brain_router", "memory_router", "sync_router"]
