"""API routes for NeuralMemory server."""

from neural_memory.server.routes.brain import router as brain_router
from neural_memory.server.routes.consolidation import router as consolidation_router
from neural_memory.server.routes.dashboard_api import router as dashboard_router
from neural_memory.server.routes.hub import router as hub_router
from neural_memory.server.routes.integration_status import (
    router as integration_status_router,
)
from neural_memory.server.routes.memory import router as memory_router
from neural_memory.server.routes.oauth import router as oauth_router
from neural_memory.server.routes.openclaw_api import router as openclaw_router
from neural_memory.server.routes.sync import router as sync_router

__all__ = [
    "brain_router",
    "consolidation_router",
    "dashboard_router",
    "hub_router",
    "integration_status_router",
    "memory_router",
    "oauth_router",
    "openclaw_router",
    "sync_router",
]
