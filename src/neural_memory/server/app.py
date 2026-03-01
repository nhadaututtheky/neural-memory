"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from neural_memory import __version__
from neural_memory.server.models import HealthResponse
from neural_memory.server.routes import (
    brain_router,
    consolidation_router,
    dashboard_router,
    hub_router,
    integration_status_router,
    memory_router,
    oauth_router,
    openclaw_router,
    sync_router,
)
from neural_memory.storage.base import NeuralStorage

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    from neural_memory.unified_config import get_shared_storage

    storage = await get_shared_storage()
    app.state.storage = storage
    yield
    await storage.close()


def create_app(
    title: str = "NeuralMemory",
    description: str = "Reflex-based memory system for AI agents",
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        title: API title
        description: API description
        cors_origins: Allowed CORS origins (default: localhost origins)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description=description,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    if cors_origins is None:
        from neural_memory.utils.config import get_config

        config = get_config()
        cors_origins = list(config.cors_origins)

        # If trusted networks are configured, add their origins to CORS
        # (never auto-upgrade to wildcard — CORS is a separate security layer)
        if config.trusted_networks:
            for net_str in config.trusted_networks:
                try:
                    import ipaddress

                    net = ipaddress.ip_network(net_str, strict=False)
                    origin = f"http://{net.network_address}:*"
                    if origin not in cors_origins:
                        cors_origins.append(origin)
                except ValueError:
                    pass

    is_wildcard = cors_origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=not is_wildcard,  # Don't allow creds with wildcard
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Override storage dependency using the shared module
    from neural_memory.server.dependencies import get_storage as shared_get_storage

    async def get_storage() -> NeuralStorage:
        storage: NeuralStorage = app.state.storage
        return storage

    app.dependency_overrides[shared_get_storage] = get_storage

    # Versioned API routes
    api_v1 = APIRouter(prefix="/api/v1")
    api_v1.include_router(memory_router)
    api_v1.include_router(brain_router)
    api_v1.include_router(sync_router)
    api_v1.include_router(consolidation_router)
    api_v1.include_router(hub_router)
    app.include_router(api_v1)

    # Legacy unversioned routes (backward compat)
    app.include_router(memory_router)
    app.include_router(brain_router)
    app.include_router(sync_router)
    app.include_router(consolidation_router)
    app.include_router(hub_router)

    # Dashboard API routes (unversioned — dashboard-specific)
    app.include_router(dashboard_router)
    app.include_router(integration_status_router)
    app.include_router(oauth_router)
    app.include_router(openclaw_router)

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="healthy", version=__version__)

    # Root endpoint
    @app.get("/", tags=["health"])
    async def root() -> dict[str, str]:
        """Root endpoint with API info."""
        return {
            "name": title,
            "description": description,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "ui": "/ui",
        }

    # Graph visualization API (supports limit/offset for progressive loading)
    @app.get("/api/graph", tags=["visualization"])
    async def get_graph_data(
        storage: NeuralStorage = Depends(shared_get_storage),
        limit: int = Query(default=500, ge=1, le=2000),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        """Get graph data for visualization with pagination."""
        capped_limit = min(limit, 2000)

        # Fetch paginated neurons (offset + limit + 1 to detect if more exist)
        all_neurons = await storage.find_neurons(limit=offset + capped_limit)
        total_neurons = len(all_neurons)
        paginated = all_neurons[offset : offset + capped_limit]

        # Fetch synapses with a capped limit
        synapses = await storage.get_all_synapses()
        capped_synapses = synapses[:2000] if len(synapses) > 2000 else synapses
        total_synapses = len(synapses)
        fibers = await storage.get_fibers(limit=1000)

        # Build neuron ID set for filtering synapses to visible nodes
        neuron_ids = {n.id for n in paginated}
        visible_synapses = [
            s for s in capped_synapses if s.source_id in neuron_ids and s.target_id in neuron_ids
        ]

        return {
            "neurons": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "content": n.content or "",
                    "metadata": n.metadata or {},
                }
                for n in paginated
            ],
            "synapses": [
                {
                    "id": s.id,
                    "source_id": s.source_id,
                    "target_id": s.target_id,
                    "type": s.type.value,
                    "weight": s.weight,
                    "direction": s.direction.value,
                }
                for s in visible_synapses
            ],
            "fibers": [
                {
                    "id": f.id,
                    "summary": f.summary or f.id[:20],
                    "neuron_count": len(f.neuron_ids) if f.neuron_ids else 0,
                }
                for f in fibers
            ],
            "total_neurons": total_neurons,
            "total_synapses": total_synapses,
            "stats": {
                "neuron_count": len(paginated),
                "synapse_count": len(visible_synapses),
                "fiber_count": len(fibers),
            },
        }

    # UI endpoint (legacy vis.js graph)
    @app.get("/ui", tags=["visualization"])
    async def ui() -> FileResponse:
        """Serve the visualization UI."""
        return FileResponse(STATIC_DIR / "index.html")

    # Dashboard endpoint (full SPA)
    @app.get("/dashboard", tags=["dashboard"])
    async def dashboard() -> FileResponse:
        """Serve the NeuralMemory dashboard."""
        return FileResponse(STATIC_DIR / "dashboard.html")

    # Mount static files (for potential future assets)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


# Create default app instance for uvicorn
app = create_app()
