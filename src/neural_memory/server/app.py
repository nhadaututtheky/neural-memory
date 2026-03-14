"""FastAPI application factory."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from neural_memory import __version__
from neural_memory.server.models import HealthResponse, ReadyResponse
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
from neural_memory.storage.sqlite_schema import SCHEMA_VERSION

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler with optional background consolidation."""
    import asyncio
    import logging

    from neural_memory.unified_config import get_config, get_shared_storage

    _logger = logging.getLogger(__name__)

    storage = await get_shared_storage()
    app.state.storage = storage
    app.state.startup_time = time.monotonic()

    # Start background consolidation daemon if enabled
    consolidation_task: asyncio.Task[None] | None = None
    config = get_config()
    maint = config.maintenance
    if maint.enabled and maint.scheduled_consolidation_enabled:
        consolidation_task = asyncio.create_task(_consolidation_loop(storage, maint))
        _logger.info(
            "Background consolidation daemon started: every %dh",
            maint.scheduled_consolidation_interval_hours,
        )

    yield

    if consolidation_task is not None and not consolidation_task.done():
        consolidation_task.cancel()
        try:
            await consolidation_task
        except asyncio.CancelledError:
            pass
    await storage.close()


async def _consolidation_loop(
    storage: NeuralStorage,
    maint: Any,
) -> None:
    """Background loop: run consolidation on a fixed interval.

    First run waits one full interval to avoid triggering on every
    server restart. Logs each run with summary stats.
    """
    import asyncio
    import logging

    from neural_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy

    _logger = logging.getLogger(__name__)
    interval_seconds = maint.scheduled_consolidation_interval_hours * 3600
    strategies = [ConsolidationStrategy(s) for s in maint.scheduled_consolidation_strategies]

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            brain_id = storage.brain_id
            if not brain_id:
                _logger.debug("Consolidation daemon skipped: no brain context set")
                continue

            engine = ConsolidationEngine(storage)
            report = await engine.run(strategies=strategies)
            _logger.info(
                "Background consolidation complete: %s",
                report.summary(),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.error("Background consolidation failed", exc_info=True)


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

        # If trusted networks are configured, add common localhost origins to CORS
        # Note: CORS does not support port wildcards — enumerate common dev ports
        common_ports = (3000, 3001, 5173, 5174, 8000, 8080, 8888)
        if config.trusted_networks:
            for net_str in config.trusted_networks:
                try:
                    import ipaddress

                    net = ipaddress.ip_network(net_str, strict=False)
                    addr = str(net.network_address)
                    for port in common_ports:
                        origin = f"http://{addr}:{port}"
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
        """Return server status, current brain name, uptime, and schema version."""
        storage: NeuralStorage = app.state.storage
        brain_name: str | None = getattr(storage, "brain_name", None)
        startup_time: float = getattr(app.state, "startup_time", time.monotonic())
        uptime = time.monotonic() - startup_time
        return HealthResponse(
            status="ok",
            version=__version__,
            brain_name=brain_name,
            uptime_seconds=round(uptime, 3),
            schema_version=SCHEMA_VERSION,
        )

    # Readiness check endpoint
    @app.get("/ready", response_model=ReadyResponse, tags=["health"])
    async def ready_check() -> ReadyResponse:
        """Return 200 when storage is initialized, 503 otherwise."""
        from fastapi.responses import JSONResponse

        storage: NeuralStorage | None = getattr(app.state, "storage", None)
        if storage is None:
            return cast(
                "ReadyResponse",
                JSONResponse(
                    status_code=503,
                    content=ReadyResponse(
                        ready=False, detail="storage not initialized"
                    ).model_dump(),
                ),
            )
        return ReadyResponse(ready=True, detail="ok")

    # Root endpoint
    @app.get("/", tags=["dashboard"])
    async def root() -> RedirectResponse:
        """Redirect root to dashboard."""
        return RedirectResponse(url="/ui", status_code=302)

    # Graph visualization API (supports limit/offset for progressive loading)
    from neural_memory.server.dependencies import require_local_request

    @app.get("/api/graph", tags=["visualization"], dependencies=[Depends(require_local_request)])
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

    # React SPA dist directory
    spa_dist = STATIC_DIR / "dist"

    def _serve_spa() -> Response:
        """Serve React SPA index.html."""
        spa_index = spa_dist / "index.html"
        if spa_index.exists():
            return FileResponse(spa_index)
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {"error": "Dashboard not built. Run: cd dashboard && npm run build"},
            status_code=404,
        )

    # Primary UI endpoint — React SPA
    @app.get("/ui", tags=["dashboard"])
    async def ui() -> Response:
        """Serve the NeuralMemory dashboard."""
        return _serve_spa()

    # SPA catch-all for /ui client-side routing
    @app.get("/ui/{path:path}", tags=["dashboard"])
    async def ui_spa_catchall(path: str) -> Response:
        """Catch-all for React SPA client-side routing under /ui."""
        return _serve_spa()

    # /dashboard alias (same SPA)
    @app.get("/dashboard", tags=["dashboard"])
    async def dashboard() -> Response:
        """Serve the NeuralMemory React dashboard."""
        return _serve_spa()

    # SPA catch-all for /dashboard client-side routing
    @app.get("/dashboard/{path:path}", tags=["dashboard"])
    async def dashboard_spa_catchall(path: str) -> Response:
        """Catch-all for React SPA client-side routing under /dashboard."""
        return _serve_spa()

    # Mount SPA static assets (JS/CSS bundles)
    if spa_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(spa_dist / "assets")), name="spa-assets")

    return app


# Create default app instance for uvicorn
app = create_app()
