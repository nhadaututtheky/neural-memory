"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neural_memory import __version__
from neural_memory.server.models import HealthResponse
from neural_memory.server.routes import brain_router, memory_router, sync_router
from neural_memory.storage.memory_store import InMemoryStorage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    app.state.storage = InMemoryStorage()
    yield
    # Shutdown
    pass


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
        cors_origins: Allowed CORS origins (default: ["*"])

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
        cors_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Override storage dependency using the shared module
    from neural_memory.server.dependencies import get_storage as shared_get_storage

    async def get_storage() -> InMemoryStorage:
        return app.state.storage

    app.dependency_overrides[shared_get_storage] = get_storage

    # Include routers
    app.include_router(memory_router)
    app.include_router(brain_router)
    app.include_router(sync_router)

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="healthy", version=__version__)

    # Root endpoint
    @app.get("/", tags=["health"])
    async def root() -> dict:
        """Root endpoint with API info."""
        return {
            "name": title,
            "description": description,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create default app instance for uvicorn
app = create_app()
