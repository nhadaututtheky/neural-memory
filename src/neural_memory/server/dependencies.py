"""Shared dependencies for API routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from neural_memory.core.brain import Brain
from neural_memory.storage.base import NeuralStorage


async def require_local_request(request: Request) -> None:
    """Reject non-localhost requests to protect local-only endpoints."""
    if request.client is None:
        return  # ASGI test clients / internal calls
    client_host = request.client.host
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="Forbidden")


async def get_storage() -> NeuralStorage:
    """
    Dependency to get storage instance.

    This is overridden by the application at startup.
    """
    raise NotImplementedError("Storage not configured")


async def get_brain(
    brain_id: Annotated[str, Header(alias="X-Brain-ID")],
    storage: Annotated[NeuralStorage, Depends(get_storage)],
) -> Brain:
    """Dependency to get and validate brain from header."""
    brain = await storage.get_brain(brain_id)
    if brain is None:
        # Fallback: brain_id might be a name, not a UUID
        brain = await storage.find_brain_by_name(brain_id)
    if brain is None:
        raise HTTPException(status_code=404, detail="Brain not found")

    # Set brain context using the actual brain ID
    storage.set_brain(brain.id)
    return brain
