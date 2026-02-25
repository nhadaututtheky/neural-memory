"""OpenClaw configuration API routes.

CRUD for API keys, Telegram/Discord channels,
function toggles, and security restrictions.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from neural_memory.integrations.openclaw_config import (
    DiscordConfig,
    OpenClawConfig,
    OpenClawConfigManager,
    SecurityConfig,
    TelegramConfig,
)
from neural_memory.server.dependencies import require_local_request

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/openclaw",
    tags=["openclaw"],
    dependencies=[Depends(require_local_request)],
)


def _manager() -> OpenClawConfigManager:
    """Create config manager instance."""
    return OpenClawConfigManager()


# ── Config CRUD ───────────────────────────────────────────────


@router.get(
    "/config",
    summary="Get OpenClaw configuration",
)
async def get_config() -> dict[str, Any]:
    """Load current OpenClaw configuration."""
    config = _manager().load()
    # Mask API key values for security
    data = config.model_dump(mode="json")
    for key_entry in data.get("api_keys", []):
        raw = key_entry.get("key", "")
        if len(raw) > 8:
            key_entry["key"] = raw[:4] + "..." + raw[-4:]
    return data


@router.post(
    "/config",
    summary="Save full OpenClaw configuration",
)
async def save_config(config: OpenClawConfig) -> dict[str, str]:
    """Save the entire OpenClaw configuration."""
    _manager().save(config)
    return {"status": "saved"}


# ── API Keys ──────────────────────────────────────────────────


class ApiKeyRequest(BaseModel):
    """Request to add/update an API key."""

    provider: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)
    label: str = ""


@router.post(
    "/apikeys",
    summary="Add or update API key",
)
async def upsert_api_key(request: ApiKeyRequest) -> dict[str, str]:
    """Add or update an API key for a provider."""
    _manager().update_api_key(request.provider, request.key, request.label)
    return {"status": "saved", "provider": request.provider}


@router.delete(
    "/apikeys/{provider}",
    summary="Remove API key",
)
async def delete_api_key(provider: str) -> dict[str, str]:
    """Remove an API key for a provider."""
    _manager().remove_api_key(provider)
    return {"status": "deleted", "provider": provider}


# ── Telegram ──────────────────────────────────────────────────


@router.get(
    "/telegram",
    summary="Get Telegram config",
)
async def get_telegram() -> dict[str, Any]:
    """Get current Telegram configuration."""
    config = _manager().load()
    data = config.telegram.model_dump(mode="json")
    # Mask bot token
    raw = data.get("bot_token", "")
    if len(raw) > 10:
        data["bot_token"] = raw[:6] + "..." + raw[-4:]
    return data


@router.post(
    "/telegram",
    summary="Update Telegram config",
)
async def update_telegram(telegram: TelegramConfig) -> dict[str, str]:
    """Update Telegram bot configuration."""
    _manager().update_telegram(telegram)
    return {"status": "saved"}


# ── Discord ───────────────────────────────────────────────────


@router.get(
    "/discord",
    summary="Get Discord config",
)
async def get_discord() -> dict[str, Any]:
    """Get current Discord configuration."""
    config = _manager().load()
    data = config.discord.model_dump(mode="json")
    # Mask bot token
    raw = data.get("bot_token", "")
    if len(raw) > 10:
        data["bot_token"] = raw[:6] + "..." + raw[-4:]
    return data


@router.post(
    "/discord",
    summary="Update Discord config",
)
async def update_discord(discord: DiscordConfig) -> dict[str, str]:
    """Update Discord bot configuration."""
    _manager().update_discord(discord)
    return {"status": "saved"}


# ── Functions ─────────────────────────────────────────────────


@router.get(
    "/functions",
    summary="List available functions",
)
async def list_functions() -> list[dict[str, Any]]:
    """List all configured functions with their settings."""
    config = _manager().load()
    return [fn.model_dump(mode="json") for fn in config.functions]


class FunctionToggleRequest(BaseModel):
    """Request to toggle a function."""

    enabled: bool


@router.post(
    "/functions/{name}",
    summary="Toggle or configure function",
)
async def toggle_function(name: str, request: FunctionToggleRequest) -> dict[str, Any]:
    """Toggle a function on/off."""
    config = _manager().toggle_function(name, request.enabled)
    fn = next((f for f in config.functions if f.name == name), None)
    if fn is None:
        raise HTTPException(status_code=404, detail="Function not found")
    return fn.model_dump(mode="json")


# ── Security ──────────────────────────────────────────────────


@router.get(
    "/security",
    summary="Get security config",
)
async def get_security() -> dict[str, Any]:
    """Get current security restrictions."""
    config = _manager().load()
    return config.security.model_dump(mode="json")


@router.post(
    "/security",
    summary="Update security config",
)
async def update_security(security: SecurityConfig) -> dict[str, str]:
    """Update security restrictions."""
    config = _manager().load()
    updated = config.model_copy(update={"security": security})
    _manager().save(updated)
    return {"status": "saved"}
