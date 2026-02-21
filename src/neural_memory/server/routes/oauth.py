"""OAuth proxy routes — forward OAuth flows to CLIProxyAPI."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


async def _require_local_request(request: Request) -> None:
    """Reject non-localhost requests to protect OAuth endpoints."""
    client_host = request.client.host if request.client else ""
    allowed = {"127.0.0.1", "::1", "localhost"}
    if client_host not in allowed:
        raise HTTPException(status_code=403, detail="Forbidden")


router = APIRouter(
    prefix="/api/oauth",
    tags=["oauth"],
    dependencies=[Depends(_require_local_request)],
)

# CLIProxyAPI base URL (Go service running separately, restricted to localhost)
_CLIPROXY_BASE = os.environ.get("CLIPROXY_URL", "http://127.0.0.1:8317")
if not any(
    _CLIPROXY_BASE.startswith(prefix)
    for prefix in ("http://127.0.0.1", "http://localhost", "http://[::1]")
):
    logger.warning("CLIPROXY_URL must point to localhost, falling back to default")
    _CLIPROXY_BASE = "http://127.0.0.1:8317"

# Provider metadata — matches CLIProxyAPI supported providers
PROVIDERS: list[dict[str, Any]] = [
    {
        "id": "claude",
        "name": "Claude",
        "icon": "brain",
        "port": 54545,
        "description": "Anthropic Claude Code OAuth",
    },
    {
        "id": "gemini",
        "name": "Gemini",
        "icon": "sparkles",
        "port": 8085,
        "description": "Google Gemini CLI OAuth",
    },
    {
        "id": "codex",
        "name": "OpenAI Codex",
        "icon": "code",
        "port": 1455,
        "description": "OpenAI Codex CLI OAuth",
    },
    {
        "id": "qwen",
        "name": "Qwen",
        "icon": "message-square",
        "port": None,
        "description": "Alibaba Qwen API key",
    },
    {
        "id": "iflow",
        "name": "iFlow",
        "icon": "workflow",
        "port": None,
        "description": "iFlow API key",
    },
    {
        "id": "antigravity",
        "name": "AntiGravity",
        "icon": "rocket",
        "port": None,
        "description": "AntiGravity platform OAuth",
    },
]


class OAuthInitiateRequest(BaseModel):
    """Request to start an OAuth session."""

    provider: str = Field(..., description="Provider ID (claude, gemini, codex, ...)")


class OAuthInitiateResponse(BaseModel):
    """Response from OAuth initiation."""

    session_id: str
    auth_url: str
    provider: str
    expires_in: int = 600


class OAuthCallbackRequest(BaseModel):
    """Request to complete OAuth callback."""

    provider: str
    code: str
    state: str | None = None


class OAuthStatusResponse(BaseModel):
    """Provider authentication status."""

    provider: str
    authenticated: bool
    token_preview: str | None = None
    expires_at: str | None = None


async def _proxy_post(path: str, json_body: dict[str, Any]) -> dict[str, Any]:
    """POST to CLIProxyAPI and return JSON response."""
    import httpx

    url = f"{_CLIPROXY_BASE}{path}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=json_body)
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail="OAuth proxy not reachable",
        )
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "CLIProxyAPI POST %s returned %s: %s", path, exc.response.status_code, exc.response.text
        )
        raise HTTPException(status_code=exc.response.status_code, detail="Upstream proxy error")


async def _proxy_get(path: str) -> dict[str, Any]:
    """GET from CLIProxyAPI and return JSON response."""
    import httpx

    url = f"{_CLIPROXY_BASE}{path}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail="CLIProxyAPI not reachable.",
        )
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "CLIProxyAPI GET %s returned %s: %s", path, exc.response.status_code, exc.response.text
        )
        raise HTTPException(status_code=exc.response.status_code, detail="Upstream proxy error")


@router.get(
    "/providers",
    summary="List supported OAuth providers",
)
async def list_providers() -> list[dict[str, Any]]:
    """Return all supported OAuth providers with metadata."""
    # Check auth status for each provider
    results: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        entry = {**provider, "authenticated": False}
        try:
            status = await _check_provider_auth(provider["id"])
            entry["authenticated"] = status
        except Exception:
            logger.debug("Auth check failed for provider %s", provider["id"], exc_info=True)
        results.append(entry)
    return results


@router.post(
    "/initiate",
    response_model=OAuthInitiateResponse,
    summary="Start OAuth session",
)
async def initiate_oauth(request: OAuthInitiateRequest) -> OAuthInitiateResponse:
    """Initiate OAuth flow via CLIProxyAPI."""
    valid_ids = {p["id"] for p in PROVIDERS}
    if request.provider not in valid_ids:
        raise HTTPException(status_code=400, detail="Unknown provider")

    data = await _proxy_post(
        "/v0/management/oauth-sessions/start",
        {"provider": request.provider},
    )

    return OAuthInitiateResponse(
        session_id=data.get("session_id", ""),
        auth_url=data.get("auth_url", ""),
        provider=request.provider,
        expires_in=data.get("expires_in", 600),
    )


@router.post(
    "/callback",
    summary="OAuth callback handler",
)
async def oauth_callback(request: OAuthCallbackRequest) -> dict[str, Any]:
    """Handle OAuth callback from provider."""
    data = await _proxy_post(
        "/v0/management/oauth/callback",
        {
            "provider": request.provider,
            "code": request.code,
            "state": request.state,
        },
    )
    return {
        "status": "authenticated",
        "provider": request.provider,
        "token_type": data.get("token_type"),
        "expires_in": data.get("expires_in"),
    }


@router.get(
    "/status/{provider}",
    response_model=OAuthStatusResponse,
    summary="Check provider auth status",
)
async def get_provider_status(provider: str) -> OAuthStatusResponse:
    """Check if a provider has valid authentication."""
    authenticated = await _check_provider_auth(provider)
    return OAuthStatusResponse(
        provider=provider,
        authenticated=authenticated,
    )


async def _check_provider_auth(provider_id: str) -> bool:
    """Check if provider has stored auth token via CLIProxyAPI."""
    try:
        data = await _proxy_get("/v0/management/auth-files")
        files = data.get("files", [])
        return any(f.get("provider") == provider_id for f in files)
    except Exception:
        return False
