"""Streamable HTTP transport for the NeuralMemory MCP server.

Allows multiple MCP clients (agents, sub-agents) to share a single
server process instead of each spawning its own stdio process.

Usage:
    nmem-mcp --http                    # default port 8765
    nmem-mcp --http 9000              # custom port

Client configuration (Claude Code ~/.claude.json):
    {
        "mcpServers": {
            "neural-memory": {
                "url": "http://127.0.0.1:8765/mcp"
            }
        }
    }

Note: All clients share a single MCPServer instance. This means:
    - Session state (nmem_session) is shared — last writer wins
    - Passive capture rate limit (3/60s) is shared across all clients
    - Eternal context message counter is shared
    These are acceptable tradeoffs for multi-agent setups on the same
    brain, and match the behavior of sharing a single SQLite database.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)

_MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB


async def run_http_server(port: int = 8765, host: str = "127.0.0.1") -> None:
    """Start the MCP server with Streamable HTTP transport.

    Binds to 127.0.0.1 by default for security (no external access).
    Uses the same MCPServer and handle_message() as stdio transport.
    """
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route

    from neural_memory.mcp.server import (
        _lazy_init,
        create_mcp_server,
        handle_message,
    )

    _lazy_init()
    server = create_mcp_server()

    # Start background tasks (same as stdio transport)
    try:
        await server.maybe_start_mem0_sync()
    except Exception:
        logger.debug("Mem0 auto-sync startup failed (non-critical)", exc_info=True)

    try:
        await server.maybe_start_scheduled_consolidation()
    except Exception:
        logger.debug("Scheduled consolidation startup failed", exc_info=True)

    try:
        await server.maybe_start_version_check()
    except Exception:
        logger.debug("Version check startup failed", exc_info=True)

    async def mcp_endpoint(request: Request) -> Response:
        """Handle MCP JSON-RPC requests over HTTP POST."""
        if request.method == "GET":
            # Health check / info
            return JSONResponse(
                {
                    "status": "ok",
                    "server": "neural-memory-mcp",
                    "transport": "streamable-http",
                    "hint": "POST JSON-RPC messages to this endpoint",
                }
            )

        # POST — JSON-RPC message
        body = await request.body()
        if len(body) > _MAX_MESSAGE_SIZE:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32000, "message": "Message too large"},
                },
                status_code=413,
            )

        try:
            message = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                },
                status_code=400,
            )

        response = await handle_message(server, message)

        if response is None:
            # Notifications return no response (204 No Content)
            return PlainTextResponse("", status_code=204)

        return JSONResponse(response)

    async def health_endpoint(request: Request) -> Response:
        """Simple health check."""
        return JSONResponse({"status": "ok"})

    app = Starlette(
        routes=[
            Route("/mcp", mcp_endpoint, methods=["GET", "POST"]),
            Route("/health", health_endpoint, methods=["GET"]),
        ],
    )

    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    uv_server = uvicorn.Server(config)

    logger.info("NeuralMemory MCP HTTP transport on http://%s:%d/mcp", host, port)
    print(
        f"NeuralMemory MCP server (HTTP) listening on http://{host}:{port}/mcp",
        flush=True,
    )

    try:
        await uv_server.serve()
    finally:
        server.cancel_mem0_sync()
        server.cancel_expiry_cleanup()
        server.cancel_scheduled_consolidation()
        server.cancel_version_check()

        if server._storage is not None:
            await server._storage.close()
