"""Tests for MCP HTTP transport."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture()
def mock_server():
    """Create a mock MCPServer for testing."""
    from neural_memory.mcp.server import MCPServer

    server = AsyncMock(spec=MCPServer)
    server.get_tools.return_value = [{"name": "nmem_recall"}]
    server.get_resources.return_value = []
    server.maybe_start_mem0_sync = AsyncMock()
    server.maybe_start_scheduled_consolidation = AsyncMock()
    server.maybe_start_version_check = AsyncMock()
    server.cancel_mem0_sync = AsyncMock()
    server.cancel_expiry_cleanup = AsyncMock()
    server.cancel_scheduled_consolidation = AsyncMock()
    server.cancel_version_check = AsyncMock()
    server._storage = None
    server._post_tool_capture = AsyncMock()
    return server


@pytest.fixture()
def app(mock_server):
    """Create Starlette test app using mocked server."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route

    from neural_memory.mcp.server import handle_message

    async def mcp_endpoint(request):
        if request.method == "GET":
            return JSONResponse(
                {
                    "status": "ok",
                    "server": "neural-memory-mcp",
                    "transport": "streamable-http",
                }
            )

        body = await request.body()
        if len(body) > 10 * 1024 * 1024:
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
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
                status_code=400,
            )

        response = await handle_message(mock_server, message)
        if response is None:
            return PlainTextResponse("", status_code=204)
        return JSONResponse(response)

    async def health_endpoint(request):
        return JSONResponse({"status": "ok"})

    return Starlette(
        routes=[
            Route("/mcp", mcp_endpoint, methods=["GET", "POST"]),
            Route("/health", health_endpoint, methods=["GET"]),
        ]
    )


@pytest.fixture()
def client(app):
    """Create test client."""
    from starlette.testclient import TestClient

    return TestClient(app)


class TestHTTPTransport:
    """Test MCP HTTP transport endpoints."""

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_mcp_get_info(self, client):
        resp = client.get("/mcp")
        assert resp.status_code == 200
        data = resp.json()
        assert data["transport"] == "streamable-http"
        assert data["server"] == "neural-memory-mcp"

    def test_initialize(self, client):
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == 1
        assert data["result"]["serverInfo"]["name"] == "neural-memory"
        assert data["result"]["protocolVersion"] == "2024-11-05"

    def test_tools_list(self, client, mock_server):
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data["result"]
        mock_server.get_tools.assert_called()

    def test_notification_returns_204(self, client):
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
        )
        assert resp.status_code == 204

    def test_invalid_json_returns_400(self, client):
        resp = client.post(
            "/mcp", content=b"not json", headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == -32700

    def test_unknown_method_returns_error(self, client):
        resp = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "unknown/method",
                "params": {},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"]["code"] == -32601

    def test_main_parses_http_flag(self):
        """Test that main() parses --http flag correctly."""
        with (
            patch("neural_memory.mcp.server.asyncio") as mock_asyncio,
            patch("neural_memory.mcp.http_transport.run_http_server"),
        ):
            mock_asyncio.run = lambda coro: None
            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["nmem-mcp", "--http"]
                from neural_memory.mcp.server import main

                main()
                # Verify asyncio.run was called (with http server coroutine)
            finally:
                sys.argv = old_argv

    def test_main_default_is_stdio(self):
        """Test that main() defaults to stdio transport."""
        with patch("neural_memory.mcp.server.asyncio") as mock_asyncio:
            mock_asyncio.run = lambda coro: None
            import sys

            old_argv = sys.argv
            try:
                sys.argv = ["nmem-mcp"]
                from neural_memory.mcp.server import main

                main()
            finally:
                sys.argv = old_argv
