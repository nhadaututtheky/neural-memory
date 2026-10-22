"""The /sync/ws route must resolve its router-level locality dependency."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from neural_memory.server.routes.sync import SyncManager, router


def _client(host: str) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, client=(host, 50000))


@pytest.fixture(autouse=True)
def _fresh_manager() -> Iterator[None]:
    SyncManager._instance = None
    yield
    SyncManager._instance = None


def test_websocket_connects_from_localhost() -> None:
    with _client("127.0.0.1").websocket_connect("/sync/ws") as ws:
        ws.send_text(json.dumps({"action": "connect"}))
        assert json.loads(ws.receive_text())["type"] == "connected"
        ws.send_text(json.dumps({"action": "ping"}))
        assert json.loads(ws.receive_text())["type"] == "pong"


def test_websocket_rejected_from_untrusted_host() -> None:
    with pytest.raises(WebSocketDisconnect):
        with _client("203.0.113.9").websocket_connect("/sync/ws"):
            pass
