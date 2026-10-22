"""Tests for the API key middleware."""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from neural_memory.server.auth import APIKeyMiddleware


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("NEURAL_MEMORY_API_KEY", "secret")
    app = Starlette(routes=[Route("/x", lambda request: PlainTextResponse("ok"))])
    app.add_middleware(APIKeyMiddleware)
    return TestClient(app, raise_server_exceptions=False)


def test_valid_key_allowed(client: TestClient) -> None:
    assert client.get("/x", headers={"X-API-Key": "secret"}).status_code == 200


def test_wrong_key_rejected(client: TestClient) -> None:
    assert client.get("/x", headers={"X-API-Key": "nope"}).status_code == 403


def test_non_ascii_key_rejected_not_500(client: TestClient) -> None:
    """hmac.compare_digest raises TypeError on non-ASCII str; must be a 403."""
    response = client.get("/x", headers={"X-API-Key": "s\xe9cret".encode("latin-1")})
    assert response.status_code == 403
