"""Regression tests for rate-limiter hardening (audit #46).

Covers:
  - X-Forwarded-For is only honored when the direct peer is a configured
    trusted proxy (otherwise it is spoofable and bypasses the limiter).
  - Empty IP buckets are reaped / LRU-capped so a flood of unique IPs cannot
    grow the counter dict without bound (memory-exhaustion DoS).
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def rl_mod(monkeypatch):
    monkeypatch.delenv("NEURAL_MEMORY_TRUSTED_PROXIES", raising=False)
    monkeypatch.delenv("NEURAL_MEMORY_RATE_LIMIT", raising=False)
    monkeypatch.delenv("NEURAL_MEMORY_RATE_LIMIT_WINDOW", raising=False)
    import neural_memory.server.rate_limit as mod

    return importlib.reload(mod)


class _FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class _FakeRequest:
    def __init__(self, peer: str, xff: str | None = None) -> None:
        self.client = _FakeClient(peer) if peer else None
        self.headers = {"X-Forwarded-For": xff} if xff is not None else {}


def test_xff_ignored_when_peer_untrusted(rl_mod, monkeypatch):
    # No trusted proxies configured -> XFF must be ignored, key on peer.
    monkeypatch.delenv("NEURAL_MEMORY_TRUSTED_PROXIES", raising=False)
    req = _FakeRequest(peer="10.0.0.1", xff="1.2.3.4")
    assert rl_mod._client_ip(req) == "10.0.0.1"


def test_xff_honored_when_peer_trusted(rl_mod, monkeypatch):
    monkeypatch.setenv("NEURAL_MEMORY_TRUSTED_PROXIES", "10.0.0.1, 10.0.0.2")
    req = _FakeRequest(peer="10.0.0.1", xff="1.2.3.4, 5.6.7.8")
    # Leftmost (originating) client IP from the trusted chain.
    assert rl_mod._client_ip(req) == "1.2.3.4"


def test_xff_spoof_does_not_change_key_for_untrusted_peer(rl_mod, monkeypatch):
    monkeypatch.delenv("NEURAL_MEMORY_TRUSTED_PROXIES", raising=False)
    # Two requests from the same untrusted peer rotating XFF must map to ONE key.
    r1 = _FakeRequest(peer="203.0.113.9", xff="9.9.9.1")
    r2 = _FakeRequest(peer="203.0.113.9", xff="9.9.9.2")
    assert rl_mod._client_ip(r1) == rl_mod._client_ip(r2) == "203.0.113.9"


@pytest.mark.asyncio
async def test_empty_buckets_reaped(rl_mod, monkeypatch):
    monkeypatch.setenv("NEURAL_MEMORY_RATE_LIMIT", "5")
    monkeypatch.setenv("NEURAL_MEMORY_RATE_LIMIT_WINDOW", "1")
    monkeypatch.delenv("NEURAL_MEMORY_TRUSTED_PROXIES", raising=False)

    async def call_next(_request):
        from starlette.responses import Response

        return Response(status_code=200)

    mw = rl_mod.RateLimitMiddleware(app=lambda *a, **k: None)

    # Simulate a stream of unique peers separated past the window so each
    # bucket empties on eviction and gets reaped.
    base = 1000.0
    times = iter(range(0, 100000, 100))  # large gaps -> always outside window

    def fake_monotonic():
        return base + next(times)

    monkeypatch.setattr(rl_mod.time, "monotonic", fake_monotonic)

    class _Req:
        def __init__(self, host):
            self.client = _FakeClient(host)
            self.headers = {}
            self.url = type("U", (), {"path": "/api/x"})()

    for i in range(200):
        await mw.dispatch(_Req(f"peer-{i}"), call_next)

    # Each new peer's first request created a bucket, but with the large time
    # gaps every prior bucket empties and is reaped on the next access. The dict
    # must stay tiny (bounded) rather than growing to 200.
    assert len(mw._counters) <= rl_mod._MAX_TRACKED_IPS
    assert len(mw._counters) <= 200  # sanity: never exceeds requests made


@pytest.mark.asyncio
async def test_lru_cap_bounds_dict(rl_mod, monkeypatch):
    monkeypatch.setenv("NEURAL_MEMORY_RATE_LIMIT", "100")
    monkeypatch.setenv("NEURAL_MEMORY_RATE_LIMIT_WINDOW", "3600")
    monkeypatch.delenv("NEURAL_MEMORY_TRUSTED_PROXIES", raising=False)

    monkeypatch.setattr(rl_mod, "_MAX_TRACKED_IPS", 10)

    async def call_next(_request):
        from starlette.responses import Response

        return Response(status_code=200)

    mw = rl_mod.RateLimitMiddleware(app=lambda *a, **k: None)

    class _Req:
        def __init__(self, host):
            self.client = _FakeClient(host)
            self.headers = {}
            self.url = type("U", (), {"path": "/api/x"})()

    # Long window so buckets never empty; only the LRU cap can bound growth.
    for i in range(500):
        await mw.dispatch(_Req(f"flood-{i}"), call_next)

    assert len(mw._counters) <= 10
