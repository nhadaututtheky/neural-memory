"""Tests for ``neural_memory.utils.sandbox`` probe + entry-point guards.

Covers issue #151: storage-backed commands must fail fast in restricted
environments where ``aiosqlite`` cross-thread loop wakeups are blocked,
not hang silently.
"""

from __future__ import annotations

import asyncio
import gc
import os
import warnings
from typing import Any
from unittest.mock import patch

import pytest

from neural_memory.utils import sandbox


@pytest.fixture(autouse=True)
def _reset_cache() -> Any:
    """Reset the module-level probe cache before and after every test."""
    sandbox.reset_probe_cache()
    os.environ.pop(sandbox.BYPASS_ENV, None)
    yield
    sandbox.reset_probe_cache()
    os.environ.pop(sandbox.BYPASS_ENV, None)


class TestProbeBasics:
    def test_probe_succeeds_in_dev_env(self) -> None:
        """Sanity check: aiosqlite works on the test runner."""
        result = sandbox.probe_aiosqlite_compat()
        assert result.ok
        assert "succeeded" in result.detail or "bypassed" in result.detail

    def test_result_is_cached(self) -> None:
        first = sandbox.probe_aiosqlite_compat()
        with patch.object(sandbox, "_probe_coro") as fake_probe:
            second = sandbox.probe_aiosqlite_compat()
            fake_probe.assert_not_called()
        assert first is second

    def test_force_reruns_probe(self) -> None:
        sandbox.probe_aiosqlite_compat()  # warm cache

        async def _fake_probe(_timeout: float) -> sandbox.ProbeResult:
            return sandbox.ProbeResult(ok=True, detail="forced")

        with patch.object(sandbox, "_probe_coro", _fake_probe):
            result = sandbox.probe_aiosqlite_compat(force=True)
        assert result.detail == "forced"

    def test_reset_cache_clears_state(self) -> None:
        sandbox.probe_aiosqlite_compat()
        sandbox.reset_probe_cache()
        assert sandbox._cached_result is None


class TestProbeBypass:
    def test_bypass_env_returns_ok_without_probing(self) -> None:
        os.environ[sandbox.BYPASS_ENV] = "1"
        with patch.object(sandbox, "_probe_coro") as fake_probe:
            result = sandbox.probe_aiosqlite_compat()
            fake_probe.assert_not_called()
        assert result.ok
        assert "bypass" in result.detail.lower()

    @pytest.mark.parametrize("value", ["1", "true", "YES", "On"])
    def test_bypass_accepts_truthy_values(self, value: str) -> None:
        os.environ[sandbox.BYPASS_ENV] = value
        assert sandbox._is_bypassed()

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_bypass_rejects_falsy_values(self, value: str) -> None:
        os.environ[sandbox.BYPASS_ENV] = value
        assert not sandbox._is_bypassed()


class TestProbeFailurePaths:
    def test_timeout_returns_failure_result(self) -> None:
        async def _hang(_timeout: float) -> sandbox.ProbeResult:
            return sandbox.ProbeResult(
                ok=False,
                detail="aiosqlite connect timed out after 2.0s",
                error_class="TimeoutError",
            )

        with patch.object(sandbox, "_probe_coro", _hang):
            result = sandbox.probe_aiosqlite_compat(force=True)
        assert not result.ok
        assert result.error_class == "TimeoutError"

    def test_actual_timeout_in_probe_coro(self) -> None:
        """Real timeout path: connect that never completes."""

        class _StuckConnect:
            def __await__(self) -> Any:
                async def _never() -> None:
                    await asyncio.sleep(60)

                return _never().__await__()

        with patch("aiosqlite.connect", return_value=_StuckConnect()):
            result = asyncio.run(sandbox._probe_coro(timeout=0.05))
        assert not result.ok
        assert result.error_class == "TimeoutError"
        assert "timed out" in result.detail


class TestEnsureHelpers:
    def test_ensure_cli_passes_when_ok(self) -> None:
        sandbox._cached_result = sandbox.ProbeResult(ok=True, detail="ok")
        sandbox.ensure_aiosqlite_or_exit_cli()  # must not raise

    def test_ensure_cli_exits_when_failed(self) -> None:
        import typer

        sandbox._cached_result = sandbox.ProbeResult(
            ok=False, detail="aiosqlite hung", error_class="TimeoutError"
        )
        with pytest.raises(typer.Exit) as exc_info:
            sandbox.ensure_aiosqlite_or_exit_cli()
        assert exc_info.value.exit_code == 2

    def test_ensure_raise_passes_when_ok(self) -> None:
        sandbox._cached_result = sandbox.ProbeResult(ok=True, detail="ok")
        sandbox.ensure_aiosqlite_or_raise()

    def test_ensure_raise_throws_when_failed(self) -> None:
        sandbox._cached_result = sandbox.ProbeResult(
            ok=False, detail="aiosqlite hung", error_class="TimeoutError"
        )
        with pytest.raises(sandbox.SandboxIncompatibleError) as exc_info:
            sandbox.ensure_aiosqlite_or_raise()
        assert exc_info.value.detail == "aiosqlite hung"


class TestDoctorIntegration:
    def test_dependencies_reports_sandbox_failure(self) -> None:
        from neural_memory.cli.doctor import _check_dependencies

        sandbox._cached_result = sandbox.ProbeResult(
            ok=False,
            detail="aiosqlite connect timed out after 2.0s",
            error_class="TimeoutError",
        )
        result = _check_dependencies()
        assert result["status"] == "fail"
        assert "aiosqlite" in result["detail"]
        assert "issues/151" in result.get("fix", "")

    def test_dependencies_ok_when_probe_passes(self) -> None:
        from neural_memory.cli.doctor import _check_dependencies

        sandbox._cached_result = sandbox.ProbeResult(ok=True, detail="ok")
        result = _check_dependencies()
        assert result["status"] == "ok"


class TestCliRunAsyncIntegration:
    def test_run_async_aborts_when_probe_fails(self) -> None:
        import typer

        from neural_memory.cli._helpers import run_async

        sandbox._cached_result = sandbox.ProbeResult(
            ok=False, detail="hung", error_class="TimeoutError"
        )

        async def _noop() -> int:
            return 1

        with pytest.raises(typer.Exit):
            run_async(_noop())

    def test_run_async_closes_unawaited_coro_when_probe_fails(self) -> None:
        import typer

        from neural_memory.cli._helpers import run_async

        sandbox._cached_result = sandbox.ProbeResult(
            ok=False, detail="hung", error_class="TimeoutError"
        )

        async def _context() -> int:
            return 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            with pytest.raises(typer.Exit):
                run_async(_context())
            gc.collect()

        warning_messages = [str(w.message) for w in caught]
        assert not any(
            "coroutine" in msg and "was never awaited" in msg for msg in warning_messages
        )

    def test_run_async_executes_when_probe_ok(self) -> None:
        from neural_memory.cli._helpers import run_async

        sandbox._cached_result = sandbox.ProbeResult(ok=True, detail="ok")

        async def _coro() -> int:
            return 42

        assert run_async(_coro()) == 42
