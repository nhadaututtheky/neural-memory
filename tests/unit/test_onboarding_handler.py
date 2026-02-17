"""Tests for the fresh-brain onboarding handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.mcp.onboarding_handler import ONBOARDING_STEPS, OnboardingHandler


def _make_handler(
    *, neuron_count: int = 0, fiber_count: int = 0, stats_error: bool = False
) -> OnboardingHandler:
    """Create a minimal OnboardingHandler with mocked storage."""
    handler = OnboardingHandler()
    handler.config = MagicMock()  # type: ignore[attr-defined]

    mock_storage = AsyncMock()
    mock_storage._current_brain_id = "test-brain"

    if stats_error:
        mock_storage.get_stats = AsyncMock(side_effect=RuntimeError("db error"))
    else:
        mock_storage.get_stats = AsyncMock(
            return_value={
                "neuron_count": neuron_count,
                "fiber_count": fiber_count,
                "synapse_count": 0,
            }
        )

    handler.get_storage = AsyncMock(return_value=mock_storage)  # type: ignore[attr-defined]
    return handler


class TestCheckOnboarding:
    """Tests for _check_onboarding method."""

    @pytest.mark.asyncio
    async def test_fresh_brain_returns_onboarding(self) -> None:
        """Empty brain returns structured onboarding data."""
        handler = _make_handler(neuron_count=0, fiber_count=0)
        result = await handler._check_onboarding()

        assert result is not None
        assert result["onboarding"] is True
        assert "Welcome" in result["message"]
        assert len(result["steps"]) == len(ONBOARDING_STEPS)

    @pytest.mark.asyncio
    async def test_onboarding_steps_structure(self) -> None:
        """Each step has required fields."""
        handler = _make_handler()
        result = await handler._check_onboarding()

        assert result is not None
        for step in result["steps"]:
            assert "step" in step
            assert "title" in step
            assert "description" in step
            assert "example_tool" in step
            assert "example_args" in step

    @pytest.mark.asyncio
    async def test_non_empty_brain_returns_none(self) -> None:
        """Brain with data returns None."""
        handler = _make_handler(neuron_count=10, fiber_count=5)
        result = await handler._check_onboarding()

        assert result is None

    @pytest.mark.asyncio
    async def test_neurons_only_returns_none(self) -> None:
        """Brain with neurons but no fibers returns None."""
        handler = _make_handler(neuron_count=5, fiber_count=0)
        result = await handler._check_onboarding()

        assert result is None

    @pytest.mark.asyncio
    async def test_flag_prevents_repeat(self) -> None:
        """Second call always returns None."""
        handler = _make_handler(neuron_count=0, fiber_count=0)

        first = await handler._check_onboarding()
        assert first is not None

        second = await handler._check_onboarding()
        assert second is None

    @pytest.mark.asyncio
    async def test_flag_set_on_non_empty_brain(self) -> None:
        """Flag set even when brain has data (so we don't recheck)."""
        handler = _make_handler(neuron_count=10, fiber_count=5)

        result = await handler._check_onboarding()
        assert result is None
        assert handler._onboarding_shown is True

    @pytest.mark.asyncio
    async def test_stats_failure_returns_none(self) -> None:
        """Storage error returns None gracefully."""
        handler = _make_handler(stats_error=True)
        result = await handler._check_onboarding()

        assert result is None

    @pytest.mark.asyncio
    async def test_stats_failure_does_not_set_flag(self) -> None:
        """Storage error doesn't set the flag, allowing retry."""
        handler = _make_handler(stats_error=True)
        await handler._check_onboarding()

        assert handler._onboarding_shown is False


class TestOnboardingIntegration:
    """Tests for onboarding injection into tool handlers."""

    @pytest.fixture
    def server(self) -> MagicMock:
        with patch("neural_memory.mcp.server.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                current_brain="test-brain",
                get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            )
            from neural_memory.mcp.server import MCPServer

            return MCPServer()

    @pytest.mark.asyncio
    async def test_context_empty_includes_onboarding(self, server: MagicMock) -> None:
        """_context on empty brain includes onboarding key."""
        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test-brain"
        mock_storage.get_fibers = AsyncMock(return_value=[])
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 0, "fiber_count": 0, "synapse_count": 0}
        )

        with (
            patch.object(server, "get_storage", return_value=mock_storage),
            patch.object(server, "_record_tool_action", new_callable=AsyncMock),
        ):
            result = await server.call_tool("nmem_context", {"limit": 5})

        assert result["count"] == 0
        assert "onboarding" in result
        assert result["onboarding"]["onboarding"] is True

    @pytest.mark.asyncio
    async def test_stats_includes_onboarding_on_fresh(self, server: MagicMock) -> None:
        """_stats on fresh brain includes onboarding key."""
        mock_storage = AsyncMock()
        mock_storage._current_brain_id = "test-brain"
        mock_storage.get_brain = AsyncMock(
            return_value=MagicMock(id="test-brain", name="test-brain", config=MagicMock())
        )
        mock_storage.get_enhanced_stats = AsyncMock(
            return_value={
                "neuron_count": 0,
                "synapse_count": 0,
                "fiber_count": 0,
                "db_size_bytes": 0,
                "today_fibers_count": 0,
                "hot_neurons": [],
                "newest_memory": None,
            }
        )
        mock_storage.get_synapses = AsyncMock(return_value=[])
        mock_storage.get_stats = AsyncMock(
            return_value={"neuron_count": 0, "fiber_count": 0, "synapse_count": 0}
        )

        with patch.object(server, "get_storage", return_value=mock_storage):
            result = await server.call_tool("nmem_stats", {})

        assert "onboarding" in result
        assert result["onboarding"]["onboarding"] is True
