"""Tests for Telegram backup integration."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.integration.telegram import (
    TelegramClient,
    TelegramConfig,
    TelegramError,
    TelegramStatus,
    get_bot_token,
    get_telegram_config,
    get_telegram_status,
)


# ── Config tests ──────────────────────────────────────────────

class TestTelegramConfig:
    def test_default_config(self) -> None:
        cfg = TelegramConfig()
        assert cfg.enabled is False
        assert cfg.chat_ids == ()
        assert cfg.max_file_size_mb == 50
        assert cfg.backup_on_consolidation is False

    def test_config_immutable(self) -> None:
        cfg = TelegramConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = True  # type: ignore[misc]

    def test_config_with_values(self) -> None:
        cfg = TelegramConfig(
            enabled=True,
            chat_ids=("123", "456"),
            max_file_size_mb=100,
            backup_on_consolidation=True,
        )
        assert cfg.enabled is True
        assert cfg.chat_ids == ("123", "456")
        assert cfg.max_file_size_mb == 100
        assert cfg.backup_on_consolidation is True

    def test_unified_config_from_dict(self) -> None:
        """TelegramConfig.from_dict lives on unified_config module."""
        from neural_memory.unified_config import TelegramConfig as UnifiedTelegramConfig

        data = {
            "enabled": True,
            "chat_ids": ["123", "456"],
            "max_file_size_mb": 100,
            "backup_on_consolidation": True,
        }
        cfg = UnifiedTelegramConfig.from_dict(data)
        assert cfg.enabled is True
        assert cfg.chat_ids == ("123", "456")

    def test_unified_config_to_dict(self) -> None:
        from neural_memory.unified_config import TelegramConfig as UnifiedTelegramConfig

        cfg = UnifiedTelegramConfig(enabled=True, chat_ids=("111",))
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert d["chat_ids"] == ["111"]

    def test_config_replace(self) -> None:
        cfg = TelegramConfig()
        new = replace(cfg, enabled=True)
        assert new.enabled is True
        assert cfg.enabled is False  # original unchanged


# ── Token tests ──────────────────────────────────────────────

class TestGetBotToken:
    def test_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NMEM_TELEGRAM_BOT_TOKEN", "test-token-123")
        assert get_bot_token() == "test-token-123"

    def test_token_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NMEM_TELEGRAM_BOT_TOKEN", raising=False)
        assert get_bot_token() is None


# ── Client tests ──────────────────────────────────────────────

class TestTelegramClient:
    @pytest.fixture
    def client(self) -> TelegramClient:
        return TelegramClient("test-bot-token")

    @pytest.mark.asyncio
    async def test_get_me_success(self, client: TelegramClient) -> None:
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={
            "ok": True,
            "result": {"id": 123, "first_name": "TestBot", "username": "testbot"},
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.get_me()
            assert result["username"] == "testbot"
            assert result["first_name"] == "TestBot"

    @pytest.mark.asyncio
    async def test_api_error_raises(self, client: TelegramClient) -> None:
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={
            "ok": False,
            "description": "Unauthorized",
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(TelegramError, match="Unauthorized"):
                await client.get_me()

    @pytest.mark.asyncio
    async def test_send_message(self, client: TelegramClient) -> None:
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={
            "ok": True,
            "result": {"message_id": 1},
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await client.send_message("123", "Hello")
            assert result["message_id"] == 1

    @pytest.mark.asyncio
    async def test_message_auto_split(self, client: TelegramClient) -> None:
        """Messages > 4096 chars should be split."""
        long_text = "x" * 5000

        call_count = 0
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={
            "ok": True,
            "result": {"message_id": 1},
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()

        def track_post(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return mock_resp

        mock_session.post = MagicMock(side_effect=track_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await client.send_message("123", long_text)
            assert call_count >= 2


# ── Status tests ──────────────────────────────────────────────

class TestTelegramStatus:
    @pytest.mark.asyncio
    async def test_status_not_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NMEM_TELEGRAM_BOT_TOKEN", raising=False)
        status = await get_telegram_status()
        assert status.configured is False
        assert status.bot_name is None

    @pytest.mark.asyncio
    async def test_status_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NMEM_TELEGRAM_BOT_TOKEN", "test-token")

        mock_config = TelegramConfig(
            enabled=True,
            chat_ids=("111",),
            backup_on_consolidation=True,
        )

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={
            "ok": True,
            "result": {"first_name": "TestBot", "username": "testbot"},
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("neural_memory.integration.telegram.get_telegram_config", return_value=mock_config),
            patch("aiohttp.ClientSession", return_value=mock_session),
        ):
            status = await get_telegram_status()
            assert status.configured is True
            assert status.bot_name == "TestBot"
            assert status.bot_username == "testbot"
            assert status.chat_ids == ["111"]
            assert status.backup_on_consolidation is True


# ── MCP handler tests ──────────────────────────────────────────

class TestTelegramHandler:
    @pytest.mark.asyncio
    async def test_backup_no_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NMEM_TELEGRAM_BOT_TOKEN", raising=False)

        from neural_memory.mcp.telegram_handler import TelegramHandler

        class FakeServer(TelegramHandler):
            pass

        server = FakeServer()
        result = await server._telegram_backup({})
        assert "error" in result
        assert "NMEM_TELEGRAM_BOT_TOKEN" in result["error"]
