"""Telegram Bot API client for brain backup and notifications.

Bot token: NMEM_TELEGRAM_BOT_TOKEN env var (never stored in config file).
Chat IDs: config.toml [telegram] section.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"


@dataclass(frozen=True)
class TelegramConfig:
    """Telegram integration configuration."""

    enabled: bool = False
    chat_ids: tuple[str, ...] = ()
    max_file_size_mb: int = 50
    backup_on_consolidation: bool = False


def get_telegram_config() -> TelegramConfig:
    """Load Telegram config from unified config."""
    from neural_memory.unified_config import get_config

    cfg = get_config()
    tg: TelegramConfig | None = getattr(cfg, "telegram", None)
    if tg is not None:
        return tg
    return TelegramConfig()


def get_bot_token() -> str | None:
    """Get bot token from environment variable only."""
    return os.environ.get("NMEM_TELEGRAM_BOT_TOKEN")


class TelegramClient:
    """Minimal async Telegram Bot API wrapper using aiohttp."""

    def __init__(self, bot_token: str) -> None:
        self._base_url = f"{API_BASE}/bot{bot_token}"

    async def _call(self, method: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a Telegram Bot API call."""
        from neural_memory.utils.ssl_helper import safe_client_session

        url = f"{self._base_url}/{method}"
        async with safe_client_session() as session:
            async with session.post(url, json=data) as resp:
                result = await resp.json()
                if not result.get("ok"):
                    desc = result.get("description", "Unknown error")
                    raise TelegramError(f"Telegram API {method}: {desc}")
                out: dict[str, Any] = result.get("result", {})
                return out

    async def _call_form(self, method: str, data: Any) -> dict[str, Any]:
        """Make a multipart form Telegram Bot API call."""
        from neural_memory.utils.ssl_helper import safe_client_session

        url = f"{self._base_url}/{method}"
        async with safe_client_session() as session:
            async with session.post(url, data=data) as resp:
                result = await resp.json()
                if not result.get("ok"):
                    desc = result.get("description", "Unknown error")
                    raise TelegramError(f"Telegram API {method}: {desc}")
                out: dict[str, Any] = result.get("result", {})
                return out

    async def get_me(self) -> dict[str, Any]:
        """Verify bot token and get bot info."""
        return await self._call("getMe")

    async def send_message(self, chat_id: str, text: str) -> dict[str, Any]:
        """Send a text message to a chat. Auto-splits messages > 4096 chars."""
        max_len = 4096
        if len(text) <= max_len:
            return await self._call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )

        # Split into chunks
        last_result: dict[str, Any] = {}
        for i in range(0, len(text), max_len):
            chunk = text[i : i + max_len]
            last_result = await self._call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
        return last_result

    async def send_document(
        self,
        chat_id: str,
        file_path: Path,
        caption: str | None = None,
    ) -> dict[str, Any]:
        """Send a file as a document to a chat."""
        import aiohttp

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        config = get_telegram_config()
        if file_size_mb > config.max_file_size_mb:
            raise TelegramError(
                f"File too large: {file_size_mb:.1f}MB (max: {config.max_file_size_mb}MB)"
            )

        data = aiohttp.FormData()
        data.add_field("chat_id", chat_id)
        data.add_field(
            "document",
            open(file_path, "rb"),
            filename=file_path.name,
            content_type="application/octet-stream",
        )
        if caption:
            data.add_field("caption", caption)
            data.add_field("parse_mode", "HTML")

        return await self._call_form("sendDocument", data)

    async def backup_brain(self, brain_name: str | None = None) -> dict[str, Any]:
        """Send brain database file as backup to all configured chat IDs."""
        from neural_memory.unified_config import get_config

        cfg = get_config()
        tg_config = get_telegram_config()

        if not tg_config.chat_ids:
            raise TelegramError("No chat IDs configured for Telegram backup")

        name = brain_name or cfg.current_brain
        db_path = Path(cfg.get_brain_db_path(name))

        if not db_path.exists():
            raise TelegramError(f"Brain database not found: {db_path}")

        from neural_memory.utils.timeutils import utcnow

        ts = utcnow().strftime("%Y%m%d-%H%M%S")
        caption = f"🧠 <b>Brain Backup</b>\n\nBrain: <code>{name}</code>\nTime: <code>{ts}</code>"

        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for chat_id in tg_config.chat_ids:
            try:
                result = await self.send_document(chat_id, db_path, caption)
                results.append(result)
                logger.info("Backup sent to chat %s for brain %s", chat_id, name)
            except Exception as exc:
                error_msg = f"Failed to send to chat {chat_id}: {exc}"
                errors.append(error_msg)
                logger.error(error_msg)

        return {
            "brain": name,
            "file": str(db_path),
            "size_bytes": db_path.stat().st_size,
            "sent_to": len(results),
            "failed": len(errors),
            "errors": errors,
        }


class TelegramError(Exception):
    """Telegram API error."""


@dataclass
class TelegramStatus:
    """Status information for Telegram integration."""

    configured: bool = False
    bot_name: str | None = None
    bot_username: str | None = None
    chat_ids: list[str] = field(default_factory=list)
    backup_on_consolidation: bool = False
    error: str | None = None


async def get_telegram_status() -> TelegramStatus:
    """Get current Telegram integration status."""
    token = get_bot_token()
    config = get_telegram_config()

    if not token:
        return TelegramStatus(
            configured=False,
            chat_ids=list(config.chat_ids),
            backup_on_consolidation=config.backup_on_consolidation,
            error="NMEM_TELEGRAM_BOT_TOKEN not set",
        )

    try:
        client = TelegramClient(token)
        bot_info = await client.get_me()
        return TelegramStatus(
            configured=True,
            bot_name=bot_info.get("first_name"),
            bot_username=bot_info.get("username"),
            chat_ids=list(config.chat_ids),
            backup_on_consolidation=config.backup_on_consolidation,
        )
    except Exception as exc:
        return TelegramStatus(
            configured=False,
            chat_ids=list(config.chat_ids),
            backup_on_consolidation=config.backup_on_consolidation,
            error=str(exc),
        )
