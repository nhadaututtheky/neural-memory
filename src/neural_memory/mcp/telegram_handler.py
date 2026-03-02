"""MCP tool handler for Telegram backup operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class TelegramHandler:
    """Mixin providing Telegram backup MCP tool handler."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _telegram_backup(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_telegram_backup tool call."""
        try:
            from neural_memory.integration.telegram import (
                TelegramClient,
                TelegramError,
                get_bot_token,
            )

            token = get_bot_token()
            if not token:
                return {
                    "error": "NMEM_TELEGRAM_BOT_TOKEN environment variable not set.",
                }

            brain_name = args.get("brain_name")
            client = TelegramClient(token)

            result = await client.backup_brain(brain_name)

            sent = result["sent_to"]
            failed = result["failed"]
            size_mb = result["size_bytes"] / (1024 * 1024)

            response: dict[str, Any] = {
                "status": "success" if sent > 0 else "failed",
                "brain": result["brain"],
                "size_mb": round(size_mb, 1),
                "sent_to": sent,
                "failed": failed,
            }

            if result.get("errors"):
                response["errors"] = result["errors"]

            return response

        except TelegramError as exc:
            logger.error("Telegram backup failed: %s", exc)
            return {"error": str(exc)}
        except Exception:
            logger.error("Telegram backup failed unexpectedly", exc_info=True)
            return {"error": "Telegram backup failed unexpectedly."}
