"""CLI commands for Telegram backup integration."""

from __future__ import annotations

import asyncio
from typing import Any

import typer

app = typer.Typer(help="Telegram backup integration")


@app.command("status")
def telegram_status() -> None:
    """Show Telegram integration status."""
    from neural_memory.integration.telegram import get_telegram_status

    status = asyncio.run(get_telegram_status())

    if status.configured:
        typer.secho("Telegram: Configured", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"  Bot: {status.bot_name} (@{status.bot_username})")
    else:
        typer.secho("Telegram: Not configured", fg=typer.colors.YELLOW, bold=True)
        if status.error:
            typer.echo(f"  Error: {status.error}")

    if status.chat_ids:
        typer.echo(f"  Chat IDs: {', '.join(status.chat_ids)}")
    else:
        typer.echo("  Chat IDs: (none configured)")

    typer.echo(f"  Backup on consolidation: {status.backup_on_consolidation}")


@app.command("test")
def telegram_test() -> None:
    """Send a test message to verify configuration."""
    from neural_memory.integration.telegram import (
        TelegramClient,
        TelegramError,
        get_bot_token,
        get_telegram_config,
    )

    token = get_bot_token()
    if not token:
        typer.secho(
            "Error: NMEM_TELEGRAM_BOT_TOKEN environment variable not set.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    config = get_telegram_config()
    if not config.chat_ids:
        typer.secho(
            "Error: No chat IDs configured. Add [telegram] chat_ids to config.toml.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    client = TelegramClient(token)

    async def _test() -> None:
        bot_info = await client.get_me()
        typer.echo(f"Bot: {bot_info.get('first_name')} (@{bot_info.get('username')})")

        for chat_id in config.chat_ids:
            try:
                await client.send_message(
                    chat_id,
                    "🧠 <b>Neural Memory</b> — Test message\n\nTelegram integration is working!",
                )
                typer.secho(f"  Sent to {chat_id}", fg=typer.colors.GREEN)
            except TelegramError as exc:
                typer.secho(f"  Failed for {chat_id}: {exc}", fg=typer.colors.RED)

    try:
        asyncio.run(_test())
    except TelegramError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command("backup")
def telegram_backup(
    brain: str = typer.Option(None, "--brain", "-b", help="Brain name (default: active brain)"),
) -> None:
    """Send brain database file as backup to Telegram."""
    from neural_memory.integration.telegram import (
        TelegramClient,
        TelegramError,
        get_bot_token,
    )

    token = get_bot_token()
    if not token:
        typer.secho(
            "Error: NMEM_TELEGRAM_BOT_TOKEN environment variable not set.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    client = TelegramClient(token)

    async def _backup() -> dict[str, Any]:
        return await client.backup_brain(brain)

    try:
        result = asyncio.run(_backup())
        sent = result["sent_to"]
        failed = result["failed"]
        size_mb = result["size_bytes"] / (1024 * 1024)

        if sent > 0:
            typer.secho(
                f"Backup sent! Brain: {result['brain']}, Size: {size_mb:.1f}MB, "
                f"Sent to {sent} chat(s)",
                fg=typer.colors.GREEN,
            )
        if failed > 0:
            typer.secho(f"Failed: {failed} chat(s)", fg=typer.colors.YELLOW)
            for err in result.get("errors", []):
                typer.echo(f"  {err}")
    except TelegramError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)
