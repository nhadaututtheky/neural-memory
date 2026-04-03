"""Shared mode commands for remote brain synchronization."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

import typer

from neural_memory.cli._helpers import get_config, get_storage, run_async

logger = logging.getLogger(__name__)

shared_app = typer.Typer(help="Real-time brain sharing configuration")


@shared_app.command("enable")
def shared_enable(
    server_url: Annotated[
        str, typer.Argument(help="NeuralMemory server URL (e.g., http://localhost:8000)")
    ],
    api_key: Annotated[
        str | None, typer.Option("--api-key", "-k", help="API key for authentication")
    ] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", "-t", help="Request timeout in seconds")
    ] = 30.0,
) -> None:
    """Enable shared mode to connect to a remote NeuralMemory server.

    When shared mode is enabled, all memory operations (remember, recall, etc.)
    will use the remote server instead of local storage.

    Examples:
        nmem shared enable http://localhost:8000
        nmem shared enable https://memory.example.com --api-key mykey
        nmem shared enable http://server:8000 --timeout 60
    """
    config = get_config()
    config.shared.enabled = True
    config.shared.server_url = server_url.rstrip("/")
    config.shared.api_key = api_key
    config.shared.timeout = timeout
    config.save()

    typer.secho("Shared mode enabled!", fg=typer.colors.GREEN)
    typer.echo(f"  Server: {config.shared.server_url}")
    if api_key:
        typer.echo(f"  API Key: {'*' * 8}...{api_key[-4:] if len(api_key) > 4 else '****'}")
    typer.echo(f"  Timeout: {timeout}s")
    typer.echo("")
    typer.secho("All memory commands will now use the remote server.", fg=typer.colors.BRIGHT_BLACK)
    typer.secho(
        "Use 'nmem shared disable' to switch back to local storage.", fg=typer.colors.BRIGHT_BLACK
    )


@shared_app.command("disable")
def shared_disable() -> None:
    """Disable shared mode and use local storage.

    Examples:
        nmem shared disable
    """
    config = get_config()
    config.shared.enabled = False
    config.save()

    typer.secho("Shared mode disabled.", fg=typer.colors.GREEN)
    typer.echo("Memory commands will now use local storage.")


@shared_app.command("status")
def shared_status(
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show shared mode status and configuration.

    Examples:
        nmem shared status
        nmem shared status --json
    """
    config = get_config()

    status = {
        "enabled": config.shared.enabled,
        "server_url": config.shared.server_url,
        "api_key_set": config.shared.api_key is not None,
        "timeout": config.shared.timeout,
    }

    if json_output:
        typer.echo(json.dumps(status, indent=2))
    else:
        if config.shared.enabled:
            typer.secho("[ENABLED] Shared mode is active", fg=typer.colors.GREEN)
        else:
            typer.secho("[DISABLED] Using local storage", fg=typer.colors.YELLOW)

        typer.echo(f"\nServer URL: {config.shared.server_url}")
        typer.echo(f"API Key: {'configured' if config.shared.api_key else 'not set'}")
        typer.echo(f"Timeout: {config.shared.timeout}s")

        # License info
        from neural_memory.unified_config import get_config as get_unified_config

        uconfig = get_unified_config()
        typer.echo()
        typer.secho("License", bold=True)
        if uconfig.is_pro():
            typer.echo(f"  Tier:     {uconfig.license.tier}")
            typer.echo(f"  Expires:  {uconfig.license.expires_at or 'perpetual'}")
        else:
            from neural_memory.mcp.sync_handler import PRO_LANDING_URL

            typer.echo("  Tier:     free")
            typer.echo(f"  Upgrade:  {PRO_LANDING_URL}")


@shared_app.command("test")
def shared_test() -> None:
    """Test connection to the shared server.

    Examples:
        nmem shared test
    """
    config = get_config()

    if not config.shared.server_url:
        typer.secho(
            "No server URL configured. Use 'nmem shared enable <url>' first.", fg=typer.colors.RED
        )
        raise typer.Exit(1)

    async def _test() -> dict[str, Any]:
        import aiohttp

        url = f"{config.shared.server_url}/health"
        headers = {}
        if config.shared.api_key:
            headers["Authorization"] = f"Bearer {config.shared.api_key}"

        try:
            from neural_memory.utils.ssl_helper import safe_client_session

            conn_timeout = aiohttp.ClientTimeout(total=config.shared.timeout)
            async with safe_client_session(timeout=conn_timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "status": data.get("status", "unknown"),
                            "version": data.get("version", "unknown"),
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Server returned status {response.status}",
                        }
        except aiohttp.ClientError as e:
            logger.error("Connection test failed: %s", e)
            return {"success": False, "error": "Connection failed: server unreachable"}
        except Exception as e:
            logger.error("Connection test failed: %s", e)
            return {"success": False, "error": "Connection test failed unexpectedly"}

    typer.echo(f"Testing connection to {config.shared.server_url}...")
    result = run_async(_test())

    if result["success"]:
        typer.secho("[OK] Connection successful!", fg=typer.colors.GREEN)
        typer.echo(f"  Server status: {result['status']}")
        typer.echo(f"  Server version: {result['version']}")
    else:
        typer.secho("[FAILED] Connection failed!", fg=typer.colors.RED)
        typer.echo(f"  Error: {result['error']}")
        raise typer.Exit(1)


@shared_app.command("activate")
def shared_activate(
    key: Annotated[str, typer.Option("--key", "-k", help="License key (NM-PRO-XXXX-XXXX-XXXX)")],
) -> None:
    """Activate a NeuralMemory Pro license key.

    Examples:
        nmem shared activate --key NM-PRO-XXXX-XXXX-XXXX
    """
    # Normalize XLabs format (NM-PRO-XXXX) → nm_pro_xxxx
    license_key = key.strip()
    if license_key.startswith("NM-"):
        license_key = license_key.replace("-", "_").lower()
    if not license_key.startswith("nm_"):
        typer.secho(
            "Invalid license key format. Expected NM-PRO-* or nm_pro_*", fg=typer.colors.RED
        )
        raise typer.Exit(1)

    async def _activate() -> dict[str, Any]:
        import aiohttp

        from neural_memory.mcp.sync_handler import DEFAULT_PAY_URL
        from neural_memory.utils.ssl_helper import safe_client_session

        pay_url = f"{DEFAULT_PAY_URL}/verify"
        headers = {
            "Content-Type": "application/json",
        }
        try:
            async with safe_client_session() as session:
                async with session.post(
                    pay_url,
                    json={"key": key.strip()},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        return {
                            "success": False,
                            "error": data.get("error", f"Activation failed ({resp.status})"),
                        }

                    return {
                        "success": True,
                        "tier": str(data.get("tier", "pro")).lower(),
                        "expires_at": data.get("expires_at"),
                        "features": data.get("features", []),
                        "message": data.get("message", "License activated!"),
                    }
        except aiohttp.ClientError as e:
            logger.error("Activation failed: %s", e)
            return {"success": False, "error": "Connection failed: hub unreachable"}
        except Exception as e:
            logger.error("Activation failed: %s", e)
            return {"success": False, "error": "Activation failed unexpectedly"}

    typer.echo("Activating license key...")
    result = run_async(_activate())

    if not result["success"]:
        typer.secho(f"[FAILED] {result['error']}", fg=typer.colors.RED)
        raise typer.Exit(1)

    # Persist license to config.toml
    from dataclasses import replace

    from neural_memory.unified_config import LicenseConfig, set_config
    from neural_memory.unified_config import get_config as get_unified_config
    from neural_memory.utils.timeutils import utcnow

    uconfig = get_unified_config(reload=True)
    new_license = LicenseConfig.from_dict(
        {
            "tier": result["tier"],
            "activated_at": utcnow().isoformat(),
            "expires_at": result.get("expires_at", ""),
        }
    )
    updated_config = replace(uconfig, license=new_license)
    updated_config.save()
    set_config(updated_config)

    typer.secho(f"[OK] {result['message']}", fg=typer.colors.GREEN)
    typer.echo(f"  Tier: {result['tier'].upper()}")
    if result.get("expires_at"):
        typer.echo(f"  Expires: {result['expires_at']}")
    if result.get("features"):
        typer.echo(f"  Features: {', '.join(result['features'])}")
    typer.echo()

    # Guide to InfinityDB if on SQLite
    if uconfig.storage_backend == "sqlite":
        typer.echo("  Next: unlock InfinityDB (HNSW, tiered compression, cone queries)")
        typer.echo("    nmem storage status")
        typer.echo("    nmem migrate infinitydb")
        typer.echo("    nmem storage switch infinitydb")
        typer.echo()

    # Check if Pro deps are installed, offer to install if missing
    from neural_memory.pro import get_missing_deps

    missing = get_missing_deps()
    if missing:
        typer.secho(
            f"  Pro dependencies missing: {', '.join(missing)}",
            fg=typer.colors.YELLOW,
        )
        install_now = typer.confirm("  Install them now? (pip install neural-memory[pro])")
        if install_now:
            import subprocess
            import sys

            typer.echo("  Installing Pro dependencies...")
            result_pip = subprocess.run(
                [sys.executable, "-m", "pip", "install", "neural-memory[pro]"],
                capture_output=True,
                text=True,
            )
            if result_pip.returncode == 0:
                typer.secho("  [OK] Pro dependencies installed!", fg=typer.colors.GREEN)
            else:
                typer.secho("  [FAILED] Install failed. Run manually:", fg=typer.colors.RED)
                typer.echo("    pip install neural-memory[pro]")
        else:
            typer.echo("  Run later: pip install neural-memory[pro]")
        typer.echo()

    typer.secho("Restart your AI tool to apply Pro features.", fg=typer.colors.BRIGHT_BLACK)


@shared_app.command("sync")
def shared_sync(
    direction: Annotated[
        str, typer.Option("--direction", "-d", help="Sync direction: push, pull, or both")
    ] = "both",
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Manually sync local brain with remote server.

    Directions:
        push  - Upload local brain to server
        pull  - Download brain from server to local
        both  - Full bidirectional sync (default)

    Examples:
        nmem shared sync
        nmem shared sync --direction push
        nmem shared sync --direction pull
    """
    config = get_config()

    if not config.shared.server_url:
        typer.secho(
            "No server URL configured. Use 'nmem shared enable <url>' first.", fg=typer.colors.RED
        )
        raise typer.Exit(1)

    async def _sync() -> dict[str, Any]:
        from neural_memory.storage.shared_store import SharedStorage

        # Load local storage (force local to avoid recursion into shared mode)
        local_storage = await get_storage(config, force_local=True)

        # Connect to remote
        remote = SharedStorage(
            server_url=config.shared.server_url,
            brain_id=config.current_brain,
            api_key=config.shared.api_key,
            timeout=config.shared.timeout,
        )
        await remote.connect()

        sync_result = {"direction": direction, "success": True}

        try:
            if direction in ("push", "both"):
                # Export local and push to remote
                snapshot = await local_storage.export_brain(local_storage.brain_id or "")
                await remote.import_brain(snapshot, config.current_brain)
                sync_result["pushed"] = True
                sync_result["neurons_pushed"] = len(snapshot.neurons)
                sync_result["synapses_pushed"] = len(snapshot.synapses)
                sync_result["fibers_pushed"] = len(snapshot.fibers)

            if direction in ("pull", "both"):
                # Pull from remote and import locally
                try:
                    snapshot = await remote.export_brain(config.current_brain)
                    await local_storage.import_brain(snapshot, local_storage.brain_id or "")
                    sync_result["pulled"] = True
                    sync_result["neurons_pulled"] = len(snapshot.neurons)
                    sync_result["synapses_pulled"] = len(snapshot.synapses)
                    sync_result["fibers_pulled"] = len(snapshot.fibers)
                except Exception as e:
                    if direction == "pull":
                        raise
                    # For "both", pulling may fail if brain doesn't exist on server
                    sync_result["pulled"] = False
                    sync_result["pull_error"] = str(e)

        finally:
            await remote.disconnect()

        return sync_result

    typer.echo(f"Syncing brain '{config.current_brain}' with {config.shared.server_url}...")
    result = run_async(_sync())

    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        if result.get("pushed"):
            typer.secho(
                f"[PUSHED] {result.get('neurons_pushed', 0)} neurons, "
                f"{result.get('synapses_pushed', 0)} synapses, "
                f"{result.get('fibers_pushed', 0)} fibers",
                fg=typer.colors.GREEN,
            )

        if result.get("pulled"):
            typer.secho(
                f"[PULLED] {result.get('neurons_pulled', 0)} neurons, "
                f"{result.get('synapses_pulled', 0)} synapses, "
                f"{result.get('fibers_pulled', 0)} fibers",
                fg=typer.colors.GREEN,
            )
        elif result.get("pull_error"):
            typer.secho(f"[PULL FAILED] {result['pull_error']}", fg=typer.colors.YELLOW)

        typer.secho("\nSync complete!", fg=typer.colors.GREEN)
