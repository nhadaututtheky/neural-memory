"""MCP sync tool handler for multi-device sync operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.sqlite_store import SQLiteStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class SyncToolHandler:
    """Mixin providing sync-related MCP tool handlers."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> SQLiteStorage:
            raise NotImplementedError

    async def _sync(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_sync tool call."""
        try:
            action = args.get("action", "full")
            if action not in ("push", "pull", "full"):
                return {"error": "Invalid action. Use: push, pull, full"}

            storage = await self.get_storage()

            # Check if sync is configured
            if not self.config.sync.enabled:
                return {
                    "status": "disabled",
                    "message": "Sync is not enabled. Use nmem_sync_config to enable it.",
                }

            hub_url = args.get("hub_url") or self.config.sync.hub_url
            if not hub_url:
                return {
                    "status": "error",
                    "message": "No hub URL configured. Set it via nmem_sync_config.",
                }

            # Get strategy
            from neural_memory.sync.protocol import ConflictStrategy
            strategy_str = args.get("strategy") or self.config.sync.conflict_strategy
            try:
                strategy = ConflictStrategy(strategy_str)
            except ValueError:
                strategy = ConflictStrategy.PREFER_RECENT

            # Create sync engine
            from neural_memory.sync.sync_engine import SyncEngine
            device_id = self.config.device_id
            engine = SyncEngine(storage, device_id, strategy)

            brain_id = storage.current_brain_id
            if not brain_id:
                return {"error": "No brain context set"}

            # Prepare request
            request = await engine.prepare_sync_request(brain_id)

            # Send to hub
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{hub_url.rstrip('/')}/hub/sync",
                    json={
                        "device_id": request.device_id,
                        "brain_id": request.brain_id,
                        "last_sequence": request.last_sequence,
                        "changes": [
                            {
                                "sequence": c.sequence,
                                "entity_type": c.entity_type,
                                "entity_id": c.entity_id,
                                "operation": c.operation,
                                "device_id": c.device_id,
                                "changed_at": c.changed_at,
                                "payload": c.payload,
                            }
                            for c in request.changes
                        ],
                        "strategy": request.strategy.value,
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        return {"status": "error", "message": f"Hub returned status {resp.status}"}
                    response_data = await resp.json()

            # Process response
            from neural_memory.sync.protocol import SyncChange, SyncConflict, SyncResponse, SyncStatus
            remote_changes = [
                SyncChange(
                    sequence=c["sequence"],
                    entity_type=c["entity_type"],
                    entity_id=c["entity_id"],
                    operation=c["operation"],
                    device_id=c.get("device_id", ""),
                    changed_at=c.get("changed_at", ""),
                    payload=c.get("payload", {}),
                )
                for c in response_data.get("changes", [])
            ]
            conflicts = [
                SyncConflict(
                    entity_type=c["entity_type"],
                    entity_id=c["entity_id"],
                    local_device=c.get("local_device", ""),
                    remote_device=c.get("remote_device", ""),
                    resolution=c.get("resolution", ""),
                )
                for c in response_data.get("conflicts", [])
            ]
            sync_response = SyncResponse(
                hub_sequence=response_data.get("hub_sequence", 0),
                changes=remote_changes,
                conflicts=conflicts,
                status=SyncStatus(response_data.get("status", "success")),
                message=response_data.get("message", ""),
            )

            result = await engine.process_sync_response(sync_response)
            return {
                "status": "success",
                "action": action,
                "changes_pushed": len(request.changes),
                "changes_pulled": result["applied"],
                "conflicts": result["conflicts"],
                "hub_sequence": result["hub_sequence"],
            }

        except ImportError:
            return {
                "status": "error",
                "message": "aiohttp not installed. Install with: pip install aiohttp",
            }
        except Exception:
            logger.error("Sync failed", exc_info=True)
            return {"status": "error", "message": "Sync operation failed"}

    async def _sync_status(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_sync_status tool call."""
        try:
            storage = await self.get_storage()

            # Change log stats
            change_stats = await storage.get_change_log_stats()

            # Device list
            devices_raw = await storage.list_devices()
            devices = [
                {
                    "device_id": d.device_id,
                    "device_name": d.device_name,
                    "last_sync_at": d.last_sync_at.isoformat() if d.last_sync_at else None,
                    "last_sync_sequence": d.last_sync_sequence,
                    "registered_at": d.registered_at.isoformat(),
                }
                for d in devices_raw
            ]

            return {
                "sync_enabled": self.config.sync.enabled,
                "hub_url": self.config.sync.hub_url or "(not set)",
                "device_id": self.config.device_id,
                "auto_sync": self.config.sync.auto_sync,
                "conflict_strategy": self.config.sync.conflict_strategy,
                "change_log": change_stats,
                "devices": devices,
                "device_count": len(devices),
            }
        except Exception:
            logger.error("Sync status failed", exc_info=True)
            return {"error": "Failed to get sync status"}

    async def _sync_config(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_sync_config tool call."""
        try:
            action = args.get("action", "get")

            if action == "get":
                return {
                    "enabled": self.config.sync.enabled,
                    "hub_url": self.config.sync.hub_url or "(not set)",
                    "auto_sync": self.config.sync.auto_sync,
                    "sync_interval_seconds": self.config.sync.sync_interval_seconds,
                    "conflict_strategy": self.config.sync.conflict_strategy,
                    "device_id": self.config.device_id,
                }

            if action == "set":
                from dataclasses import replace as dc_replace

                new_sync = self.config.sync
                if "enabled" in args:
                    new_sync = dc_replace(new_sync, enabled=bool(args["enabled"]))
                if "hub_url" in args:
                    url = str(args["hub_url"]).strip()
                    if url and not url.startswith(("http://", "https://")):
                        return {"error": "hub_url must start with http:// or https://"}
                    new_sync = dc_replace(new_sync, hub_url=url[:256])
                if "auto_sync" in args:
                    new_sync = dc_replace(new_sync, auto_sync=bool(args["auto_sync"]))
                if "sync_interval_seconds" in args:
                    interval = max(10, min(86400, int(args["sync_interval_seconds"])))
                    new_sync = dc_replace(new_sync, sync_interval_seconds=interval)
                if "conflict_strategy" in args:
                    valid = {"prefer_recent", "prefer_local", "prefer_remote", "prefer_stronger"}
                    strategy = str(args["conflict_strategy"])
                    if strategy not in valid:
                        return {"error": f"Invalid strategy. Use: {', '.join(sorted(valid))}"}
                    new_sync = dc_replace(new_sync, conflict_strategy=strategy)

                self.config = dc_replace(self.config, sync=new_sync)
                self.config.save()

                return {
                    "status": "updated",
                    "enabled": self.config.sync.enabled,
                    "hub_url": self.config.sync.hub_url or "(not set)",
                    "auto_sync": self.config.sync.auto_sync,
                    "sync_interval_seconds": self.config.sync.sync_interval_seconds,
                    "conflict_strategy": self.config.sync.conflict_strategy,
                }

            return {"error": "Invalid action. Use: get, set"}
        except Exception:
            logger.error("Sync config failed", exc_info=True)
            return {"error": "Failed to manage sync config"}
