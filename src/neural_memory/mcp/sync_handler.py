"""MCP sync tool handler for multi-device sync operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

# No default shared hub — users deploy their own Cloudflare Worker
# See docs/guides/cloud-sync.md for self-hosted setup
DEFAULT_HUB_URL = ""


def _mask_key(api_key: str) -> str:
    """Mask API key for display: nmk_a1b2****."""
    if not api_key or len(api_key) < 12:
        return "(not set)"
    return f"{api_key[:12]}****"


def _build_sync_url(hub_url: str) -> str:
    """Build the sync endpoint URL with version prefix.

    Cloud hub uses /v1/hub/sync, local hub uses /hub/sync.
    """
    base = hub_url.rstrip("/")
    if "localhost" in base or "127.0.0.1" in base:
        return f"{base}/hub/sync"
    return f"{base}/v1/hub/sync"


def _build_hub_url(hub_url: str, path: str) -> str:
    """Build a hub endpoint URL with version prefix."""
    base = hub_url.rstrip("/")
    if "localhost" in base or "127.0.0.1" in base:
        return f"{base}{path}"
    return f"{base}/v1{path}"


class SyncToolHandler:
    """Mixin providing sync-related MCP tool handlers."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _sync(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_sync tool call."""
        try:
            action = args.get("action", "full")
            if action not in ("push", "pull", "full", "seed"):
                return {"error": "Invalid action. Use: push, pull, full, seed"}

            storage = await self.get_storage()

            # Seed: populate change_log from existing entities (for initial sync)
            if action == "seed":
                device_id = self.config.device_id
                counts = await storage.seed_change_log(device_id=device_id)
                total = counts["neurons"] + counts["synapses"] + counts["fibers"]
                return {
                    "status": "success",
                    "action": "seed",
                    "seeded": counts,
                    "total": total,
                    "message": (
                        f"Seeded {total} entities into change log. "
                        "Run nmem_sync(action='push') to push to hub."
                        if total > 0
                        else "No new entities to seed — change log already up to date."
                    ),
                }

            # Check if sync is configured
            if not self.config.sync.enabled:
                return {
                    "status": "disabled",
                    "message": "Sync is not enabled. Use nmem_sync_config(action='setup') to get started.",
                }

            hub_url = args.get("hub_url") or self.config.sync.hub_url
            if not hub_url:
                return {
                    "status": "error",
                    "message": "No hub URL configured. Use nmem_sync_config(action='setup') to configure.",
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

            # Build headers with API key auth
            api_key = args.get("api_key") or self.config.sync.api_key
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            # Refuse non-HTTPS for cloud hubs
            if hub_url and not hub_url.startswith("https://"):
                if "localhost" not in hub_url and "127.0.0.1" not in hub_url:
                    return {
                        "status": "error",
                        "message": "Cloud hub requires HTTPS. Use https:// URL.",
                    }

            # Try Merkle delta sync if Pro (single round, ~49KB)
            if self.config.is_pro() and action in ("full", "pull"):
                merkle_result = await self._try_merkle_sync(
                    engine, brain_id, hub_url, args, strategy
                )
                if merkle_result is not None:
                    return merkle_result
                # Fallback: Merkle failed or unavailable, continue with change-log sync
                logger.debug("Merkle sync unavailable, falling back to change-log")

            # Send to hub (change-log sync)
            import aiohttp

            sync_url = _build_sync_url(hub_url)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    sync_url,
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
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        return _handle_http_error(resp.status)
                    response_data = await resp.json()

            # Process response
            from neural_memory.sync.protocol import (
                SyncChange,
                SyncConflict,
                SyncResponse,
                SyncStatus,
            )

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
            sync_result: dict[str, Any] = {
                "status": "success",
                "action": action,
                "sync_mode": "changelog",
                "changes_pushed": len(request.changes),
                "changes_pulled": result["applied"],
                "conflicts": result["conflicts"],
                "hub_sequence": result["hub_sequence"],
            }

            # Pass through Pro upsell hints from hub
            hub_hints = response_data.get("hints")
            if hub_hints:
                sync_result["hints"] = hub_hints

            return sync_result

        except ImportError:
            return {
                "status": "error",
                "message": "aiohttp not installed. Install with: pip install aiohttp",
            }
        except Exception:
            logger.error("Sync failed", exc_info=True)
            return {"status": "error", "message": "Sync operation failed"}

    async def _try_merkle_sync(
        self,
        engine: Any,
        brain_id: str,
        hub_url: str,
        args: dict[str, Any],
        strategy: Any,
    ) -> dict[str, Any] | None:
        """Attempt Merkle delta sync. Returns result dict or None to fallback."""
        try:
            import aiohttp

            request = await engine.prepare_merkle_request(brain_id, is_pro=True)
            if request is None:
                return None

            merkle_url = _build_hub_url(hub_url, "/hub/sync/merkle")
            api_key = args.get("api_key") or self.config.sync.api_key
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    merkle_url,
                    json={
                        "device_id": request.device_id,
                        "brain_id": request.brain_id,
                        "root_hash": request.root_hash,
                        "buckets": request.buckets,
                        "strategy": request.strategy.value,
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 403:
                        # Hub says not Pro — fall back
                        return None
                    if resp.status != 200:
                        logger.debug("Merkle sync HTTP %d, falling back", resp.status)
                        return None
                    data = await resp.json()

            if data.get("status") == "in_sync":
                return {
                    "status": "success",
                    "action": "full",
                    "sync_mode": "merkle",
                    "changes_pushed": 0,
                    "changes_pulled": 0,
                    "conflicts": 0,
                    "hub_sequence": data.get("hub_sequence", 0),
                    "message": "In sync (Merkle hash match)",
                }

            # Process diffs
            from neural_memory.sync.protocol import MerkleBucketDiff, MerkleSyncResponse

            diffs = [
                MerkleBucketDiff(
                    entity_type=d["entity_type"],
                    prefix=d["prefix"],
                    entity_ids=d.get("entity_ids", []),
                    entities=d.get("entities", []),
                )
                for d in data.get("diffs", [])
            ]
            response = MerkleSyncResponse(
                status=data.get("status", "diff"),
                hub_root_hash=data.get("hub_root_hash", ""),
                changed_prefixes=data.get("changed_prefixes", []),
                diffs=diffs,
                hub_sequence=data.get("hub_sequence", 0),
            )

            result = await engine.process_merkle_response(response, request.buckets)
            return {
                "status": "success",
                "action": "full",
                "sync_mode": "merkle",
                "changes_pulled": result.get("applied", 0),
                "deleted": result.get("deleted", 0),
                "changed_prefixes": result.get("changed_prefixes", 0),
                "conflicts": 0,
                "hub_sequence": result.get("hub_sequence", 0),
            }

        except Exception:
            logger.debug("Merkle sync failed, falling back to change-log", exc_info=True)
            return None

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

            result: dict[str, Any] = {
                "sync_enabled": self.config.sync.enabled,
                "hub_url": self.config.sync.hub_url or "(not set)",
                "api_key": _mask_key(self.config.sync.api_key),
                "device_id": self.config.device_id,
                "auto_sync": self.config.sync.auto_sync,
                "conflict_strategy": self.config.sync.conflict_strategy,
                "change_log": change_stats,
                "devices": devices,
                "device_count": len(devices),
            }

            # If connected to cloud hub, fetch tier info
            if self.config.sync.enabled and self.config.sync.api_key:
                cloud_info = await self._fetch_cloud_profile()
                if cloud_info:
                    result["cloud"] = cloud_info

            return result
        except Exception:
            logger.error("Sync status failed", exc_info=True)
            return {"error": "Failed to get sync status"}

    async def _fetch_cloud_profile(self) -> dict[str, Any] | None:
        """Fetch cloud hub profile (/v1/auth/me) for status display."""
        try:
            import aiohttp

            url = _build_hub_url(self.config.sync.hub_url, "/auth/me")
            headers = {"Authorization": f"Bearer {self.config.sync.api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "email": data.get("email", ""),
                            "tier": data.get("tier", ""),
                            "brains": data.get("usage", {}).get("brains", 0),
                            "devices": data.get("usage", {}).get("devices", 0),
                        }
        except Exception:
            logger.debug("Failed to fetch cloud usage", exc_info=True)
        return None

    async def _sync_config(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_sync_config tool call."""
        try:
            action = args.get("action", "get")

            if action == "get":
                return {
                    "enabled": self.config.sync.enabled,
                    "hub_url": self.config.sync.hub_url or "(not set)",
                    "api_key": _mask_key(self.config.sync.api_key),
                    "auto_sync": self.config.sync.auto_sync,
                    "sync_interval_seconds": self.config.sync.sync_interval_seconds,
                    "conflict_strategy": self.config.sync.conflict_strategy,
                    "device_id": self.config.device_id,
                }

            if action == "setup":
                if self.config.sync.enabled and self.config.sync.api_key:
                    return {
                        "status": "already_configured",
                        "hub_url": self.config.sync.hub_url,
                        "api_key": _mask_key(self.config.sync.api_key),
                        "message": "Cloud sync is already configured. Use action='get' to view settings.",
                    }
                return {
                    "status": "setup_needed",
                    "steps": [
                        "1. Deploy your own sync hub: cd sync-hub && npx wrangler deploy (requires free Cloudflare account)",
                        "2. Register: curl -X POST https://YOUR-WORKER.workers.dev/v1/auth/register -H 'Content-Type: application/json' -d '{\"email\":\"you@example.com\"}'",
                        "3. Save the API key from the response (shown once, starts with nmk_)",
                        "4. Run: nmem_sync_config(action='set', hub_url='https://YOUR-WORKER.workers.dev', api_key='nmk_YOUR_KEY')",
                        "5. Run: nmem_sync(action='seed') to prepare existing memories",
                        "6. Run: nmem_sync(action='push') to upload to cloud",
                    ],
                    "note": "Your data stays on YOUR Cloudflare account. See docs/guides/cloud-sync.md for details.",
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
                if "api_key" in args:
                    key = str(args["api_key"]).strip()
                    if key and not key.startswith("nmk_"):
                        return {"error": "API key must start with 'nmk_'"}
                    new_sync = dc_replace(new_sync, api_key=key)
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

                # Auto-enable when both hub_url and api_key are provided
                if new_sync.hub_url and new_sync.api_key and not new_sync.enabled:
                    new_sync = dc_replace(new_sync, enabled=True)

                self.config = dc_replace(self.config, sync=new_sync)
                self.config.save()

                result: dict[str, Any] = {
                    "status": "updated",
                    "enabled": self.config.sync.enabled,
                    "hub_url": self.config.sync.hub_url or "(not set)",
                    "api_key": _mask_key(self.config.sync.api_key),
                    "auto_sync": self.config.sync.auto_sync,
                    "sync_interval_seconds": self.config.sync.sync_interval_seconds,
                    "conflict_strategy": self.config.sync.conflict_strategy,
                }

                # Test connectivity if newly configured
                if self.config.sync.enabled and self.config.sync.api_key:
                    connectivity = await self._test_hub_connectivity()
                    result["connectivity"] = connectivity

                return result

            if action == "activate":
                license_key = str(args.get("license_key", "")).strip()
                if not license_key:
                    return {"error": "Missing license_key parameter"}
                # Normalize XLabs format (NM-PRO-XXXX) → nm_pro_XXXX
                if license_key.startswith("NM-"):
                    license_key = license_key.replace("-", "_").lower()
                if not license_key.startswith("nm_"):
                    return {"error": "Invalid license key format. Expected nm_pro_* or nm_team_*"}

                hub_url = self.config.sync.hub_url
                api_key = self.config.sync.api_key
                if not hub_url or not api_key:
                    return {
                        "error": "Sync must be configured first. Run nmem_sync_config(action='setup').",
                    }

                import aiohttp

                activate_url = _build_hub_url(hub_url, "/hub/activate")
                headers: dict[str, str] = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        activate_url,
                        json={"license_key": license_key},
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            return {
                                "status": "error",
                                "message": data.get("error", f"Activation failed ({resp.status})"),
                            }

                        # Persist license tier to config.toml
                        from dataclasses import replace as _dc_replace

                        from neural_memory.unified_config import (
                            LicenseConfig,
                            set_config,
                        )

                        activated_tier = str(data.get("tier", "pro")).lower()
                        new_license = LicenseConfig.from_dict(
                            {
                                "tier": activated_tier,
                                "activated_at": data.get("activatedAt", ""),
                                "expires_at": data.get("expiresAt", ""),
                            }
                        )
                        self.config = _dc_replace(self.config, license=new_license)
                        self.config.save()
                        set_config(self.config)  # Update singleton for REST API

                        return {
                            "status": "activated",
                            "tier": activated_tier,
                            "expires_at": data.get("expiresAt"),
                            "features": data.get("features", []),
                            "message": data.get("message", "License activated!"),
                        }

            return {"error": "Invalid action. Use: get, set, setup, activate"}
        except Exception:
            logger.error("Sync config failed", exc_info=True)
            return {"error": "Failed to manage sync config"}

    async def _test_hub_connectivity(self) -> dict[str, Any]:
        """Test hub connectivity with a health check."""
        try:
            import aiohttp

            url = _build_hub_url(self.config.sync.hub_url, "/health")
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return {"status": "ok", "message": "Hub is reachable"}
                    return {"status": "error", "message": f"Hub returned {resp.status}"}
        except Exception as e:
            return {"status": "error", "message": f"Cannot reach hub: {e}"}


def _handle_http_error(status: int) -> dict[str, Any]:
    """Map HTTP status codes to user-friendly error messages."""
    messages: dict[int, str] = {
        401: "Invalid or expired API key. Run nmem_sync_config(action='setup') to reconfigure.",
        403: "Access denied. You don't own this brain or need to upgrade your plan.",
        413: "Payload too large. Reduce the number of changes per sync.",
        422: "Invalid request format. Check brain_id and device_id.",
        429: "Rate limited. Try again in a few seconds.",
    }
    message = messages.get(status, f"Hub returned status {status}")
    return {"status": "error", "http_status": status, "message": message}
