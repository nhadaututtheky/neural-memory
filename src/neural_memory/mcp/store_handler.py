"""MCP handler mixin for Brain Store operations.

Provides browse, preview, import, and export actions for the
community brain marketplace via the nmem_store tool.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neural_memory.engine.brain_package import (
    create_brain_package,
    preview_brain_package,
    validate_brain_package,
)
from neural_memory.engine.brain_registry import BrainRegistryClient
from neural_memory.mcp.tool_handler_utils import _require_brain_id
from neural_memory.safety.brain_scanner import scan_brain_package

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

# Shared registry client (reused across calls, 5min cache)
_registry = BrainRegistryClient()


class StoreHandler:
    """Mixin providing Brain Store tool implementations."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _store(self, args: dict[str, Any]) -> dict[str, Any]:
        """Unified Brain Store tool dispatcher."""
        action = args.get("action", "browse")
        try:
            if action == "browse":
                return await self._store_browse(args)
            if action == "preview":
                return await self._store_preview(args)
            if action == "import":
                return await self._store_import(args)
            if action == "export":
                return await self._store_export(args)
            if action == "publish":
                return await self._store_publish(args)
            if action == "delete":
                return await self._store_delete(args)
            return {"error": f"Unknown store action: {action}"}
        except Exception as e:
            logger.error("Store %s failed: %s", action, e)
            return {"error": f"Store {action} failed"}

    async def _store_browse(self, args: dict[str, Any]) -> dict[str, Any]:
        """Browse the community brain registry."""
        manifests = await _registry.fetch_index()
        # Coerce limit defensively: a non-int arg (e.g. a string) would otherwise
        # raise TypeError inside min() and surface as a generic failure (#72).
        try:
            limit = min(int(args.get("limit", 20)), 50)
        except (TypeError, ValueError):
            limit = 20
        filtered = _registry.filter_index(
            manifests,
            category=args.get("category"),
            search=args.get("search"),
            tag=args.get("tag"),
            sort_by=args.get("sort_by", "created_at"),
            limit=limit,
        )

        if not filtered:
            return {"brains": [], "total": 0, "message": "No brains found matching your criteria"}

        results = []
        for m in filtered:
            stats = m.get("stats", {})
            results.append(
                {
                    "name": m.get("name", ""),
                    "display_name": m.get("display_name", ""),
                    "description": m.get("description", "")[:200],
                    "author": m.get("author", ""),
                    "category": m.get("category", ""),
                    "neurons": stats.get("neuron_count", 0),
                    "size_tier": m.get("size_tier", ""),
                    "rating": m.get("rating_avg", 0),
                    "tags": m.get("tags", [])[:5],
                }
            )

        return {"brains": results, "total": len(results)}

    async def _store_preview(self, args: dict[str, Any]) -> dict[str, Any]:
        """Preview a brain from the store."""
        brain_name = args.get("brain_name")
        if not brain_name:
            return {"error": "brain_name is required"}

        data = await _registry.fetch_brain(brain_name)
        if data is None:
            return {"error": f"Brain '{brain_name}' not found in registry"}

        preview = preview_brain_package(data)
        manifest = preview.get("manifest", {})
        scan = preview.get("scan_result", {})

        return {
            "name": manifest.get("display_name", brain_name),
            "description": manifest.get("description", ""),
            "author": manifest.get("author", ""),
            "version": manifest.get("version", ""),
            "license": manifest.get("license", ""),
            "stats": manifest.get("stats", {}),
            "neuron_types": preview.get("neuron_type_breakdown", {}),
            "top_tags": preview.get("top_tags", []),
            "sample_neurons": [
                {"type": n.get("type", ""), "content": n.get("content", "")}
                for n in preview.get("sample_neurons", [])[:5]
            ],
            "scan": {
                "safe": scan.get("safe", False),
                "risk_level": scan.get("risk_level", "unknown"),
                "finding_count": scan.get("finding_count", 0),
            },
        }

    async def _store_import(self, args: dict[str, Any]) -> dict[str, Any]:
        """Import a brain from the store."""
        brain_name = args.get("brain_name")
        if not brain_name:
            return {"error": "brain_name is required"}

        data = await _registry.fetch_brain(brain_name)
        if data is None:
            return {"error": f"Brain '{brain_name}' not found in registry"}

        # Validate
        valid, errors = validate_brain_package(data)
        if not valid:
            return {"error": f"Invalid package: {'; '.join(errors[:3])}"}

        # Security scan
        scan_result = scan_brain_package(data)
        if scan_result.risk_level in ("high", "critical"):
            return {
                "error": "Brain blocked by security scan",
                "risk_level": scan_result.risk_level,
                "findings": [f.description for f in scan_result.findings[:5]],
            }

        # Import
        snapshot_data = data.get("snapshot", {})
        manifest = data.get("manifest", {})

        import uuid

        from neural_memory.core.brain import BrainSnapshot
        from neural_memory.utils.timeutils import utcnow

        snapshot = BrainSnapshot(
            brain_id=str(uuid.uuid4()),
            brain_name=manifest.get("name", brain_name),
            exported_at=utcnow(),
            version=str(snapshot_data.get("version", "1")),
            neurons=snapshot_data.get("neurons", []),
            synapses=snapshot_data.get("synapses", []),
            fibers=snapshot_data.get("fibers", []),
            config=snapshot_data.get("config", {}),
            metadata=snapshot_data.get("metadata", {}),
        )

        storage = await self.get_storage()
        brain_id = await storage.import_brain(snapshot)

        return {
            "status": "imported",
            "brain_id": brain_id,
            "brain_name": manifest.get("display_name", brain_name),
            "neurons_imported": len(snapshot_data.get("neurons", [])),
            "synapses_imported": len(snapshot_data.get("synapses", [])),
            "fibers_imported": len(snapshot_data.get("fibers", [])),
            "scan_safe": scan_result.safe,
        }

    async def _store_export(self, args: dict[str, Any]) -> dict[str, Any]:
        """Export the current brain as a .brain package."""
        display_name = args.get("display_name")
        if not display_name:
            return {"error": "display_name is required"}

        description = args.get("description", "")
        author = args.get("author", "anonymous")

        storage = await self.get_storage()
        brain_id = _require_brain_id(storage)
        snapshot = await storage.export_brain(brain_id)

        import neural_memory

        try:
            package = create_brain_package(
                snapshot,
                display_name=display_name,
                description=description,
                author=author,
                tags=args.get("tags", []),
                category=args.get("category", "general"),
                nmem_version=getattr(neural_memory, "__version__", ""),
            )
        except ValueError as e:
            return {"error": str(e)}

        manifest = package.get("manifest", {})

        # Save to disk if output_path provided
        output_path = args.get("output_path")
        if output_path:
            path = Path(output_path).resolve()
            # Restrict to user's home directory or current working directory
            home = Path.home().resolve()
            cwd = Path.cwd().resolve()
            if not (path.is_relative_to(home) or path.is_relative_to(cwd)):
                return {
                    "error": "output_path must be within home directory or current working directory"
                }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(package, default=str, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return {
                "status": "exported",
                "path": str(path),
                "display_name": display_name,
                "neurons": manifest.get("stats", {}).get("neuron_count", 0),
                "size_bytes": manifest.get("size_bytes", 0),
                "size_tier": manifest.get("size_tier", ""),
            }

        return {
            "status": "exported",
            "display_name": display_name,
            "manifest": {
                "name": manifest.get("name", ""),
                "neurons": manifest.get("stats", {}).get("neuron_count", 0),
                "synapses": manifest.get("stats", {}).get("synapse_count", 0),
                "fibers": manifest.get("stats", {}).get("fiber_count", 0),
                "size_bytes": manifest.get("size_bytes", 0),
                "size_tier": manifest.get("size_tier", ""),
                "content_hash": manifest.get("content_hash", ""),
            },
            "hint": "Use output_path to save the .brain file to disk, or action='publish' to share with the community",
        }

    async def _store_delete(self, args: dict[str, Any]) -> dict[str, Any]:
        """Delete a local brain and all its data."""
        brain_id = args.get("brain_id")
        if not brain_id:
            return {"error": "brain_id is required for delete"}

        storage = await self.get_storage()

        # Verify the brain exists before deleting
        brain = await storage.get_brain(brain_id)
        if brain is None:
            return {"error": f"Brain '{brain_id}' not found"}

        # Prevent deleting the currently active brain
        current_brain_id = getattr(storage, "_brain_id", None)
        if current_brain_id == brain_id:
            return {
                "error": "Cannot delete the currently active brain. Switch to another brain first."
            }

        # Get stats for preview / confirmation
        stats = await storage.get_stats(brain_id)

        # Without confirm=true, return a preview of what would be deleted
        if not args.get("confirm"):
            return {
                "status": "pending_confirmation",
                "brain_id": brain_id,
                "brain_name": brain.name,
                "will_delete": {
                    "neurons": stats.get("neuron_count", 0),
                    "synapses": stats.get("synapse_count", 0),
                    "fibers": stats.get("fiber_count", 0),
                },
                "message": "This will permanently delete the brain and all its data. "
                "Call again with confirm=true to proceed.",
            }

        await storage.clear(brain_id)

        return {
            "status": "deleted",
            "brain_id": brain_id,
            "deleted": {
                "neurons": stats.get("neuron_count", 0),
                "synapses": stats.get("synapse_count", 0),
                "fibers": stats.get("fiber_count", 0),
            },
            "message": "Brain and all associated data permanently deleted",
        }

    async def _store_publish(self, args: dict[str, Any]) -> dict[str, Any]:
        """Publish the current brain to the community store."""
        display_name = args.get("display_name")
        if not display_name:
            return {"error": "display_name is required"}

        description = args.get("description", "")
        if not description:
            return {"error": "description is required"}

        author = args.get("author", "anonymous")

        storage = await self.get_storage()
        brain_id = _require_brain_id(storage)
        snapshot = await storage.export_brain(brain_id)

        import neural_memory

        try:
            package = create_brain_package(
                snapshot,
                display_name=display_name,
                description=description,
                author=author,
                tags=args.get("tags", []),
                category=args.get("category", "general"),
                version=args.get("version", "1.0.0"),
                license_=args.get("license", "CC-BY-4.0"),
                nmem_version=getattr(neural_memory, "__version__", ""),
            )
        except ValueError as e:
            return {"error": str(e)}

        # Security scan
        scan_result = scan_brain_package(package)
        if scan_result.risk_level in ("high", "critical"):
            return {
                "error": "Brain failed security scan — cannot publish",
                "risk_level": scan_result.risk_level,
                "findings": [f.description for f in scan_result.findings[:5]],
            }

        # Get API key
        api_key = getattr(self.config, "api_key", "") or ""
        if not api_key:
            return {
                "error": "API key required for publishing. Set with: nmem config set api_key YOUR_KEY"
            }

        try:
            result = await _registry.publish_brain(package, api_key)
        except ValueError as e:
            return {"error": str(e)}

        return {
            "status": "published",
            "name": result.get("name", ""),
            "display_name": display_name,
            "pr_url": result.get("pr_url", ""),
            "message": result.get("message", "Brain submitted for review"),
        }
