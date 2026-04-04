"""MCP handler for semantic drift detection tool (nmem_drift)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.tool_handler_utils import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class DriftHandler:
    """Mixin providing the nmem_drift tool handler for MCPServer."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _drift(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_drift tool calls.

        Semantic drift detection — find tag clusters that should be
        merged or aliased using Jaccard similarity on co-occurrence data.
        """
        action = args.get("action", "detect")
        valid_actions = ("detect", "list", "merge", "alias", "dismiss")
        if action not in valid_actions:
            return {"error": f"Invalid action: {action}. Must be one of {valid_actions}."}

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for drift action '%s'", action)
            return {"error": "No brain configured"}

        if action == "detect":
            return await self._drift_detect(storage)
        elif action == "list":
            return await self._drift_list(storage, args)
        elif action in ("merge", "alias", "dismiss"):
            return await self._drift_resolve(storage, action, args)

        return {"error": f"Unhandled action: {action}"}

    async def _drift_detect(self, storage: NeuralStorage) -> dict[str, Any]:
        """Run drift detection analysis."""
        try:
            from neural_memory.engine.drift_detection import run_drift_detection

            result = await run_drift_detection(storage)

            clusters = result.get("clusters", [])
            summary = result.get("summary", {})
            temporal = result.get("temporal_drifts", [])

            if not clusters and not temporal:
                return {
                    "status": "clean",
                    "message": "No semantic drift detected. Tag usage is consistent.",
                }

            return {
                "status": "drift_detected",
                "clusters": clusters,
                "temporal_drifts": temporal,
                "summary": summary,
                "hint": (
                    "Use nmem_drift(action='merge', cluster_id='...') to merge synonyms, "
                    "or 'alias' to mark as related, or 'dismiss' to ignore."
                ),
            }
        except Exception as e:
            logger.error("Drift detection failed: %s", e, exc_info=True)
            return {"error": "Drift detection failed"}

    async def _drift_list(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        """List existing drift clusters."""
        status_filter = args.get("status")
        try:
            clusters = await storage.get_drift_clusters(  # type: ignore[attr-defined]
                status=status_filter,
                limit=50,
            )
            if not clusters:
                return {
                    "clusters": [],
                    "message": "No drift clusters found."
                    + (f" (filter: status={status_filter})" if status_filter else ""),
                }
            return {"clusters": clusters, "count": len(clusters)}
        except Exception as e:
            logger.error("Drift list failed: %s", e, exc_info=True)
            return {"error": "Failed to list drift clusters"}

    async def _drift_resolve(
        self,
        storage: NeuralStorage,
        action: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve a drift cluster (merge/alias/dismiss)."""
        cluster_id = args.get("cluster_id")
        if not cluster_id:
            return {"error": f"cluster_id is required for '{action}' action"}

        # Map action to status
        status_map = {"merge": "merged", "alias": "aliased", "dismiss": "dismissed"}
        new_status = status_map[action]

        try:
            updated = await storage.resolve_drift_cluster(  # type: ignore[attr-defined]
                cluster_id=cluster_id,
                status=new_status,
            )
            if not updated:
                return {"error": f"Cluster '{cluster_id}' not found"}

            return {
                "status": "resolved",
                "cluster_id": cluster_id,
                "resolution": new_status,
                "message": f"Cluster {cluster_id} marked as {new_status}.",
            }
        except Exception as e:
            logger.error("Drift resolve failed: %s", e, exc_info=True)
            return {"error": f"Failed to {action} cluster"}
