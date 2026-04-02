"""Pro MCP tools — exclusive tools for Neural Memory Pro users.

These tools are registered via the plugin system and dispatched
by the free tier's MCP server when Pro is installed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Tool Schemas ──────────────────────────────────────────


PRO_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "nmem_cone_query",
        "description": (
            "Pro: Cone recall — find memories within a semantic cone around a query. "
            "Uses HNSW approximate nearest neighbors for fast retrieval. "
            "Returns results ranked by combined similarity + activation score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to search for",
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum cosine similarity threshold (0.0-1.0). Default: 0.7",
                    "default": 0.7,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results. Default: 10",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "nmem_tier_info",
        "description": (
            "Pro: View memory tier distribution and compression stats. "
            "Shows how memories are distributed across tiers "
            "(ACTIVE/WARM/COOL/FROZEN/CRYSTAL) and estimated storage savings."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["stats", "sweep"],
                    "description": (
                        "'stats' to view tier distribution, "
                        "'sweep' to run a demote sweep (moves stale memories to lower tiers). "
                        "Default: stats"
                    ),
                    "default": "stats",
                },
            },
        },
    },
    {
        "name": "nmem_pro_merge",
        "description": (
            "Pro: Smart merge — HNSW-accelerated memory consolidation. "
            "Finds clusters of semantically similar memories and merges them, "
            "keeping the highest-priority memory as anchor. "
            "Use dry_run=true to preview merge plan without executing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "similarity_threshold": {
                    "type": "number",
                    "description": "Cosine similarity threshold (0.0-1.0). Default: 0.9",
                    "default": 0.9,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview merge plan without executing. Default: true",
                    "default": True,
                },
                "max_merges": {
                    "type": "integer",
                    "description": "Maximum number of merge actions. Default: 50",
                    "default": 50,
                },
            },
        },
    },
]


# ── Tool Handlers ─────────────────────────────────────────


async def handle_cone_query(server: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle nmem_cone_query tool call."""
    query = arguments.get("query", "")
    threshold = float(arguments.get("threshold", 0.7))
    max_results = int(arguments.get("max_results", 10))

    if not query:
        return {"error": "query is required"}

    try:
        storage = server.storage
        if not hasattr(storage, "search_similar"):
            return {"error": "Pro storage (InfinityDB) not active. Using SQLite backend."}

        # Get query embedding
        encoder = server.encoder
        embedding = await encoder.get_embedding(query)
        if embedding is None:
            return {"error": "Embedding not available. Configure an embedding provider."}

        from neural_memory.pro.retrieval.cone_queries import cone_recall

        results = await cone_recall(
            query_embedding=embedding,
            db=storage.db if hasattr(storage, "db") else storage,
            threshold=threshold,
            max_results=max_results,
        )

        return {
            "results": [
                {
                    "neuron_id": r.neuron_id,
                    "content": r.content,
                    "similarity": round(r.similarity, 4),
                    "activation": round(r.activation, 4),
                    "combined_score": round(r.combined_score, 4),
                    "type": r.neuron_type,
                }
                for r in results
            ],
            "count": len(results),
            "threshold": threshold,
            "pro": True,
        }
    except Exception as e:
        logger.error("nmem_cone_query failed: %s", e)
        return {"error": "Cone query failed. Ensure Pro storage is configured."}


async def handle_tier_info(server: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle nmem_tier_info tool call."""
    action = arguments.get("action", "stats")

    try:
        storage = server.storage
        if not hasattr(storage, "get_tier_stats"):
            return {"error": "Pro storage (InfinityDB) not active. Using SQLite backend."}

        if action == "sweep":
            result = await storage.demote_sweep()
            return {
                "action": "sweep",
                "result": result,
                "pro": True,
            }

        stats = await storage.get_tier_stats()
        return {
            "action": "stats",
            "tiers": stats.get("tiers", {}),
            "savings": stats.get("savings", {}),
            "pro": True,
        }
    except Exception as e:
        logger.error("nmem_tier_info failed: %s", e)
        return {"error": "Tier info failed. Ensure Pro storage is configured."}


async def handle_pro_merge(server: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle nmem_pro_merge tool call."""
    similarity_threshold = float(arguments.get("similarity_threshold", 0.9))
    dry_run = arguments.get("dry_run", True)
    max_merges = int(arguments.get("max_merges", 50))

    try:
        storage = server.storage
        db = storage.db if hasattr(storage, "db") else storage

        if not hasattr(db, "search_similar"):
            return {"error": "Pro storage (InfinityDB) not active. Using SQLite backend."}

        from neural_memory.pro.consolidation.smart_merge import smart_merge

        result = await smart_merge(
            db,
            similarity_threshold=similarity_threshold,
            dry_run=dry_run,
            max_merges=max_merges,
        )

        return {
            "status": result.get("status", "unknown"),
            "clusters_found": result.get("clusters_found", 0),
            "merges": result.get("merges", 0),
            "merge_actions": result.get("merge_actions", 0),
            "dry_run": dry_run,
            "pro": True,
        }
    except Exception as e:
        logger.error("nmem_pro_merge failed: %s", e)
        return {"error": "Smart merge failed. Ensure Pro storage is configured."}


# ── Dispatch Table ────────────────────────────────────────

TOOL_HANDLERS: dict[str, Any] = {
    "nmem_cone_query": handle_cone_query,
    "nmem_tier_info": handle_tier_info,
    "nmem_pro_merge": handle_pro_merge,
}
