"""InfinityDB — Custom spatial database engine for Neural Memory Pro.

Multi-dimensional vector storage with HNSW indexing, graph-native synapses,
tiered compression, and crash-safe WAL. Designed for 1M+ neurons at <100ms recall.

Requires: numpy, hnswlib, msgpack (bundled with neural-memory)
"""

from __future__ import annotations

# Lazy exports — only import when Pro deps are available
__all__ = [
    "InfinityDB",
    "GraphStore",
    "FiberStore",
    "CompressionTier",
    "VectorCompressor",
    "TierConfig",
    "TierManager",
    "TierStats",
    "QueryPlan",
    "MigrationStats",
    "SQLiteToInfinityMigrator",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy import to avoid ImportError when deps not installed."""
    if name in __all__:
        from neural_memory.pro import PRO_INSTALL_HINT, is_pro_deps_installed

        if not is_pro_deps_installed():
            raise ImportError(
                f"InfinityDB requires Pro dependencies. Install with: {PRO_INSTALL_HINT}"
            )

        # Import the actual modules
        if name == "InfinityDB":
            from neural_memory.pro.infinitydb.engine import InfinityDB

            return InfinityDB
        if name in ("CompressionTier", "VectorCompressor"):
            from neural_memory.pro.infinitydb import compressor

            return getattr(compressor, name)
        if name == "FiberStore":
            from neural_memory.pro.infinitydb.fiber_store import FiberStore

            return FiberStore
        if name == "GraphStore":
            from neural_memory.pro.infinitydb.graph_store import GraphStore

            return GraphStore
        if name in ("MigrationStats", "SQLiteToInfinityMigrator"):
            from neural_memory.pro.infinitydb import migrator

            return getattr(migrator, name)
        if name == "QueryPlan":
            from neural_memory.pro.infinitydb.query_planner import QueryPlan

            return QueryPlan
        if name in ("TierConfig", "TierManager", "TierStats"):
            from neural_memory.pro.infinitydb import tier_manager

            return getattr(tier_manager, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
