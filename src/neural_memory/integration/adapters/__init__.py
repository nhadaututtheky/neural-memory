"""Adapter registry for external source integrations."""

from __future__ import annotations

from typing import Any

from neural_memory.integration.adapter import SourceAdapter

# Registry of available adapters by system name
_ADAPTER_REGISTRY: dict[str, type] = {}


def register_adapter(name: str, adapter_cls: type) -> None:
    """Register an adapter class by system name."""
    _ADAPTER_REGISTRY[name] = adapter_cls


def get_adapter(name: str, **kwargs: Any) -> SourceAdapter:
    """Get an adapter instance by system name.

    Args:
        name: System name (e.g., 'chromadb', 'mem0')
        **kwargs: Adapter-specific configuration

    Returns:
        Configured SourceAdapter instance

    Raises:
        ValueError: If adapter not found or dependency missing
    """
    adapter_cls = _ADAPTER_REGISTRY.get(name)

    if adapter_cls is None:
        adapter_cls = _lazy_load_adapter(name)

    if adapter_cls is None:
        raise ValueError(f"Unknown adapter: {name!r}")

    return adapter_cls(**kwargs)  # type: ignore[no-any-return]


def _lazy_load_adapter(name: str) -> type | None:
    """Attempt to lazy-load a known adapter (handles optional deps)."""
    if name == "chromadb":
        try:
            from neural_memory.integration.adapters.chromadb_adapter import ChromaDBAdapter

            register_adapter("chromadb", ChromaDBAdapter)
            return ChromaDBAdapter
        except ImportError as e:
            msg = (
                "ChromaDB adapter requires 'chromadb' package. "
                "Install with: pip install neural-memory[chromadb]"
            )
            raise ValueError(msg) from e

    if name == "mem0":
        try:
            from neural_memory.integration.adapters.mem0_adapter import Mem0Adapter

            register_adapter("mem0", Mem0Adapter)
            return Mem0Adapter
        except ImportError as e:
            msg = (
                "Mem0 adapter requires 'mem0ai' package. "
                "Install with: pip install neural-memory[mem0]"
            )
            raise ValueError(msg) from e

    if name == "mem0_self_hosted":
        try:
            from neural_memory.integration.adapters.mem0_adapter import Mem0SelfHostedAdapter

            register_adapter("mem0_self_hosted", Mem0SelfHostedAdapter)
            return Mem0SelfHostedAdapter
        except ImportError as e:
            msg = (
                "Mem0 self-hosted adapter requires 'mem0ai' package. "
                "Install with: pip install neural-memory[mem0]"
            )
            raise ValueError(msg) from e

    if name == "awf":
        try:
            from neural_memory.integration.adapters.awf_adapter import AWFAdapter

            register_adapter("awf", AWFAdapter)
            return AWFAdapter
        except ImportError as e:
            msg = "AWF adapter requires 'aiohttp' package. Install with: pip install neural-memory"
            raise ValueError(msg) from e

    if name == "cognee":
        try:
            from neural_memory.integration.adapters.cognee_adapter import CogneeAdapter

            register_adapter("cognee", CogneeAdapter)
            return CogneeAdapter
        except ImportError as e:
            msg = (
                "Cognee adapter requires 'cognee' package. "
                "Install with: pip install neural-memory[cognee]"
            )
            raise ValueError(msg) from e

    if name == "graphiti":
        try:
            from neural_memory.integration.adapters.graphiti_adapter import GraphitiAdapter

            register_adapter("graphiti", GraphitiAdapter)
            return GraphitiAdapter
        except ImportError as e:
            msg = (
                "Graphiti adapter requires 'graphiti-core' package. "
                "Install with: pip install neural-memory[graphiti]"
            )
            raise ValueError(msg) from e

    if name == "llamaindex":
        try:
            from neural_memory.integration.adapters.llamaindex_adapter import LlamaIndexAdapter

            register_adapter("llamaindex", LlamaIndexAdapter)
            return LlamaIndexAdapter
        except ImportError as e:
            msg = (
                "LlamaIndex adapter requires 'llama-index-core' package. "
                "Install with: pip install neural-memory[llamaindex]"
            )
            raise ValueError(msg) from e

    return None


def list_adapters() -> list[str]:
    """List all available adapter names (including lazy-loadable)."""
    known = {"chromadb", "mem0", "mem0_self_hosted", "awf", "cognee", "graphiti", "llamaindex"}
    return sorted(known | set(_ADAPTER_REGISTRY.keys()))
