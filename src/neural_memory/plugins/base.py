"""Base class for Neural Memory plugins.

All plugins must subclass ProPlugin and implement the required methods.
Return empty dicts/None for features your plugin doesn't provide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class ProPlugin(ABC):
    """Extension point for third-party plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version string."""
        ...

    @abstractmethod
    def get_retrieval_strategies(self) -> dict[str, Callable[..., Any]]:
        """Return named retrieval strategies.

        Keys are strategy names (e.g. 'cone', 'ppr_pro').
        Values are async callables matching the retrieval interface.
        """
        ...

    @abstractmethod
    def get_compression_fn(self) -> Callable[..., Any] | None:
        """Return a compression function, or None if not provided.

        Should match: async def compress(content: str, level: FidelityLevel, embed_fn) -> str
        """
        ...

    @abstractmethod
    def get_consolidation_strategies(self) -> dict[str, Callable[..., Any]]:
        """Return named consolidation strategies.

        Keys are strategy names (e.g. 'smart_merge', 'priority_aware').
        Values are async callables for consolidation.
        """
        ...

    def get_tools(self) -> list[dict[str, Any]]:
        """Return additional MCP tool schemas provided by this plugin.

        Optional — override to add Pro-only MCP tools.
        Default: no extra tools.
        """
        return []

    def get_tool_handler(self, tool_name: str) -> Callable[..., Any] | None:
        """Return handler for a plugin-provided MCP tool.

        Optional — override if get_tools() returns tools.
        Default: None (tool not handled by this plugin).
        """
        return None

    def get_storage_class(self) -> type | None:
        """Return a NeuralStorage subclass for Pro storage backend.

        Optional — override to provide an alternative storage engine.
        The returned class must accept (base_dir, brain_id, dimensions) and
        implement NeuralStorage + async open()/close().
        Default: None (use SQLite).
        """
        return None
