"""TTL-based neuron lookup cache for repeated find_neurons calls.

Caches exact-match neuron lookups (content_exact) which are the most
repeated pattern during encoding pipelines. Invalidated on neuron
add/update/delete to prevent stale reads.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_memory.core.neuron import Neuron

logger = logging.getLogger(__name__)


class NeuronLookupCache:
    """Simple TTL cache for neuron exact-match lookups.

    Thread-safe for single-writer async patterns (no concurrent mutations).
    Keys are ``(content, type_value)`` tuples; values are neuron lists.

    Attributes:
        _ttl: Time-to-live in seconds for each entry.
        _max_entries: Maximum cache entries before eviction.
    """

    def __init__(self, ttl_seconds: float = 30.0, max_entries: int = 500) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._cache: dict[tuple[str, str | None], tuple[float, list[Neuron]]] = {}
        self._hits = 0
        self._misses = 0

    def get(self, content: str, neuron_type: str | None = None) -> list[Neuron] | None:
        """Look up cached neurons for an exact content match.

        Returns None on cache miss or expired entry.
        """
        key = (content, neuron_type)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        ts, neurons = entry
        if time.monotonic() - ts > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return neurons

    def put(self, content: str, neuron_type: str | None, neurons: list[Neuron]) -> None:
        """Cache neurons for an exact content match."""
        if len(self._cache) >= self._max_entries:
            self._evict_oldest()
        self._cache[(content, neuron_type)] = (time.monotonic(), neurons)

    def invalidate(self) -> None:
        """Clear all cached entries (called on neuron add/update/delete)."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of entries currently in cache."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction (0.0-1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def _evict_oldest(self) -> None:
        """Remove the oldest entry to make room."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
        del self._cache[oldest_key]
