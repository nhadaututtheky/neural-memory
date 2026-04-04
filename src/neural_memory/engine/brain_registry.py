"""Brain Store registry client — fetches and caches the brain catalog from GitHub.

The registry is a GitHub repository containing:
- index.json: array of BrainPackageManifest dicts (no snapshot data)
- brains/{name}/brain.json: full .brain packages

Distribution uses raw.githubusercontent.com for zero-auth, zero-cost access.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────

DEFAULT_REGISTRY_REPO = "neural-memory/brain-store"
DEFAULT_REGISTRY_BRANCH = "main"
CACHE_TTL_SECONDS = 300  # 5 minutes
MAX_INDEX_SIZE = 5 * 1024 * 1024  # 5MB max for index.json


def _raw_url(repo: str, branch: str, path: str) -> str:
    """Build a raw.githubusercontent.com URL."""
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


# ── Cache ───────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """A cached registry response with TTL."""

    data: list[dict[str, Any]]
    fetched_at: float
    ttl: float = CACHE_TTL_SECONDS

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.fetched_at) > self.ttl


@dataclass
class RegistryCache:
    """Simple in-memory cache for registry data."""

    _index: CacheEntry | None = field(default=None, repr=False)
    _brains: dict[str, tuple[dict[str, Any], float]] = field(default_factory=dict, repr=False)

    def get_index(self) -> list[dict[str, Any]] | None:
        """Get cached index if not expired."""
        if self._index is not None and not self._index.expired:
            return self._index.data
        return None

    def set_index(self, data: list[dict[str, Any]]) -> None:
        """Cache the index."""
        self._index = CacheEntry(data=data, fetched_at=time.monotonic())

    def get_brain(self, name: str) -> dict[str, Any] | None:
        """Get a cached brain package if not expired."""
        if name in self._brains:
            data, fetched_at = self._brains[name]
            if (time.monotonic() - fetched_at) <= CACHE_TTL_SECONDS:
                return data
            del self._brains[name]
        return None

    def set_brain(self, name: str, data: dict[str, Any]) -> None:
        """Cache a brain package."""
        self._brains[name] = (data, time.monotonic())

    def clear(self) -> None:
        """Clear all cached data."""
        self._index = None
        self._brains.clear()


# ── Registry Client ─────────────────────────────────────────────


class BrainRegistryClient:
    """Client for the GitHub-based brain registry.

    Fetches index.json and individual brain packages from
    raw.githubusercontent.com with in-memory caching.
    """

    def __init__(
        self,
        repo: str = DEFAULT_REGISTRY_REPO,
        branch: str = DEFAULT_REGISTRY_BRANCH,
    ) -> None:
        self.repo = repo
        self.branch = branch
        self.cache = RegistryCache()

    def _index_url(self) -> str:
        return _raw_url(self.repo, self.branch, "index.json")

    def _brain_url(self, name: str) -> str:
        return _raw_url(self.repo, self.branch, f"brains/{name}/brain.json")

    async def fetch_index(
        self,
        *,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch the registry index (catalog of manifests).

        Returns cached data if available and not expired.
        Falls back to cached data on fetch errors.
        """
        if not force_refresh:
            cached = self.cache.get_index()
            if cached is not None:
                return cached

        try:
            import aiohttp

            url = self._index_url()
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("Registry fetch failed: HTTP %d from %s", resp.status, url)
                        return self._fallback_index()

                    content_length = resp.content_length or 0
                    if content_length > MAX_INDEX_SIZE:
                        logger.warning("Registry index too large: %d bytes", content_length)
                        return self._fallback_index()

                    data = await resp.json()

        except Exception as e:
            logger.warning("Registry fetch error: %s", e)
            return self._fallback_index()

        if not isinstance(data, list):
            logger.warning("Registry index is not a list")
            return self._fallback_index()

        self.cache.set_index(data)
        return data

    async def fetch_brain(self, name: str) -> dict[str, Any] | None:
        """Fetch a full brain package by name.

        Returns None if not found or on error.
        """
        cached = self.cache.get_brain(name)
        if cached is not None:
            return cached

        try:
            import aiohttp

            url = self._brain_url(name)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 404:
                        return None
                    if resp.status != 200:
                        logger.warning("Brain fetch failed: HTTP %d for %s", resp.status, name)
                        return None

                    data = await resp.json()

        except Exception as e:
            logger.warning("Brain fetch error for %s: %s", name, e)
            return None

        if isinstance(data, dict):
            self.cache.set_brain(name, data)
            return data

        return None

    async def fetch_brain_from_url(self, url: str) -> dict[str, Any] | None:
        """Fetch a brain package from an arbitrary URL.

        Used for importing brains from direct URLs (not just the registry).
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.warning("Brain URL fetch failed: HTTP %d from %s", resp.status, url)
                        return None

                    data = await resp.json()

        except Exception as e:
            logger.warning("Brain URL fetch error: %s", e)
            return None

        return data if isinstance(data, dict) else None

    def _fallback_index(self) -> list[dict[str, Any]]:
        """Return stale cache or empty list on errors."""
        if self.cache._index is not None:
            logger.info("Using stale registry cache")
            return self.cache._index.data
        return []

    def filter_index(
        self,
        manifests: list[dict[str, Any]],
        *,
        category: str | None = None,
        search: str | None = None,
        tag: str | None = None,
        sort_by: str = "created_at",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Filter and sort the registry index.

        Args:
            manifests: List of manifest dicts from fetch_index().
            category: Filter by category (exact match).
            search: Search in name, display_name, description, author.
            tag: Filter by tag (must contain this tag).
            sort_by: Sort field — created_at, rating_avg, download_count.
            limit: Max results to return.

        Returns:
            Filtered and sorted list of manifests.
        """
        results = list(manifests)

        if category:
            results = [m for m in results if m.get("category") == category]

        if tag:
            results = [m for m in results if tag in m.get("tags", [])]

        if search:
            query = search.lower()
            results = [
                m
                for m in results
                if query in m.get("name", "").lower()
                or query in m.get("display_name", "").lower()
                or query in m.get("description", "").lower()
                or query in m.get("author", "").lower()
                or any(query in t.lower() for t in m.get("tags", []))
            ]

        # Sort
        if sort_by == "rating_avg":
            results.sort(key=lambda m: m.get("rating_avg", 0), reverse=True)
        elif sort_by == "download_count":
            results.sort(key=lambda m: m.get("download_count", 0), reverse=True)
        else:
            results.sort(key=lambda m: m.get("created_at", ""), reverse=True)

        return results[: min(limit, 100)]
