"""Brain Store registry client — browse, fetch, and publish community brains.

Architecture:
- **GitHub repo** stores brain packages (unlimited free storage)
- **Hub API** proxies browse (with ratings) and publish (creates GitHub PRs)
- **Fallback**: direct raw.githubusercontent.com if hub is down

Flow:
  browse → Hub /v1/store/browse → returns index + ratings
  fetch  → Hub /v1/store/brain/:name → proxies from GitHub raw
  publish → Hub /v1/store/publish → creates PR to brain-store repo
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────

DEFAULT_REGISTRY_REPO = "nhadaututtheky/brain-store"
DEFAULT_REGISTRY_BRANCH = "main"
DEFAULT_HUB_URL = "https://neural-memory-sync-hub.congnguyenit.workers.dev"
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
    """Client for the community brain registry.

    Primary: Hub API (browse with ratings, publish via GitHub PR).
    Fallback: raw.githubusercontent.com (direct GitHub access).
    """

    def __init__(
        self,
        repo: str = DEFAULT_REGISTRY_REPO,
        branch: str = DEFAULT_REGISTRY_BRANCH,
        hub_url: str = DEFAULT_HUB_URL,
    ) -> None:
        self.repo = repo
        self.branch = branch
        self.hub_url = hub_url.rstrip("/")
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

        Tries Hub API first (includes ratings), falls back to GitHub raw.
        Returns cached data if available and not expired.
        """
        if not force_refresh:
            cached = self.cache.get_index()
            if cached is not None:
                return cached

        # Try Hub API first (includes ratings data)
        data = await self._fetch_index_from_hub()
        if data is None:
            # Fall back to GitHub raw
            data = await self._fetch_index_from_github()

        if data is None:
            return self._fallback_index()

        self.cache.set_index(data)
        return data

    async def _fetch_index_from_hub(self) -> list[dict[str, Any]] | None:
        """Fetch index from Hub API (browse endpoint)."""
        try:
            import aiohttp

            url = f"{self.hub_url}/v1/store/browse?limit=100"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.debug("Hub browse failed: HTTP %d", resp.status)
                        return None

                    body = await resp.read()
                    if len(body) > MAX_INDEX_SIZE:
                        return None

                    import json as _json

                    result = _json.loads(body)

            if isinstance(result, dict) and isinstance(result.get("brains"), list):
                brains: list[dict[str, Any]] = result["brains"]
                return brains
        except Exception as e:
            logger.debug("Hub browse error: %s", e)

        return None

    async def _fetch_index_from_github(self) -> list[dict[str, Any]] | None:
        """Fetch index.json directly from GitHub raw (fallback)."""
        try:
            import aiohttp

            url = self._index_url()
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("GitHub index fetch failed: HTTP %d", resp.status)
                        return None

                    body = await resp.read()
                    if len(body) > MAX_INDEX_SIZE:
                        logger.warning("GitHub index too large: %d bytes", len(body))
                        return None

                    import json as _json

                    data = _json.loads(body)

            if isinstance(data, list):
                return data
        except Exception as e:
            logger.warning("GitHub index fetch error: %s", e)

        return None

    async def fetch_brain(self, name: str) -> dict[str, Any] | None:
        """Fetch a full brain package by name.

        Tries Hub API first, falls back to GitHub raw.
        Returns None if not found or on error.
        """
        cached = self.cache.get_brain(name)
        if cached is not None:
            return cached

        data = await self._fetch_brain_from_hub(name)
        if data is None:
            data = await self._fetch_brain_from_github(name)

        if data is not None and isinstance(data, dict):
            self.cache.set_brain(name, data)
            return data

        return None

    async def _fetch_brain_from_hub(self, name: str) -> dict[str, Any] | None:
        """Fetch a brain package from the Hub API."""
        try:
            import aiohttp

            url = f"{self.hub_url}/v1/store/brain/{name}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 404:
                        return None
                    if resp.status != 200:
                        logger.debug("Hub brain fetch failed: HTTP %d for %s", resp.status, name)
                        return None

                    data: dict[str, Any] = await resp.json()
                    return data if isinstance(data, dict) else None

        except Exception as e:
            logger.debug("Hub brain fetch error for %s: %s", name, e)
            return None

    async def _fetch_brain_from_github(self, name: str) -> dict[str, Any] | None:
        """Fetch a brain package directly from GitHub raw (fallback)."""
        try:
            import aiohttp

            url = self._brain_url(name)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 404:
                        return None
                    if resp.status != 200:
                        logger.warning(
                            "GitHub brain fetch failed: HTTP %d for %s", resp.status, name
                        )
                        return None

                    data: dict[str, Any] = await resp.json()
                    return data if isinstance(data, dict) else None

        except Exception as e:
            logger.warning("GitHub brain fetch error for %s: %s", name, e)
            return None

    async def fetch_brain_from_url(self, url: str) -> dict[str, Any] | None:
        """Fetch a brain package from a validated HTTPS URL.

        Only allows HTTPS scheme and blocks private/loopback IPs to prevent SSRF.
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme != "https":
            logger.warning("Rejected non-HTTPS brain URL: %s", parsed.scheme)
            return None

        hostname = parsed.hostname or ""
        if not hostname:
            return None

        # Block private/loopback/link-local IPs
        import ipaddress

        try:
            addr = ipaddress.ip_address(hostname)
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                logger.warning("Rejected private/loopback brain URL: %s", hostname)
                return None
        except ValueError:
            pass  # hostname is a domain name, not an IP — OK

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.warning("Brain URL fetch failed: HTTP %d from %s", resp.status, url)
                        return None

                    # Enforce body size limit before deserialization
                    body = await resp.read()
                    if len(body) > MAX_INDEX_SIZE:
                        logger.warning("Remote brain too large: %d bytes", len(body))
                        return None

                    import json as _json

                    data = _json.loads(body)

        except Exception as e:
            logger.warning("Brain URL fetch error: %s", e)
            return None

        return data if isinstance(data, dict) else None

    async def publish_brain(
        self,
        package: dict[str, Any],
        api_key: str,
    ) -> dict[str, Any]:
        """Publish a brain package to the community store via Hub.

        The Hub creates a PR to the brain-store GitHub repo.

        Args:
            package: The full .brain package dict (manifest + snapshot).
            api_key: Neural Memory API key for authentication.

        Returns:
            Response dict with status, PR URL, etc.

        Raises:
            ValueError: If publishing fails.
        """
        import json as _json

        import aiohttp

        url = f"{self.hub_url}/v1/store/publish"
        body = _json.dumps(package, default=str, ensure_ascii=False)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    result: dict[str, Any] = await resp.json()

                    if resp.status == 201:
                        # Invalidate cache so next browse shows new brain
                        self.cache.clear()
                        return result

                    error_msg: str = result.get("error", "Publishing failed")
                    raise ValueError(f"Publish failed ({resp.status}): {error_msg}")

        except aiohttp.ClientError as e:
            raise ValueError(f"Connection error: {e}") from e

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
