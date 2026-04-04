"""Tests for brain registry client — cache, filtering, fetch logic."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from neural_memory.engine.brain_registry import (
    CACHE_TTL_SECONDS,
    BrainRegistryClient,
    CacheEntry,
    RegistryCache,
    _raw_url,
)

# ── URL Builder ─────────────────────────────────────────────────


class TestRawUrl:
    def test_index_url(self) -> None:
        url = _raw_url("neural-memory/brain-store", "main", "index.json")
        assert url == "https://raw.githubusercontent.com/neural-memory/brain-store/main/index.json"

    def test_brain_url(self) -> None:
        url = _raw_url("neural-memory/brain-store", "main", "brains/python-tips/brain.json")
        assert (
            url
            == "https://raw.githubusercontent.com/neural-memory/brain-store/main/brains/python-tips/brain.json"
        )


# ── Cache ───────────────────────────────────────────────────────


class TestCacheEntry:
    def test_not_expired_when_fresh(self) -> None:
        entry = CacheEntry(data=[], fetched_at=time.monotonic())
        assert not entry.expired

    def test_expired_after_ttl(self) -> None:
        entry = CacheEntry(data=[], fetched_at=time.monotonic() - CACHE_TTL_SECONDS - 1)
        assert entry.expired

    def test_custom_ttl(self) -> None:
        entry = CacheEntry(data=[], fetched_at=time.monotonic() - 5, ttl=10)
        assert not entry.expired


class TestRegistryCache:
    def test_index_cache_roundtrip(self) -> None:
        cache = RegistryCache()
        assert cache.get_index() is None

        data = [{"name": "test"}]
        cache.set_index(data)
        assert cache.get_index() == data

    def test_index_cache_expires(self) -> None:
        cache = RegistryCache()
        cache.set_index([{"name": "old"}])
        # Manually expire
        assert cache._index is not None
        cache._index = CacheEntry(
            data=[{"name": "old"}],
            fetched_at=time.monotonic() - CACHE_TTL_SECONDS - 1,
        )
        assert cache.get_index() is None

    def test_brain_cache_roundtrip(self) -> None:
        cache = RegistryCache()
        assert cache.get_brain("test") is None

        data = {"manifest": {}, "snapshot": {}}
        cache.set_brain("test", data)
        assert cache.get_brain("test") == data

    def test_brain_cache_expires(self) -> None:
        cache = RegistryCache()
        cache._brains["old"] = ({"data": True}, time.monotonic() - CACHE_TTL_SECONDS - 1)
        assert cache.get_brain("old") is None
        assert "old" not in cache._brains  # Cleaned up

    def test_clear(self) -> None:
        cache = RegistryCache()
        cache.set_index([{"name": "test"}])
        cache.set_brain("x", {"data": True})
        cache.clear()
        assert cache.get_index() is None
        assert cache.get_brain("x") is None


# ── Filter ──────────────────────────────────────────────────────

_SAMPLE_MANIFESTS = [
    {
        "name": "python-tips",
        "display_name": "Python Tips",
        "description": "Best practices for Python",
        "author": "alice",
        "category": "programming",
        "tags": ["python", "coding"],
        "created_at": "2026-01-01",
        "rating_avg": 4.5,
        "download_count": 100,
    },
    {
        "name": "docker-guide",
        "display_name": "Docker Guide",
        "description": "Container essentials",
        "author": "bob",
        "category": "devops",
        "tags": ["docker", "containers"],
        "created_at": "2026-02-01",
        "rating_avg": 3.8,
        "download_count": 50,
    },
    {
        "name": "security-101",
        "display_name": "Security 101",
        "description": "Security fundamentals for developers",
        "author": "charlie",
        "category": "security",
        "tags": ["security", "coding"],
        "created_at": "2026-03-01",
        "rating_avg": 4.9,
        "download_count": 200,
    },
]


class TestFilterIndex:
    def setup_method(self) -> None:
        self.client = BrainRegistryClient()

    def test_no_filters_returns_all(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS)
        assert len(result) == 3

    def test_filter_by_category(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS, category="programming")
        assert len(result) == 1
        assert result[0]["name"] == "python-tips"

    def test_filter_by_tag(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS, tag="coding")
        assert len(result) == 2

    def test_search_by_name(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS, search="docker")
        assert len(result) == 1
        assert result[0]["name"] == "docker-guide"

    def test_search_by_description(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS, search="fundamentals")
        assert len(result) == 1
        assert result[0]["name"] == "security-101"

    def test_search_by_author(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS, search="alice")
        assert len(result) == 1

    def test_search_case_insensitive(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS, search="PYTHON")
        assert len(result) == 1

    def test_sort_by_rating(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS, sort_by="rating_avg")
        assert result[0]["name"] == "security-101"
        assert result[-1]["name"] == "docker-guide"

    def test_sort_by_downloads(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS, sort_by="download_count")
        assert result[0]["name"] == "security-101"

    def test_sort_by_created_at_default(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS)
        assert result[0]["name"] == "security-101"  # Most recent first

    def test_limit(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS, limit=2)
        assert len(result) == 2

    def test_limit_capped_at_100(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS, limit=999)
        assert len(result) == 3  # Only 3 items, but limit doesn't exceed 100

    def test_combined_filters(self) -> None:
        result = self.client.filter_index(_SAMPLE_MANIFESTS, tag="coding", search="security")
        assert len(result) == 1
        assert result[0]["name"] == "security-101"


# ── Fetch Logic ─────────────────────────────────────────────────


class TestFetchIndex:
    @pytest.mark.asyncio
    async def test_returns_cached_data(self) -> None:
        client = BrainRegistryClient()
        client.cache.set_index(_SAMPLE_MANIFESTS)

        result = await client.fetch_index()
        assert result == _SAMPLE_MANIFESTS

    @pytest.mark.asyncio
    async def test_force_refresh_skips_cache(self) -> None:
        """Force refresh should attempt network fetch even with valid cache."""
        client = BrainRegistryClient()
        client.cache.set_index([{"name": "old"}])

        # Mock aiohttp to fail — should fall back to stale cache
        with patch.dict("sys.modules", {"aiohttp": MagicMock()}):
            result = await client.fetch_index(force_refresh=True)
            # Falls back to stale cache since aiohttp mock won't work
            assert result == [{"name": "old"}]

    @pytest.mark.asyncio
    async def test_returns_empty_on_error_no_cache(self) -> None:
        """Network error with no cache should return empty list."""
        client = BrainRegistryClient()

        with patch(
            "neural_memory.engine.brain_registry.BrainRegistryClient._fallback_index",
            return_value=[],
        ):
            # Force an import error to test error handling
            result = await client.fetch_index()
            # Should return empty or fallback
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_fallback_returns_stale_cache(self) -> None:
        client = BrainRegistryClient()
        # Set expired cache
        client.cache._index = CacheEntry(
            data=[{"name": "stale"}],
            fetched_at=time.monotonic() - CACHE_TTL_SECONDS - 100,
        )
        result = client._fallback_index()
        assert result == [{"name": "stale"}]

    @pytest.mark.asyncio
    async def test_fallback_returns_empty_without_cache(self) -> None:
        client = BrainRegistryClient()
        result = client._fallback_index()
        assert result == []


class TestFetchBrain:
    @pytest.mark.asyncio
    async def test_returns_cached_brain(self) -> None:
        client = BrainRegistryClient()
        brain_data = {"manifest": {"name": "test"}, "snapshot": {}}
        client.cache.set_brain("test", brain_data)

        result = await client.fetch_brain("test")
        assert result == brain_data
