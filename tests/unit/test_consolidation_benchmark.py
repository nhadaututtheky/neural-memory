"""Smoke tests for consolidation benchmark module."""

from __future__ import annotations

from scripts.benchmark_consolidation import _rss_mb


def test_rss_helper_returns_float() -> None:
    assert isinstance(_rss_mb(), float)
