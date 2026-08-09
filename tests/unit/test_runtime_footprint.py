"""Footprint helper tests."""

from __future__ import annotations

from scripts.measure_runtime_footprint import base_dep_count, schema_stats


def test_base_dep_count_within_budget() -> None:
    n = base_dep_count()
    assert 1 <= n <= 4


def test_standard_schema_smaller_than_full() -> None:
    std = schema_stats("standard")
    full = schema_stats("full")
    assert std["tool_count"] == 10
    assert full["tool_count"] >= 60
    assert std["schema_bytes"] < full["schema_bytes"]
