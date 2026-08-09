"""Unit tests for write-path benchmark helpers (Phase 5)."""

from __future__ import annotations

from scripts.benchmark_write_path import ACK_P95_MS, CONVERGENCE_P95_MS, percentile


def test_percentile_empty() -> None:
    assert percentile([], 50) == 0.0


def test_percentile_single() -> None:
    assert percentile([12.0], 95) == 12.0


def test_percentile_mid() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0


def test_gate_constants() -> None:
    assert ACK_P95_MS == 75.0
    assert CONVERGENCE_P95_MS == 5000.0
