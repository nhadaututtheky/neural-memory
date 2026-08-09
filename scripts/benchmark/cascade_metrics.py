"""Phase 4 cascade evidence helpers — route/stage latency aggregation.

Used by scale benchmarks and evidence reports. Marks metrics with
insufficient samples as ``inconclusive`` (never auto-pass).
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

# Minimum samples before a route/stage percentile is considered conclusive.
MIN_SAMPLES_PER_ROUTE = 5

# Objective gates from the Cognitive Efficiency scorecard (ms).
SIMPLE_P95_MS = 100.0
COGNITIVE_P95_MS = 250.0

SIMPLE_ROUTES = frozenset({"exact", "semantic"})
COGNITIVE_ROUTES = frozenset({"temporal", "causal"})


def percentile(values: Sequence[float], p: float) -> float | None:
    """Return the p-th percentile (0-100) or None if empty."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    if p <= 0:
        return ordered[0]
    if p >= 100:
        return ordered[-1]
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def aggregate_route_latencies(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_samples: int = MIN_SAMPLES_PER_ROUTE,
) -> dict[str, Any]:
    """Aggregate per-route latency distributions from query result rows.

    Each row may include:
      - ``route`` / ``cascade_route``: exact|semantic|temporal|causal
      - ``stage`` / ``cascade_stage``: candidate_exit|bounded_graph|...
      - ``latency_ms``: wall latency for the query
      - ``phase_timings_ms``: optional dict of cumulative phase marks

    Returns a report dict with per-route p50/p95/p99 and gate evaluation.
    """
    by_route: dict[str, list[float]] = {}
    by_stage: dict[str, list[float]] = {}
    simple: list[float] = []
    cognitive: list[float] = []

    for row in rows:
        route = str(row.get("route") or row.get("cascade_route") or "unknown")
        stage = str(row.get("stage") or row.get("cascade_stage") or "unknown")
        try:
            latency = float(row.get("latency_ms", 0.0))
        except (TypeError, ValueError):
            continue
        if latency < 0 or latency != latency:  # NaN guard
            continue
        by_route.setdefault(route, []).append(latency)
        by_stage.setdefault(stage, []).append(latency)
        if route in SIMPLE_ROUTES:
            simple.append(latency)
        elif route in COGNITIVE_ROUTES:
            cognitive.append(latency)

    def _bucket(samples: list[float]) -> dict[str, Any]:
        n = len(samples)
        conclusive = n >= min_samples
        p50 = percentile(samples, 50)
        p95 = percentile(samples, 95)
        p99 = percentile(samples, 99)
        return {
            "samples": n,
            "conclusive": conclusive,
            "p50_ms": None if p50 is None else round(p50, 3),
            "p95_ms": None if p95 is None else round(p95, 3),
            "p99_ms": None if p99 is None else round(p99, 3),
            "mean_ms": None if n == 0 else round(statistics.fmean(samples), 3),
        }

    routes_report = {route: _bucket(vals) for route, vals in sorted(by_route.items())}
    stages_report = {stage: _bucket(vals) for stage, vals in sorted(by_stage.items())}
    simple_bucket = _bucket(simple)
    cognitive_bucket = _bucket(cognitive)

    simple_gate = _eval_gate(simple_bucket, SIMPLE_P95_MS, label="simple")
    cognitive_gate = _eval_gate(cognitive_bucket, COGNITIVE_P95_MS, label="cognitive")

    return {
        "schema_version": 1,
        "min_samples": min_samples,
        "routes": routes_report,
        "stages": stages_report,
        "simple": simple_bucket,
        "cognitive": cognitive_bucket,
        "gates": {
            "simple_p95_ms": simple_gate,
            "cognitive_p95_ms": cognitive_gate,
        },
        "all_gates_pass": simple_gate["status"] == "pass" and cognitive_gate["status"] == "pass",
    }


def _eval_gate(bucket: Mapping[str, Any], limit_ms: float, *, label: str) -> dict[str, Any]:
    if not bucket.get("conclusive"):
        return {
            "label": label,
            "status": "inconclusive",
            "limit_ms": limit_ms,
            "p95_ms": bucket.get("p95_ms"),
            "samples": bucket.get("samples", 0),
            "reason": "insufficient_samples",
        }
    p95 = bucket.get("p95_ms")
    if p95 is None:
        return {
            "label": label,
            "status": "inconclusive",
            "limit_ms": limit_ms,
            "p95_ms": None,
            "samples": bucket.get("samples", 0),
            "reason": "missing_p95",
        }
    passed = float(p95) <= limit_ms
    return {
        "label": label,
        "status": "pass" if passed else "fail",
        "limit_ms": limit_ms,
        "p95_ms": p95,
        "samples": bucket.get("samples", 0),
        "reason": "within_budget" if passed else "exceeds_budget",
    }


def rows_from_retrieval_results(
    results: Sequence[Any],
) -> list[dict[str, Any]]:
    """Extract cascade rows from ``RetrievalResult``-like objects."""
    rows: list[dict[str, Any]] = []
    for result in results:
        meta = getattr(result, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        rows.append(
            {
                "route": meta.get("cascade_route"),
                "stage": meta.get("cascade_stage"),
                "latency_ms": getattr(result, "latency_ms", 0.0),
                "phase_timings_ms": meta.get("cascade_phase_timings_ms")
                or meta.get("phase_timings_ms"),
            }
        )
    return rows
