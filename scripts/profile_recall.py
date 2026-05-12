"""Profile recall latency on the user's real brain.

Runs N queries, collects phase_timings_ms from each,
prints per-phase breakdown (mean, p50, p95).

Usage:
    python scripts/profile_recall.py [--queries 20]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

QUERIES = [
    "Neural Memory architecture decisions",
    "Python async patterns",
    "config migration upgrade path",
    "InfinityDB performance optimization",
    "MCP tool handler implementation",
    "Telegram bot session management",
    "authentication and license activation",
    "consolidation quality tuning",
    "spreading activation lateral inhibition",
    "brain store publishing workflow",
    "error handling and logging patterns",
    "SQLite vector search embedding",
    "dashboard React components",
    "CI/CD pipeline GitHub Actions",
    "context compiler dedup merge",
    "fidelity decay ghost neurons",
    "sync hub Cloudflare Workers",
    "OpenClaw plugin development",
    "session reflection significance",
    "layered consciousness global brain",
]


async def main(num_queries: int) -> None:
    from neural_memory.engine.retrieval import ReflexPipeline
    from neural_memory.storage import SQLiteStorage
    from neural_memory.unified_config import UnifiedConfig

    config = UnifiedConfig.load()
    brain_name = config.current_brain or "default"

    # Try multiple DB path patterns (NM uses various layouts)
    candidates = [
        config.data_dir / "brains" / f"{brain_name}.db",
        config.data_dir / "brains" / brain_name / "brain.db",
        config.data_dir / brain_name / "brain.db",
        config.data_dir / f"{brain_name}.db",
    ]
    db_path = None
    for c in candidates:
        if c.exists():
            db_path = c
            break

    if db_path is None:
        print(f"Brain DB not found. Tried: {[str(c) for c in candidates]}")
        return

    storage = SQLiteStorage(db_path=db_path)
    await storage.initialize()

    brain = await storage.get_brain(brain_name)
    if brain is None:
        print(f"Brain '{brain_name}' not found in storage")
        return

    storage.set_brain(brain_name)
    pipeline = ReflexPipeline(storage, brain.config)

    phase_data: dict[str, list[float]] = {}
    total_latencies: list[float] = []

    queries = (QUERIES * ((num_queries // len(QUERIES)) + 1))[:num_queries]

    stats = await storage.get_stats(brain_name)
    print(f"Profiling {num_queries} queries on brain: {brain_name}")
    print(f"Neurons: {stats.get('neuron_count', '?')}, Fibers: {stats.get('fiber_count', '?')}")
    print()

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        result = await pipeline.query(q)
        total_ms = (time.perf_counter() - t0) * 1000
        total_latencies.append(total_ms)

        timings = result.metadata.get("phase_timings_ms", {}) if result.metadata else {}

        for phase, ms in timings.items():
            phase_data.setdefault(phase, []).append(ms)

        if (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{num_queries}] last={total_ms:.1f}ms")

    print()
    print("=" * 75)
    print(f"{'Phase':<25} {'Mean':>8} {'P50':>8} {'P95':>8} {'Max':>8}  (ms, cumulative)")
    print("-" * 75)

    # Convert cumulative timings to per-phase deltas
    sorted_phases = _get_phase_order(phase_data)
    prev_means: dict[str, float] = {}

    for phase in sorted_phases:
        values = phase_data[phase]
        mean_val = statistics.mean(values)
        p50 = sorted(values)[len(values) // 2]
        p95 = sorted(values)[int(len(values) * 0.95)]
        max_val = max(values)
        print(f"  {phase:<23} {mean_val:>7.1f} {p50:>7.1f} {p95:>7.1f} {max_val:>7.1f}")

    print("-" * 75)
    mean_total = statistics.mean(total_latencies)
    p50_total = sorted(total_latencies)[len(total_latencies) // 2]
    p95_total = sorted(total_latencies)[int(len(total_latencies) * 0.95)]
    max_total = max(total_latencies)
    print(
        f"  {'TOTAL (wall clock)':<23} {mean_total:>7.1f} {p50_total:>7.1f} {p95_total:>7.1f} {max_total:>7.1f}"
    )

    # Per-phase DELTA breakdown
    print()
    print("=" * 75)
    print(f"{'Phase':<25} {'Delta Mean':>10} {'% of Total':>10}  (per-phase cost)")
    print("-" * 75)

    prev_phase_mean = 0.0
    for phase in sorted_phases:
        values = phase_data[phase]
        mean_val = statistics.mean(values)
        delta = mean_val - prev_phase_mean
        pct = (delta / mean_total * 100) if mean_total > 0 else 0
        bar = "#" * max(1, int(pct / 2))
        print(f"  {phase:<23} {delta:>9.1f} {pct:>9.1f}%  {bar}")
        prev_phase_mean = mean_val

    # Unaccounted time (total - last phase)
    if sorted_phases:
        last_phase_mean = statistics.mean(phase_data[sorted_phases[-1]])
        unaccounted = mean_total - last_phase_mean
        pct = (unaccounted / mean_total * 100) if mean_total > 0 else 0
        bar = "#" * max(1, int(pct / 2))
        print(f"  {'(post-pipeline)':<23} {unaccounted:>9.1f} {pct:>9.1f}%  {bar}")

    print("=" * 75)

    await storage.close()


def _get_phase_order(phase_data: dict[str, list[float]]) -> list[str]:
    """Sort phases by their mean cumulative time (ascending)."""
    return sorted(phase_data.keys(), key=lambda p: statistics.mean(phase_data[p]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.queries))
