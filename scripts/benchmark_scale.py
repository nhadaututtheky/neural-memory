"""Benchmark: NeuralMemory recall latency at scale (10K / 50K / 100K neurons).

Measures:
  1. Encode throughput  — memories/sec at each neuron scale
  2. Recall latency     — p50 / p95 / p99 over 50 diverse queries
  3. Consolidation time — full ALL-strategy pass
  4. Memory usage       — RSS delta from empty → seeded brain

Run:
    python scripts/benchmark_scale.py                        # default: 1K, 10K
    python scripts/benchmark_scale.py --scales 1000,10000,50000
    python scripts/benchmark_scale.py --scales 100000 --runs 3

Output:
    stdout   — markdown table summary
    scripts/benchmark_scale_results.json — full results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil

# Ensure src is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Suppress noisy loggers
for _name in [
    "neural_memory.engine",
    "neural_memory.storage",
    "neural_memory.safety",
    "neural_memory.extraction",
    "neural_memory.core",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark-scale")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    """Latency percentiles in milliseconds."""

    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    samples: int = 0


@dataclass
class ScaleResult:
    """Benchmark results for a single neuron scale."""

    target_neurons: int = 0
    actual_neurons: int = 0
    actual_fibers: int = 0
    actual_synapses: int = 0
    encode_memories: int = 0
    encode_elapsed_sec: float = 0.0
    encode_rate: float = 0.0  # memories/sec
    recall_latency: LatencyStats = field(default_factory=LatencyStats)
    consolidation_ms: float = 0.0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_delta_mb: float = 0.0


@dataclass
class BenchmarkReport:
    """Full benchmark report across all scales."""

    timestamp: str = ""
    python_version: str = ""
    platform: str = ""
    scales: list[ScaleResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Seed generator
# ---------------------------------------------------------------------------

TOPICS = [
    "Python", "JavaScript", "Rust", "Go", "TypeScript", "C++", "Java",
    "Kotlin", "Swift", "Ruby", "Elixir", "Scala", "Haskell", "Clojure",
    "PostgreSQL", "Redis", "MongoDB", "SQLite", "MySQL", "Cassandra",
    "Docker", "Kubernetes", "Terraform", "Ansible", "Prometheus",
    "React", "Vue", "Angular", "Svelte", "Next.js", "Nuxt", "Remix",
    "FastAPI", "Django", "Flask", "Express", "Spring Boot", "Actix",
    "GraphQL", "gRPC", "REST", "WebSocket", "MQTT", "AMQP",
    "JWT", "OAuth2", "SAML", "CORS", "HTTPS", "mTLS",
]

ACTIONS = [
    "supports", "implements", "requires", "provides", "enables",
    "handles", "manages", "processes", "validates", "optimizes",
    "caches", "streams", "serializes", "encrypts", "compresses",
]

FEATURES = [
    "concurrent request handling with async I/O",
    "type-safe data validation using schemas",
    "automatic OpenAPI schema generation",
    "efficient memory management via arena allocation",
    "distributed caching with TTL-based eviction",
    "real-time event streaming over persistent connections",
    "structured error handling with typed exceptions",
    "automated test discovery and parallel execution",
    "incremental compilation for faster builds",
    "hot module replacement during development",
    "connection pooling with health checks",
    "query optimization via prepared statements",
    "load balancing with least-connections strategy",
    "rate limiting using token bucket algorithm",
    "health monitoring with circuit breaker patterns",
    "graceful shutdown with drain timeout",
    "zero-downtime deployment via rolling updates",
    "distributed tracing with OpenTelemetry spans",
    "schema migration with rollback support",
    "content-based routing for microservices",
]

CONTEXTS = [
    "in production environments",
    "for high-throughput scenarios",
    "when handling 10K+ concurrent users",
    "during peak traffic windows",
    "for latency-sensitive applications",
    "in containerized deployments",
    "across multi-region setups",
    "with strict compliance requirements",
    "for real-time data pipelines",
    "in event-driven architectures",
]

DECISIONS = [
    "We chose {t1} over {t2} because {f}",
    "The team decided to use {t1} for {f}",
    "After benchmarking, {t1} outperformed {t2} {c}",
    "{t1} was selected as the primary solution for {f}",
    "We migrated from {t2} to {t1} to improve {f}",
]

ERRORS = [
    "ConnectionError in {t1}: timeout after 30s {c}",
    "TypeError in {t1} integration: expected dict got None",
    "ImportError: {t1} module not found after upgrade",
    "MemoryError: {t1} OOM when processing large batch {c}",
    "ValueError: invalid config for {t1} {f}",
]

INSIGHTS = [
    "{t1} {a} {f} — discovered this improves throughput by 3x {c}",
    "Pattern: combining {t1} with {t2} for {f} reduces latency",
    "Key insight: {t1} {f} only works reliably {c}",
]


def generate_memories(n: int) -> list[str]:
    """Generate N unique, diverse memory strings."""
    memories: list[str] = []
    templates = []

    # Build template pool
    for i, topic in enumerate(TOPICS):
        for j, action in enumerate(ACTIONS):
            for k, feature in enumerate(FEATURES):
                ctx = CONTEXTS[(i + j + k) % len(CONTEXTS)]
                templates.append(f"{topic} {action} {feature} {ctx}")

    # Add decision/error/insight templates
    for i in range(200):
        t1 = TOPICS[i % len(TOPICS)]
        t2 = TOPICS[(i + 7) % len(TOPICS)]
        f = FEATURES[i % len(FEATURES)]
        a = ACTIONS[i % len(ACTIONS)]
        c = CONTEXTS[i % len(CONTEXTS)]

        tpl = DECISIONS[i % len(DECISIONS)]
        templates.append(tpl.format(t1=t1, t2=t2, f=f, a=a, c=c))

        tpl = ERRORS[i % len(ERRORS)]
        templates.append(tpl.format(t1=t1, t2=t2, f=f, a=a, c=c))

        tpl = INSIGHTS[i % len(INSIGHTS)]
        templates.append(tpl.format(t1=t1, t2=t2, f=f, a=a, c=c))

    # Cycle through templates, add index for uniqueness
    for i in range(n):
        base = templates[i % len(templates)]
        if i >= len(templates):
            memories.append(f"{base} (variant #{i})")
        else:
            memories.append(base)

    return memories


# ---------------------------------------------------------------------------
# Query set
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES = [
    # Exact keyword (10)
    "Python", "Redis", "Docker", "GraphQL", "JWT",
    "Kubernetes", "PostgreSQL", "React", "FastAPI", "WebSocket",
    # Multi-word (10)
    "Python concurrency async", "Redis caching TTL eviction",
    "Docker container orchestration", "GraphQL schema generation",
    "JWT token authentication security", "Kubernetes pod scaling",
    "PostgreSQL query optimization", "React component rendering",
    "FastAPI middleware validation", "WebSocket real-time streaming",
    # Question-style (10)
    "What framework handles concurrent requests?",
    "Which database supports JSON storage?",
    "How does authentication work?",
    "What tool manages container deployments?",
    "Which system handles rate limiting?",
    "What provides distributed caching?",
    "How are errors handled in production?",
    "What handles schema migrations?",
    "Which framework supports hot reload?",
    "What monitors service health?",
    # Cross-topic (10)
    "connection between Redis and PostgreSQL caching",
    "Docker and Kubernetes deployment pipeline",
    "FastAPI with JWT authentication flow",
    "React and GraphQL data fetching",
    "Python type safety and validation",
    "WebSocket and Redis pub/sub integration",
    "MongoDB vs PostgreSQL migration decision",
    "Terraform infrastructure as code patterns",
    "OAuth2 and CORS security configuration",
    "gRPC vs REST API performance comparison",
    # Broad/vague (10)
    "performance optimization", "error handling patterns",
    "security best practices", "deployment strategy",
    "data validation", "monitoring and observability",
    "testing automation", "memory management",
    "connection handling", "configuration management",
]


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def get_rss_mb() -> float:
    """Get current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


async def seed_brain(
    n_memories: int,
    db_path: str,
    progress_interval: int = 500,
) -> tuple["SQLiteStorage", "MemoryEncoder", float]:
    """Seed a brain with n_memories and return (storage, encoder, elapsed_sec)."""
    from neural_memory.core.brain import Brain, BrainConfig
    from neural_memory.engine.encoder import MemoryEncoder
    from neural_memory.storage.sqlite_store import SQLiteStorage

    storage = SQLiteStorage(db_path=db_path)
    await storage.initialize()

    config = BrainConfig(
        decay_rate=0.1,
        reinforcement_delta=0.05,
        activation_threshold=0.15,
        max_spread_hops=3,
        max_context_tokens=1500,
    )
    brain = Brain.create(name="benchmark", config=config)
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    encoder = MemoryEncoder(storage=storage, config=brain.config)
    memories = generate_memories(n_memories)

    print(f"  Seeding {n_memories} memories...", flush=True)
    t0 = time.perf_counter()

    for i, content in enumerate(memories):
        await encoder.encode(content)
        if (i + 1) % progress_interval == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"    [{i + 1}/{n_memories}] {rate:.0f} mem/s", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  Seeded {n_memories} memories in {elapsed:.1f}s ({n_memories / elapsed:.0f} mem/s)")

    return storage, encoder, elapsed


async def bench_recall(
    storage: "SQLiteStorage",
    queries: list[str],
    runs: int = 1,
) -> LatencyStats:
    """Measure recall latency across queries * runs."""
    from neural_memory.engine.retrieval import ReflexPipeline

    brain = await storage.get_brain(storage._current_brain_id)  # type: ignore[arg-type]
    assert brain is not None
    pipeline = ReflexPipeline(storage=storage, config=brain.config)

    # Warmup — 3 queries to stabilize caches
    for q in queries[:3]:
        await pipeline.query(q)

    latencies_ms: list[float] = []

    for run_idx in range(runs):
        for q in queries:
            t0 = time.perf_counter()
            await pipeline.query(q)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)

        if runs > 1:
            print(f"    Recall run {run_idx + 1}/{runs} done", flush=True)

    latencies_ms.sort()
    n = len(latencies_ms)

    return LatencyStats(
        p50=latencies_ms[int(n * 0.50)],
        p95=latencies_ms[int(n * 0.95)],
        p99=latencies_ms[int(n * 0.99)] if n >= 100 else latencies_ms[-1],
        mean=statistics.mean(latencies_ms),
        min=latencies_ms[0],
        max=latencies_ms[-1],
        samples=n,
    )


async def bench_consolidation(storage: "SQLiteStorage") -> float:
    """Run full consolidation and return duration in ms."""
    from neural_memory.engine.consolidation import (
        ConsolidationEngine,
        ConsolidationStrategy,
    )

    engine = ConsolidationEngine(storage=storage)

    t0 = time.perf_counter()
    await engine.run(strategies=[ConsolidationStrategy.ALL])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return elapsed_ms


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def format_markdown(report: BenchmarkReport) -> str:
    """Format results as a markdown table."""
    lines = [
        f"# NeuralMemory Scale Benchmark",
        f"",
        f"**Date**: {report.timestamp}",
        f"**Python**: {report.python_version}",
        f"**Platform**: {report.platform}",
        f"",
        f"## Results",
        f"",
        f"| Scale | Neurons | Fibers | Synapses | Encode (mem/s) | Recall p50 | Recall p95 | Recall p99 | Consolidation | RSS delta |",
        f"|------:|--------:|-------:|---------:|---------------:|-----------:|-----------:|-----------:|--------------:|----------:|",
    ]

    for s in report.scales:
        lines.append(
            f"| {s.target_neurons:,} "
            f"| {s.actual_neurons:,} "
            f"| {s.actual_fibers:,} "
            f"| {s.actual_synapses:,} "
            f"| {s.encode_rate:,.0f} "
            f"| {s.recall_latency.p50:,.1f}ms "
            f"| {s.recall_latency.p95:,.1f}ms "
            f"| {s.recall_latency.p99:,.1f}ms "
            f"| {s.consolidation_ms:,.0f}ms "
            f"| {s.rss_delta_mb:,.1f}MB |"
        )

    lines.extend([
        "",
        "## Recall Latency Detail",
        "",
        "| Scale | Samples | Mean | Min | Max | p50 | p95 | p99 |",
        "|------:|--------:|-----:|----:|----:|----:|----:|----:|",
    ])

    for s in report.scales:
        lat = s.recall_latency
        lines.append(
            f"| {s.target_neurons:,} "
            f"| {lat.samples} "
            f"| {lat.mean:,.1f}ms "
            f"| {lat.min:,.1f}ms "
            f"| {lat.max:,.1f}ms "
            f"| {lat.p50:,.1f}ms "
            f"| {lat.p95:,.1f}ms "
            f"| {lat.p99:,.1f}ms |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def estimate_memories_for_neurons(target_neurons: int) -> int:
    """Estimate how many memories to encode to reach target neuron count.

    Each memory typically creates 3-7 neurons (content + entity + type nodes).
    We use a conservative estimate of 4 neurons/memory.
    """
    return max(target_neurons // 4, 100)


async def run_scale(
    target_neurons: int,
    queries: list[str],
    runs: int,
    tmp_dir: str,
) -> ScaleResult:
    """Run benchmark at a single scale."""
    n_memories = estimate_memories_for_neurons(target_neurons)

    print(f"\n{'='*60}")
    print(f"Scale: {target_neurons:,} neurons (est. {n_memories:,} memories)")
    print(f"{'='*60}")

    result = ScaleResult(target_neurons=target_neurons)

    db_path = os.path.join(tmp_dir, f"bench_{target_neurons}.db")

    # 1. Seed
    rss_before = get_rss_mb()
    result.rss_before_mb = rss_before

    storage, encoder, encode_elapsed = await seed_brain(n_memories, db_path)
    result.encode_memories = n_memories
    result.encode_elapsed_sec = encode_elapsed
    result.encode_rate = n_memories / encode_elapsed

    # Get actual counts
    brain_id = storage._current_brain_id
    assert brain_id is not None
    stats = await storage.get_stats(brain_id)
    result.actual_neurons = stats.get("neuron_count", 0)
    result.actual_fibers = stats.get("fiber_count", 0)
    result.actual_synapses = stats.get("synapse_count", 0)

    rss_after_seed = get_rss_mb()
    result.rss_after_mb = rss_after_seed
    result.rss_delta_mb = rss_after_seed - rss_before

    print(f"  Actual: {result.actual_neurons:,} neurons, "
          f"{result.actual_fibers:,} fibers, "
          f"{result.actual_synapses:,} synapses")
    print(f"  RSS: {rss_before:.0f}MB -> {rss_after_seed:.0f}MB "
          f"(+{result.rss_delta_mb:.1f}MB)")

    # 2. Recall benchmark
    print(f"  Running recall benchmark ({len(queries)} queries x {runs} runs)...")
    result.recall_latency = await bench_recall(storage, queries, runs)
    print(f"  Recall: p50={result.recall_latency.p50:.1f}ms "
          f"p95={result.recall_latency.p95:.1f}ms "
          f"p99={result.recall_latency.p99:.1f}ms")

    # 3. Consolidation benchmark
    print(f"  Running consolidation benchmark...")
    result.consolidation_ms = await bench_consolidation(storage)
    print(f"  Consolidation: {result.consolidation_ms:.0f}ms")

    await storage.close()

    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralMemory scale benchmark")
    parser.add_argument(
        "--scales",
        type=str,
        default="1000,10000",
        help="Comma-separated neuron counts (default: 1000,10000)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of recall runs per scale (default: 2)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=50,
        help="Number of queries per run (default: 50, max 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "benchmark_scale_results.json"),
        help="Output JSON path",
    )

    args = parser.parse_args()
    scales = [int(s.strip()) for s in args.scales.split(",")]
    n_queries = min(args.queries, len(BENCHMARK_QUERIES))
    queries = BENCHMARK_QUERIES[:n_queries]

    print("NeuralMemory Scale Benchmark")
    print(f"Scales: {', '.join(f'{s:,}' for s in scales)} neurons")
    print(f"Queries: {n_queries} x {args.runs} runs each")
    print(f"Output: {args.output}")

    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        python_version=sys.version.split()[0],
        platform=sys.platform,
    )

    with tempfile.TemporaryDirectory(prefix="nm_bench_") as tmp_dir:
        for scale in scales:
            result = await run_scale(scale, queries, args.runs, tmp_dir)
            report.scales.append(result)

    # Output
    md = format_markdown(report)
    print(f"\n{'='*60}")
    print(md)

    # Save JSON
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
