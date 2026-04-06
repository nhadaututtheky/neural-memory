"""Benchmark: SQLite vs InfinityDB — side-by-side recall latency at scale.

Measures per-phase timing breakdown for the full recall pipeline on both backends.

Run:
    python scripts/benchmark_backends.py                          # SQLite only, 1K
    python scripts/benchmark_backends.py --backends sqlite,infdb  # both backends
    python scripts/benchmark_backends.py --scales 1000,10000 --backends sqlite,infdb
    python scripts/benchmark_backends.py --scales 1000 --backends infdb --runs 3

Output:
    stdout   — markdown table summary with per-phase breakdown
    scripts/benchmark_backends_results.json — full results
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
from typing import Any

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
    "neural_memory.pro",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark-backends")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PhaseTimings:
    """Per-phase cumulative timings in ms (wall-clock from query start)."""

    parse: float = 0.0
    simhash_prefilter: float = 0.0
    anchors_rrf: float = 0.0
    activation: float = 0.0
    post_activation: float = 0.0
    fibers: float = 0.0
    reconstruction: float = 0.0


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
    """Benchmark results for a single (backend, scale) pair."""

    backend: str = ""
    target_neurons: int = 0
    actual_neurons: int = 0
    actual_fibers: int = 0
    actual_synapses: int = 0
    encode_memories: int = 0
    encode_elapsed_sec: float = 0.0
    encode_rate: float = 0.0
    recall_latency: LatencyStats = field(default_factory=LatencyStats)
    phase_timings_avg: PhaseTimings = field(default_factory=PhaseTimings)
    consolidation_ms: float = 0.0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_delta_mb: float = 0.0


@dataclass
class BenchmarkReport:
    """Full benchmark report across all scales and backends."""

    timestamp: str = ""
    python_version: str = ""
    platform: str = ""
    scales: list[ScaleResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Seed generator (reuse from benchmark_scale)
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
    templates: list[str] = []

    for i, topic in enumerate(TOPICS):
        for j, action in enumerate(ACTIONS):
            for k, feature in enumerate(FEATURES):
                ctx = CONTEXTS[(i + j + k) % len(CONTEXTS)]
                templates.append(f"{topic} {action} {feature} {ctx}")

    for i in range(200):
        t1 = TOPICS[i % len(TOPICS)]
        t2 = TOPICS[(i + 7) % len(TOPICS)]
        f = FEATURES[i % len(FEATURES)]
        a = ACTIONS[i % len(ACTIONS)]
        c = CONTEXTS[i % len(CONTEXTS)]

        templates.append(DECISIONS[i % len(DECISIONS)].format(t1=t1, t2=t2, f=f, a=a, c=c))
        templates.append(ERRORS[i % len(ERRORS)].format(t1=t1, t2=t2, f=f, a=a, c=c))
        templates.append(INSIGHTS[i % len(INSIGHTS)].format(t1=t1, t2=t2, f=f, a=a, c=c))

    for i in range(n):
        base = templates[i % len(templates)]
        if i >= len(templates):
            memories.append(f"{base} (variant #{i})")
        else:
            memories.append(base)

    return memories


# ---------------------------------------------------------------------------
# Query set (same as benchmark_scale.py for comparability)
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES = [
    "Python", "Redis", "Docker", "GraphQL", "JWT",
    "Kubernetes", "PostgreSQL", "React", "FastAPI", "WebSocket",
    "Python concurrency async", "Redis caching TTL eviction",
    "Docker container orchestration", "GraphQL schema generation",
    "JWT token authentication security", "Kubernetes pod scaling",
    "PostgreSQL query optimization", "React component rendering",
    "FastAPI middleware validation", "WebSocket real-time streaming",
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
    "performance optimization", "error handling patterns",
    "security best practices", "deployment strategy",
    "data validation", "monitoring and observability",
    "testing automation", "memory management",
    "connection handling", "configuration management",
]


# ---------------------------------------------------------------------------
# Storage factories
# ---------------------------------------------------------------------------


async def make_sqlite(db_path: str, brain_id: str) -> Any:
    """Create and initialize a SQLite storage backend."""
    from neural_memory.storage.sqlite_store import SQLiteStorage

    storage = SQLiteStorage(db_path=db_path)
    await storage.initialize()
    return storage


async def make_infinitydb(base_dir: str, brain_id: str) -> Any:
    """Create and initialize an InfinityDB storage backend."""
    from neural_memory.pro.storage_adapter import InfinityDBStorage

    storage = InfinityDBStorage(base_dir=base_dir, brain_id=brain_id, dimensions=384)
    await storage.open()
    return storage


def check_infdb_available() -> bool:
    """Check if InfinityDB dependencies are installed."""
    try:
        from neural_memory.pro import is_pro_deps_installed
        return is_pro_deps_installed()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def estimate_memories_for_neurons(target_neurons: int) -> int:
    return max(target_neurons // 4, 100)


async def seed_brain(
    n_memories: int,
    storage: Any,
    brain_id: str,
    progress_interval: int = 500,
) -> float:
    """Seed a brain with n_memories on the given storage. Returns elapsed_sec."""
    from neural_memory.core.brain import Brain, BrainConfig
    from neural_memory.engine.encoder import MemoryEncoder

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
    return elapsed


async def bench_recall(
    storage: Any,
    queries: list[str],
    runs: int = 1,
) -> tuple[LatencyStats, PhaseTimings, list[dict[str, float]]]:
    """Measure recall latency + per-phase timings. Returns (stats, avg_phases, all_phase_timings)."""
    from neural_memory.engine.retrieval import ReflexPipeline

    # Get brain — InfinityDB uses db.brain_id, SQLite uses _current_brain_id
    brain_id = getattr(storage, "_current_brain_id", None)
    if brain_id is None and hasattr(storage, "db"):
        brain_id = storage.db.brain_id
    assert brain_id is not None, "No brain_id set on storage"
    brain = await storage.get_brain(brain_id)
    if brain is None:
        # InfinityDB may not find brain by arbitrary ID — use default config
        from neural_memory.core.brain import BrainConfig
        config = BrainConfig(
            decay_rate=0.1,
            reinforcement_delta=0.05,
            activation_threshold=0.15,
            max_spread_hops=3,
            max_context_tokens=1500,
        )
    else:
        config = brain.config
    pipeline = ReflexPipeline(storage=storage, config=config)

    # Warmup
    for q in queries[:3]:
        await pipeline.query(q)

    latencies_ms: list[float] = []
    all_phases: list[dict[str, float]] = []

    for run_idx in range(runs):
        for q in queries:
            t0 = time.perf_counter()
            result = await pipeline.query(q)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)

            # Extract phase timings from result metadata
            phase_data = result.metadata.get("phase_timings_ms", {})
            all_phases.append(phase_data)

        if runs > 1:
            print(f"    Recall run {run_idx + 1}/{runs} done", flush=True)

    latencies_ms.sort()
    n = len(latencies_ms)

    stats = LatencyStats(
        p50=latencies_ms[int(n * 0.50)],
        p95=latencies_ms[int(n * 0.95)],
        p99=latencies_ms[int(n * 0.99)] if n >= 100 else latencies_ms[-1],
        mean=statistics.mean(latencies_ms),
        min=latencies_ms[0],
        max=latencies_ms[-1],
        samples=n,
    )

    # Average phase timings (convert cumulative → delta)
    phase_keys = ["parse", "simhash_prefilter", "anchors_rrf", "activation",
                  "post_activation", "fibers", "reconstruction"]
    avg_phases = PhaseTimings()
    if all_phases:
        for key in phase_keys:
            values = [p.get(key, 0.0) for p in all_phases if key in p]
            if values:
                setattr(avg_phases, key, statistics.mean(values))

    return stats, avg_phases, all_phases


async def bench_consolidation(storage: Any) -> float:
    """Run full consolidation and return duration in ms."""
    from neural_memory.engine.consolidation import (
        ConsolidationEngine,
        ConsolidationStrategy,
    )

    engine = ConsolidationEngine(storage=storage)
    t0 = time.perf_counter()
    try:
        await engine.run(strategies=[ConsolidationStrategy.ALL])
    except (AttributeError, NotImplementedError) as e:
        logger.warning("Consolidation not fully supported on this backend: %s", e)
        return -1.0
    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def format_markdown(report: BenchmarkReport) -> str:
    """Format results as markdown tables."""
    lines = [
        "# NeuralMemory Backend Comparison Benchmark",
        "",
        f"**Date**: {report.timestamp}",
        f"**Python**: {report.python_version}",
        f"**Platform**: {report.platform}",
        "",
        "## Summary",
        "",
        "| Backend | Scale | Neurons | Encode (mem/s) | Recall p50 | Recall p95 | Recall p99 | Consol. | RSS delta |",
        "|---------|------:|--------:|---------------:|-----------:|-----------:|-----------:|--------:|------:|",
    ]

    for s in report.scales:
        lines.append(
            f"| {s.backend} "
            f"| {s.target_neurons:,} "
            f"| {s.actual_neurons:,} "
            f"| {s.encode_rate:,.0f} "
            f"| {s.recall_latency.p50:,.1f}ms "
            f"| {s.recall_latency.p95:,.1f}ms "
            f"| {s.recall_latency.p99:,.1f}ms "
            f"| {s.consolidation_ms:,.0f}ms "
            f"| {s.rss_delta_mb:,.1f}MB |"
        )

    # Phase timing breakdown
    lines.extend([
        "",
        "## Phase Timing Breakdown (avg cumulative ms from query start)",
        "",
        "| Backend | Scale | Parse | SimHash | Anchors+RRF | Activation | Post-Act | Fibers | Reconstruction |",
        "|---------|------:|------:|--------:|------------:|-----------:|---------:|-------:|---------------:|",
    ])

    for s in report.scales:
        pt = s.phase_timings_avg
        lines.append(
            f"| {s.backend} "
            f"| {s.target_neurons:,} "
            f"| {pt.parse:,.1f} "
            f"| {pt.simhash_prefilter:,.1f} "
            f"| {pt.anchors_rrf:,.1f} "
            f"| {pt.activation:,.1f} "
            f"| {pt.post_activation:,.1f} "
            f"| {pt.fibers:,.1f} "
            f"| {pt.reconstruction:,.1f} |"
        )

    # Delta breakdown (phase N - phase N-1 = actual time spent in each phase)
    lines.extend([
        "",
        "## Phase Delta Breakdown (avg ms spent IN each phase)",
        "",
        "| Backend | Scale | Parse | SimHash | Anchors+RRF | Activation | Post-Act | Fibers | Recon |",
        "|---------|------:|------:|--------:|------------:|-----------:|---------:|-------:|------:|",
    ])

    for s in report.scales:
        pt = s.phase_timings_avg
        d_parse = pt.parse
        d_simhash = pt.simhash_prefilter - pt.parse
        d_anchors = pt.anchors_rrf - pt.simhash_prefilter
        d_activation = pt.activation - pt.anchors_rrf
        d_post = pt.post_activation - pt.activation
        d_fibers = pt.fibers - pt.post_activation
        d_recon = pt.reconstruction - pt.fibers
        lines.append(
            f"| {s.backend} "
            f"| {s.target_neurons:,} "
            f"| {d_parse:,.1f} "
            f"| {d_simhash:,.1f} "
            f"| {d_anchors:,.1f} "
            f"| {d_activation:,.1f} "
            f"| {d_post:,.1f} "
            f"| {d_fibers:,.1f} "
            f"| {d_recon:,.1f} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_scale(
    backend: str,
    target_neurons: int,
    queries: list[str],
    runs: int,
    tmp_dir: str,
) -> ScaleResult:
    """Run benchmark at a single (backend, scale) pair."""
    n_memories = estimate_memories_for_neurons(target_neurons)

    print(f"\n{'='*60}")
    print(f"Backend: {backend.upper()} | Scale: {target_neurons:,} neurons (est. {n_memories:,} memories)")
    print(f"{'='*60}")

    result = ScaleResult(backend=backend, target_neurons=target_neurons)

    rss_before = get_rss_mb()
    result.rss_before_mb = rss_before

    # Create storage
    if backend == "sqlite":
        db_path = os.path.join(tmp_dir, f"bench_{backend}_{target_neurons}.db")
        storage = await make_sqlite(db_path, "benchmark")
    elif backend == "infdb":
        base_dir = os.path.join(tmp_dir, f"bench_{backend}_{target_neurons}")
        storage = await make_infinitydb(base_dir, "benchmark")
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # 1. Seed
    encode_elapsed = await seed_brain(n_memories, storage, "benchmark")
    result.encode_memories = n_memories
    result.encode_elapsed_sec = encode_elapsed
    result.encode_rate = n_memories / encode_elapsed

    # Flush InfinityDB stores (including Tantivy text index) before recall
    if hasattr(storage, "db") and hasattr(storage.db, "flush"):
        await storage.db.flush()

    # Get actual counts
    brain_id = getattr(storage, "_current_brain_id", None)
    if brain_id is None and hasattr(storage, "db"):
        brain_id = storage.db.brain_id
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
    recall_stats, avg_phases, _all_phases = await bench_recall(storage, queries, runs)
    result.recall_latency = recall_stats
    result.phase_timings_avg = avg_phases
    print(f"  Recall: p50={recall_stats.p50:.1f}ms "
          f"p95={recall_stats.p95:.1f}ms "
          f"p99={recall_stats.p99:.1f}ms")

    # Print phase delta breakdown
    pt = avg_phases
    print(f"  Phases (delta): parse={pt.parse:.1f}ms "
          f"simhash={pt.simhash_prefilter - pt.parse:.1f}ms "
          f"anchors={pt.anchors_rrf - pt.simhash_prefilter:.1f}ms "
          f"activation={pt.activation - pt.anchors_rrf:.1f}ms "
          f"post={pt.post_activation - pt.activation:.1f}ms "
          f"fibers={pt.fibers - pt.post_activation:.1f}ms "
          f"recon={pt.reconstruction - pt.fibers:.1f}ms")

    # 3. Consolidation benchmark
    print(f"  Running consolidation benchmark...")
    result.consolidation_ms = await bench_consolidation(storage)
    print(f"  Consolidation: {result.consolidation_ms:.0f}ms")

    await storage.close()

    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralMemory backend comparison benchmark")
    parser.add_argument(
        "--scales",
        type=str,
        default="1000",
        help="Comma-separated neuron counts (default: 1000)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="sqlite",
        help="Comma-separated backends: sqlite,infdb (default: sqlite)",
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
        default=str(Path(__file__).parent / "benchmark_backends_results.json"),
        help="Output JSON path",
    )

    args = parser.parse_args()
    scales = [int(s.strip()) for s in args.scales.split(",")]
    backends = [b.strip() for b in args.backends.split(",")]
    n_queries = min(args.queries, len(BENCHMARK_QUERIES))
    queries = BENCHMARK_QUERIES[:n_queries]

    # Validate backends
    if "infdb" in backends and not check_infdb_available():
        print("ERROR: InfinityDB dependencies not installed. Install with: pip install neural-memory[pro]")
        sys.exit(1)

    print("NeuralMemory Backend Comparison Benchmark")
    print(f"Backends: {', '.join(b.upper() for b in backends)}")
    print(f"Scales: {', '.join(f'{s:,}' for s in scales)} neurons")
    print(f"Queries: {n_queries} x {args.runs} runs each")
    print(f"Output: {args.output}")

    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        python_version=sys.version.split()[0],
        platform=sys.platform,
    )

    with tempfile.TemporaryDirectory(prefix="nm_bench_", ignore_cleanup_errors=True) as tmp_dir:
        for scale in scales:
            for backend in backends:
                result = await run_scale(backend, scale, queries, args.runs, tmp_dir)
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
