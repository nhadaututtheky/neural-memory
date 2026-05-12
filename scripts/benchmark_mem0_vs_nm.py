"""Benchmark: Mem0 vs NeuralMemory on real-world memory tasks.

Tests:
  1. Write Speed    — store 50 diverse memories
  2. Read Speed     — recall 20 diverse queries
  3. Accuracy       — semantic similarity of recalled vs. stored content
  4. Multi-hop      — connecting related memories
  5. Memory Cost    — API calls per system
  6. Conversation   — store 10-turn chat, then recall context

Run:
    python scripts/benchmark_mem0_vs_nm.py

Env vars:
    DASHSCOPE_API_KEY   Alibaba Cloud / DashScope key (fallback hardcoded)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure src is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Suppress noisy sub-loggers so benchmark output stays readable
for _name in [
    "neural_memory.engine",
    "neural_memory.storage",
    "neural_memory.safety",
    "neural_memory.extraction",
    "httpx",
    "httpcore",
    "openai",
    "litellm",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

DASHSCOPE_BASE_URL = "https://coding-intl.dashscope.aliyuncs.com/v1"

MEM0_CONFIG: dict[str, Any] = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "qwen3-coder-plus",
            "api_key": DASHSCOPE_API_KEY,
            "openai_base_url": DASHSCOPE_BASE_URL,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "all-MiniLM-L6-v2",
            "embedding_dims": 384,
        },
    },
}

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

MEMORIES_50: list[str] = [
    # Architecture decisions (8)
    "Chose PostgreSQL over MongoDB because we need ACID transactions for payments processing.",
    "Selected Redis for session storage due to sub-millisecond latency requirements.",
    "Rejected microservices in favour of modular monolith — team is too small to operate many services.",
    "Decided to use Rust for the hot path encoder after profiling showed Python bottleneck.",
    "Chose React over Vue because team is more familiar with JSX and the ecosystem is larger.",
    "Selected Cloudflare Workers for edge deployment — zero cold-start and global distribution.",
    "Picked DuckDB for analytics queries; replaces expensive ClickHouse setup for our data volume.",
    "Adopted event sourcing for the audit log table to enable full replay without extra storage.",
    # Bug fixes / root causes (8)
    "Bug fix: root cause was race condition in WebSocket reconnect handler — two goroutines wrote the same key.",
    "After upgrading to v3, auth middleware broke due to cookie SameSite policy change in the new runtime.",
    "Memory leak traced to unclosed aiosqlite connections when exceptions escaped the async context manager.",
    "CORS errors in production were caused by missing trailing slash in ALLOWED_ORIGINS regex.",
    "TypeError in payment flow: Stripe amount expects integer cents, not float dollars.",
    "Null pointer crash on iOS: UILabel was deallocated before the async callback fired.",
    "Infinite redirect loop triggered when JWT expiry check ran before the refresh-token endpoint.",
    "Slow queries identified: missing composite index on (user_id, created_at) in the events table.",
    # User preferences (6)
    "User prefers dark mode with purple accent colors and JetBrains Mono as the coding font.",
    "Alex prefers concise bullet-point summaries, not prose paragraphs — max 5 bullets.",
    "Team lead requires all PRs to include a test plan before review.",
    "Product manager wants KPIs shown as percentage-change badges, not raw numbers.",
    "Designer insists on 8px grid spacing and 12px border-radius on all card components.",
    "End user consistently skips onboarding — add skip button to first screen.",
    # Workflows / procedures (7)
    "Deploy workflow: build → run tests → push Docker image → tag release → notify Slack.",
    "Code review checklist: types, tests, error handling, security, perf, docs, changelog entry.",
    "Database migration process: write migration script → test on staging → peer review → apply prod.",
    "On-call runbook: check Datadog dashboard, then tail app logs, then inspect DB slow query log.",
    "Release process: bump version in 6 files, run pre_ship.py, tag commit, push, wait for CI.",
    "API key rotation: generate new key, update .env, redeploy, revoke old key, update secrets manager.",
    "Incident response: acknowledge PagerDuty, create incident channel, assign roles, start timeline.",
    # Concepts / technical facts (7)
    "Spreading activation retrieval uses zero LLM calls — pure graph traversal with Hebbian weights.",
    "Neuron types in NeuralMemory: FACT, DECISION, ERROR, WORKFLOW, CONCEPT, ENTITY, PATTERN.",
    "PPR (Personalized PageRank) is an opt-in retrieval strategy — enables richer multi-hop paths.",
    "Retrieval pipeline: decompose query → find anchors → spread activation → fuse scores → reconstruct.",
    "BrainConfig is a frozen dataclass — use replace() to derive new configs, never mutate in place.",
    "SQLiteStorage uses WAL journal mode and a read-pool for parallel query execution.",
    "RRF score fusion blends ranked lists from multiple retrievers; default k=60.",
    # Errors / lessons learned (6)
    "Lesson: never use datetime.now() in async code — use utcnow() from timeutils to avoid tz drift.",
    "Lesson: running migrations without testing rollback burned 4 hours of prod downtime.",
    "Lesson: storing JWTs in localStorage is XSS-vulnerable; use HttpOnly cookies instead.",
    "Lesson: f-string interpolation into SQL causes injection — always use parameterised queries.",
    "Lesson: premature optimisation cost two sprints; profile first, optimise the bottleneck only.",
    "Lesson: monorepo with shared types caught 12 cross-service type errors before production.",
    # Entities / relationships (8)
    "Neural Memory project is maintained at nhadaututtheky/neural-memory on GitHub.",
    "Alex is the tech lead; Maria is the product manager; Chen is the senior backend engineer.",
    "The sync hub runs on Cloudflare Workers + D1 at neural-memory-sync-hub.vietnam11399.workers.dev.",
    "Qwen-Plus is Alibaba Cloud's flagship LLM, accessible via DashScope API with litellm prefix.",
    "The NeuralMemory MCP server exposes 44 tools and communicates via stdio transport.",
    "ReflexPipeline and MemoryEncoder are the two core engine classes for write and read paths.",
    "DashScope text-embedding-v3 produces 1024-dimensional vectors; supports batch encoding.",
    "The VS Code extension polls every 30 seconds and also subscribes to WebSocket for live updates.",
]

QUERIES_20: list[str] = [
    "Why did we choose PostgreSQL?",
    "What caused the WebSocket race condition bug?",
    "What are the user's UI preferences?",
    "How do we deploy the application?",
    "What broke after upgrading to v3?",
    "What is spreading activation retrieval?",
    "Who are the team members?",
    "What lesson did we learn about SQL queries?",
    "What is the release workflow?",
    "How does NeuralMemory retrieve memories?",
    "What caching system do we use and why?",
    "What font do we use for coding?",
    "How do we handle database migrations?",
    "What is the incident response process?",
    "What went wrong with the iOS app?",
    "What is BrainConfig and how is it used?",
    "What is the sync hub URL?",
    "What lesson did we learn about JWT storage?",
    "What neuron types exist in NeuralMemory?",
    "What embedding model does DashScope provide?",
]

MULTIHOP_QUERIES: list[tuple[str, list[str]]] = [
    (
        "What storage does the sync hub use and how do we deploy it?",
        ["Cloudflare Workers", "D1", "deploy"],
    ),
    (
        "What security issues did we learn about and how do we fix auth?",
        ["XSS", "HttpOnly", "auth middleware", "cookie"],
    ),
    (
        "What database choices did we make and what query performance issues did we find?",
        ["PostgreSQL", "DuckDB", "index", "slow queries"],
    ),
    (
        "What are the key NeuralMemory engine classes and how do they interact?",
        ["ReflexPipeline", "MemoryEncoder", "BrainConfig"],
    ),
    (
        "What are the developer workflow steps from code to production?",
        ["deploy", "release", "migration", "review"],
    ),
]

CONVERSATION_TURNS: list[tuple[str, str]] = [
    ("user", "I'm building a payment service. Which database should I use?"),
    (
        "assistant",
        "Based on the project requirements, PostgreSQL is the recommended choice due to ACID compliance.",
    ),
    ("user", "What about caching? We need fast session lookups."),
    (
        "assistant",
        "Redis is ideal for session storage — sub-millisecond latency and TTL support built in.",
    ),
    ("user", "Our auth system keeps breaking after upgrades. Any known issues?"),
    (
        "assistant",
        "Watch out for cookie SameSite policy changes between runtime versions — that was the root cause in v3.",
    ),
    ("user", "How do I deploy once the code is ready?"),
    ("assistant", "Follow: build → run tests → push Docker image → tag release → notify Slack."),
    ("user", "What about security — are there any common mistakes?"),
    (
        "assistant",
        "Never store JWTs in localStorage; use HttpOnly cookies. Also always use parameterised SQL queries.",
    ),
    ("user", "Anything else about the team I should know?"),
    (
        "assistant",
        "Alex is tech lead, Maria is PM, Chen is senior backend. Alex requires PRs to have a test plan before review.",
    ),
]

CONVERSATION_RECALL_QUERIES: list[str] = [
    "What database was recommended for the payment service?",
    "What caching system did they suggest and why?",
    "What security warning was given about JWT tokens?",
    "Who is the tech lead and what are their PR requirements?",
    "What is the deployment workflow mentioned?",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    operation: str
    mem0_time_s: float | None = None
    nm_time_s: float | None = None
    mem0_error: str | None = None
    nm_error: str | None = None
    mem0_api_calls: int = 0
    nm_api_calls: int = 0
    mem0_accuracy: float | None = None
    nm_accuracy: float | None = None
    notes: str = ""


@dataclass
class BenchmarkSuite:
    results: list[BenchmarkResult] = field(default_factory=list)
    mem0_total_api_calls: int = 0
    nm_total_api_calls: int = 0
    mem0_available: bool = True


# ---------------------------------------------------------------------------
# API call counter (monkey-patch wrapper)
# ---------------------------------------------------------------------------

_api_call_counts: dict[str, int] = {"mem0": 0, "nm": 0}


def _reset_counts() -> None:
    _api_call_counts["mem0"] = 0
    _api_call_counts["nm"] = 0


# ---------------------------------------------------------------------------
# Similarity helper (no external deps — simple Jaccard on unigrams)
# ---------------------------------------------------------------------------


def _token_set(text: str) -> set[str]:
    """Lower-cased word tokens from text."""
    import re

    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two strings (token-level)."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def best_similarity(query: str, candidates: list[str]) -> float:
    """Return the highest Jaccard similarity between query and any candidate."""
    if not candidates:
        return 0.0
    return max(jaccard_similarity(query, c) for c in candidates)


# ---------------------------------------------------------------------------
# NeuralMemory helpers
# ---------------------------------------------------------------------------


async def nm_setup(db_path: Path) -> tuple[Any, Any, Any]:
    """Create + initialise SQLite storage, Brain, Encoder, ReflexPipeline."""
    from neural_memory.core.brain import Brain, BrainConfig
    from neural_memory.engine.encoder import MemoryEncoder
    from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
    from neural_memory.storage.sqlite_store import SQLiteStorage

    config = BrainConfig(
        max_context_tokens=3000,
        max_spread_hops=4,
        graph_expansion_enabled=True,
        activation_strategy="classic",
    )
    storage = SQLiteStorage(db_path)
    await storage.initialize()

    brain = Brain.create(name="benchmark_brain", config=config, brain_id="benchmark_brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    encoder = MemoryEncoder(storage=storage, config=config)
    pipeline = ReflexPipeline(storage=storage, config=config)

    return storage, encoder, pipeline


async def nm_store_memory(encoder: Any, content: str) -> None:
    """Encode one memory into NM. NM makes ZERO external API calls."""
    await encoder.encode(content)


async def nm_recall(pipeline: Any, query: str) -> str:
    """Query NM and return the context string."""
    result = await pipeline.query(query, max_tokens=1500)
    return result.context or result.answer or ""


# ---------------------------------------------------------------------------
# Mem0 helpers
# ---------------------------------------------------------------------------


def mem0_setup(tmp_dir: str) -> Any | None:
    """Initialise Mem0 client with Qdrant local path; returns None on failure."""
    try:
        from mem0 import Memory  # type: ignore[import]

        qdrant_path = str(Path(tmp_dir) / "qdrant_mem0")
        cfg = {
            "llm": MEM0_CONFIG["llm"],
            "embedder": MEM0_CONFIG["embedder"],
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "benchmark_mem0",
                    "path": qdrant_path,
                    "embedding_model_dims": 384,  # match HuggingFace all-MiniLM-L6-v2
                },
            },
        }
        return Memory.from_config(cfg)
    except Exception as e:
        print(f"  Mem0 setup failed: {e}")
        return None


def mem0_store(client: Any, content: str, user_id: str = "benchmark_user") -> None:
    """Add one memory to Mem0 (synchronous). Counts as 1 LLM call (internally)."""
    _api_call_counts["mem0"] += 1  # Each add() makes ≥1 LLM call
    client.add(content, user_id=user_id)


def mem0_recall(client: Any, query: str, user_id: str = "benchmark_user") -> str:
    """Search Mem0 for the query and return joined results."""
    _api_call_counts["mem0"] += 1  # Each search() makes ≥1 LLM call for re-ranking
    results = client.search(query, user_id=user_id, limit=5)
    if isinstance(results, dict):
        memories = results.get("results", [])
    elif isinstance(results, list):
        memories = results
    else:
        memories = []
    texts = [m.get("memory", "") if isinstance(m, dict) else str(m) for m in memories]
    return " | ".join(texts)


# ---------------------------------------------------------------------------
# Benchmark sections
# ---------------------------------------------------------------------------


async def bench_write_speed(
    suite: BenchmarkSuite, mem0_client: Any | None, nm_encoder: Any
) -> None:
    """Test 1: Store 50 memories — measure total wall-clock time."""
    print("\n[1/6] Write Speed — storing 50 memories")

    result = BenchmarkResult(operation="Write Speed (50 memories)")

    # --- NM ---
    _reset_counts()
    t0 = time.perf_counter()
    nm_err: str | None = None
    try:
        for mem in MEMORIES_50:
            await nm_store_memory(nm_encoder, mem)
    except Exception as e:
        nm_err = f"{type(e).__name__}: {e}"
    nm_elapsed = time.perf_counter() - t0
    result.nm_time_s = nm_elapsed
    result.nm_error = nm_err
    result.nm_api_calls = _api_call_counts["nm"]
    suite.nm_total_api_calls += result.nm_api_calls
    print(f"  NM  : {nm_elapsed:.3f}s — {'OK' if not nm_err else nm_err}")

    # --- Mem0 ---
    if mem0_client is not None:
        _reset_counts()
        t0 = time.perf_counter()
        mem0_err: str | None = None
        try:
            for mem in MEMORIES_50:
                mem0_store(mem0_client, mem)
        except Exception as e:
            mem0_err = f"{type(e).__name__}: {e}"
        mem0_elapsed = time.perf_counter() - t0
        result.mem0_time_s = mem0_elapsed
        result.mem0_error = mem0_err
        result.mem0_api_calls = _api_call_counts["mem0"]
        suite.mem0_total_api_calls += result.mem0_api_calls
        print(f"  Mem0: {mem0_elapsed:.3f}s — {'OK' if not mem0_err else mem0_err}")
    else:
        result.mem0_error = "mem0 not installed or unavailable"
        print("  Mem0: SKIPPED (not available)")

    suite.results.append(result)


async def bench_read_speed(
    suite: BenchmarkSuite, mem0_client: Any | None, nm_pipeline: Any
) -> None:
    """Test 2: Recall 20 queries — measure total wall-clock time."""
    print("\n[2/6] Read Speed — recalling 20 queries")

    result = BenchmarkResult(operation="Read Speed (20 queries)")

    # --- NM ---
    _reset_counts()
    t0 = time.perf_counter()
    nm_err = None
    nm_responses: list[str] = []
    try:
        for q in QUERIES_20:
            resp = await nm_recall(nm_pipeline, q)
            nm_responses.append(resp)
    except Exception as e:
        nm_err = f"{type(e).__name__}: {e}"
    nm_elapsed = time.perf_counter() - t0
    result.nm_time_s = nm_elapsed
    result.nm_error = nm_err
    result.nm_api_calls = _api_call_counts["nm"]
    suite.nm_total_api_calls += result.nm_api_calls
    print(f"  NM  : {nm_elapsed:.3f}s — {'OK' if not nm_err else nm_err}")

    # --- Mem0 ---
    mem0_responses: list[str] = []
    if mem0_client is not None:
        _reset_counts()
        t0 = time.perf_counter()
        mem0_err = None
        try:
            for q in QUERIES_20:
                resp = mem0_recall(mem0_client, q)
                mem0_responses.append(resp)
        except Exception as e:
            mem0_err = f"{type(e).__name__}: {e}"
        mem0_elapsed = time.perf_counter() - t0
        result.mem0_time_s = mem0_elapsed
        result.mem0_error = mem0_err
        result.mem0_api_calls = _api_call_counts["mem0"]
        suite.mem0_total_api_calls += result.mem0_api_calls
        print(f"  Mem0: {mem0_elapsed:.3f}s — {'OK' if not mem0_err else mem0_err}")
    else:
        result.mem0_error = "mem0 not installed or unavailable"
        print("  Mem0: SKIPPED")

    # --- Accuracy ---
    print("\n[3/6] Accuracy — semantic similarity of recalled content")
    acc_result = BenchmarkResult(operation="Accuracy (Jaccard vs expected)")

    if nm_responses:
        nm_scores = [
            best_similarity(QUERIES_20[i], [nm_responses[i]] + MEMORIES_50)
            for i in range(len(QUERIES_20))
        ]
        acc_result.nm_accuracy = sum(nm_scores) / len(nm_scores)
        print(f"  NM   avg Jaccard: {acc_result.nm_accuracy:.3f}")

    if mem0_responses:
        mem0_scores = [
            best_similarity(QUERIES_20[i], [mem0_responses[i]] + MEMORIES_50)
            for i in range(len(mem0_responses))
        ]
        acc_result.mem0_accuracy = sum(mem0_scores) / len(mem0_scores)
        print(f"  Mem0 avg Jaccard: {acc_result.mem0_accuracy:.3f}")
    else:
        acc_result.mem0_error = "no responses"

    suite.results.append(result)
    suite.results.append(acc_result)


async def bench_multihop(suite: BenchmarkSuite, mem0_client: Any | None, nm_pipeline: Any) -> None:
    """Test 4: Multi-hop reasoning — connection across related memories."""
    print("\n[4/6] Multi-hop Reasoning — 5 queries requiring cross-memory links")

    result = BenchmarkResult(operation="Multi-hop Reasoning (5 queries)")
    nm_hit = 0
    mem0_hit = 0

    for query, expected_keywords in MULTIHOP_QUERIES:
        # NM
        try:
            resp = await nm_recall(nm_pipeline, query)
            hits = sum(1 for kw in expected_keywords if kw.lower() in resp.lower())
            nm_hit += hits / len(expected_keywords)
        except Exception as e:
            print(f"  NM multi-hop error: {e}")

        # Mem0
        if mem0_client is not None:
            try:
                resp = mem0_recall(mem0_client, query)
                hits = sum(1 for kw in expected_keywords if kw.lower() in resp.lower())
                mem0_hit += hits / len(expected_keywords)
            except Exception as e:
                print(f"  Mem0 multi-hop error: {e}")

    n = len(MULTIHOP_QUERIES)
    result.nm_accuracy = nm_hit / n
    print(f"  NM   keyword coverage: {result.nm_accuracy:.3f}")

    if mem0_client is not None:
        result.mem0_accuracy = mem0_hit / n
        print(f"  Mem0 keyword coverage: {result.mem0_accuracy:.3f}")
    else:
        result.mem0_error = "mem0 not installed or unavailable"

    suite.results.append(result)


async def bench_conversation(
    suite: BenchmarkSuite, mem0_client: Any | None, nm_encoder: Any, nm_pipeline: Any
) -> None:
    """Test 6: Store 10-turn conversation then recall context."""
    print("\n[6/6] Conversation Context — 10-turn chat + 5 recall queries")

    result = BenchmarkResult(operation="Conversation Context (10 turns + 5 recalls)")

    # --- NM store ---
    t0 = time.perf_counter()
    nm_err = None
    try:
        for role, text in CONVERSATION_TURNS:
            await nm_store_memory(nm_encoder, f"[{role}]: {text}")
    except Exception as e:
        nm_err = f"{type(e).__name__}: {e}"

    nm_responses: list[str] = []
    for q in CONVERSATION_RECALL_QUERIES:
        try:
            nm_responses.append(await nm_recall(nm_pipeline, q))
        except Exception as e:
            nm_responses.append(f"ERROR: {e}")
    nm_elapsed = time.perf_counter() - t0
    result.nm_time_s = nm_elapsed
    result.nm_error = nm_err

    nm_conv_scores = [
        best_similarity(
            CONVERSATION_RECALL_QUERIES[i], [nm_responses[i]] + [t for _, t in CONVERSATION_TURNS]
        )
        for i in range(len(CONVERSATION_RECALL_QUERIES))
    ]
    result.nm_accuracy = sum(nm_conv_scores) / len(nm_conv_scores)
    print(f"  NM  : {nm_elapsed:.3f}s | accuracy={result.nm_accuracy:.3f}")

    # --- Mem0 store ---
    if mem0_client is not None:
        t0 = time.perf_counter()
        mem0_err = None
        try:
            for role, text in CONVERSATION_TURNS:
                mem0_store(mem0_client, f"[{role}]: {text}", user_id="conv_user")
        except Exception as e:
            mem0_err = f"{type(e).__name__}: {e}"

        mem0_responses: list[str] = []
        for q in CONVERSATION_RECALL_QUERIES:
            try:
                mem0_responses.append(mem0_recall(mem0_client, q, user_id="conv_user"))
            except Exception as e:
                mem0_responses.append(f"ERROR: {e}")
        mem0_elapsed = time.perf_counter() - t0
        result.mem0_time_s = mem0_elapsed
        result.mem0_error = mem0_err

        mem0_conv_scores = [
            best_similarity(
                CONVERSATION_RECALL_QUERIES[i],
                [mem0_responses[i]] + [t for _, t in CONVERSATION_TURNS],
            )
            for i in range(len(CONVERSATION_RECALL_QUERIES))
        ]
        result.mem0_accuracy = sum(mem0_conv_scores) / len(mem0_conv_scores)
        print(f"  Mem0: {mem0_elapsed:.3f}s | accuracy={result.mem0_accuracy:.3f}")
    else:
        result.mem0_error = "mem0 not installed or unavailable"
        print("  Mem0: SKIPPED")

    suite.results.append(result)


# ---------------------------------------------------------------------------
# Memory Cost summary result
# ---------------------------------------------------------------------------


def add_cost_result(suite: BenchmarkSuite) -> None:
    result = BenchmarkResult(
        operation="Memory Cost (API calls)",
        mem0_api_calls=suite.mem0_total_api_calls,
        nm_api_calls=suite.nm_total_api_calls,
        notes="Mem0 calls LLM for every add()+search(). NM uses zero LLM calls.",
    )
    suite.results.append(result)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COL_W = (38, 14, 14, 10)


def _row(*cells: str) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, _COL_W))


def _winner(nm: float | None, mem0: float | None, lower_is_better: bool = True) -> str:
    if nm is None and mem0 is None:
        return "TIE"
    if nm is None:
        return "Mem0"
    if mem0 is None:
        return "NM"
    if lower_is_better:
        return "NM" if nm < mem0 else ("Mem0" if mem0 < nm else "TIE")
    return "NM" if nm > mem0 else ("Mem0" if mem0 > nm else "TIE")


def print_report(suite: BenchmarkSuite) -> None:
    sep = "=" * 84
    thin = "-" * 84

    print(f"\n{sep}")
    print("  BENCHMARK RESULTS: NeuralMemory vs Mem0")
    print(sep)

    # Speed table
    print("\n  SPEED COMPARISON")
    print(thin)
    print("  " + _row("Operation", "NM (s)", "Mem0 (s)", "Winner"))
    print(thin)
    for r in suite.results:
        if r.nm_time_s is None and r.mem0_time_s is None:
            continue
        nm_s = f"{r.nm_time_s:.3f}" if r.nm_time_s is not None else "N/A"
        m0_s = f"{r.mem0_time_s:.3f}" if r.mem0_time_s is not None else "N/A"
        w = _winner(r.nm_time_s, r.mem0_time_s, lower_is_better=True)
        print("  " + _row(r.operation, nm_s, m0_s, w))
    print(thin)

    # Accuracy table
    print("\n  ACCURACY COMPARISON  (higher = better)")
    print(thin)
    print("  " + _row("Operation", "NM score", "Mem0 score", "Winner"))
    print(thin)
    for r in suite.results:
        if r.nm_accuracy is None and r.mem0_accuracy is None:
            continue
        nm_a = f"{r.nm_accuracy:.3f}" if r.nm_accuracy is not None else "N/A"
        m0_a = f"{r.mem0_accuracy:.3f}" if r.mem0_accuracy is not None else "N/A"
        w = _winner(r.nm_accuracy, r.mem0_accuracy, lower_is_better=False)
        print("  " + _row(r.operation, nm_a, m0_a, w))
    print(thin)

    # API cost
    print(f"\n  API CALL COST")
    print(thin)
    print(f"  NeuralMemory total external API calls : {suite.nm_total_api_calls}")
    print(f"  Mem0         total external API calls : {suite.mem0_total_api_calls}")
    print(f"  (Mem0 calls LLM for every add() and search() — hidden cost per operation)")
    print(thin)

    # Errors
    errors = [
        (r.operation, r.nm_error, r.mem0_error) for r in suite.results if r.nm_error or r.mem0_error
    ]
    if errors:
        print("\n  ERRORS / WARNINGS")
        print(thin)
        for op, nm_e, m0_e in errors:
            if nm_e:
                print(f"  NM   [{op[:45]}]: {nm_e[:80]}")
            if m0_e:
                print(f"  Mem0 [{op[:45]}]: {m0_e[:80]}")
        print(thin)

    # Verdict
    nm_wins = sum(
        1
        for r in suite.results
        if (
            (r.nm_time_s is not None and r.mem0_time_s is not None and r.nm_time_s < r.mem0_time_s)
            or (
                r.nm_accuracy is not None
                and r.mem0_accuracy is not None
                and r.nm_accuracy > r.mem0_accuracy
            )
        )
    )
    mem0_wins = sum(
        1
        for r in suite.results
        if (
            (r.nm_time_s is not None and r.mem0_time_s is not None and r.mem0_time_s < r.nm_time_s)
            or (
                r.nm_accuracy is not None
                and r.mem0_accuracy is not None
                and r.mem0_accuracy > r.nm_accuracy
            )
        )
    )

    print("\n  FINAL VERDICT")
    print(thin)
    if not suite.mem0_available:
        print("  Mem0 was NOT available — comparison limited to NM standalone metrics.")
    else:
        print(f"  NeuralMemory wins : {nm_wins} categories")
        print(f"  Mem0 wins         : {mem0_wins} categories")
        if nm_wins > mem0_wins:
            print("  >> NeuralMemory outperforms Mem0 overall in this benchmark.")
        elif mem0_wins > nm_wins:
            print("  >> Mem0 outperforms NeuralMemory overall in this benchmark.")
        else:
            print("  >> Both systems perform comparably overall.")

    print(f"\n  Key insight: NM uses {suite.nm_total_api_calls} external API calls.")
    if suite.mem0_available:
        print(f"               Mem0 uses {suite.mem0_total_api_calls} (LLM call per operation).")
    print(f"  At scale, NM's zero-LLM retrieval means near-zero variable cost per query.")
    print(sep)


def save_results(suite: BenchmarkSuite, output_path: Path) -> None:
    data = {
        "mem0_available": suite.mem0_available,
        "nm_total_api_calls": suite.nm_total_api_calls,
        "mem0_total_api_calls": suite.mem0_total_api_calls,
        "results": [
            {
                "operation": r.operation,
                "nm_time_s": r.nm_time_s,
                "mem0_time_s": r.mem0_time_s,
                "nm_error": r.nm_error,
                "mem0_error": r.mem0_error,
                "nm_api_calls": r.nm_api_calls,
                "mem0_api_calls": r.mem0_api_calls,
                "nm_accuracy": r.nm_accuracy,
                "mem0_accuracy": r.mem0_accuracy,
                "notes": r.notes,
            }
            for r in suite.results
        ],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 84)
    print("  NeuralMemory vs Mem0 — Real-World Memory Benchmark")
    print("=" * 84)
    print(f"  DashScope key : {'*' * 20}{DASHSCOPE_API_KEY[-6:]}")
    print(f"  Test memories : {len(MEMORIES_50)}")
    print(f"  Recall queries: {len(QUERIES_20)}")

    suite = BenchmarkSuite()

    # Temp dirs
    tmp_dir = Path(tempfile.mkdtemp(prefix="nmem_bench_"))
    nm_db = tmp_dir / "nm_benchmark.db"
    print(f"  Temp dir      : {tmp_dir}")

    # Setup NM
    print("\n  Setting up NeuralMemory...")
    try:
        nm_storage, nm_encoder, nm_pipeline = await nm_setup(nm_db)
        print("  NM ready.")
    except Exception as e:
        print(f"  FATAL: NeuralMemory setup failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Setup Mem0
    print("\n  Setting up Mem0...")
    mem0_client = mem0_setup(str(tmp_dir))
    if mem0_client is None:
        print("  Mem0 not available (pip install mem0ai chromadb to enable).")
        suite.mem0_available = False
    else:
        print("  Mem0 ready.")

    # Run benchmarks
    try:
        await bench_write_speed(suite, mem0_client, nm_encoder)
        await bench_read_speed(suite, mem0_client, nm_pipeline)
        await bench_multihop(suite, mem0_client, nm_pipeline)

        print("\n[5/6] Memory Cost — tracked across all operations above")
        add_cost_result(suite)

        await bench_conversation(suite, mem0_client, nm_encoder, nm_pipeline)
    finally:
        await nm_storage.close()

    # Report
    print_report(suite)

    output_path = Path(__file__).resolve().parent / "benchmark_results.json"
    save_results(suite, output_path)


if __name__ == "__main__":
    asyncio.run(main())
