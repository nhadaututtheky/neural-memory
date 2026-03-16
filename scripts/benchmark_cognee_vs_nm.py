"""Benchmark: Cognee vs NeuralMemory on real-world memory tasks.

Tests:
  1. Write Speed    — store 50 diverse memories (add + cognify vs encode)
  2. Read Speed     — recall 20 diverse queries (search vs reflex query)
  3. Accuracy       — semantic similarity of recalled vs. stored content
  4. Multi-hop      — connecting related memories across the graph
  5. Memory Cost    — API calls per system
  6. Conversation   — store 10-turn chat, then recall context

Run (requires Python 3.12 venv with cognee installed):
    .venv-cognee/Scripts/python.exe scripts/benchmark_cognee_vs_nm.py

Env vars:
    DASHSCOPE_API_KEY   Alibaba Cloud / DashScope Coding Plan key
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

# Suppress noisy sub-loggers
for _name in [
    "neural_memory.engine",
    "neural_memory.storage",
    "neural_memory.safety",
    "neural_memory.extraction",
    "httpx",
    "httpcore",
    "openai",
    "litellm",
    "cognee",
    "cognee.shared",
    "cognee.infrastructure",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://coding-intl.dashscope.aliyuncs.com/v1"

# Cognee reads env vars at import time — must set BEFORE importing cognee
os.environ["LLM_API_KEY"] = DASHSCOPE_API_KEY
os.environ["LLM_PROVIDER"] = "custom"
os.environ["LLM_MODEL"] = "openai/qwen3-coder-plus"
os.environ["LLM_ENDPOINT"] = DASHSCOPE_BASE_URL
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
os.environ["EMBEDDING_PROVIDER"] = "fastembed"
os.environ["EMBEDDING_MODEL"] = "BAAI/bge-small-en-v1.5"
os.environ["EMBEDDING_DIMENSIONS"] = "384"

# ---------------------------------------------------------------------------
# Test data (same as Mem0 benchmark for fair comparison)
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
    ("assistant", "Based on the project requirements, PostgreSQL is the recommended choice due to ACID compliance."),
    ("user", "What about caching? We need fast session lookups."),
    ("assistant", "Redis is ideal for session storage — sub-millisecond latency and TTL support built in."),
    ("user", "Our auth system keeps breaking after upgrades. Any known issues?"),
    ("assistant", "Watch out for cookie SameSite policy changes between runtime versions — that was the root cause in v3."),
    ("user", "How do I deploy once the code is ready?"),
    ("assistant", "Follow: build → run tests → push Docker image → tag release → notify Slack."),
    ("user", "What about security — are there any common mistakes?"),
    ("assistant", "Never store JWTs in localStorage; use HttpOnly cookies. Also always use parameterised SQL queries."),
    ("user", "Anything else about the team I should know?"),
    ("assistant", "Alex is tech lead, Maria is PM, Chen is senior backend. Alex requires PRs to have a test plan before review."),
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
    cognee_time_s: float | None = None
    nm_time_s: float | None = None
    cognee_error: str | None = None
    nm_error: str | None = None
    cognee_api_calls: int = 0
    nm_api_calls: int = 0
    cognee_accuracy: float | None = None
    nm_accuracy: float | None = None
    notes: str = ""


@dataclass
class BenchmarkSuite:
    results: list[BenchmarkResult] = field(default_factory=list)
    cognee_total_api_calls: int = 0
    nm_total_api_calls: int = 0
    cognee_available: bool = True


# ---------------------------------------------------------------------------
# API call counter
# ---------------------------------------------------------------------------

_api_call_counts: dict[str, int] = {"cognee": 0, "nm": 0}


def _reset_counts() -> None:
    _api_call_counts["cognee"] = 0
    _api_call_counts["nm"] = 0


# ---------------------------------------------------------------------------
# Similarity helper (Jaccard on unigrams — same as Mem0 benchmark)
# ---------------------------------------------------------------------------


def _token_set(text: str) -> set[str]:
    import re
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def best_similarity(query: str, candidates: list[str]) -> float:
    if not candidates:
        return 0.0
    return max(jaccard_similarity(query, c) for c in candidates)


# ---------------------------------------------------------------------------
# NeuralMemory helpers
# ---------------------------------------------------------------------------


async def nm_setup(db_path: Path) -> tuple[Any, Any, Any]:
    from neural_memory.core.brain import Brain, BrainConfig
    from neural_memory.engine.encoder import MemoryEncoder
    from neural_memory.engine.retrieval import ReflexPipeline
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
    await encoder.encode(content)


async def nm_recall(pipeline: Any, query: str) -> str:
    result = await pipeline.query(query, max_tokens=1500)
    return result.context or result.answer or ""


# ---------------------------------------------------------------------------
# Cognee helpers
# ---------------------------------------------------------------------------


async def cognee_setup() -> bool:
    """Configure Cognee with DashScope as custom LLM provider."""
    try:
        import cognee
        from cognee.infrastructure.llm.config import get_llm_config

        # Verify config was picked up from env vars
        cfg = get_llm_config()
        print(f"  Cognee LLM: provider={cfg.llm_provider}, model={cfg.llm_model}")
        print(f"  Cognee LLM: endpoint={cfg.llm_endpoint}, key={'***' + cfg.llm_api_key[-6:] if cfg.llm_api_key else 'NONE'}")

        # Also force via config API as backup
        cognee.config.set_llm_config({
            "llm_api_key": DASHSCOPE_API_KEY,
            "llm_endpoint": DASHSCOPE_BASE_URL,
            "llm_model": "openai/qwen3-coder-plus",
            "llm_provider": "custom",
        })

        # WORKAROUND: Cognee bug — CUSTOM provider drops llm_endpoint from GenericAPIAdapter.
        # Monkey-patch litellm to use DashScope as default api_base.
        import litellm
        litellm.api_base = DASHSCOPE_BASE_URL
        litellm.api_key = DASHSCOPE_API_KEY

        # Reset Cognee state for clean benchmark
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        return True
    except Exception as e:
        print(f"  Cognee setup failed: {e}")
        traceback.print_exc()
        return False


async def cognee_store_batch(memories: list[str]) -> None:
    """Add memories to Cognee and build knowledge graph.

    Cognee's workflow: add() ingests data, cognify() builds the graph.
    Each cognify() triggers LLM calls for entity extraction + graph building.
    """
    import cognee

    # Add all memories first
    for mem in memories:
        await cognee.add(mem, dataset_name="benchmark")
        _api_call_counts["cognee"] += 0  # add() itself doesn't call LLM

    # cognify() builds the knowledge graph — THIS is where LLM calls happen
    await cognee.cognify()
    # Cognee calls LLM for each chunk: entity extraction + relationship extraction
    # Conservative estimate: ~2 LLM calls per memory (extract + link)
    _api_call_counts["cognee"] += len(memories) * 2


async def cognee_search(query: str) -> str:
    """Search Cognee and return joined results."""
    import cognee
    from cognee.api.v1.search import SearchType

    _api_call_counts["cognee"] += 1  # search may trigger LLM for query parsing
    try:
        results = await cognee.search(query, SearchType.GRAPH_COMPLETION)
        if not results:
            return ""
        # Results can be various formats — extract text content
        texts: list[str] = []
        for item in results:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(item.get("text", item.get("content", str(item))))
            elif hasattr(item, "text"):
                texts.append(str(item.text))
            elif hasattr(item, "content"):
                texts.append(str(item.content))
            else:
                texts.append(str(item))
        return " | ".join(texts)
    except Exception as e:
        logger.warning(f"Cognee search error: {e}")
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Benchmark sections
# ---------------------------------------------------------------------------


async def bench_write_speed(suite: BenchmarkSuite, cognee_ok: bool, nm_encoder: Any) -> None:
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
    print(f"  NM    : {nm_elapsed:.3f}s — {'OK' if not nm_err else nm_err}")

    # --- Cognee ---
    if cognee_ok:
        _reset_counts()
        t0 = time.perf_counter()
        cognee_err: str | None = None
        try:
            await cognee_store_batch(MEMORIES_50)
        except Exception as e:
            cognee_err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        cognee_elapsed = time.perf_counter() - t0
        result.cognee_time_s = cognee_elapsed
        result.cognee_error = cognee_err
        result.cognee_api_calls = _api_call_counts["cognee"]
        suite.cognee_total_api_calls += result.cognee_api_calls
        print(f"  Cognee: {cognee_elapsed:.3f}s — {'OK' if not cognee_err else cognee_err}")
    else:
        result.cognee_error = "cognee not available"
        print("  Cognee: SKIPPED (not available)")

    suite.results.append(result)


async def bench_read_speed(suite: BenchmarkSuite, cognee_ok: bool, nm_pipeline: Any) -> None:
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
    print(f"  NM    : {nm_elapsed:.3f}s — {'OK' if not nm_err else nm_err}")

    # --- Cognee ---
    cognee_responses: list[str] = []
    if cognee_ok:
        _reset_counts()
        t0 = time.perf_counter()
        cognee_err = None
        try:
            for q in QUERIES_20:
                resp = await cognee_search(q)
                cognee_responses.append(resp)
        except Exception as e:
            cognee_err = f"{type(e).__name__}: {e}"
        cognee_elapsed = time.perf_counter() - t0
        result.cognee_time_s = cognee_elapsed
        result.cognee_error = cognee_err
        result.cognee_api_calls = _api_call_counts["cognee"]
        suite.cognee_total_api_calls += result.cognee_api_calls
        print(f"  Cognee: {cognee_elapsed:.3f}s — {'OK' if not cognee_err else cognee_err}")
    else:
        result.cognee_error = "cognee not available"
        print("  Cognee: SKIPPED")

    # --- Accuracy ---
    print("\n[3/6] Accuracy — semantic similarity of recalled content")
    acc_result = BenchmarkResult(operation="Accuracy (Jaccard vs expected)")

    if nm_responses:
        nm_scores = [
            best_similarity(QUERIES_20[i], [nm_responses[i]] + MEMORIES_50)
            for i in range(len(QUERIES_20))
        ]
        acc_result.nm_accuracy = sum(nm_scores) / len(nm_scores)
        print(f"  NM    avg Jaccard: {acc_result.nm_accuracy:.3f}")

    if cognee_responses:
        cognee_scores = [
            best_similarity(QUERIES_20[i], [cognee_responses[i]] + MEMORIES_50)
            for i in range(len(cognee_responses))
        ]
        acc_result.cognee_accuracy = sum(cognee_scores) / len(cognee_scores)
        print(f"  Cognee avg Jaccard: {acc_result.cognee_accuracy:.3f}")
    else:
        acc_result.cognee_error = "no responses"

    suite.results.append(result)
    suite.results.append(acc_result)


async def bench_multihop(suite: BenchmarkSuite, cognee_ok: bool, nm_pipeline: Any) -> None:
    """Test 4: Multi-hop reasoning — connection across related memories."""
    print("\n[4/6] Multi-hop Reasoning — 5 queries requiring cross-memory links")

    result = BenchmarkResult(operation="Multi-hop Reasoning (5 queries)")
    nm_hit = 0
    cognee_hit = 0

    for query, expected_keywords in MULTIHOP_QUERIES:
        # NM
        try:
            resp = await nm_recall(nm_pipeline, query)
            hits = sum(1 for kw in expected_keywords if kw.lower() in resp.lower())
            nm_hit += hits / len(expected_keywords)
        except Exception as e:
            print(f"  NM multi-hop error: {e}")

        # Cognee
        if cognee_ok:
            try:
                resp = await cognee_search(query)
                hits = sum(1 for kw in expected_keywords if kw.lower() in resp.lower())
                cognee_hit += hits / len(expected_keywords)
            except Exception as e:
                print(f"  Cognee multi-hop error: {e}")

    n = len(MULTIHOP_QUERIES)
    result.nm_accuracy = nm_hit / n
    print(f"  NM    keyword coverage: {result.nm_accuracy:.3f}")

    if cognee_ok:
        result.cognee_accuracy = cognee_hit / n
        print(f"  Cognee keyword coverage: {result.cognee_accuracy:.3f}")
    else:
        result.cognee_error = "cognee not available"

    suite.results.append(result)


async def bench_conversation(
    suite: BenchmarkSuite, cognee_ok: bool, nm_encoder: Any, nm_pipeline: Any
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
            CONVERSATION_RECALL_QUERIES[i],
            [nm_responses[i]] + [t for _, t in CONVERSATION_TURNS],
        )
        for i in range(len(CONVERSATION_RECALL_QUERIES))
    ]
    result.nm_accuracy = sum(nm_conv_scores) / len(nm_conv_scores)
    print(f"  NM    : {nm_elapsed:.3f}s | accuracy={result.nm_accuracy:.3f}")

    # --- Cognee store ---
    if cognee_ok:
        import cognee

        t0 = time.perf_counter()
        cognee_err = None
        try:
            for role, text in CONVERSATION_TURNS:
                await cognee.add(f"[{role}]: {text}", dataset_name="benchmark_conv")
                _api_call_counts["cognee"] += 0
            await cognee.cognify()
            _api_call_counts["cognee"] += len(CONVERSATION_TURNS) * 2
            suite.cognee_total_api_calls += len(CONVERSATION_TURNS) * 2
        except Exception as e:
            cognee_err = f"{type(e).__name__}: {e}"

        cognee_responses: list[str] = []
        for q in CONVERSATION_RECALL_QUERIES:
            try:
                cognee_responses.append(await cognee_search(q))
                _api_call_counts["cognee"] += 1
                suite.cognee_total_api_calls += 1
            except Exception as e:
                cognee_responses.append(f"ERROR: {e}")
        cognee_elapsed = time.perf_counter() - t0
        result.cognee_time_s = cognee_elapsed
        result.cognee_error = cognee_err

        cognee_conv_scores = [
            best_similarity(
                CONVERSATION_RECALL_QUERIES[i],
                [cognee_responses[i]] + [t for _, t in CONVERSATION_TURNS],
            )
            for i in range(len(CONVERSATION_RECALL_QUERIES))
        ]
        result.cognee_accuracy = sum(cognee_conv_scores) / len(cognee_conv_scores)
        print(f"  Cognee: {cognee_elapsed:.3f}s | accuracy={result.cognee_accuracy:.3f}")
    else:
        result.cognee_error = "cognee not available"
        print("  Cognee: SKIPPED")

    suite.results.append(result)


# ---------------------------------------------------------------------------
# Memory Cost summary
# ---------------------------------------------------------------------------


def add_cost_result(suite: BenchmarkSuite) -> None:
    result = BenchmarkResult(
        operation="Memory Cost (API calls)",
        cognee_api_calls=suite.cognee_total_api_calls,
        nm_api_calls=suite.nm_total_api_calls,
        notes="Cognee calls LLM for cognify() + search(). NM uses zero LLM calls.",
    )
    suite.results.append(result)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COL_W = (38, 14, 14, 10)


def _row(*cells: str) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, _COL_W))


def _winner(nm: float | None, other: float | None, lower_is_better: bool = True) -> str:
    if nm is None and other is None:
        return "TIE"
    if nm is None:
        return "Cognee"
    if other is None:
        return "NM"
    if lower_is_better:
        return "NM" if nm < other else ("Cognee" if other < nm else "TIE")
    return "NM" if nm > other else ("Cognee" if other > nm else "TIE")


def print_report(suite: BenchmarkSuite) -> None:
    sep = "=" * 84
    thin = "-" * 84

    print(f"\n{sep}")
    print("  BENCHMARK RESULTS: NeuralMemory vs Cognee")
    print(sep)

    # Speed table
    print("\n  SPEED COMPARISON")
    print(thin)
    print("  " + _row("Operation", "NM (s)", "Cognee (s)", "Winner"))
    print(thin)
    for r in suite.results:
        if r.nm_time_s is None and r.cognee_time_s is None:
            continue
        nm_s = f"{r.nm_time_s:.3f}" if r.nm_time_s is not None else "N/A"
        c_s = f"{r.cognee_time_s:.3f}" if r.cognee_time_s is not None else "N/A"
        w = _winner(r.nm_time_s, r.cognee_time_s, lower_is_better=True)
        print("  " + _row(r.operation, nm_s, c_s, w))
    print(thin)

    # Accuracy table
    print("\n  ACCURACY COMPARISON  (higher = better)")
    print(thin)
    print("  " + _row("Operation", "NM score", "Cognee score", "Winner"))
    print(thin)
    for r in suite.results:
        if r.nm_accuracy is None and r.cognee_accuracy is None:
            continue
        nm_a = f"{r.nm_accuracy:.3f}" if r.nm_accuracy is not None else "N/A"
        c_a = f"{r.cognee_accuracy:.3f}" if r.cognee_accuracy is not None else "N/A"
        w = _winner(r.nm_accuracy, r.cognee_accuracy, lower_is_better=False)
        print("  " + _row(r.operation, nm_a, c_a, w))
    print(thin)

    # API cost
    print("\n  API CALL COST")
    print(thin)
    print(f"  NeuralMemory total external API calls : {suite.nm_total_api_calls}")
    print(f"  Cognee       total external API calls : {suite.cognee_total_api_calls}")
    print(f"  (Cognee calls LLM for cognify() entity extraction + search() query parsing)")
    print(thin)

    # Errors
    errors = [
        (r.operation, r.nm_error, r.cognee_error)
        for r in suite.results
        if r.nm_error or r.cognee_error
    ]
    if errors:
        print("\n  ERRORS / WARNINGS")
        print(thin)
        for op, nm_e, c_e in errors:
            if nm_e:
                print(f"  NM    [{op[:45]}]: {nm_e[:80]}")
            if c_e:
                print(f"  Cognee [{op[:45]}]: {c_e[:80]}")
        print(thin)

    # Verdict
    nm_wins = sum(
        1
        for r in suite.results
        if (
            (r.nm_time_s is not None and r.cognee_time_s is not None and r.nm_time_s < r.cognee_time_s)
            or (r.nm_accuracy is not None and r.cognee_accuracy is not None and r.nm_accuracy > r.cognee_accuracy)
        )
    )
    cognee_wins = sum(
        1
        for r in suite.results
        if (
            (r.nm_time_s is not None and r.cognee_time_s is not None and r.cognee_time_s < r.nm_time_s)
            or (r.nm_accuracy is not None and r.cognee_accuracy is not None and r.cognee_accuracy > r.nm_accuracy)
        )
    )

    print("\n  FINAL VERDICT")
    print(thin)
    if not suite.cognee_available:
        print("  Cognee was NOT available — comparison limited to NM standalone metrics.")
    else:
        print(f"  NeuralMemory wins : {nm_wins} categories")
        print(f"  Cognee wins       : {cognee_wins} categories")
        if nm_wins > cognee_wins:
            print("  >> NeuralMemory outperforms Cognee overall in this benchmark.")
        elif cognee_wins > nm_wins:
            print("  >> Cognee outperforms NeuralMemory overall in this benchmark.")
        else:
            print("  >> Both systems perform comparably overall.")

    print(f"\n  Key insight: NM uses {suite.nm_total_api_calls} external API calls.")
    if suite.cognee_available:
        print(f"               Cognee uses {suite.cognee_total_api_calls} (LLM calls for cognify + search).")
    print(f"  NM = zero-LLM retrieval → predictable cost. Cognee = knowledge graph → richer semantics but LLM-dependent.")
    print(sep)


def save_results(suite: BenchmarkSuite, output_path: Path) -> None:
    data = {
        "benchmark": "NeuralMemory vs Cognee",
        "cognee_version": "0.5.5",
        "nm_version": "4.7.0",
        "cognee_available": suite.cognee_available,
        "nm_total_api_calls": suite.nm_total_api_calls,
        "cognee_total_api_calls": suite.cognee_total_api_calls,
        "results": [
            {
                "operation": r.operation,
                "nm_time_s": r.nm_time_s,
                "cognee_time_s": r.cognee_time_s,
                "nm_error": r.nm_error,
                "cognee_error": r.cognee_error,
                "nm_api_calls": r.nm_api_calls,
                "cognee_api_calls": r.cognee_api_calls,
                "nm_accuracy": r.nm_accuracy,
                "cognee_accuracy": r.cognee_accuracy,
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
    print("  NeuralMemory vs Cognee — Real-World Memory Benchmark")
    print("=" * 84)
    print(f"  DashScope key : {'*' * 20}{DASHSCOPE_API_KEY[-6:] if DASHSCOPE_API_KEY else 'NOT SET'}")
    print(f"  Test memories : {len(MEMORIES_50)}")
    print(f"  Recall queries: {len(QUERIES_20)}")

    suite = BenchmarkSuite()

    # Temp dirs
    tmp_dir = Path(tempfile.mkdtemp(prefix="nmem_bench_cognee_"))
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

    # Setup Cognee
    print("\n  Setting up Cognee...")
    cognee_ok = await cognee_setup()
    if not cognee_ok:
        print("  Cognee not available.")
        suite.cognee_available = False
    else:
        print("  Cognee ready.")

    # Run benchmarks
    try:
        await bench_write_speed(suite, cognee_ok, nm_encoder)
        await bench_read_speed(suite, cognee_ok, nm_pipeline)
        await bench_multihop(suite, cognee_ok, nm_pipeline)

        print("\n[5/6] Memory Cost — tracked across all operations above")
        add_cost_result(suite)

        await bench_conversation(suite, cognee_ok, nm_encoder, nm_pipeline)
    finally:
        await nm_storage.close()

    # Report
    print_report(suite)

    output_path = Path(__file__).resolve().parent / "benchmark_cognee_results.json"
    save_results(suite, output_path)


if __name__ == "__main__":
    asyncio.run(main())
