"""Baseline retrievers for LongMemEval — used to compare NM vs dumb approaches.

Four baselines implemented:
    1. Naive: deterministic query-token overlap across complete sessions
    2. FTS5: SQLite full-text search (BM25) — the "dumb but fast" baseline
    3. Embedding: sentence-transformers + cosine — the "semantic but shallow" baseline
    4. Full-context (recency): top-k most-recent sessions, no relevance ranking

All four return a deduplicated list of session_ids ranked by relevance.
No LLM API calls needed — this runs entirely offline.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import time
from dataclasses import dataclass

from scripts.benchmark.data_loader import LMEInstance, Session
from scripts.benchmark.evidence import PINNED_EMBEDDING_REVISION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _session_text(session: Session) -> str:
    """Concatenate all turns of a session into one searchable text blob."""
    lines: list[str] = []
    for turn in session.turns:
        lines.append(f"{turn.role}: {turn.content}")
    return "\n".join(lines)


@dataclass
class RetrievalOutput:
    """Result of a baseline retrieval."""

    session_ids: list[str]
    elapsed_sec: float
    extra: dict[str, object]


# ---------------------------------------------------------------------------
# Baseline 1: Naive token overlap
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE))


def retrieve_naive(instance: LMEInstance, top_k: int = 10) -> RetrievalOutput:
    """Rank complete sessions by query-token overlap with stable ID tie-breaking."""
    t0 = time.perf_counter()
    query_tokens = _tokens(instance.question)
    ranked = sorted(
        (
            (len(query_tokens & _tokens(_session_text(session))), session.session_id)
            for session in instance.sessions
        ),
        key=lambda item: (-item[0], item[1]),
    )[: max(0, top_k)]
    return RetrievalOutput(
        session_ids=[session_id for _, session_id in ranked],
        elapsed_sec=time.perf_counter() - t0,
        extra={
            "method": "naive_token_overlap",
            "scores": [score for score, _ in ranked],
        },
    )


# ---------------------------------------------------------------------------
# Baseline 2: FTS5 BM25
# ---------------------------------------------------------------------------


def retrieve_fts5(instance: LMEInstance, top_k: int = 10) -> RetrievalOutput:
    """Rank sessions by SQLite FTS5 BM25 against the question.

    Builds an in-memory FTS5 table with one row per session, inserts concatenated
    session text, runs MATCH on the question. Returns top-k session_ids by BM25.
    Ties are broken by session recency (more recent first).
    """
    t0 = time.perf_counter()

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE sessions USING fts5(session_id UNINDEXED, text, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        rows = [(s.session_id, _session_text(s)) for s in instance.sessions]
        conn.executemany("INSERT INTO sessions (session_id, text) VALUES (?, ?)", rows)

        query = _fts5_escape(instance.question)
        cursor = conn.execute(
            "SELECT session_id, bm25(sessions) AS score "
            "FROM sessions WHERE sessions MATCH ? "
            "ORDER BY score, session_id LIMIT ?",
            (query, top_k),
        )
        ranked = [str(r[0]) for r in cursor.fetchall()]
    finally:
        conn.close()

    elapsed = time.perf_counter() - t0

    return RetrievalOutput(
        session_ids=ranked,
        elapsed_sec=elapsed,
        extra={"method": "fts5_bm25"},
    )


def _fts5_escape(query: str) -> str:
    """Convert a natural-language question into an FTS5 MATCH query.

    We tokenize on word boundaries, drop FTS5 special chars, and join with OR.
    This gives a permissive "any-of-these-words" match against BM25.
    """
    import re

    tokens = re.findall(r"[A-Za-z0-9]+", query)
    # Drop stop-ish tokens to improve BM25 signal
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "do",
        "does",
        "did",
        "what",
        "when",
        "where",
        "who",
        "whom",
        "why",
        "how",
        "which",
        "i",
        "you",
        "me",
        "my",
        "your",
        "it",
        "of",
        "in",
        "on",
        "at",
        "for",
        "to",
        "with",
        "from",
        "this",
        "that",
        "and",
        "or",
        "but",
        "not",
    }
    filtered = [f'"{t}"' for t in tokens if t.lower() not in stop and len(t) > 1]
    if not filtered:
        filtered = [f'"{t}"' for t in tokens[:3]] or ['"x"']
    return " OR ".join(filtered)


# ---------------------------------------------------------------------------
# Baseline 3: Embedding (sentence-transformers + cosine)
# ---------------------------------------------------------------------------


_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _get_embedder(
    model_name: str = "all-MiniLM-L6-v2",
    revision: str = PINNED_EMBEDDING_REVISION,
) -> object:
    """Lazy-load and cache the sentence-transformer model."""
    cache_key = (model_name, revision)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision)
    _MODEL_CACHE[cache_key] = model
    return model


def retrieve_embedding(
    instance: LMEInstance,
    top_k: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
    revision: str = PINNED_EMBEDDING_REVISION,
) -> RetrievalOutput:
    """Rank sessions by cosine similarity between embeddings of session + question."""
    import numpy as np

    t0 = time.perf_counter()

    embedder = _get_embedder(model_name, revision)

    session_texts = [_session_text(s) for s in instance.sessions]
    session_ids = [s.session_id for s in instance.sessions]

    # Encode in one batch — much faster than per-call
    session_embs = embedder.encode(  # type: ignore[attr-defined]
        session_texts, normalize_embeddings=True, show_progress_bar=False
    )
    query_emb = embedder.encode(  # type: ignore[attr-defined]
        [instance.question], normalize_embeddings=True, show_progress_bar=False
    )[0]

    scores = np.asarray(session_embs) @ np.asarray(query_emb)
    order = sorted(
        range(len(session_ids)),
        key=lambda index: (-float(scores[index]), session_ids[index]),
    )[:top_k]
    ranked = [session_ids[index] for index in order]

    elapsed = time.perf_counter() - t0

    return RetrievalOutput(
        session_ids=ranked,
        elapsed_sec=elapsed,
        extra={
            "method": "embedding_cosine",
            "model": model_name,
            "revision": revision,
            "top_score": float(scores[order[0]]) if order else 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Baseline 4: Recency (top-k most-recent sessions — strawman for full-context)
# ---------------------------------------------------------------------------


def retrieve_recency(instance: LMEInstance, top_k: int = 10) -> RetrievalOutput:
    """Return top-k most-recent sessions as a ranked list.

    This is the dumbest strawman: no content analysis, just "what did we talk about
    recently?". It stress-tests how much recall comes purely from recency bias.
    """
    t0 = time.perf_counter()

    sorted_sessions = sorted(instance.sessions, key=lambda s: s.timestamp, reverse=True)
    ranked = [s.session_id for s in sorted_sessions[:top_k]]

    elapsed = time.perf_counter() - t0

    return RetrievalOutput(
        session_ids=ranked,
        elapsed_sec=elapsed,
        extra={"method": "recency"},
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


BASELINES = {
    "naive": retrieve_naive,
    "fts5": retrieve_fts5,
    "embedding": retrieve_embedding,
    "recency": retrieve_recency,
}


async def retrieve(instance: LMEInstance, method: str, top_k: int = 10) -> RetrievalOutput:
    """Async wrapper so orchestrator can `await` all baselines uniformly."""
    if method not in BASELINES:
        raise ValueError(f"Unknown baseline {method!r}. Available: {list(BASELINES)}")

    fn = BASELINES[method]
    # These are CPU-bound and sync — run in executor to avoid blocking the loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(instance, top_k))
