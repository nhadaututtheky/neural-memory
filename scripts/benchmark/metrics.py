"""Metrics computation for LongMemEval benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics at various cutoffs."""

    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0


@dataclass
class QuestionResult:
    """Per-question benchmark result."""

    question_id: str
    question_type: str
    hypothesis: str
    # None when running in --retrieval-only mode
    correct: bool | None
    retrieved_session_ids: list[str]
    answer_session_ids: list[str]
    # True if at least one answer session appears in retrieved sessions
    retrieval_hit: bool
    elapsed_sec: float

    def to_dict(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "hypothesis": self.hypothesis,
            "correct": self.correct,
            "retrieved_session_ids": self.retrieved_session_ids,
            "answer_session_ids": self.answer_session_ids,
            "retrieval_hit": self.retrieval_hit,
            "elapsed_sec": self.elapsed_sec,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "QuestionResult":
        return cls(
            question_id=str(d["question_id"]),
            question_type=str(d["question_type"]),
            hypothesis=str(d.get("hypothesis", "")),
            correct=d.get("correct"),  # type: ignore[arg-type]
            retrieved_session_ids=list(d.get("retrieved_session_ids", [])),  # type: ignore[arg-type]
            answer_session_ids=list(d.get("answer_session_ids", [])),  # type: ignore[arg-type]
            retrieval_hit=bool(d.get("retrieval_hit", False)),
            elapsed_sec=float(d.get("elapsed_sec", 0.0)),
        )


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


def compute_recall_at_k(results: list[QuestionResult], k: int) -> float:
    """Compute mean Recall@k across all results.

    A result is a hit if at least one answer_session_id appears in the top-k
    retrieved_session_ids.
    """
    if not results:
        return 0.0

    hits = 0
    for r in results:
        top_k = r.retrieved_session_ids[:k]
        if any(sid in top_k for sid in r.answer_session_ids):
            hits += 1

    return hits / len(results)


# ---------------------------------------------------------------------------
# NDCG@k
# ---------------------------------------------------------------------------


def _dcg(relevant_flags: list[bool]) -> float:
    """Compute DCG for a ranked list of relevant flags."""
    dcg = 0.0
    for i, rel in enumerate(relevant_flags):
        if rel:
            dcg += 1.0 / math.log2(i + 2)  # rank is 1-indexed → log2(rank+1)
    return dcg


def compute_ndcg_at_k(results: list[QuestionResult], k: int) -> float:
    """Compute mean NDCG@k across all results.

    Binary relevance: a retrieved session is relevant if its ID appears in
    answer_session_ids.
    """
    if not results:
        return 0.0

    ndcg_scores: list[float] = []
    for r in results:
        top_k = r.retrieved_session_ids[:k]
        answer_set = set(r.answer_session_ids)

        # Ideal DCG: all relevant docs at top positions
        n_relevant = min(len(answer_set), k)
        ideal_flags = [True] * n_relevant + [False] * (k - n_relevant)
        ideal = _dcg(ideal_flags)

        if ideal == 0.0:
            # No relevant docs exist — skip (counts as 0)
            ndcg_scores.append(0.0)
            continue

        actual_flags = [sid in answer_set for sid in top_k]
        actual = _dcg(actual_flags)
        ndcg_scores.append(actual / ideal)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


# ---------------------------------------------------------------------------
# Per-type breakdown
# ---------------------------------------------------------------------------


def compute_metrics_by_type(results: list[QuestionResult]) -> dict[str, dict[str, float]]:
    """Compute accuracy and retrieval metrics grouped by question_type.

    Returns a dict mapping question_type → dict of metric_name → value.
    """
    by_type: dict[str, list[QuestionResult]] = {}
    for r in results:
        by_type.setdefault(r.question_type, []).append(r)

    out: dict[str, dict[str, float]] = {}
    for qtype, type_results in sorted(by_type.items()):
        scored = [r for r in type_results if r.correct is not None]
        accuracy = sum(1 for r in scored if r.correct) / len(scored) if scored else float("nan")

        out[qtype] = {
            "count": float(len(type_results)),
            "accuracy": accuracy,
            "recall_at_1": compute_recall_at_k(type_results, 1),
            "recall_at_3": compute_recall_at_k(type_results, 3),
            "recall_at_5": compute_recall_at_k(type_results, 5),
            "recall_at_10": compute_recall_at_k(type_results, 10),
            "ndcg_at_5": compute_ndcg_at_k(type_results, 5),
            "ndcg_at_10": compute_ndcg_at_k(type_results, 10),
        }

    return out


def compute_retrieval_metrics(results: list[QuestionResult]) -> RetrievalMetrics:
    """Compute aggregate retrieval metrics."""
    return RetrievalMetrics(
        recall_at_1=compute_recall_at_k(results, 1),
        recall_at_3=compute_recall_at_k(results, 3),
        recall_at_5=compute_recall_at_k(results, 5),
        recall_at_10=compute_recall_at_k(results, 10),
        ndcg_at_5=compute_ndcg_at_k(results, 5),
        ndcg_at_10=compute_ndcg_at_k(results, 10),
    )
