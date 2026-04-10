"""Report generation for LongMemEval benchmark results."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from scripts.benchmark.metrics import (
    QuestionResult,
    compute_metrics_by_type,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_retrieval_metrics,
)

logger = logging.getLogger(__name__)


def _overall_accuracy(results: list[QuestionResult]) -> float:
    scored = [r for r in results if r.correct is not None]
    if not scored:
        return float("nan")
    return sum(1 for r in scored if r.correct) / len(scored)


def print_report(results: list[QuestionResult], config: object) -> None:
    """Print a markdown-formatted report to stdout."""
    from scripts.benchmark.config import BenchmarkConfig

    assert isinstance(config, BenchmarkConfig)

    rm = compute_retrieval_metrics(results)
    accuracy = _overall_accuracy(results)
    by_type = compute_metrics_by_type(results)
    scored_count = sum(1 for r in results if r.correct is not None)

    print()
    print("# LongMemEval Benchmark Results")
    print()
    print(f"- **Variant**: {config.variant}")
    print(f"- **Reader**: {config.reader}")
    print(f"- **Judge**: {config.judge}")
    print(f"- **Backend**: {config.backend}")
    print(f"- **Instances**: {len(results)}")
    print(f"- **Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print()

    # --- Retrieval metrics ---
    print("## Retrieval Metrics")
    print()
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Recall@1  | {rm.recall_at_1:.3f} |")
    print(f"| Recall@3  | {rm.recall_at_3:.3f} |")
    print(f"| Recall@5  | {rm.recall_at_5:.3f} |")
    print(f"| Recall@10 | {rm.recall_at_10:.3f} |")
    print(f"| NDCG@5    | {rm.ndcg_at_5:.3f} |")
    print(f"| NDCG@10   | {rm.ndcg_at_10:.3f} |")
    print()

    # --- Answer accuracy ---
    if scored_count > 0:
        print("## Answer Accuracy")
        print()
        print(f"- **Overall accuracy**: {accuracy:.3f} ({scored_count}/{len(results)} scored)")
        print()

        # Per-type table
        print("### By Question Type")
        print()
        print("| Type | Count | Accuracy | R@1 | R@5 | NDCG@5 |")
        print("|------|-------|----------|-----|-----|--------|")
        for qtype, m in by_type.items():
            acc_str = f"{m['accuracy']:.3f}" if not _is_nan(m["accuracy"]) else "N/A"
            print(
                f"| {qtype} | {int(m['count'])} | {acc_str} "
                f"| {m['recall_at_1']:.3f} | {m['recall_at_5']:.3f} | {m['ndcg_at_5']:.3f} |"
            )
        print()
    else:
        print("*(Retrieval-only mode -- no judge scores)*")
        print()
        # Still show per-type retrieval breakdown
        print("## Retrieval by Question Type")
        print()
        print("| Type | Count | R@1 | R@3 | R@5 | NDCG@5 |")
        print("|------|-------|-----|-----|-----|--------|")
        for qtype, m in by_type.items():
            print(
                f"| {qtype} | {int(m['count'])} "
                f"| {m['recall_at_1']:.3f} | {m['recall_at_3']:.3f} "
                f"| {m['recall_at_5']:.3f} | {m['ndcg_at_5']:.3f} |"
            )
        print()

    # --- Timing ---
    elapsed_values = [r.elapsed_sec for r in results]
    if elapsed_values:
        avg_elapsed = sum(elapsed_values) / len(elapsed_values)
        total_elapsed = sum(elapsed_values)
        print("## Timing")
        print()
        print(f"- **Total elapsed**: {total_elapsed:.1f}s")
        print(f"- **Avg per instance**: {avg_elapsed:.2f}s")
        print()


def save_report(
    results: list[QuestionResult],
    config: object,
    output_dir: Path,
) -> None:
    """Save JSON results and markdown report to output_dir."""
    from scripts.benchmark.config import BenchmarkConfig

    assert isinstance(config, BenchmarkConfig)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stem = f"lme_{config.variant}_{config.reader}_{timestamp}"

    # --- JSON ---
    json_path = output_dir / f"{stem}.json"
    rm = compute_retrieval_metrics(results)
    by_type = compute_metrics_by_type(results)

    report_data = {
        "config": {
            "variant": config.variant,
            "reader": config.reader,
            "judge": config.judge,
            "backend": config.backend,
            "limit": config.limit,
            "retrieval_only": config.retrieval_only,
            "claude_model": config.claude_model,
        },
        "summary": {
            "instances": len(results),
            "overall_accuracy": _overall_accuracy(results),
            "recall_at_1": rm.recall_at_1,
            "recall_at_3": rm.recall_at_3,
            "recall_at_5": rm.recall_at_5,
            "recall_at_10": rm.recall_at_10,
            "ndcg_at_5": rm.ndcg_at_5,
            "ndcg_at_10": rm.ndcg_at_10,
        },
        "by_type": by_type,
        "results": [r.to_dict() for r in results],
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=_json_default)

    logger.info("Saved JSON results to %s", json_path)

    # --- Markdown ---
    md_path = output_dir / f"{stem}.md"
    import io
    import sys

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        print_report(results, config)
    finally:
        sys.stdout = old_stdout

    with md_path.open("w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    logger.info("Saved markdown report to %s", md_path)
    print(f"\nResults saved to:\n  {json_path}\n  {md_path}")


def _is_nan(v: float) -> bool:
    import math

    return math.isnan(v)


def _json_default(obj: object) -> object:
    import math

    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
