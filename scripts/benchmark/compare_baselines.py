"""Compare NM vs dumb baselines on LongMemEval — retrieval-only (no API cost).

Runs FTS5, embedding (MiniLM), and recency baselines on the same instances
we benchmark NM on. Outputs a side-by-side report of R@1/3/5/10 and NDCG@5/10.

Usage:
    python scripts/benchmark/compare_baselines.py --variant s --limit 20
    python scripts/benchmark/compare_baselines.py --variant s --instance-ids scripts/benchmark/mini_bench_ids.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Make imports work when run directly
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root))

# Quiet noisy loggers
for _name in [
    "neural_memory.engine",
    "neural_memory.storage",
    "neural_memory.safety",
    "sentence_transformers",
    "transformers",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("compare_baselines")

from scripts.benchmark.baselines import BASELINES, retrieve
from scripts.benchmark.data_loader import LMEInstance, load_dataset
from scripts.benchmark.metrics import (
    QuestionResult,
    compute_metrics_by_type,
    compute_retrieval_metrics,
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def _run_method(
    method: str, instances: list[LMEInstance], top_k: int
) -> tuple[list[QuestionResult], float]:
    """Run one baseline across all instances, return per-instance results."""
    results: list[QuestionResult] = []
    t0 = time.perf_counter()

    for i, inst in enumerate(instances):
        out = await retrieve(inst, method, top_k)
        top = out.session_ids[:top_k]
        hit = any(sid in top for sid in inst.answer_session_ids)
        results.append(
            QuestionResult(
                question_id=inst.question_id,
                question_type=inst.question_type,
                hypothesis="",
                correct=None,
                retrieved_session_ids=top,
                answer_session_ids=inst.answer_session_ids,
                retrieval_hit=hit,
                elapsed_sec=out.elapsed_sec,
            )
        )
        if (i + 1) % 5 == 0 or i == len(instances) - 1:
            print(f"  [{method}] {i + 1}/{len(instances)}", flush=True)

    total_elapsed = time.perf_counter() - t0
    return results, total_elapsed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_report(
    all_results: dict[str, list[QuestionResult]],
    nm_baseline: dict[str, object] | None,
    total_elapsed: dict[str, float],
) -> str:
    lines: list[str] = []
    lines.append("# LongMemEval — Baseline Comparison Report\n")
    lines.append(f"**Instances**: {len(next(iter(all_results.values())))}\n")
    lines.append("**Mode**: Retrieval-only (no LLM reader/judge)\n\n")

    # Build header dynamically
    methods = list(all_results.keys())
    if nm_baseline is not None:
        methods = ["NM (current)"] + methods

    lines.append("## Aggregate Retrieval Metrics\n")
    lines.append("| Method | R@1 | R@3 | R@5 | R@10 | NDCG@5 | NDCG@10 | Total sec |")
    lines.append("|---|---|---|---|---|---|---|---|")

    if nm_baseline is not None:
        nm = nm_baseline
        lines.append(
            f"| NM (current) | {nm['recall_at_1']:.3f} | {nm['recall_at_3']:.3f} | "
            f"{nm['recall_at_5']:.3f} | {nm['recall_at_10']:.3f} | "
            f"{nm['ndcg_at_5']:.3f} | {nm['ndcg_at_10']:.3f} | "
            f"{nm.get('total_elapsed_sec', '—')} |"
        )

    for method in all_results:
        results = all_results[method]
        m = compute_retrieval_metrics(results)
        t = total_elapsed.get(method, 0.0)
        lines.append(
            f"| {method} | {m.recall_at_1:.3f} | {m.recall_at_3:.3f} | "
            f"{m.recall_at_5:.3f} | {m.recall_at_10:.3f} | "
            f"{m.ndcg_at_5:.3f} | {m.ndcg_at_10:.3f} | {t:.1f}s |"
        )
    lines.append("")

    # Per-question-type breakdown
    lines.append("## Per-Question-Type Breakdown (Recall@5)\n")

    # Collect all types and methods
    all_types: set[str] = set()
    by_method_type: dict[str, dict[str, dict[str, float]]] = {}
    for method, results in all_results.items():
        by_type = compute_metrics_by_type(results)
        by_method_type[method] = by_type
        all_types.update(by_type.keys())

    if nm_baseline is not None and "by_type" in nm_baseline:
        by_method_type["NM (current)"] = nm_baseline["by_type"]  # type: ignore[assignment]
        all_types.update(nm_baseline["by_type"].keys())  # type: ignore[arg-type]

    method_order = ["NM (current)"] if nm_baseline is not None else []
    method_order += list(all_results.keys())

    header = "| Question Type | Count | " + " | ".join(method_order) + " |"
    sep = "|---|---|" + "---|" * len(method_order)
    lines.append(header)
    lines.append(sep)

    for qtype in sorted(all_types):
        count = 0
        row = []
        for method in method_order:
            data = by_method_type.get(method, {}).get(qtype)
            if data is None:
                row.append("—")
            else:
                count = int(data.get("count", 0))
                row.append(f"{data['recall_at_5']:.3f}")
        lines.append(f"| {qtype} | {count} | " + " | ".join(row) + " |")
    lines.append("")

    # Latency breakdown
    lines.append("## Average Retrieval Latency (per instance)\n")
    lines.append("| Method | Mean sec | Total sec |")
    lines.append("|---|---|---|")
    for method, results in all_results.items():
        mean = sum(r.elapsed_sec for r in results) / len(results) if results else 0.0
        total = total_elapsed.get(method, 0.0)
        lines.append(f"| {method} | {mean:.4f} | {total:.1f} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NM baseline loader (parses prior NM results JSON if available)
# ---------------------------------------------------------------------------


def _load_nm_baseline(path: Path | None, instance_ids: set[str]) -> dict[str, object] | None:
    """Load prior NM benchmark results and filter to the chosen instance subset."""
    if path is None or not path.exists():
        logger.info("NM baseline JSON not provided — skipping NM row")
        return None

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    per_question = data.get("results") or data.get("per_question") or []
    filtered: list[QuestionResult] = []
    for item in per_question:
        qid = item.get("question_id")
        if qid is None or (instance_ids and qid not in instance_ids):
            continue
        try:
            filtered.append(QuestionResult.from_dict(item))
        except Exception:  # noqa: BLE001
            logger.debug("Could not parse NM result item: %s", item)

    if not filtered:
        logger.warning("NM baseline had no matching instances — skipping NM row")
        return None

    m = compute_retrieval_metrics(filtered)
    by_type = compute_metrics_by_type(filtered)
    total_elapsed = sum(r.elapsed_sec for r in filtered)

    return {
        "recall_at_1": m.recall_at_1,
        "recall_at_3": m.recall_at_3,
        "recall_at_5": m.recall_at_5,
        "recall_at_10": m.recall_at_10,
        "ndcg_at_5": m.ndcg_at_5,
        "ndcg_at_10": m.ndcg_at_10,
        "by_type": by_type,
        "total_elapsed_sec": f"{total_elapsed:.0f}",
        "n": len(filtered),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NM vs baselines on LongMemEval")
    parser.add_argument("--variant", choices=["oracle", "s", "m"], default="s")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--instance-ids",
        type=Path,
        default=None,
        help="JSON file with instance_ids array (e.g. mini_bench_ids.json). Overrides --limit.",
    )
    parser.add_argument(
        "--nm-results",
        type=Path,
        default=None,
        help="Prior NM benchmark JSON (from longmemeval.py). Filters to same instances.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(BASELINES.keys()),
        help=f"Baseline methods to run (default: all of {list(BASELINES.keys())})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    # Load dataset
    instances = load_dataset(args.variant, Path(__file__).resolve().parent / "data")

    # Filter to requested instance set
    if args.instance_ids is not None:
        with args.instance_ids.open(encoding="utf-8") as f:
            id_data = json.load(f)
        allowed = set(id_data["instance_ids"])
        instances = [i for i in instances if i.question_id in allowed]
        print(f"Filtered to {len(instances)} instances from {args.instance_ids.name}")
    else:
        instances = instances[: args.limit]
        print(f"Using first {len(instances)} instances (variant={args.variant})")

    instance_id_set = {i.question_id for i in instances}

    # Load NM baseline if provided
    nm_baseline = _load_nm_baseline(args.nm_results, instance_id_set)
    if nm_baseline is not None:
        print(
            f"NM baseline loaded: {nm_baseline['n']} instances, "
            f"R@5={nm_baseline['recall_at_5']:.3f}"
        )

    # Run each baseline
    all_results: dict[str, list[QuestionResult]] = {}
    total_elapsed: dict[str, float] = {}

    for method in args.methods:
        if method not in BASELINES:
            print(f"Skipping unknown method: {method}")
            continue
        print(f"\n=== Running {method} ===", flush=True)
        results, elapsed = await _run_method(method, instances, args.top_k)
        all_results[method] = results
        total_elapsed[method] = elapsed
        m = compute_retrieval_metrics(results)
        print(
            f"  {method}: R@5={m.recall_at_5:.3f} R@10={m.recall_at_10:.3f} "
            f"NDCG@5={m.ndcg_at_5:.3f} ({elapsed:.1f}s total)"
        )

    # Generate report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    md_path = args.output_dir / f"compare_baselines_{timestamp}.md"
    json_path = args.output_dir / f"compare_baselines_{timestamp}.json"

    report = _format_report(all_results, nm_baseline, total_elapsed)
    md_path.write_text(report, encoding="utf-8")
    print(f"\n[OK] Report saved: {md_path}")

    # Structured JSON for downstream analysis
    out_json: dict[str, object] = {
        "variant": args.variant,
        "n_instances": len(instances),
        "instance_ids": sorted(instance_id_set),
        "methods": {},
        "nm_baseline": nm_baseline,
    }
    methods_out: dict[str, object] = out_json["methods"]  # type: ignore[assignment]
    for method, results in all_results.items():
        m = compute_retrieval_metrics(results)
        methods_out[method] = {
            "recall_at_1": m.recall_at_1,
            "recall_at_3": m.recall_at_3,
            "recall_at_5": m.recall_at_5,
            "recall_at_10": m.recall_at_10,
            "ndcg_at_5": m.ndcg_at_5,
            "ndcg_at_10": m.ndcg_at_10,
            "total_elapsed_sec": total_elapsed.get(method, 0.0),
            "by_type": compute_metrics_by_type(results),
            "per_question": [r.to_dict() for r in results],
        }
    json_path.write_text(json.dumps(out_json, indent=2, default=str), encoding="utf-8")
    print(f"[OK] JSON saved: {json_path}")

    # Print summary
    print("\n" + report.split("\n## Per-Question")[0])


if __name__ == "__main__":
    asyncio.run(main())
