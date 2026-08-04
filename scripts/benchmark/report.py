"""Report generation for LongMemEval benchmark results."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from scripts.benchmark.evidence import (
    EvidenceManifest,
    bootstrap_interval,
    ground_truth_sha256,
)
from scripts.benchmark.metrics import (
    QuestionResult,
    compute_metrics_by_type,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_retrieval_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


class EvidenceContractError(ValueError):
    """Raised when methods were not evaluated on the manifest's exact ID set."""


def _metric_samples(results: Sequence[QuestionResult]) -> dict[str, list[float]]:
    return {
        "recall_at_1": [compute_recall_at_k([result], 1) for result in results],
        "recall_at_3": [compute_recall_at_k([result], 3) for result in results],
        "recall_at_5": [compute_recall_at_k([result], 5) for result in results],
        "recall_at_10": [compute_recall_at_k([result], 10) for result in results],
        "ndcg_at_5": [compute_ndcg_at_k([result], 5) for result in results],
        "ndcg_at_10": [compute_ndcg_at_k([result], 10) for result in results],
        "elapsed_sec": [result.elapsed_sec for result in results],
    }


def _method_evidence(
    results: Sequence[QuestionResult],
    *,
    seed: int,
    total_elapsed: float,
) -> dict[str, object]:
    if not math.isfinite(total_elapsed) or total_elapsed < 0.0:
        raise EvidenceContractError("total elapsed must be finite non-negative")
    for result in results:
        if not math.isfinite(result.elapsed_sec) or result.elapsed_sec < 0.0:
            raise EvidenceContractError(
                f"{result.question_id} elapsed time must be finite non-negative"
            )
    metrics = compute_retrieval_metrics(list(results))
    samples = _metric_samples(results)
    confidence_intervals = {
        name: list(bootstrap_interval(values, seed=seed)) for name, values in samples.items()
    }
    elapsed_values = samples["elapsed_sec"]
    ground_truth_hash = ground_truth_sha256(
        [
            (result.question_id, result.question_type, result.answer_session_ids)
            for result in results
        ]
    )
    return {
        "instance_ids": [result.question_id for result in results],
        "ground_truth_sha256": ground_truth_hash,
        "summary": {
            **asdict(metrics),
            "instances": len(results),
            "total_elapsed_sec": total_elapsed,
            "mean_elapsed_sec": (
                sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0
            ),
        },
        "confidence_intervals": confidence_intervals,
        "by_type": _sanitize_json(compute_metrics_by_type(list(results))),
        "raw_timing_rows": [
            {"question_id": result.question_id, "elapsed_sec": result.elapsed_sec}
            for result in results
        ],
        "per_question": [deepcopy(result.to_dict()) for result in results],
    }


def build_evidence_report(
    manifest: EvidenceManifest,
    method_results: Mapping[str, Sequence[QuestionResult]],
    *,
    total_elapsed: Mapping[str, float],
    canonical: bool = False,
    non_canonical_reasons: Sequence[str] = (),
) -> dict[str, object]:
    """Build a versioned report after enforcing one ordered ID set for every method."""
    expected_ids = list(manifest.instance_ids)
    expected_ground_truth_hash = manifest.config.get("ground_truth_sha256")
    if not isinstance(expected_ground_truth_hash, str):
        raise EvidenceContractError("manifest ground truth hash is missing")
    methods: dict[str, object] = {}
    for method, results in method_results.items():
        actual_ids = [result.question_id for result in results]
        if actual_ids != expected_ids:
            missing_ids = [
                question_id for question_id in expected_ids if question_id not in actual_ids
            ]
            unexpected_ids = [
                question_id for question_id in actual_ids if question_id not in expected_ids
            ]
            raise EvidenceContractError(
                f"{method} instance IDs differ from manifest; "
                f"missing={missing_ids}, unexpected={unexpected_ids}, "
                f"expected_order={expected_ids}, actual_order={actual_ids}"
            )
        actual_ground_truth_hash = ground_truth_sha256(
            [
                (result.question_id, result.question_type, result.answer_session_ids)
                for result in results
            ]
        )
        if actual_ground_truth_hash != expected_ground_truth_hash:
            raise EvidenceContractError(
                f"{method} ground truth differs from the manifest's canonical sample"
            )
        if method not in total_elapsed:
            raise EvidenceContractError(f"missing total elapsed for method {method!r}")
        method_elapsed = total_elapsed[method]
        if not math.isfinite(method_elapsed) or method_elapsed < 0.0:
            raise EvidenceContractError(
                f"total elapsed for method {method!r} must be finite non-negative"
            )
        methods[method] = _method_evidence(
            results,
            seed=manifest.seed,
            total_elapsed=method_elapsed,
        )

    return {
        "schema_version": manifest.schema_version,
        "generated_at": datetime.now(UTC).isoformat(),
        "canonical": canonical,
        "non_canonical_reasons": list(non_canonical_reasons),
        "manifest": _sanitize_json(asdict(manifest)),
        "methods": methods,
    }


def _sanitize_json(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    return value


def _report_float(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise EvidenceContractError(f"expected numeric report value, got {value!r}")
    return float(value)


def _evidence_markdown(report: Mapping[str, object]) -> str:
    manifest = cast("Mapping[str, object]", report["manifest"])
    methods = cast("Mapping[str, object]", report["methods"])
    lines = [
        "# Benchmark Evidence Report",
        "",
        f"- **Schema**: v{report['schema_version']}",
        f"- **Canonical**: {str(report['canonical']).lower()}",
        f"- **Git SHA**: {manifest['git_sha']}",
        f"- **Corpus SHA-256**: `{manifest['corpus_sha256']}`",
        f"- **Instances**: {len(cast('Sequence[object]', manifest['instance_ids']))}",
        "",
        "| Method | N | R@5 | NDCG@5 | Total sec |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, method_value in methods.items():
        method_data = cast("Mapping[str, object]", method_value)
        summary = cast("Mapping[str, object]", method_data["summary"])
        lines.append(
            f"| {method} | {summary['instances']} | "
            f"{_report_float(summary['recall_at_5']):.3f} | "
            f"{_report_float(summary['ndcg_at_5']):.3f} | "
            f"{_report_float(summary['total_elapsed_sec']):.3f} |"
        )
    reasons = cast("Sequence[str]", report["non_canonical_reasons"])
    if reasons:
        lines.extend(["", "## Non-Canonical Reasons", ""])
        lines.extend(f"- `{reason}`" for reason in reasons)
    gate = report.get("gate")
    if isinstance(gate, dict):
        lines.extend(["", "## Evidence Gate", "", f"- **Passed**: {gate.get('passed')}"])
        gate_failures = gate.get("failures")
        if isinstance(gate_failures, (list, tuple)):
            lines.extend(f"- {failure}" for failure in gate_failures)
    return "\n".join(lines) + "\n"


def save_evidence_report(
    report: Mapping[str, object],
    output_dir: Path,
    *,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    """Write one immutable JSON artifact and its human-readable Markdown view."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    stem = f"evidence_v{report['schema_version']}_{run_timestamp}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_payload = json.dumps(_sanitize_json(dict(report)), indent=2, allow_nan=False) + "\n"
    markdown_payload = _evidence_markdown(report)
    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        for suffix, payload in (("json", json_payload), ("md", markdown_payload)):
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_dir,
                prefix=f".{stem}.{suffix}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_paths.append(Path(temporary_file.name))

        # Publish JSON last so its presence proves the Markdown pair is complete.
        for temporary_path, final_path in (
            (temporary_paths[1], markdown_path),
            (temporary_paths[0], json_path),
        ):
            os.link(temporary_path, final_path)
            published_paths.append(final_path)
    except Exception:
        for published_path in reversed(published_paths):
            published_path.unlink(missing_ok=True)
        raise
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
    return json_path, markdown_path


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
    print(f"- **Date**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
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
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
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
