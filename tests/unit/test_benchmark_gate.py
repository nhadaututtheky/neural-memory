from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts.benchmark.evidence import (
    PINNED_BACKEND,
    PINNED_EMBEDDING_MODEL,
    PINNED_EMBEDDING_REVISION,
    PINNED_LAST_GOOD,
    PINNED_QUALITY_FLOORS,
    PINNED_RETRIEVAL_PROFILE,
    PINNED_SOURCE_ARTIFACT_SHA256,
    bootstrap_interval,
    evaluate_evidence,
    ground_truth_sha256,
)
from scripts.benchmark.metrics import (
    QuestionResult,
    compute_metrics_by_type,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_retrieval_metrics,
)


def _baseline() -> dict[str, object]:
    ground_truth = ground_truth_sha256(
        [
            ("q-1", "single-session-user", ["answer-q-1"]),
            ("q-2", "single-session-user", ["answer-q-2"]),
        ]
    )
    return {
        "schema_version": 1,
        "variant": "s",
        "backend": PINNED_BACKEND,
        "source_artifact_sha256": PINNED_SOURCE_ARTIFACT_SHA256,
        "source_git_sha": "b" * 40,
        "source_git_sha_basis": "test fixture",
        "source_provenance": "legacy_regression_anchor",
        "legacy_metadata_missing": [
            "artifact_git_sha",
            "artifact_git_dirty",
            "seed",
            "backend",
            "top_k",
            "embedding_model",
            "embedding_revision",
            "retrieval_profile",
            "python_version",
            "package_versions",
            "hardware",
        ],
        "corpus_sha256": "fixed-corpus",
        "seed": 42,
        "top_k": 10,
        "embedding_model": PINNED_EMBEDDING_MODEL,
        "embedding_revision": PINNED_EMBEDDING_REVISION,
        "retrieval_profile": PINNED_RETRIEVAL_PROFILE,
        "instance_ids": ["q-1", "q-2"],
        "ground_truth_sha256": ground_truth,
        "quality_floors": dict(PINNED_QUALITY_FLOORS),
        "metrics": dict(PINNED_LAST_GOOD),
        "required_same_run_methods": ["naive", "fts5", "embedding"],
    }


def _baseline_hash(baseline: dict[str, object]) -> str:
    payload = json.dumps(
        baseline,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _method(hits: tuple[bool, bool]) -> dict[str, object]:
    results: list[QuestionResult] = []
    for question_id, hit in zip(("q-1", "q-2"), hits, strict=True):
        answer_id = f"answer-{question_id}"
        results.append(
            QuestionResult(
                question_id=question_id,
                question_type="single-session-user",
                hypothesis="",
                correct=None,
                retrieved_session_ids=[answer_id] if hit else ["other"],
                answer_session_ids=[answer_id],
                retrieval_hit=hit,
                elapsed_sec=0.1,
            )
        )
    metrics = compute_retrieval_metrics(results)
    metric_samples = {
        "recall_at_1": [compute_recall_at_k([result], 1) for result in results],
        "recall_at_3": [compute_recall_at_k([result], 3) for result in results],
        "recall_at_5": [compute_recall_at_k([result], 5) for result in results],
        "recall_at_10": [compute_recall_at_k([result], 10) for result in results],
        "ndcg_at_5": [compute_ndcg_at_k([result], 5) for result in results],
        "ndcg_at_10": [compute_ndcg_at_k([result], 10) for result in results],
        "elapsed_sec": [result.elapsed_sec for result in results],
    }
    return {
        "instance_ids": [result.question_id for result in results],
        "ground_truth_sha256": ground_truth_sha256(
            [
                (result.question_id, result.question_type, result.answer_session_ids)
                for result in results
            ]
        ),
        "summary": {
            **asdict(metrics),
            "instances": len(results),
            "total_elapsed_sec": 0.2,
            "mean_elapsed_sec": 0.1,
        },
        "by_type": compute_metrics_by_type(results),
        "confidence_intervals": {
            metric: list(bootstrap_interval(values, seed=42))
            for metric, values in metric_samples.items()
        },
        "raw_timing_rows": [
            {"question_id": result.question_id, "elapsed_sec": result.elapsed_sec}
            for result in results
        ],
        "per_question": [result.to_dict() for result in results],
    }


def _report(baseline: dict[str, object] | None = None) -> dict[str, object]:
    pinned = baseline or _baseline()
    return {
        "schema_version": 1,
        "canonical": True,
        "non_canonical_reasons": [],
        "baseline_sha256": _baseline_hash(pinned),
        "manifest": {
            "git_sha": "a" * 40,
            "git_dirty": False,
            "corpus_sha256": "fixed-corpus",
            "seed": 42,
            "instance_ids": ["q-1", "q-2"],
            "config": {
                "variant": "s",
                "backend": PINNED_BACKEND,
                "retrieval_only": True,
                "top_k": 10,
                "embedding_model": PINNED_EMBEDDING_MODEL,
                "embedding_revision": PINNED_EMBEDDING_REVISION,
                "retrieval_profile": PINNED_RETRIEVAL_PROFILE,
                "ground_truth_sha256": pinned["ground_truth_sha256"],
            },
        },
        "methods": {
            "nm": _method((True, True)),
            "naive": _method((True, False)),
            "fts5": _method((False, False)),
            "embedding": _method((True, False)),
        },
    }


def test_evaluate_evidence_accepts_canonical_absolute_and_relative_wins() -> None:
    baseline = _baseline()
    result = evaluate_evidence(_report(baseline), baseline)

    assert result.passed is True
    assert result.failures == ()


def test_evaluate_evidence_enforces_floors_even_when_same_run_baselines_are_weaker() -> None:
    baseline = _baseline()
    report = _report(baseline)
    report["methods"]["nm"] = _method((False, False))
    report["methods"]["naive"] = _method((False, False))
    report["methods"]["embedding"] = _method((False, False))

    result = evaluate_evidence(report, baseline)

    assert result.passed is False
    assert any(
        "absolute floor" in failure and "recall_at_5" in failure for failure in result.failures
    )


def test_evaluate_evidence_rejects_last_good_regression_and_same_run_loss() -> None:
    baseline = _baseline()
    report = _report(baseline)
    report["methods"]["nm"] = _method((True, False))
    report["methods"]["embedding"] = _method((True, True))

    result = evaluate_evidence(report, baseline)

    assert any("last-good" in failure and "ndcg_at_5" in failure for failure in result.failures)
    assert any("embedding" in failure and "recall_at_5" in failure for failure in result.failures)


def test_evaluate_evidence_fails_closed_on_sample_or_contract_mismatch() -> None:
    baseline = _baseline()
    report = _report(baseline)
    report["canonical"] = False
    report["non_canonical_reasons"] = ["git_worktree_dirty"]
    report["manifest"]["instance_ids"] = ["q-2", "q-1"]
    report["methods"].pop("fts5")

    result = evaluate_evidence(report, deepcopy(baseline))

    assert any("non-canonical" in failure for failure in result.failures)
    assert any("instance IDs" in failure for failure in result.failures)
    assert any("required method 'fts5'" in failure for failure in result.failures)


def test_evaluate_evidence_rejects_seed_cutoff_and_backend_mismatch() -> None:
    baseline = _baseline()
    report = _report(baseline)
    report["manifest"]["seed"] = 7
    report["manifest"]["config"]["top_k"] = 5
    report["manifest"]["config"]["backend"] = "infinitydb"

    result = evaluate_evidence(report, baseline)

    assert any("seed mismatch" in failure for failure in result.failures)
    assert any("top_k mismatch" in failure for failure in result.failures)
    assert any("backend mismatch" in failure for failure in result.failures)


def test_evaluate_evidence_revalidates_every_evidence_row() -> None:
    baseline = _baseline()
    report = _report(baseline)
    report["methods"]["embedding"]["instance_ids"] = ["q-2", "q-1"]
    report["methods"]["nm"]["per_question"] = []
    report["methods"]["naive"]["raw_timing_rows"] = []

    result = evaluate_evidence(report, baseline)

    assert any("embedding" in failure and "instance IDs" in failure for failure in result.failures)
    assert any("nm" in failure and "per_question" in failure for failure in result.failures)
    assert any("naive" in failure and "raw timing" in failure for failure in result.failures)


def test_evaluate_evidence_rejects_method_specific_ground_truth() -> None:
    baseline = _baseline()
    report = _report(baseline)
    nm_row = report["methods"]["nm"]["per_question"][0]
    nm_row["answer_session_ids"] = ["forged-answer"]
    nm_row["retrieved_session_ids"] = ["forged-answer"]
    nm_row["retrieval_hit"] = True

    result = evaluate_evidence(report, baseline)

    assert any("nm" in failure and "ground truth" in failure for failure in result.failures)


def test_evaluate_evidence_rejects_non_finite_or_self_declared_metrics() -> None:
    baseline = _baseline()
    report = _report(baseline)
    report["methods"]["nm"]["summary"]["recall_at_5"] = math.nan
    report["methods"]["nm"]["summary"]["ndcg_at_5"] = math.inf
    report["methods"]["nm"]["raw_timing_rows"][0]["elapsed_sec"] = math.inf

    result = evaluate_evidence(report, baseline)

    assert result.passed is False
    assert any("finite" in failure and "recall_at_5" in failure for failure in result.failures)
    assert any("finite" in failure and "ndcg_at_5" in failure for failure in result.failures)
    assert any("finite" in failure and "elapsed_sec" in failure for failure in result.failures)


def test_evaluate_evidence_recomputes_derived_evidence_fields() -> None:
    baseline = _baseline()
    report = _report(baseline)
    nm = report["methods"]["nm"]
    nm["summary"]["mean_elapsed_sec"] = 9.0
    nm["confidence_intervals"]["recall_at_5"] = [0.9, 1.0]
    nm["by_type"]["single-session-user"]["recall_at_5"] = 0.0

    result = evaluate_evidence(report, baseline)

    assert any("mean_elapsed_sec" in failure for failure in result.failures)
    assert any("confidence interval" in failure for failure in result.failures)
    assert any("by_type" in failure for failure in result.failures)


def test_evaluate_evidence_rejects_weakened_or_unidentified_baseline() -> None:
    baseline = _baseline()
    baseline["quality_floors"]["recall_at_5"] = -1.0
    baseline["required_same_run_methods"] = []
    report = _report(baseline)
    report["baseline_sha256"] = "wrong-hash"

    result = evaluate_evidence(report, baseline)

    assert any("baseline SHA-256" in failure for failure in result.failures)
    assert any("required_same_run_methods" in failure for failure in result.failures)
    assert any("quality floor" in failure for failure in result.failures)


def test_evaluate_evidence_rejects_dirty_or_unknown_git_provenance() -> None:
    baseline = _baseline()
    report = _report(baseline)
    report["manifest"]["git_dirty"] = True
    report["manifest"]["git_sha"] = "unknown"

    result = evaluate_evidence(report, baseline)

    assert any("Git worktree" in failure for failure in result.failures)
    assert any("Git SHA" in failure for failure in result.failures)


def test_pinned_baseline_matches_source_artifact_and_corpus() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    baseline_path = repo_root / "scripts" / "benchmark" / "evidence_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    source_path = repo_root / baseline["source_artifact"]
    corpus_path = repo_root / "scripts" / "benchmark" / "data" / "longmemeval_s_cleaned.json"
    assert baseline["source_provenance"] == "legacy_regression_anchor"
    assert {
        "artifact_git_sha",
        "seed",
        "backend",
        "top_k",
        "embedding_revision",
        "retrieval_profile",
    } <= set(baseline["legacy_metadata_missing"])
    assert source_path.exists()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_results = [QuestionResult.from_dict(row) for row in source["results"]]
    source_metrics = compute_retrieval_metrics(source_results)
    source_ids = [row["question_id"] for row in source["results"]]

    # Hash Git-canonical LF bytes so Windows autocrlf checkouts match CI/Linux.
    source_bytes = source_path.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(source_bytes).hexdigest() == baseline["source_artifact_sha256"]
    if corpus_path.exists():
        corpus_bytes = corpus_path.read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(corpus_bytes).hexdigest() == baseline["corpus_sha256"]
    # Recomputed floats can drift 1 ULP across platforms; pin equality within eps.
    assert source_metrics.recall_at_5 == pytest.approx(
        float(baseline["metrics"]["recall_at_5"]), abs=1e-12
    )
    assert source_metrics.ndcg_at_5 == pytest.approx(
        float(baseline["metrics"]["ndcg_at_5"]), abs=1e-12
    )
    assert source_ids == baseline["source_instance_ids"]
    assert len(source_ids) == len(set(source_ids))
    assert set(source_ids) == set(baseline["instance_ids"])
    source_by_id = {row.question_id: row for row in source_results}
    assert (
        ground_truth_sha256(
            [
                (
                    question_id,
                    source_by_id[question_id].question_type,
                    source_by_id[question_id].answer_session_ids,
                )
                for question_id in baseline["instance_ids"]
            ]
        )
        == baseline["ground_truth_sha256"]
    )
