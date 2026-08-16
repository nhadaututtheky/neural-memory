from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.benchmark.cascade_metrics import (
    COGNITIVE_P95_MS,
    SIMPLE_P95_MS,
    aggregate_route_latencies,
    percentile,
)
from scripts.benchmark.evidence import EvidenceManifest, ground_truth_sha256
from scripts.benchmark.metrics import QuestionResult
from scripts.benchmark.report import (
    EvidenceContractError,
    build_evidence_report,
    save_evidence_report,
)


def _manifest() -> EvidenceManifest:
    ground_truth = ground_truth_sha256(
        [
            ("q-1", "single-session-user", ["answer-q-1"]),
            ("q-2", "single-session-user", ["answer-q-2"]),
        ]
    )
    return EvidenceManifest(
        schema_version=1,
        git_sha="abc123",
        git_dirty=False,
        corpus_sha256="corpus-hash",
        instance_ids=("q-1", "q-2"),
        seed=42,
        config={
            "variant": "s",
            "retrieval_only": True,
            "ground_truth_sha256": ground_truth,
        },
        packages={"neural-memory": "4.62.0"},
        hardware={"platform": "test"},
        warmup_runs=1,
    )


def _result(question_id: str, hit: bool, elapsed: float) -> QuestionResult:
    answer_id = f"answer-{question_id}"
    return QuestionResult(
        question_id=question_id,
        question_type="single-session-user",
        hypothesis="",
        correct=None,
        retrieved_session_ids=[answer_id] if hit else ["other"],
        answer_session_ids=[answer_id],
        retrieval_hit=hit,
        elapsed_sec=elapsed,
    )


def test_build_evidence_report_persists_identical_ids_raw_rows_and_intervals() -> None:
    method_results = {
        "nm": [_result("q-1", True, 0.2), _result("q-2", False, 0.4)],
        "naive": [_result("q-1", False, 0.01), _result("q-2", True, 0.02)],
    }

    report = build_evidence_report(
        _manifest(),
        method_results,
        total_elapsed={"nm": 0.7, "naive": 0.04},
        canonical=False,
        non_canonical_reasons=("embedding_not_run",),
    )

    assert report["schema_version"] == 1
    assert report["manifest"]["instance_ids"] == ["q-1", "q-2"]
    assert report["canonical"] is False
    assert report["non_canonical_reasons"] == ["embedding_not_run"]
    assert report["methods"]["nm"]["instance_ids"] == ["q-1", "q-2"]
    assert report["methods"]["nm"]["summary"]["recall_at_5"] == 0.5
    assert report["methods"]["nm"]["confidence_intervals"]["recall_at_5"] == [0.0, 1.0]
    assert report["methods"]["naive"]["raw_timing_rows"] == [
        {"question_id": "q-1", "elapsed_sec": 0.01},
        {"question_id": "q-2", "elapsed_sec": 0.02},
    ]
    assert len(report["methods"]["nm"]["per_question"]) == 2


def test_build_evidence_report_rejects_method_id_mismatch() -> None:
    with pytest.raises(EvidenceContractError, match=r"naive.*q-2"):
        build_evidence_report(
            _manifest(),
            {"naive": [_result("q-1", True, 0.1)]},
            total_elapsed={"naive": 0.1},
        )


def test_build_evidence_report_defaults_to_non_canonical() -> None:
    report = build_evidence_report(
        _manifest(),
        {"nm": [_result("q-1", True, 0.1), _result("q-2", True, 0.1)]},
        total_elapsed={"nm": 0.2},
    )

    assert report["canonical"] is False


def test_build_evidence_report_rejects_method_ground_truth_mismatch() -> None:
    poisoned = _result("q-1", True, 0.1)
    poisoned.answer_session_ids = ["forged-answer"]
    poisoned.retrieved_session_ids = ["forged-answer"]

    with pytest.raises(EvidenceContractError, match="ground truth"):
        build_evidence_report(
            _manifest(),
            {"nm": [poisoned, _result("q-2", True, 0.1)]},
            total_elapsed={"nm": 0.2},
        )


def test_build_evidence_report_rejects_missing_or_non_finite_timing() -> None:
    with pytest.raises(EvidenceContractError, match="missing total elapsed"):
        build_evidence_report(
            _manifest(),
            {"nm": [_result("q-1", True, 0.1), _result("q-2", True, 0.1)]},
            total_elapsed={},
        )

    with pytest.raises(EvidenceContractError, match="finite non-negative"):
        build_evidence_report(
            _manifest(),
            {"nm": [_result("q-1", True, float("inf")), _result("q-2", True, 0.1)]},
            total_elapsed={"nm": 0.2},
        )


def test_save_evidence_report_writes_versioned_json_and_markdown(tmp_path: Path) -> None:
    report = build_evidence_report(
        _manifest(),
        {"nm": [_result("q-1", True, 0.1), _result("q-2", True, 0.1)]},
        total_elapsed={"nm": 0.2},
    )

    json_path, markdown_path = save_evidence_report(
        report,
        tmp_path,
        timestamp="20260102_030405",
    )

    assert json_path.name == "evidence_v1_20260102_030405.json"
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == 1
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Benchmark Evidence Report" in markdown
    assert "| nm | 2 | 1.000 | 1.000 |" in markdown


def test_method_intervals_do_not_depend_on_other_method_order() -> None:
    nm_results = [_result("q-1", True, 0.1), _result("q-2", False, 0.2)]
    naive_results = [_result("q-1", False, 0.1), _result("q-2", True, 0.2)]

    nm_first = build_evidence_report(
        _manifest(),
        {"nm": nm_results, "naive": naive_results},
        total_elapsed={"nm": 0.3, "naive": 0.3},
    )
    nm_second = build_evidence_report(
        _manifest(),
        {"naive": naive_results, "nm": nm_results},
        total_elapsed={"nm": 0.3, "naive": 0.3},
    )

    assert (
        nm_first["methods"]["nm"]["confidence_intervals"]
        == nm_second["methods"]["nm"]["confidence_intervals"]
    )


def test_evidence_report_defensively_copies_rows_and_refuses_overwrite(tmp_path: Path) -> None:
    results = [_result("q-1", True, 0.1), _result("q-2", True, 0.1)]
    report = build_evidence_report(
        _manifest(),
        {"nm": results},
        total_elapsed={"nm": 0.2},
    )
    results[0].retrieved_session_ids.append("mutated-after-build")

    assert report["methods"]["nm"]["per_question"][0]["retrieved_session_ids"] == ["answer-q-1"]
    save_evidence_report(report, tmp_path, timestamp="collision")
    with pytest.raises(FileExistsError):
        save_evidence_report(report, tmp_path, timestamp="collision")


def test_evidence_report_rolls_back_when_publish_fails(tmp_path: Path) -> None:
    report = build_evidence_report(
        _manifest(),
        {"nm": [_result("q-1", True, 0.1), _result("q-2", True, 0.1)]},
        total_elapsed={"nm": 0.2},
    )
    real_link = os.link
    calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        real_link(source, destination)

    with (
        patch("scripts.benchmark.report.os.link", side_effect=fail_second_link),
        pytest.raises(OSError, match="injected"),
    ):
        save_evidence_report(report, tmp_path, timestamp="partial")

    assert not (tmp_path / "evidence_v1_partial.json").exists()
    assert not (tmp_path / "evidence_v1_partial.md").exists()
    assert not list(tmp_path.glob(".evidence_v1_partial.*.tmp"))


def test_cascade_percentile_basic() -> None:
    assert percentile([], 50) is None
    assert percentile([10.0], 95) == 10.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0


def test_aggregate_route_latencies_inconclusive_under_min_samples() -> None:
    rows = [
        {"route": "exact", "stage": "candidate_exit", "latency_ms": 12.0},
        {"route": "causal", "stage": "bounded_graph", "latency_ms": 80.0},
    ]
    report = aggregate_route_latencies(rows, min_samples=5)
    assert report["gates"]["simple_p95_ms"]["status"] == "inconclusive"
    assert report["gates"]["cognitive_p95_ms"]["status"] == "inconclusive"
    assert report["all_gates_pass"] is False
    assert report["routes"]["exact"]["samples"] == 1


def test_aggregate_route_latencies_pass_fail_gates() -> None:
    simple_rows = [
        {"cascade_route": "exact", "cascade_stage": "candidate_exit", "latency_ms": 20.0 + i}
        for i in range(8)
    ]
    cognitive_rows = [
        {"route": "causal", "stage": "bounded_graph", "latency_ms": 40.0 + i} for i in range(8)
    ]
    report = aggregate_route_latencies(simple_rows + cognitive_rows, min_samples=5)
    assert report["gates"]["simple_p95_ms"]["status"] == "pass"
    assert report["gates"]["cognitive_p95_ms"]["status"] == "pass"
    assert report["all_gates_pass"] is True
    assert report["simple"]["p95_ms"] is not None
    assert report["simple"]["p95_ms"] <= SIMPLE_P95_MS
    assert report["cognitive"]["p95_ms"] <= COGNITIVE_P95_MS


def test_aggregate_route_latencies_fail_when_over_budget() -> None:
    rows = [
        {"route": "exact", "stage": "candidate_exit", "latency_ms": 200.0 + i} for i in range(10)
    ]
    report = aggregate_route_latencies(rows, min_samples=5)
    assert report["gates"]["simple_p95_ms"]["status"] == "fail"
    assert report["all_gates_pass"] is False
