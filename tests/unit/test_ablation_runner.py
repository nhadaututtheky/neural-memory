"""Unit tests for default-stage ablation evidence (Phase 8)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.benchmark.run_ablation import (
    DEFAULT_STAGES,
    StageMetrics,
    default_fixture_measurements,
    evaluate_stage,
    main,
    parse_stage_metrics,
    run_ablation_matrix,
    write_ablation_report,
)


def test_default_stages_cover_required_matrix() -> None:
    assert "lexical" in DEFAULT_STAGES
    assert "vector" in DEFAULT_STAGES
    assert "graph" in DEFAULT_STAGES
    assert "priming" in DEFAULT_STAGES
    assert "reconsolidation" in DEFAULT_STAGES


def test_quality_gain_keeps_stage() -> None:
    result = evaluate_stage(
        "priming",
        StageMetrics(0.8, 100.0, 1000.0),
        StageMetrics(0.7, 95.0, 900.0),
    )
    assert result.keeps_default is True
    assert result.justification == "quality_gain"
    assert result.quality_delta == pytest.approx(0.1)


def test_latency_gain_keeps_stage() -> None:
    result = evaluate_stage(
        "priming",
        StageMetrics(0.7, 80.0, 1000.0),
        StageMetrics(0.7, 120.0, 1000.0),
    )
    assert result.keeps_default is True
    assert result.justification == "latency_gain"


def test_no_value_disables_optional_stage() -> None:
    result = evaluate_stage(
        "priming",
        StageMetrics(0.7, 100.0, 1000.0),
        StageMetrics(0.7, 100.0, 1000.0),
    )
    assert result.keeps_default is False
    assert result.justification == "no_measured_value"


def test_required_correctness_always_keeps() -> None:
    result = evaluate_stage(
        "lexical",
        StageMetrics(0.5, 100.0, 1000.0),
        StageMetrics(0.9, 50.0, 500.0),  # "better" without stage — still required
    )
    assert result.keeps_default is True
    assert result.required_correctness is True
    assert result.justification == "required_correctness"


def test_unavailable_vector_cannot_justify_default() -> None:
    result = evaluate_stage(
        "vector",
        StageMetrics(0.7, 100.0, 1000.0),
        StageMetrics(0.0, 0.0, 0.0, available=False, note="no embeddings"),
    )
    assert result.keeps_default is False
    assert "unavailable" in result.justification


def test_nan_metrics_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        parse_stage_metrics(
            {"quality": math.nan, "latency_ms": 1.0, "context_tokens": 1.0},
            label="bad",
        )


def test_fixture_matrix_all_justified() -> None:
    report = run_ablation_matrix(default_fixture_measurements(), seed=7, git_sha="a" * 40)
    assert report.all_justified is True
    assert not report.failures
    assert {s.stage for s in report.stages} == set(DEFAULT_STAGES)
    vector = next(s for s in report.stages if s.stage == "vector")
    assert vector.justification == "unavailable_optional_cannot_justify_default"


def test_missing_stage_fails_closed() -> None:
    measurements = default_fixture_measurements()
    del measurements["priming"]
    report = run_ablation_matrix(measurements)
    assert report.all_justified is False
    assert any("priming" in f for f in report.failures)


def test_write_and_cli_fixture(tmp_path: Path) -> None:
    out = tmp_path / "ablation.json"
    code = main(["--fixture", "--seed", "1", "--git-sha", "b" * 40, "--output", str(out)])
    assert code == 0
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["all_justified"] is True
    assert data["schema_version"] == 1
    assert len(data["stages"]) == len(DEFAULT_STAGES)


def test_write_ablation_report_roundtrip(tmp_path: Path) -> None:
    report = run_ablation_matrix(default_fixture_measurements())
    path = tmp_path / "a.json"
    write_ablation_report(report, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["all_justified"] is report.all_justified
