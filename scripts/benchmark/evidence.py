"""Reproducible evidence primitives for benchmark runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Never, cast

from scripts.benchmark.metrics import (
    QuestionResult,
    compute_metrics_by_type,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_retrieval_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from scripts.benchmark.config import BenchmarkConfig
    from scripts.benchmark.data_loader import LMEInstance


SCHEMA_VERSION = 1
DEFAULT_PACKAGES = ("neural-memory", "numpy", "sentence-transformers")
REQUIRED_SAME_RUN_METHODS = ("naive", "fts5", "embedding")
GATED_METRICS = ("recall_at_5", "ndcg_at_5")
PINNED_VARIANT = "s"
PINNED_BACKEND = "sqlite"
PINNED_TOP_K = 10
PINNED_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PINNED_EMBEDDING_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
PINNED_RETRIEVAL_PROFILE = "longmemeval-reflex-v1"
# Canonical pin is the Git-blob / LF bytes of the source artifact.
# Never pin a Windows autocrlf (CRLF) checkout hash — CI checks out LF.
PINNED_SOURCE_ARTIFACT_SHA256 = "80da95d7dabe772f494932d21c6940345254f68b6accf394e817d1b917e83522"
PINNED_SOURCE_PROVENANCE = "legacy_regression_anchor"
PINNED_QUALITY_FLOORS = {"recall_at_5": 0.466, "ndcg_at_5": 0.464}
PINNED_LAST_GOOD = {"recall_at_5": 0.84, "ndcg_at_5": 0.6253950480719784}
RETRIEVAL_METRICS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
)
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}", re.IGNORECASE)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)


class FrozenDict(dict[str, object]):
    """A dict-compatible immutable mapping for frozen manifest fields."""

    def _immutable(self, *args: object, **kwargs: object) -> Never:
        raise TypeError("evidence manifest mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return FrozenDict({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class EvidenceManifest:
    """Inputs and environment metadata needed to reproduce an evidence run."""

    schema_version: int
    git_sha: str
    git_dirty: bool
    corpus_sha256: str
    instance_ids: tuple[str, ...]
    seed: int
    config: dict[str, object]
    packages: dict[str, str]
    hardware: dict[str, object]
    warmup_runs: int


@dataclass(frozen=True)
class EvidenceGateResult:
    """Outcome of applying the pinned evidence contract to one report."""

    passed: bool
    failures: tuple[str, ...]


def _mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    if not all(isinstance(key, str) for key in value):
        return None
    return cast("dict[str, object]", value)


def _string_sequence(value: object) -> list[str] | None:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def _number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def canonical_json_sha256(value: dict[str, object]) -> str:
    """Hash a JSON contract independently of whitespace and object key order."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def ground_truth_sha256(
    rows: Sequence[tuple[str, str, Sequence[str]]],
) -> str:
    """Hash the ordered question IDs, types, and answer-session IDs."""
    payload = [
        {
            "question_id": question_id,
            "question_type": question_type,
            "answer_session_ids": list(answer_session_ids),
        }
        for question_id, question_type, answer_session_ids in rows
    ]
    return canonical_json_sha256({"rows": payload})


def _validated_metric(
    value: object,
    *,
    label: str,
    failures: list[str],
) -> float | None:
    metric = _number(value)
    if metric is None:
        failures.append(f"{label} must be a finite number")
        return None
    if not 0.0 <= metric <= 1.0:
        failures.append(f"{label} must be between 0 and 1")
        return None
    return metric


def _parse_question_rows(
    method: str,
    method_data: dict[str, object],
    expected_ids: list[str] | None,
    failures: list[str],
) -> list[QuestionResult]:
    rows = method_data.get("per_question")
    if not isinstance(rows, list):
        failures.append(f"required method '{method}' per_question rows are missing or malformed")
        return []

    parsed: list[QuestionResult] = []
    row_ids: list[str] = []
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row)
        if row is None:
            failures.append(f"required method '{method}' per_question row {index} is malformed")
            continue
        question_id = row.get("question_id")
        question_type = row.get("question_type")
        retrieved_ids = _string_sequence(row.get("retrieved_session_ids"))
        answer_ids = _string_sequence(row.get("answer_session_ids"))
        elapsed = _number(row.get("elapsed_sec"))
        retrieval_hit = row.get("retrieval_hit")
        if not isinstance(question_id, str) or not isinstance(question_type, str):
            failures.append(f"required method '{method}' per_question row {index} has invalid IDs")
            continue
        if retrieved_ids is None or answer_ids is None:
            failures.append(
                f"required method '{method}' per_question row {index} has invalid sessions"
            )
            continue
        if elapsed is None or elapsed < 0.0:
            failures.append(
                f"required method '{method}' per_question elapsed_sec must be finite non-negative"
            )
            continue
        expected_hit = any(session_id in retrieved_ids for session_id in answer_ids)
        if not isinstance(retrieval_hit, bool) or retrieval_hit != expected_hit:
            failures.append(
                f"required method '{method}' per_question row {index} has inconsistent retrieval_hit"
            )
            continue
        correct_value = row.get("correct")
        correct = correct_value if isinstance(correct_value, bool) else None
        row_ids.append(question_id)
        parsed.append(
            QuestionResult(
                question_id=question_id,
                question_type=question_type,
                hypothesis=str(row.get("hypothesis", "")),
                correct=correct,
                retrieved_session_ids=retrieved_ids,
                answer_session_ids=answer_ids,
                retrieval_hit=retrieval_hit,
                elapsed_sec=elapsed,
            )
        )
    if row_ids != expected_ids:
        failures.append(
            f"required method '{method}' per_question IDs do not match pinned ordered sample"
        )
    return list(parsed)


def _validate_raw_timing_rows(
    method: str,
    method_data: dict[str, object],
    expected_ids: list[str] | None,
    question_results: list[QuestionResult],
    failures: list[str],
) -> None:
    rows = method_data.get("raw_timing_rows")
    if not isinstance(rows, list):
        failures.append(f"required method '{method}' raw timing rows are missing or malformed")
        return

    timing_ids: list[str] = []
    elapsed_values: list[float] = []
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row)
        if row is None or not isinstance(row.get("question_id"), str):
            failures.append(f"required method '{method}' raw timing row {index} is malformed")
            continue
        elapsed = _number(row.get("elapsed_sec"))
        if elapsed is None or elapsed < 0.0:
            failures.append(
                f"required method '{method}' raw timing elapsed_sec must be finite non-negative"
            )
            continue
        timing_ids.append(cast("str", row["question_id"]))
        elapsed_values.append(elapsed)
    if timing_ids != expected_ids:
        failures.append(
            f"required method '{method}' raw timing IDs do not match pinned ordered sample"
        )

    question_elapsed = [result.elapsed_sec for result in question_results]
    if len(elapsed_values) == len(question_elapsed) and any(
        not math.isclose(raw, question, rel_tol=0.0, abs_tol=1e-12)
        for raw, question in zip(elapsed_values, question_elapsed, strict=True)
    ):
        failures.append(f"required method '{method}' raw timing rows differ from per_question rows")


def _validate_confidence_intervals(
    method: str,
    method_data: dict[str, object],
    question_results: list[QuestionResult],
    seed: int,
    failures: list[str],
) -> None:
    intervals = _mapping(method_data.get("confidence_intervals"))
    if intervals is None:
        failures.append(f"required method '{method}' confidence intervals are missing or malformed")
        return
    samples = {
        "recall_at_1": [compute_recall_at_k([result], 1) for result in question_results],
        "recall_at_3": [compute_recall_at_k([result], 3) for result in question_results],
        "recall_at_5": [compute_recall_at_k([result], 5) for result in question_results],
        "recall_at_10": [compute_recall_at_k([result], 10) for result in question_results],
        "ndcg_at_5": [compute_ndcg_at_k([result], 5) for result in question_results],
        "ndcg_at_10": [compute_ndcg_at_k([result], 10) for result in question_results],
        "elapsed_sec": [result.elapsed_sec for result in question_results],
    }
    for metric, values in samples.items():
        interval = intervals.get(metric)
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            failures.append(f"required method '{method}' has invalid {metric} confidence interval")
            continue
        lower = _number(interval[0])
        upper = _number(interval[1])
        if lower is None or upper is None or lower > upper:
            failures.append(
                f"required method '{method}' {metric} confidence interval must be finite and ordered"
            )
        elif metric != "elapsed_sec" and not (0.0 <= lower <= upper <= 1.0):
            failures.append(
                f"required method '{method}' {metric} confidence interval must be between 0 and 1"
            )
        elif metric == "elapsed_sec" and lower < 0.0:
            failures.append(
                f"required method '{method}' elapsed confidence interval must be non-negative"
            )
        else:
            expected_lower, expected_upper = bootstrap_interval(values, seed=seed)
            if not math.isclose(
                lower,
                expected_lower,
                rel_tol=0.0,
                abs_tol=1e-12,
            ) or not math.isclose(
                upper,
                expected_upper,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                failures.append(
                    f"required method '{method}' {metric} confidence interval "
                    "does not match recomputed raw rows"
                )


def _validate_by_type(
    method: str,
    method_data: dict[str, object],
    question_results: list[QuestionResult],
    failures: list[str],
) -> None:
    actual_by_type = _mapping(method_data.get("by_type"))
    if actual_by_type is None:
        failures.append(f"required method '{method}' by_type metrics are missing or malformed")
        return

    expected_by_type = compute_metrics_by_type(question_results)
    if set(actual_by_type) != set(expected_by_type):
        failures.append(f"required method '{method}' by_type groups do not match raw rows")
        return
    for question_type, expected_metrics in expected_by_type.items():
        actual_metrics = _mapping(actual_by_type.get(question_type))
        if actual_metrics is None:
            failures.append(f"required method '{method}' by_type row is malformed")
            continue
        for metric, expected in expected_metrics.items():
            actual = actual_metrics.get(metric)
            if math.isnan(expected):
                if actual is not None and not (isinstance(actual, float) and math.isnan(actual)):
                    failures.append(f"required method '{method}' by_type does not match raw rows")
                continue
            actual_number = _number(actual)
            if actual_number is None or not math.isclose(
                actual_number,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                failures.append(f"required method '{method}' by_type does not match raw rows")


def _validate_method(
    method: str,
    methods: dict[str, object],
    expected_ids: list[str] | None,
    expected_ground_truth_sha256: str | None,
    seed: int,
    failures: list[str],
) -> dict[str, float]:
    method_data = _mapping(methods.get(method))
    if method_data is None:
        failures.append(f"required method '{method}' is missing or malformed")
        return {}
    method_ids = _string_sequence(method_data.get("instance_ids"))
    if method_ids != expected_ids:
        failures.append(
            f"required method '{method}' instance IDs do not match pinned ordered sample"
        )

    question_results = _parse_question_rows(method, method_data, expected_ids, failures)
    method_ground_truth_sha256 = ground_truth_sha256(
        [
            (result.question_id, result.question_type, result.answer_session_ids)
            for result in question_results
        ]
    )
    if method_data.get("ground_truth_sha256") != method_ground_truth_sha256:
        failures.append(f"required method '{method}' ground truth hash is inconsistent")
    if method_ground_truth_sha256 != expected_ground_truth_sha256:
        failures.append(f"required method '{method}' ground truth does not match pinned sample")
    _validate_raw_timing_rows(method, method_data, expected_ids, question_results, failures)
    _validate_confidence_intervals(method, method_data, question_results, seed, failures)
    _validate_by_type(method, method_data, question_results, failures)

    summary = _mapping(method_data.get("summary"))
    if summary is None:
        failures.append(f"required method '{method}' has no valid summary")
        return {}
    for metric in RETRIEVAL_METRICS:
        _validated_metric(
            summary.get(metric), label=f"{method} summary {metric}", failures=failures
        )
    for timing_key in ("total_elapsed_sec", "mean_elapsed_sec"):
        timing = _number(summary.get(timing_key))
        if timing is None or timing < 0.0:
            failures.append(f"{method} summary {timing_key} must be finite non-negative")
    if summary.get("instances") != len(expected_ids or []):
        failures.append(f"{method} summary instances does not match pinned sample")
    expected_mean_elapsed = (
        statistics.fmean(result.elapsed_sec for result in question_results)
        if question_results
        else 0.0
    )
    mean_elapsed = _number(summary.get("mean_elapsed_sec"))
    if mean_elapsed is None or not math.isclose(
        mean_elapsed,
        expected_mean_elapsed,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        failures.append(f"{method} summary mean_elapsed_sec does not match raw rows")

    if len(question_results) != len(expected_ids or []):
        return {}
    computed = compute_retrieval_metrics(question_results)
    computed_metrics = {metric: float(getattr(computed, metric)) for metric in RETRIEVAL_METRICS}
    for metric, computed_value in computed_metrics.items():
        summary_value = _number(summary.get(metric))
        if summary_value is None or not math.isclose(
            summary_value, computed_value, rel_tol=0.0, abs_tol=1e-12
        ):
            failures.append(
                f"{method} summary {metric} does not match recomputed per_question value"
            )
    return computed_metrics


def _validate_baseline_contract(
    baseline: dict[str, object],
    failures: list[str],
) -> None:
    expected_scalars: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "variant": PINNED_VARIANT,
        "backend": PINNED_BACKEND,
        "top_k": PINNED_TOP_K,
        "embedding_model": PINNED_EMBEDDING_MODEL,
        "embedding_revision": PINNED_EMBEDDING_REVISION,
        "retrieval_profile": PINNED_RETRIEVAL_PROFILE,
        "source_artifact_sha256": PINNED_SOURCE_ARTIFACT_SHA256,
    }
    for key, expected in expected_scalars.items():
        if baseline.get(key) != expected:
            failures.append(f"baseline {key} does not match the pinned evidence contract")
    if baseline.get("source_provenance") != PINNED_SOURCE_PROVENANCE:
        failures.append("baseline source provenance must be a legacy regression anchor")
    if baseline.get("quality_floors") != PINNED_QUALITY_FLOORS:
        failures.append("baseline quality floors do not match the pinned evidence contract")
    if baseline.get("metrics") != PINNED_LAST_GOOD:
        failures.append("baseline last-good metrics do not match the pinned evidence contract")
    if baseline.get("required_same_run_methods") != list(REQUIRED_SAME_RUN_METHODS):
        failures.append("baseline required_same_run_methods must be exactly naive, fts5, embedding")
    source_git_sha = baseline.get("source_git_sha")
    if not isinstance(source_git_sha, str) or _GIT_SHA_PATTERN.fullmatch(source_git_sha) is None:
        failures.append("baseline source_git_sha must be a known 40-character Git SHA")
    if not isinstance(baseline.get("source_git_sha_basis"), str):
        failures.append("baseline must declare source_git_sha_basis")
    missing_metadata = _string_sequence(baseline.get("legacy_metadata_missing"))
    required_missing_metadata = {
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
    }
    if missing_metadata is None:
        failures.append("baseline must declare legacy_metadata_missing")
    elif not required_missing_metadata.issubset(missing_metadata):
        failures.append("baseline must declare every unverified legacy runtime field")
    ground_truth_hash = baseline.get("ground_truth_sha256")
    if (
        not isinstance(ground_truth_hash, str)
        or _SHA256_PATTERN.fullmatch(ground_truth_hash) is None
    ):
        failures.append("baseline ground_truth_sha256 must be a 64-character SHA-256")


def evaluate_evidence(report: dict[str, object], baseline: dict[str, object]) -> EvidenceGateResult:
    """Fail closed unless one report satisfies every absolute and relative gate."""
    failures: list[str] = []
    _validate_baseline_contract(baseline, failures)

    report_schema = report.get("schema_version")
    baseline_schema = baseline.get("schema_version")
    if report_schema != baseline_schema:
        failures.append(f"schema mismatch: report={report_schema!r}, baseline={baseline_schema!r}")
    if report.get("canonical") is not True:
        failures.append("report is non-canonical")
    reasons = _string_sequence(report.get("non_canonical_reasons"))
    if reasons is None or reasons:
        failures.append(f"canonical report has non-canonical reasons: {reasons!r}")
    try:
        baseline_sha256 = canonical_json_sha256(baseline)
    except (TypeError, ValueError):
        baseline_sha256 = "invalid"
        failures.append("baseline must contain finite JSON values")
    if report.get("baseline_sha256") != baseline_sha256:
        failures.append("report baseline SHA-256 does not match evaluated baseline")

    manifest = _mapping(report.get("manifest"))
    if manifest is None:
        failures.append("report manifest is missing or malformed")
        manifest = {}
    config = _mapping(manifest.get("config")) or {}
    if manifest.get("git_dirty") is not False:
        failures.append("canonical evidence requires a clean Git worktree")
    git_sha = manifest.get("git_sha")
    if not isinstance(git_sha, str) or _GIT_SHA_PATTERN.fullmatch(git_sha) is None:
        failures.append("canonical evidence requires a known 40-character Git SHA")
    if config.get("variant") != baseline.get("variant"):
        failures.append(
            f"variant mismatch: report={config.get('variant')!r}, "
            f"baseline={baseline.get('variant')!r}"
        )
    if config.get("retrieval_only") is not True:
        failures.append("report is not declared retrieval-only")
    if manifest.get("seed") != baseline.get("seed"):
        failures.append(
            f"seed mismatch: report={manifest.get('seed')!r}, baseline={baseline.get('seed')!r}"
        )
    if config.get("top_k") != baseline.get("top_k"):
        failures.append(
            f"top_k mismatch: report={config.get('top_k')!r}, baseline={baseline.get('top_k')!r}"
        )
    if config.get("backend") != baseline.get("backend"):
        failures.append(
            f"backend mismatch: report={config.get('backend')!r}, "
            f"baseline={baseline.get('backend')!r}"
        )
    if config.get("embedding_model") != baseline.get("embedding_model"):
        failures.append("embedding model does not match pinned baseline")
    if config.get("embedding_revision") != baseline.get("embedding_revision"):
        failures.append("embedding model revision does not match pinned baseline")
    if config.get("retrieval_profile") != baseline.get("retrieval_profile"):
        failures.append("retrieval profile does not match pinned baseline")
    expected_ground_truth_sha256 = baseline.get("ground_truth_sha256")
    if config.get("ground_truth_sha256") != expected_ground_truth_sha256:
        failures.append("manifest ground truth does not match pinned baseline")
    if manifest.get("corpus_sha256") != baseline.get("corpus_sha256"):
        failures.append("corpus SHA-256 does not match pinned baseline")

    report_ids = _string_sequence(manifest.get("instance_ids"))
    baseline_ids = _string_sequence(baseline.get("instance_ids"))
    if baseline_ids is None or not baseline_ids or len(baseline_ids) != len(set(baseline_ids)):
        failures.append("baseline instance_ids must be non-empty and unique")
    if report_ids is None or baseline_ids is None or report_ids != baseline_ids:
        failures.append(
            f"instance IDs do not match pinned ordered sample: "
            f"report={report_ids!r}, baseline={baseline_ids!r}"
        )

    methods = _mapping(report.get("methods"))
    if methods is None:
        failures.append("report methods are missing or malformed")
        methods = {}
    configured_methods = _string_sequence(baseline.get("required_same_run_methods"))
    if configured_methods != list(REQUIRED_SAME_RUN_METHODS):
        failures.append("baseline required_same_run_methods must be exactly naive, fts5, embedding")
    required_methods = list(REQUIRED_SAME_RUN_METHODS)

    quality_floors = _mapping(baseline.get("quality_floors"))
    last_good_metrics = _mapping(baseline.get("metrics"))
    if quality_floors is None:
        failures.append("baseline quality_floors is missing or malformed")
        quality_floors = {}
    if last_good_metrics is None:
        failures.append("baseline metrics are missing or malformed")
        last_good_metrics = {}

    computed_by_method = {
        method: _validate_method(
            method,
            methods,
            baseline_ids,
            expected_ground_truth_sha256 if isinstance(expected_ground_truth_sha256, str) else None,
            manifest.get("seed")
            if isinstance(manifest.get("seed"), int) and not isinstance(manifest.get("seed"), bool)
            else 0,
            failures,
        )
        for method in ["nm", *required_methods]
    }

    for metric in GATED_METRICS:
        nm_value = computed_by_method["nm"].get(metric)
        floor = _validated_metric(
            quality_floors.get(metric),
            label=f"baseline quality floor {metric}",
            failures=failures,
        )
        last_good = _validated_metric(
            last_good_metrics.get(metric),
            label=f"baseline last-good {metric}",
            failures=failures,
        )
        if floor is not None and last_good is not None and floor >= last_good:
            failures.append(f"baseline quality floor {metric} must be below last-good")
        if floor is not None and nm_value is not None and nm_value <= floor:
            failures.append(f"nm {metric}={nm_value:.6f} must exceed absolute floor {floor:.6f}")
        if last_good is not None and nm_value is not None and nm_value < last_good:
            failures.append(f"nm {metric}={nm_value:.6f} regressed below last-good {last_good:.6f}")

        for method in required_methods:
            baseline_value = computed_by_method[method].get(metric)
            if nm_value is not None and baseline_value is not None and nm_value <= baseline_value:
                failures.append(
                    f"nm {metric}={nm_value:.6f} must beat same-run {method} "
                    f"{metric}={baseline_value:.6f}"
                )

    unique_failures = tuple(dict.fromkeys(failures))
    return EvidenceGateResult(passed=not unique_failures, failures=unique_failures)


def _json_value(value: object) -> object:
    """Convert config values to stable JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _git_metadata(repo_root: Path) -> tuple[str, bool]:
    git_executable = shutil.which("git")
    if git_executable is None:
        return "unknown", False
    try:
        sha_result = subprocess.run(
            [git_executable, "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
        dirty_result = subprocess.run(
            [git_executable, "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return "unknown", False
    return sha_result.stdout.strip() or "unknown", bool(dirty_result.stdout.strip())


def _package_versions(package_names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    try:
        from neural_memory import __version__
    except ImportError:
        versions["neural-memory-source"] = "unknown"
    else:
        versions["neural-memory-source"] = __version__
    return versions


def _ram_bytes() -> int | str:
    """Return physical RAM when psutil is available, otherwise an explicit fallback."""
    try:
        import psutil
    except ImportError:
        return "unknown"
    return int(psutil.virtual_memory().total)


def _hardware_metadata() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or "unknown",
        "ram_bytes": _ram_bytes(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as corpus_file:
        for chunk in iter(lambda: corpus_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_evidence_manifest(
    config: BenchmarkConfig,
    instances: Sequence[LMEInstance],
    *,
    corpus_path: Path,
    seed: int = 0,
    warmup_runs: int = 0,
    package_names: Iterable[str] = DEFAULT_PACKAGES,
    repo_root: Path | None = None,
    run_config: Mapping[str, object] | None = None,
) -> EvidenceManifest:
    """Build deterministic provenance metadata for one fixed benchmark input set."""
    root = repo_root or Path(__file__).resolve().parents[2]
    git_sha, git_dirty = _git_metadata(root)
    config_values = {key: _json_value(value) for key, value in sorted(asdict(config).items())}
    if run_config is not None:
        config_values.update({key: _json_value(value) for key, value in sorted(run_config.items())})
    config_values["ground_truth_sha256"] = ground_truth_sha256(
        [
            (instance.question_id, instance.question_type, instance.answer_session_ids)
            for instance in instances
        ]
    )
    packages = _package_versions(package_names)
    hardware = _hardware_metadata()
    return EvidenceManifest(
        schema_version=SCHEMA_VERSION,
        git_sha=git_sha,
        git_dirty=git_dirty,
        corpus_sha256=_sha256_file(corpus_path),
        instance_ids=tuple(instance.question_id for instance in instances),
        seed=seed,
        config=cast("dict[str, object]", _freeze(config_values)),
        packages=cast("dict[str, str]", _freeze(packages)),
        hardware=cast("dict[str, object]", _freeze(hardware)),
        warmup_runs=warmup_runs,
    )


def bootstrap_interval(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = 2_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for the sample mean."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    sample_size = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(sample_size)] for _ in range(sample_size))
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, int(tail * samples))
    upper_index = min(samples - 1, int((1.0 - tail) * samples) - 1)
    return (float(means[lower_index]), float(means[upper_index]))
