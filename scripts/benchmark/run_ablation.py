"""Default retrieval-stage ablation evidence (Phase 8).

Toggles one default stage at a time against a fixed baseline configuration.
A stage may remain default only if it shows quality gain, latency gain, or is
marked required for correctness. Optional vector stages may report N/A when
embeddings are unavailable — never used to justify a default vector stage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Stages that ship enabled by default and must appear in the ablation matrix.
DEFAULT_STAGES: tuple[str, ...] = (
    "lexical",  # FTS / keyword / BM25 candidate generation
    "vector",  # optional embedding retrieval
    "graph",  # bounded graph cognition after sufficiency gate
    "priming",  # co-activation / predictive priming
    "reconsolidation",  # post-recall reconsolidation writes
)

# Stages that must not be disabled without an explicit correctness waiver.
REQUIRED_CORRECTNESS_STAGES: frozenset[str] = frozenset({"lexical", "graph"})

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StageMetrics:
    """Quality + latency snapshot for one ablation condition."""

    quality: float  # higher is better (e.g. R@5 or composite)
    latency_ms: float  # lower is better (p95)
    context_tokens: float  # lower is better when quality equal
    available: bool = True
    note: str = ""


@dataclass(frozen=True)
class StageAblationResult:
    """Comparison of full baseline vs stage-disabled condition."""

    stage: str
    baseline: StageMetrics
    disabled: StageMetrics
    quality_delta: float
    latency_delta_ms: float
    token_delta: float
    keeps_default: bool
    justification: str
    required_correctness: bool


@dataclass
class AblationReport:
    """Full matrix for all default stages on one fixed seed/config."""

    schema_version: int = SCHEMA_VERSION
    seed: int = 0
    git_sha: str = "unknown"
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z"
    )
    stages: list[StageAblationResult] = field(default_factory=list)
    all_justified: bool = False
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "git_sha": self.git_sha,
            "generated_at": self.generated_at,
            "all_justified": self.all_justified,
            "failures": list(self.failures),
            "stages": [
                {
                    "stage": s.stage,
                    "baseline": asdict(s.baseline),
                    "disabled": asdict(s.disabled),
                    "quality_delta": s.quality_delta,
                    "latency_delta_ms": s.latency_delta_ms,
                    "token_delta": s.token_delta,
                    "keeps_default": s.keeps_default,
                    "justification": s.justification,
                    "required_correctness": s.required_correctness,
                }
                for s in self.stages
            ],
        }


def _finite(value: float, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite (got {value!r})")
    return number


def parse_stage_metrics(raw: Mapping[str, Any], *, label: str) -> StageMetrics:
    """Parse and validate stage metrics; reject NaN/non-finite."""
    available = bool(raw.get("available", True))
    if not available:
        return StageMetrics(
            quality=0.0,
            latency_ms=0.0,
            context_tokens=0.0,
            available=False,
            note=str(raw.get("note") or "unavailable"),
        )
    return StageMetrics(
        quality=_finite(float(raw["quality"]), name=f"{label}.quality"),
        latency_ms=_finite(float(raw["latency_ms"]), name=f"{label}.latency_ms"),
        context_tokens=_finite(float(raw["context_tokens"]), name=f"{label}.context_tokens"),
        available=True,
        note=str(raw.get("note") or ""),
    )


def evaluate_stage(
    stage: str,
    baseline: StageMetrics,
    disabled: StageMetrics,
    *,
    quality_eps: float = 1e-9,
    latency_eps_ms: float = 0.05,
) -> StageAblationResult:
    """Decide whether a default stage is justified by the ablation.

    Rules (any one is enough to keep default):
    - quality improves with the stage enabled (baseline > disabled)
    - latency improves with the stage enabled (baseline < disabled)
    - stage is required for correctness (lexical/graph)
    - stage is optional/unavailable (vector without embeddings) → N/A, cannot
      justify enabling as a default (keeps_default=False unless required)
    """
    required = stage in REQUIRED_CORRECTNESS_STAGES
    if not disabled.available or not baseline.available:
        # Optional stage missing capability — must not justify a default enable.
        keeps = required  # required stages cannot be N/A
        reason = (
            "required_correctness" if required else "unavailable_optional_cannot_justify_default"
        )
        return StageAblationResult(
            stage=stage,
            baseline=baseline,
            disabled=disabled,
            quality_delta=0.0,
            latency_delta_ms=0.0,
            token_delta=0.0,
            keeps_default=keeps,
            justification=reason,
            required_correctness=required,
        )

    quality_delta = baseline.quality - disabled.quality
    latency_delta_ms = disabled.latency_ms - baseline.latency_ms  # positive = stage faster
    token_delta = disabled.context_tokens - baseline.context_tokens

    quality_gain = quality_delta > quality_eps
    latency_gain = latency_delta_ms > latency_eps_ms
    # Token reduction with non-worse quality counts as efficiency gain
    token_gain = token_delta > 0 and quality_delta >= -quality_eps

    if required:
        keeps = True
        justification = "required_correctness"
    elif quality_gain:
        keeps = True
        justification = "quality_gain"
    elif latency_gain:
        keeps = True
        justification = "latency_gain"
    elif token_gain:
        keeps = True
        justification = "context_token_gain"
    else:
        keeps = False
        justification = "no_measured_value"

    return StageAblationResult(
        stage=stage,
        baseline=baseline,
        disabled=disabled,
        quality_delta=quality_delta,
        latency_delta_ms=latency_delta_ms,
        token_delta=token_delta,
        keeps_default=keeps,
        justification=justification,
        required_correctness=required,
    )


def run_ablation_matrix(
    measurements: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = 0,
    git_sha: str = "unknown",
    stages: Sequence[str] = DEFAULT_STAGES,
) -> AblationReport:
    """Evaluate ablation for every default stage from fixed measurements.

    ``measurements`` keys:
      - ``baseline``: full default stack
      - each stage name: metrics with that stage disabled
    """
    report = AblationReport(seed=seed, git_sha=git_sha)
    if "baseline" not in measurements:
        report.failures.append("missing baseline measurements")
        return report

    try:
        baseline = parse_stage_metrics(measurements["baseline"], label="baseline")
    except (KeyError, TypeError, ValueError) as exc:
        report.failures.append(f"invalid baseline: {exc}")
        return report

    for stage in stages:
        if stage not in measurements:
            report.failures.append(f"missing ablation for stage '{stage}'")
            continue
        try:
            disabled = parse_stage_metrics(measurements[stage], label=stage)
        except (KeyError, TypeError, ValueError) as exc:
            report.failures.append(f"invalid metrics for stage '{stage}': {exc}")
            continue
        result = evaluate_stage(stage, baseline, disabled)
        report.stages.append(result)
        if not result.keeps_default and stage in stages:
            # Vector may be optional-off by default when unavailable — that is OK
            # only when justification is N/A (not silently enabled).
            if result.justification == "unavailable_optional_cannot_justify_default":
                continue
            report.failures.append(
                f"stage '{stage}' lacks quality/latency/correctness justification"
            )

    expected = set(stages)
    seen = {s.stage for s in report.stages}
    if expected - seen:
        for missing in sorted(expected - seen):
            if f"missing ablation for stage '{missing}'" not in report.failures:
                report.failures.append(f"missing ablation for stage '{missing}'")

    report.all_justified = not report.failures and len(report.stages) == len(stages)
    return report


def default_fixture_measurements() -> dict[str, dict[str, Any]]:
    """Deterministic fixture used by unit tests and offline release packaging.

    Models the post Phase 4 cascade stack: lexical+graph required; vector
    optional (N/A without embedding extras); priming/reconsolidation show
    modest quality or correctness contribution.
    """
    baseline = {"quality": 0.72, "latency_ms": 95.0, "context_tokens": 1200.0}
    return {
        "baseline": dict(baseline),
        # Without lexical, quality collapses
        "lexical": {"quality": 0.20, "latency_ms": 80.0, "context_tokens": 400.0},
        # Vector optional — unavailable in base install
        "vector": {
            "available": False,
            "note": "embedding extras not installed; cannot justify default vector stage",
        },
        # Without graph, cognitive queries lose quality
        "graph": {"quality": 0.55, "latency_ms": 70.0, "context_tokens": 900.0},
        # Priming improves quality slightly
        "priming": {"quality": 0.68, "latency_ms": 90.0, "context_tokens": 1150.0},
        # Reconsolidation is correctness/lifecycle (quality similar, required? no)
        # Show small quality gain when enabled
        "reconsolidation": {"quality": 0.70, "latency_ms": 94.0, "context_tokens": 1180.0},
    }


def write_ablation_report(report: AblationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run default-stage ablation matrix")
    parser.add_argument(
        "--measurements",
        type=Path,
        help="JSON file with baseline + per-stage-disabled metrics",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Use built-in deterministic fixture (no live retrieval)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/benchmark/results/ablation.json"),
    )
    args = parser.parse_args(argv)

    if args.measurements:
        raw = json.loads(args.measurements.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            print("measurements file must be a JSON object", file=sys.stderr)
            return 2
        measurements = raw
    elif args.fixture:
        measurements = default_fixture_measurements()
    else:
        print("Provide --measurements PATH or --fixture", file=sys.stderr)
        return 2

    report = run_ablation_matrix(
        measurements,
        seed=args.seed,
        git_sha=args.git_sha,
    )
    write_ablation_report(report, args.output)
    print(f"Wrote {args.output} all_justified={report.all_justified}")
    if report.failures:
        for failure in report.failures:
            print(f"  FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
