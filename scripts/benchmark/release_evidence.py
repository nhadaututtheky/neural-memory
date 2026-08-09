"""Candidate-SHA release evidence manifest (Phase 8).

Binds every optimization claim to immutable, hash-addressed artifacts generated
from a specific commit. Fail closed on stale SHA, dirty tree, missing gates,
tampered artifact hashes, NaN metrics, or inconclusive samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

# Gates that must be true for a release-ready manifest.
REQUIRED_GATES: tuple[str, ...] = (
    "pre_ship",
    "e2e_integrity",
    "ablation_justified",
    "footprint",
    "storage_adapter",
    "write_change_log",
    "outbox_restart",
    "checkpoint_resume",
)

# Scorecard metric thresholds (Phase 8 plan).
METRIC_FLOORS: dict[str, float] = {
    "curated_recall_at_5": 0.466,
    "curated_ndcg_at_5": 0.464,
    "base_dependency_count_max": 4.0,
    "standard_tool_count": 10.0,
    "simple_recall_p95_ms_max": 100.0,
    "cognitive_recall_p95_ms_max": 250.0,
    "write_ack_p95_ms_max": 75.0,
    "vector_convergence_p95_s_max": 5.0,
}


@dataclass(frozen=True)
class ReleaseEvidence:
    """Immutable release evidence document."""

    schema_version: int
    git_sha: str
    git_dirty: bool
    generated_at: datetime
    gates: dict[str, bool]
    artifacts: dict[str, str]  # logical name -> sha256
    metrics: dict[str, float]
    notes: tuple[str, ...] = ()
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "generated_at": self.generated_at.replace(tzinfo=None).isoformat() + "Z"
            if self.generated_at.tzinfo
            else self.generated_at.isoformat() + "Z",
            "gates": dict(sorted(self.gates.items())),
            "artifacts": dict(sorted(self.artifacts.items())),
            "artifact_paths": dict(sorted(self.artifact_paths.items())),
            "metrics": {k: self.metrics[k] for k in sorted(self.metrics)},
            "notes": list(self.notes),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def git_metadata(repo_root: Path) -> tuple[str, bool]:
    git = shutil.which("git")
    if git is None:
        return "unknown", False
    try:
        sha = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [git, "status", "--porcelain"],
                cwd=repo_root,
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return "unknown", False
    return sha or "unknown", dirty


def _finite_metric(value: object, *, label: str, failures: list[str]) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        failures.append(f"metric {label} must be a number")
        return None
    number = float(value)
    if not math.isfinite(number):
        failures.append(f"metric {label} must be finite (no NaN/Inf)")
        return None
    return number


def build_release_evidence(
    *,
    git_sha: str,
    git_dirty: bool,
    gates: Mapping[str, bool],
    artifacts: Mapping[str, str],
    metrics: Mapping[str, float],
    artifact_paths: Mapping[str, str] | None = None,
    notes: Sequence[str] = (),
    generated_at: datetime | None = None,
) -> ReleaseEvidence:
    """Construct a ReleaseEvidence document (does not validate thresholds)."""
    return ReleaseEvidence(
        schema_version=SCHEMA_VERSION,
        git_sha=git_sha,
        git_dirty=git_dirty,
        generated_at=generated_at or datetime.now(UTC).replace(tzinfo=None),
        gates={str(k): bool(v) for k, v in gates.items()},
        artifacts={str(k): str(v) for k, v in artifacts.items()},
        metrics={str(k): float(v) for k, v in metrics.items()},
        notes=tuple(notes),
        artifact_paths={str(k): str(v) for k, v in (artifact_paths or {}).items()},
    )


def write_release_evidence(evidence: ReleaseEvidence, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_release_evidence(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"non-finite JSON constant in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("release evidence root must be an object")
    return raw


def verify_release_evidence(
    path: Path,
    expected_sha: str,
    *,
    require_clean: bool = True,
    repo_root: Path | None = None,
    require_all_gates: bool = True,
    check_artifact_files: bool = True,
) -> list[str]:
    """Verify release evidence against candidate SHA and artifact hashes.

    Returns a list of failure messages (empty = pass).
    """
    failures: list[str] = []
    try:
        data = load_release_evidence(path)
    except ValueError as exc:
        return [str(exc)]

    if data.get("schema_version") != SCHEMA_VERSION:
        failures.append(
            f"schema_version must be {SCHEMA_VERSION}, got {data.get('schema_version')!r}"
        )

    git_sha = data.get("git_sha")
    if not isinstance(git_sha, str) or _GIT_SHA_PATTERN.fullmatch(git_sha) is None:
        failures.append("git_sha must be a 40-character hex SHA")
    elif expected_sha and git_sha.lower() != expected_sha.lower():
        failures.append(f"stale evidence: manifest git_sha={git_sha} != expected {expected_sha}")

    if require_clean and data.get("git_dirty") is not False:
        failures.append("canonical release evidence requires git_dirty=false")

    gates = data.get("gates")
    if not isinstance(gates, dict):
        failures.append("gates must be an object")
        gates = {}
    else:
        for name in REQUIRED_GATES:
            if name not in gates:
                failures.append(f"missing required gate '{name}'")
            elif gates[name] is not True and require_all_gates:
                failures.append(f"gate '{name}' is not true")
        for key, value in gates.items():
            if not isinstance(key, str) or not isinstance(value, bool):
                failures.append(f"gate {key!r} must be a bool")

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        failures.append("artifacts map must be a non-empty object")
        artifacts = {}
    else:
        for name, digest in artifacts.items():
            if not isinstance(name, str):
                failures.append("artifact names must be strings")
                continue
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                failures.append(f"artifact '{name}' hash must be 64-char sha256")

    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        failures.append("metrics must be an object")
        metrics = {}
    else:
        for key, value in metrics.items():
            _finite_metric(value, label=str(key), failures=failures)

    # Threshold checks when metrics present
    r5 = metrics.get("curated_recall_at_5") if isinstance(metrics, dict) else None
    n5 = metrics.get("curated_ndcg_at_5") if isinstance(metrics, dict) else None
    if isinstance(r5, (int, float)) and float(r5) <= METRIC_FLOORS["curated_recall_at_5"]:
        failures.append(
            f"curated_recall_at_5={r5} must exceed {METRIC_FLOORS['curated_recall_at_5']}"
        )
    if isinstance(n5, (int, float)) and float(n5) <= METRIC_FLOORS["curated_ndcg_at_5"]:
        failures.append(f"curated_ndcg_at_5={n5} must exceed {METRIC_FLOORS['curated_ndcg_at_5']}")

    deps = metrics.get("base_dependency_count") if isinstance(metrics, dict) else None
    if isinstance(deps, (int, float)) and float(deps) > METRIC_FLOORS["base_dependency_count_max"]:
        failures.append(
            f"base_dependency_count={deps} exceeds max {METRIC_FLOORS['base_dependency_count_max']}"
        )

    tools = metrics.get("standard_tool_count") if isinstance(metrics, dict) else None
    if isinstance(tools, (int, float)) and float(tools) != METRIC_FLOORS["standard_tool_count"]:
        failures.append(
            f"standard_tool_count={tools} must equal {METRIC_FLOORS['standard_tool_count']}"
        )

    # Verify on-disk artifact hashes when paths provided
    paths = data.get("artifact_paths")
    if check_artifact_files and isinstance(paths, dict) and isinstance(artifacts, dict):
        root = repo_root or path.parent
        for name, rel in paths.items():
            if name not in artifacts:
                failures.append(f"artifact_paths entry '{name}' missing from artifacts")
                continue
            file_path = Path(rel)
            if not file_path.is_absolute():
                # Prefer path relative to repo root, then relative to evidence file
                candidate = (root / file_path) if root else file_path
                if not candidate.is_file():
                    candidate = path.parent / file_path
                file_path = candidate
            if not file_path.is_file():
                failures.append(f"artifact file missing for '{name}': {rel}")
                continue
            actual = sha256_file(file_path)
            if actual.lower() != str(artifacts[name]).lower():
                failures.append(
                    f"artifact '{name}' hash mismatch: manifest={artifacts[name][:12]}… "
                    f"file={actual[:12]}…"
                )

    return list(dict.fromkeys(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify release evidence")
    parser.add_argument(
        "--verify",
        type=Path,
        help="Path to release-evidence.json to verify",
    )
    parser.add_argument(
        "--expected-sha",
        default="",
        help="Candidate git SHA (default: HEAD)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Do not fail on git_dirty=true",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--build",
        type=Path,
        help="Build a release-evidence.json at this path from --inputs",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        help="JSON describing gates/artifacts/metrics for --build",
    )
    args = parser.parse_args(argv)

    sha = args.expected_sha
    if not sha:
        sha, _ = git_metadata(args.repo_root)

    if args.verify:
        failures = verify_release_evidence(
            args.verify,
            expected_sha=sha,
            require_clean=not args.allow_dirty,
            repo_root=args.repo_root,
        )
        if failures:
            print(f"FAIL {args.verify} ({len(failures)} issue(s))", file=sys.stderr)
            for item in failures:
                print(f"  - {item}", file=sys.stderr)
            return 1
        print(f"PASS {args.verify} (sha={sha[:12]})")
        return 0

    if args.build:
        if not args.inputs or not args.inputs.is_file():
            print("--build requires --inputs JSON", file=sys.stderr)
            return 2
        payload = json.loads(args.inputs.read_text(encoding="utf-8"))
        dirty = bool(payload.get("git_dirty", False))
        if "git_dirty" not in payload:
            _, dirty = git_metadata(args.repo_root)
        evidence = build_release_evidence(
            git_sha=str(payload.get("git_sha") or sha),
            git_dirty=dirty,
            gates=payload.get("gates") or {},
            artifacts=payload.get("artifacts") or {},
            metrics=payload.get("metrics") or {},
            artifact_paths=payload.get("artifact_paths") or {},
            notes=payload.get("notes") or (),
        )
        write_release_evidence(evidence, args.build)
        print(f"Wrote {args.build}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
