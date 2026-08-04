from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from scripts.benchmark.config import BenchmarkConfig
from scripts.benchmark.data_loader import LMEInstance
from scripts.benchmark.evidence import (
    bootstrap_interval,
    build_evidence_manifest,
)


def _instance(question_id: str) -> LMEInstance:
    return LMEInstance(
        question_id=question_id,
        question_type="single-session-user",
        question="What happened?",
        answer="An event",
        question_date="2024/01/01 (Mon) 00:00",
    )


def test_build_evidence_manifest_is_complete_deterministic_and_frozen(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_bytes(b'{"dataset":"fixed"}\n')
    config = BenchmarkConfig(
        variant="s",
        limit=2,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "results",
        retrieval_only=True,
    )
    instances = [_instance("q-2"), _instance("q-1")]

    first = build_evidence_manifest(
        config,
        instances,
        corpus_path=corpus_path,
        seed=17,
        warmup_runs=2,
        package_names=("neural-memory", "package-that-does-not-exist"),
        run_config={"methods": ["nm", "naive"], "top_k": 10},
    )
    second = build_evidence_manifest(
        config,
        instances,
        corpus_path=corpus_path,
        seed=17,
        warmup_runs=2,
        package_names=("neural-memory", "package-that-does-not-exist"),
        run_config={"methods": ["nm", "naive"], "top_k": 10},
    )

    assert first == second
    assert first.schema_version == 1
    assert first.corpus_sha256 == hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    assert first.instance_ids == ("q-2", "q-1")
    assert first.seed == 17
    assert first.warmup_runs == 2
    assert first.config["data_dir"] == str(tmp_path / "data")
    assert first.config["methods"] == ("nm", "naive")
    assert first.config["top_k"] == 10
    assert first.packages["package-that-does-not-exist"] == "unknown"
    assert first.git_sha
    assert isinstance(first.git_dirty, bool)
    assert {"platform", "python", "cpu_count", "ram_bytes"} <= first.hardware.keys()

    with pytest.raises(FrozenInstanceError):
        first.seed = 23  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable"):
        first.config["top_k"] = 5


def test_build_evidence_manifest_uses_explicit_git_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unavailable(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr("scripts.benchmark.evidence.subprocess.run", unavailable)
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text("[]", encoding="utf-8")

    manifest = build_evidence_manifest(
        BenchmarkConfig(),
        [_instance("q-1")],
        corpus_path=corpus_path,
    )

    assert manifest.git_sha == "unknown"
    assert manifest.git_dirty is False


def test_bootstrap_interval_is_seeded_and_handles_degenerate_samples() -> None:
    values = [0.0, 0.25, 0.5, 0.75, 1.0]

    first = bootstrap_interval(values, seed=42, samples=500)
    second = bootstrap_interval(values, seed=42, samples=500)

    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0
    assert bootstrap_interval([0.75], seed=42) == (0.75, 0.75)
    assert bootstrap_interval([], seed=42) == (0.0, 0.0)
