"""Unit tests for candidate-bound release evidence (Phase 8)."""

from __future__ import annotations

from pathlib import Path

from scripts.benchmark.release_evidence import (
    REQUIRED_GATES,
    build_release_evidence,
    canonical_json_sha256,
    main,
    sha256_file,
    verify_release_evidence,
    write_release_evidence,
)


def _all_gates_true() -> dict[str, bool]:
    return dict.fromkeys(REQUIRED_GATES, True)


def _write_artifact(path: Path, body: str = '{"ok": true}\n') -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return sha256_file(path)


def test_required_gates_cover_integrity_surface() -> None:
    assert "pre_ship" in REQUIRED_GATES
    assert "e2e_integrity" in REQUIRED_GATES
    assert "ablation_justified" in REQUIRED_GATES
    assert "footprint" in REQUIRED_GATES


def test_build_and_verify_happy_path(tmp_path: Path) -> None:
    artifact = tmp_path / "ablation.json"
    digest = _write_artifact(artifact)
    sha = "a" * 40
    evidence = build_release_evidence(
        git_sha=sha,
        git_dirty=False,
        gates=_all_gates_true(),
        artifacts={"ablation": digest},
        artifact_paths={"ablation": str(artifact)},
        metrics={
            "curated_recall_at_5": 0.84,
            "curated_ndcg_at_5": 0.63,
            "base_dependency_count": 4,
            "standard_tool_count": 10,
        },
        notes=("unit-test fixture",),
    )
    path = tmp_path / "release-evidence.json"
    write_release_evidence(evidence, path)
    failures = verify_release_evidence(
        path,
        expected_sha=sha,
        repo_root=tmp_path,
        check_artifact_files=True,
    )
    assert failures == []


def test_stale_sha_fails(tmp_path: Path) -> None:
    evidence = build_release_evidence(
        git_sha="a" * 40,
        git_dirty=False,
        gates=_all_gates_true(),
        artifacts={"x": "b" * 64},
        metrics={
            "curated_recall_at_5": 0.5,
            "curated_ndcg_at_5": 0.5,
            "base_dependency_count": 4,
            "standard_tool_count": 10,
        },
    )
    path = tmp_path / "ev.json"
    write_release_evidence(evidence, path)
    failures = verify_release_evidence(
        path,
        expected_sha="c" * 40,
        check_artifact_files=False,
    )
    assert any("stale" in f for f in failures)


def test_dirty_tree_fails(tmp_path: Path) -> None:
    evidence = build_release_evidence(
        git_sha="a" * 40,
        git_dirty=True,
        gates=_all_gates_true(),
        artifacts={"x": "b" * 64},
        metrics={
            "curated_recall_at_5": 0.5,
            "curated_ndcg_at_5": 0.5,
            "base_dependency_count": 4,
            "standard_tool_count": 10,
        },
    )
    path = tmp_path / "ev.json"
    write_release_evidence(evidence, path)
    failures = verify_release_evidence(
        path,
        expected_sha="a" * 40,
        require_clean=True,
        check_artifact_files=False,
    )
    assert any("git_dirty" in f for f in failures)


def test_missing_gate_fails(tmp_path: Path) -> None:
    gates = _all_gates_true()
    del gates["e2e_integrity"]
    evidence = build_release_evidence(
        git_sha="a" * 40,
        git_dirty=False,
        gates=gates,
        artifacts={"x": "b" * 64},
        metrics={
            "curated_recall_at_5": 0.5,
            "curated_ndcg_at_5": 0.5,
            "base_dependency_count": 4,
            "standard_tool_count": 10,
        },
    )
    path = tmp_path / "ev.json"
    write_release_evidence(evidence, path)
    failures = verify_release_evidence(
        path,
        expected_sha="a" * 40,
        check_artifact_files=False,
    )
    assert any("e2e_integrity" in f for f in failures)


def test_quality_floor_fails(tmp_path: Path) -> None:
    evidence = build_release_evidence(
        git_sha="a" * 40,
        git_dirty=False,
        gates=_all_gates_true(),
        artifacts={"x": "b" * 64},
        metrics={
            "curated_recall_at_5": 0.1,
            "curated_ndcg_at_5": 0.1,
            "base_dependency_count": 4,
            "standard_tool_count": 10,
        },
    )
    path = tmp_path / "ev.json"
    write_release_evidence(evidence, path)
    failures = verify_release_evidence(
        path,
        expected_sha="a" * 40,
        check_artifact_files=False,
    )
    assert any("curated_recall_at_5" in f for f in failures)


def test_tampered_artifact_fails(tmp_path: Path) -> None:
    artifact = tmp_path / "a.json"
    digest = _write_artifact(artifact, '{"v": 1}\n')
    evidence = build_release_evidence(
        git_sha="a" * 40,
        git_dirty=False,
        gates=_all_gates_true(),
        artifacts={"ablation": digest},
        artifact_paths={"ablation": str(artifact.name)},
        metrics={
            "curated_recall_at_5": 0.5,
            "curated_ndcg_at_5": 0.5,
            "base_dependency_count": 4,
            "standard_tool_count": 10,
        },
    )
    path = tmp_path / "ev.json"
    write_release_evidence(evidence, path)
    # Tamper after hashing
    artifact.write_text('{"v": 2}\n', encoding="utf-8")
    failures = verify_release_evidence(
        path,
        expected_sha="a" * 40,
        repo_root=tmp_path,
        check_artifact_files=True,
    )
    assert any("hash mismatch" in f for f in failures)


def test_nan_in_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    # Strict loader rejects NaN even if someone writes it
    path.write_text(
        '{"schema_version":1,"git_sha":"'
        + "a" * 40
        + '","git_dirty":false,"gates":{},"artifacts":{},"metrics":{"x":NaN}}\n',
        encoding="utf-8",
    )
    failures = verify_release_evidence(path, expected_sha="a" * 40, check_artifact_files=False)
    assert failures


def test_canonical_json_stable() -> None:
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert canonical_json_sha256(a) == canonical_json_sha256(b)


def test_cli_verify(tmp_path: Path) -> None:
    evidence = build_release_evidence(
        git_sha="a" * 40,
        git_dirty=False,
        gates=_all_gates_true(),
        artifacts={"x": "b" * 64},
        metrics={
            "curated_recall_at_5": 0.5,
            "curated_ndcg_at_5": 0.5,
            "base_dependency_count": 4,
            "standard_tool_count": 10,
        },
    )
    path = tmp_path / "ev.json"
    write_release_evidence(evidence, path)
    code = main(["--verify", str(path), "--expected-sha", "a" * 40])
    # Artifact file check may not run without paths — should still pass structure
    assert code == 0
