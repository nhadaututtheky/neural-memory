from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from scripts.benchmark.evidence import EvidenceGateResult
from scripts.benchmark.longmemeval import _acquire_output_lock, _safe_instance_path
from scripts.benchmark.run_evidence import (
    _attach_evidence_gate,
    _nm_environment,
    _safe_result_path,
)
from scripts.benchmark.run_instance import (
    _safe_database_path,
    _verify_pinned_embedding_revision,
    _verify_vector_index,
)


def test_nm_result_path_cannot_escape_unique_temporary_directory(tmp_path: Path) -> None:
    result_path = _safe_result_path(tmp_path, "../../outside/victim")

    assert result_path.parent == tmp_path.resolve()
    assert result_path.suffix == ".json"
    assert result_path.resolve().is_relative_to(tmp_path.resolve())
    assert "victim" not in result_path.name


def test_nm_database_path_is_opaque_and_isolated_per_run(tmp_path: Path) -> None:
    first_root = tmp_path / "run-1"
    second_root = tmp_path / "run-2"

    first = _safe_database_path(first_root, "../../outside/victim")
    second = _safe_database_path(second_root, "../../outside/victim")

    assert first.parent == first_root.resolve()
    assert first.suffix == ".db"
    assert first.resolve().is_relative_to(first_root.resolve())
    assert "victim" not in first.name
    assert first.name == second.name
    assert first != second


def test_legacy_instance_paths_cannot_escape_output_root(tmp_path: Path) -> None:
    result_path = _safe_instance_path(tmp_path, "../../outside/victim", ".json")

    assert result_path.parent == tmp_path.resolve()
    assert result_path.resolve().is_relative_to(tmp_path.resolve())
    assert "victim" not in result_path.name


def test_legacy_runner_exclusively_locks_output_directory(tmp_path: Path) -> None:
    first_lock = _acquire_output_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already using output directory"):
            _acquire_output_lock(tmp_path)
    finally:
        first_lock.close()

    replacement_lock = _acquire_output_lock(tmp_path)
    replacement_lock.close()


def test_nm_strict_embedding_is_scoped_to_pinned_sqlite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NMEM_REQUIRE_EMBEDDING", "inherited")

    sqlite_environment = _nm_environment(42, tmp_path, "sqlite")
    infinity_environment = _nm_environment(42, tmp_path, "infinitydb")

    assert sqlite_environment["NMEM_REQUIRE_EMBEDDING"] == "1"
    assert "NMEM_REQUIRE_EMBEDDING" not in infinity_environment


@pytest.mark.asyncio
async def test_nm_revision_preflight_loads_pinned_cached_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AsyncMock()
    monkeypatch.setenv("NMEM_SENTENCE_TRANSFORMER_REVISION", "pinned-revision")

    with patch(
        "neural_memory.engine.semantic_discovery._create_provider",
        return_value=provider,
    ) as create_provider:
        await _verify_pinned_embedding_revision()

    assert create_provider.call_args.args[0].embedding_model == "all-MiniLM-L6-v2"
    provider.embed.assert_awaited_once_with("nmem benchmark revision probe")


@pytest.mark.asyncio
async def test_strict_benchmark_embedding_step_propagates_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neural_memory.engine.steps.embedding_step import EmbeddingStep

    monkeypatch.setenv("NMEM_REQUIRE_EMBEDDING", "1")
    provider = AsyncMock()
    provider.embed.side_effect = RuntimeError("embedding failed")
    step = EmbeddingStep(provider)
    context = SimpleNamespace(anchor_neuron=SimpleNamespace(id="neuron-1", content="content"))

    with pytest.raises(RuntimeError, match="embedding failed"):
        await step.execute(context, AsyncMock(), SimpleNamespace())


@pytest.mark.asyncio
async def test_normal_embedding_step_keeps_graceful_fallback() -> None:
    from neural_memory.engine.steps.embedding_step import EmbeddingStep

    provider = AsyncMock()
    provider.embed.side_effect = RuntimeError("embedding failed")
    step = EmbeddingStep(provider)
    context = SimpleNamespace(anchor_neuron=SimpleNamespace(id="neuron-1", content="content"))

    result = await step.execute(context, AsyncMock(), SimpleNamespace())

    assert result is context


@pytest.mark.asyncio
async def test_strict_benchmark_retrieval_propagates_query_embedding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neural_memory.engine.retrieval import ReflexPipeline

    monkeypatch.setenv("NMEM_REQUIRE_EMBEDDING", "1")
    provider = AsyncMock()
    provider.embed.side_effect = RuntimeError("query embedding failed")
    pipeline = SimpleNamespace(_embedding_provider=provider)

    with pytest.raises(RuntimeError, match="query embedding failed"):
        await ReflexPipeline._find_embedding_anchors(pipeline, "query")


@pytest.mark.asyncio
async def test_strict_benchmark_retrieval_propagates_vector_search_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neural_memory.engine.retrieval import ReflexPipeline

    monkeypatch.setenv("NMEM_REQUIRE_EMBEDDING", "1")
    provider = AsyncMock()
    provider.embed.return_value = [0.1, 0.2]
    storage = AsyncMock()
    storage.knn_search.side_effect = RuntimeError("query index failed")
    pipeline = SimpleNamespace(
        _embedding_provider=provider,
        _storage=storage,
        _config=SimpleNamespace(embedding_similarity_threshold=0.5),
    )

    with pytest.raises(RuntimeError, match="query index failed"):
        await ReflexPipeline._find_embedding_anchors(pipeline, "query")


@pytest.mark.asyncio
async def test_vector_index_preflight_requires_written_vectors(tmp_path: Path) -> None:
    provider = AsyncMock()
    provider.embed.return_value = [0.1, 0.2]
    storage = AsyncMock()
    storage.knn_search.return_value = []

    with (
        patch(
            "neural_memory.engine.semantic_discovery._create_provider",
            return_value=provider,
        ),
        patch("scripts.benchmark.longmemeval._open_storage", return_value=storage),
        pytest.raises(RuntimeError, match="no searchable vectors"),
    ):
        await _verify_vector_index(tmp_path / "brain.db", "sqlite", "brain-1")

    storage.set_brain.assert_called_once_with("brain-1")
    storage.knn_search.assert_awaited_once_with([0.1, 0.2], k=1)
    storage.close.assert_awaited_once()


def test_sql_vector_index_strict_mode_rejects_missing_hnswlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neural_memory.storage.sql.mixins.vector_search import VectorSearchMixin

    monkeypatch.setenv("NMEM_REQUIRE_EMBEDDING", "1")
    storage = VectorSearchMixin()
    storage._init_vector_search()

    with (
        patch(
            "neural_memory.engine.embedding.vector_index.is_available",
            return_value=False,
        ),
        pytest.raises(RuntimeError, match="hnswlib"),
    ):
        storage._ensure_vector_index()


def test_failed_gate_cannot_leave_report_marked_canonical() -> None:
    report: dict[str, object] = {
        "canonical": True,
        "non_canonical_reasons": [],
    }
    gate_result = EvidenceGateResult(False, ("quality regression",))

    with patch("scripts.benchmark.run_evidence.evaluate_evidence", return_value=gate_result):
        _attach_evidence_gate(report, {})

    assert report["canonical"] is False
    assert report["non_canonical_reasons"] == ["evidence_gate_failed"]
    assert report["gate"] == {
        "passed": False,
        "failures": ["quality regression"],
    }


@pytest.mark.asyncio
async def test_optional_embedding_runtime_failure_is_non_canonical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.benchmark.config import BenchmarkConfig
    from scripts.benchmark.data_loader import LMEInstance
    from scripts.benchmark.metrics import QuestionResult
    from scripts.benchmark.run_evidence import run_evidence

    instance = LMEInstance(
        question_id="q-1",
        question_type="single-session-user",
        question="Question?",
        answer="Answer",
        question_date="2024/01/01 (Mon) 00:00",
        answer_session_ids=["answer-q-1"],
    )

    async def fake_run_method(
        method: str,
        instances: list[LMEInstance],
        **kwargs: object,
    ) -> tuple[list[QuestionResult], float]:
        if method == "embedding":
            raise OSError("model unavailable")
        return (
            [
                QuestionResult(
                    question_id=item.question_id,
                    question_type=item.question_type,
                    hypothesis="",
                    correct=None,
                    retrieved_session_ids=[],
                    answer_session_ids=item.answer_session_ids,
                    retrieval_hit=False,
                    elapsed_sec=0.01,
                )
                for item in instances
            ],
            0.01,
        )

    corpus_path = tmp_path / "longmemeval_s_cleaned.json"
    corpus_path.write_text("[]", encoding="utf-8")
    baseline_path = Path(__file__).resolve().parents[2] / "scripts/benchmark/evidence_baseline.json"
    monkeypatch.setattr("scripts.benchmark.run_evidence.load_dataset", lambda *args: [instance])
    monkeypatch.setattr("scripts.benchmark.run_evidence._embedding_available", lambda: True)
    monkeypatch.setattr("scripts.benchmark.run_evidence._run_selected_method", fake_run_method)

    report, json_path, markdown_path = await run_evidence(
        BenchmarkConfig(
            variant="s",
            limit=1,
            data_dir=tmp_path,
            output_dir=tmp_path / "results",
            retrieval_only=True,
        ),
        methods=("naive", "embedding"),
        baseline_path=baseline_path,
    )

    assert report["canonical"] is False
    assert "embedding_runtime_unavailable" in report["non_canonical_reasons"]
    assert "method_not_run:embedding" in report["non_canonical_reasons"]
    assert json_path.exists()
    assert markdown_path.exists()
