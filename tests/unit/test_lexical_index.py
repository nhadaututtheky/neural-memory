"""Tests for the in-memory BM25 lexical index.

Item #1 from plan-tllr-learnings: lexical retrieval as a parallel
candidate source alongside semantic activation. Pure-Python BM25 (no
new heavy dep) — index is incrementally built and queried in <15ms at
10k neurons.
"""

from __future__ import annotations

from neural_memory.engine.lexical_index import LexicalIndex
from neural_memory.engine.tokenizers import WhitespaceTokenizer


def _build_index() -> LexicalIndex:
    return LexicalIndex(tokenizer=WhitespaceTokenizer())


def test_empty_index_returns_no_matches() -> None:
    idx = _build_index()
    assert idx.search("anything", limit=5) == []


def test_add_document_makes_it_findable() -> None:
    idx = _build_index()
    idx.add_document("doc1", "neural memory provenance footer")
    results = idx.search("provenance", limit=5)
    assert results
    assert results[0][0] == "doc1"


def test_search_returns_descending_score() -> None:
    idx = _build_index()
    idx.add_document("doc1", "rare term provenance")
    idx.add_document("doc2", "rare term provenance provenance provenance")
    idx.add_document("doc3", "completely unrelated content")
    results = idx.search("provenance", limit=5)
    assert [r[0] for r in results[:2]] == ["doc2", "doc1"]
    assert results[0][1] >= results[1][1]


def test_search_excludes_irrelevant_docs() -> None:
    idx = _build_index()
    idx.add_document("doc1", "neural memory")
    idx.add_document("doc2", "completely unrelated")
    results = idx.search("memory", limit=5)
    assert "doc2" not in {r[0] for r in results}


def test_search_respects_limit() -> None:
    idx = _build_index()
    for i in range(20):
        idx.add_document(f"doc{i}", f"shared term content {i}")
    results = idx.search("shared", limit=5)
    assert len(results) == 5


def test_search_case_insensitive_via_tokenizer() -> None:
    idx = _build_index()
    idx.add_document("doc1", "Provenance Footer")
    results = idx.search("PROVENANCE", limit=5)
    assert results and results[0][0] == "doc1"


def test_idf_penalizes_common_terms() -> None:
    """Rare terms should outrank common ones in BM25 scoring."""
    idx = _build_index()
    idx.add_document("rare", "memory unique zebra")
    for i in range(50):
        idx.add_document(f"common{i}", "memory standard noise")
    rare_results = idx.search("zebra", limit=5)
    common_results = idx.search("standard", limit=5)
    # Rare doc match score should be much higher than the common-term winners.
    assert rare_results[0][1] > common_results[0][1]


def test_remove_document_excludes_from_results() -> None:
    idx = _build_index()
    idx.add_document("doc1", "provenance footer")
    idx.add_document("doc2", "provenance metadata")
    idx.remove_document("doc1")
    results = idx.search("provenance", limit=5)
    ids = {r[0] for r in results}
    assert "doc1" not in ids
    assert "doc2" in ids


def test_remove_unknown_id_no_error() -> None:
    """Removing a document that was never added is a no-op."""
    idx = _build_index()
    idx.remove_document("ghost")  # must not raise


def test_size_tracks_document_count() -> None:
    idx = _build_index()
    assert idx.size == 0
    idx.add_document("doc1", "a")
    idx.add_document("doc2", "b")
    assert idx.size == 2
    idx.remove_document("doc1")
    assert idx.size == 1


def test_re_add_document_replaces_content() -> None:
    """Re-adding the same id should replace, not duplicate."""
    idx = _build_index()
    idx.add_document("doc1", "old content here")
    idx.add_document("doc1", "new keyword here")
    results_old = idx.search("old", limit=5)
    results_new = idx.search("keyword", limit=5)
    assert not results_old
    assert results_new and results_new[0][0] == "doc1"
    assert idx.size == 1


def test_search_empty_query_returns_no_matches() -> None:
    idx = _build_index()
    idx.add_document("doc1", "anything")
    assert idx.search("", limit=5) == []


def test_search_query_with_only_punctuation_returns_no_matches() -> None:
    idx = _build_index()
    idx.add_document("doc1", "real content")
    assert idx.search(",,, !!", limit=5) == []
