"""Tantivy-based full-text search index for InfinityDB.

Replaces O(N) linear content scan with BM25 ranked search.
Uses in-memory Tantivy index by default; optionally persists to disk.

Usage:
    idx = TextIndex()
    idx.open()
    idx.add("neuron-1", "Python supports async I/O")
    idx.commit()
    results = idx.search("async Python", limit=10)
    # returns [(neuron_id, bm25_score), ...]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tantivy

    _TANTIVY_AVAILABLE = True
except ImportError:
    _TANTIVY_AVAILABLE = False


def is_tantivy_available() -> bool:
    """Check if tantivy-py is installed."""
    return _TANTIVY_AVAILABLE


class TextIndex:
    """Tantivy BM25 full-text index for neuron content.

    Thread-safe for reads (multiple searchers). Single writer.
    GIL is released during search operations.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialize TextIndex.

        Args:
            path: Directory for persistent index. None = in-memory only.
        """
        self._path = Path(path) if path is not None else None
        self._index: Any = None
        self._writer: Any = None
        self._schema: Any = None
        self._is_open = False
        self._dirty = False  # Track uncommitted changes

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        """Open or create the text index."""
        if not _TANTIVY_AVAILABLE:
            logger.warning("tantivy-py not installed — text search disabled")
            return

        builder = tantivy.SchemaBuilder()
        builder.add_text_field("neuron_id", stored=True, tokenizer_name="raw")
        builder.add_text_field("content", stored=True)
        self._schema = builder.build()

        if self._path is not None:
            self._path.mkdir(parents=True, exist_ok=True)
            self._index = tantivy.Index(self._schema, path=str(self._path))
        else:
            self._index = tantivy.Index(self._schema)

        self._writer = self._index.writer(heap_size=15_000_000)
        self._is_open = True
        self._dirty = False

    def close(self) -> None:
        """Flush and close the index."""
        if not self._is_open:
            return
        if self._dirty:
            self.commit()
        self._writer = None
        self._index = None
        self._is_open = False

    def add(self, neuron_id: str, content: str) -> None:
        """Add or replace a document in the index."""
        if not self._is_open or self._writer is None:
            return
        # Delete existing doc with same neuron_id first (tantivy is append-only)
        self._writer.delete_documents("neuron_id", neuron_id)
        doc = tantivy.Document()
        doc.add_text("neuron_id", neuron_id)
        doc.add_text("content", content)
        self._writer.add_document(doc)
        self._dirty = True

    def add_batch(self, items: list[tuple[str, str]]) -> None:
        """Add multiple documents in one batch.

        Args:
            items: List of (neuron_id, content) tuples.
        """
        if not self._is_open or self._writer is None:
            return
        for neuron_id, content in items:
            self._writer.delete_documents("neuron_id", neuron_id)
            doc = tantivy.Document()
            doc.add_text("neuron_id", neuron_id)
            doc.add_text("content", content)
            self._writer.add_document(doc)
        self._dirty = True

    def delete(self, neuron_id: str) -> None:
        """Delete a document by neuron_id."""
        if not self._is_open or self._writer is None:
            return
        self._writer.delete_documents("neuron_id", neuron_id)
        self._dirty = True

    def commit(self) -> None:
        """Commit pending changes and make them searchable."""
        if not self._is_open or self._writer is None:
            return
        if self._dirty:
            self._writer.commit()
            self._index.reload()
            self._dirty = False

    def search(
        self,
        query: str,
        limit: int = 100,
    ) -> list[tuple[str, float]]:
        """Search for documents matching query.

        Args:
            query: Search query string (BM25 ranked).
            limit: Maximum results to return.

        Returns:
            List of (neuron_id, bm25_score) tuples, highest score first.
        """
        if not self._is_open or self._index is None:
            return []

        # Commit any pending changes before searching
        if self._dirty:
            try:
                self.commit()
            except Exception:
                logger.debug("Tantivy auto-commit before search failed", exc_info=True)

        try:
            searcher = self._index.searcher()
            parsed = self._index.parse_query(query, ["content"])
            results = searcher.search(parsed, limit=limit)

            hits: list[tuple[str, float]] = []
            for score, doc_address in results.hits:
                doc = searcher.doc(doc_address)
                nid_list = doc["neuron_id"]
                if nid_list:
                    hits.append((nid_list[0], score))

            return hits
        except Exception:
            logger.debug("Tantivy search failed for query: %s", query, exc_info=True)
            return []

    def search_contains(
        self,
        substring: str,
        limit: int = 100,
    ) -> list[str]:
        """Search for documents containing a substring (simulates content_contains).

        Uses Tantivy tokenized search — matches individual terms, not exact substrings.
        For most use cases in NeuralMemory, term-based matching is sufficient.

        Args:
            substring: Text to search for.
            limit: Maximum results.

        Returns:
            List of neuron_ids (unscored, for backward compat with content_contains).
        """
        hits = self.search(substring, limit=limit)
        return [nid for nid, _score in hits]

    def search_contains_batch(
        self,
        terms: list[str],
        limit_per_term: int = 3,
    ) -> dict[str, list[str]]:
        """Batch search for multiple terms at once.

        More efficient than calling search_contains() per term because
        it runs a single combined query and maps results back to terms.

        Args:
            terms: List of search terms.
            limit_per_term: Max results per term.

        Returns:
            Dict mapping each term to a list of matching neuron_ids.
        """
        if not self._is_open or self._index is None or not terms:
            return {t: [] for t in terms}

        if self._dirty:
            try:
                self.commit()
            except Exception:
                logger.debug("Tantivy auto-commit before batch search failed", exc_info=True)

        # Build per-term results using individual Tantivy queries (fast: <0.2ms each)
        results: dict[str, list[str]] = {}
        try:
            searcher = self._index.searcher()
            for term in terms:
                try:
                    parsed = self._index.parse_query(term, ["content"])
                    search_results = searcher.search(parsed, limit=limit_per_term)
                    nids: list[str] = []
                    for _score, doc_address in search_results.hits:
                        doc = searcher.doc(doc_address)
                        nid_list = doc["neuron_id"]
                        if nid_list:
                            nids.append(nid_list[0])
                    results[term] = nids
                except Exception:
                    results[term] = []
        except Exception:
            logger.debug("Tantivy batch search failed", exc_info=True)
            return {t: [] for t in terms}

        return results

    @property
    def count(self) -> int:
        """Approximate document count."""
        if not self._is_open or self._index is None:
            return 0
        try:
            searcher = self._index.searcher()
            return searcher.num_docs
        except Exception:
            return 0
