"""In-memory BM25 lexical index over neuron content.

Pure-Python BM25 (Okapi variant). Pluggable tokenizer so Vietnamese,
English, and code identifiers all index correctly via the same path.

The index is incrementally maintained: `add_document` and
`remove_document` keep the document-frequency tables in sync without
a full rebuild. At ~10k neurons the index is ~10MB of memory and
search returns top-K in <15ms.

Used by the retrieval pipeline as a parallel candidate source — its
ranks are fed into the existing RRF fusion in `engine/score_fusion.py`.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from neural_memory.engine.tokenizers import Tokenizer, WhitespaceTokenizer

# Standard Okapi BM25 hyperparameters.
_BM25_K1: float = 1.5  # term-frequency saturation
_BM25_B: float = 0.75  # length normalization
MAX_BM25_LIMIT: int = 200
"""Hard cap on `search(limit=...)` to bound worst-case CPU per query.

Prevents a misconfigured `BrainConfig.bm25_limit` from triggering a
full-corpus scan + sort. 200 is well above any realistic anchor-pool
size; callers needing more should reach for direct semantic search.
"""


@dataclass
class _Document:
    tokens: tuple[str, ...]
    length: int
    term_freq: dict[str, int] = field(default_factory=dict)


class LexicalIndex:
    """Append-only BM25 index keyed by neuron / document id."""

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer: Tokenizer = tokenizer or WhitespaceTokenizer()
        self._docs: dict[str, _Document] = {}
        self._doc_freq: Counter[str] = Counter()
        self._total_length: int = 0

    @property
    def size(self) -> int:
        return len(self._docs)

    @property
    def _avg_length(self) -> float:
        if not self._docs:
            return 0.0
        return self._total_length / len(self._docs)

    def add_document(self, doc_id: str, content: str) -> None:
        """Insert or replace a document.

        Re-adding an existing id is a replace, not a duplicate — avoids
        leaking stale tokens into the document-frequency tables when a
        neuron's content gets edited.
        """
        if doc_id in self._docs:
            self.remove_document(doc_id)

        tokens = self._tokenizer.tokenize(content)
        if not tokens:
            return
        tf = Counter(tokens)
        doc = _Document(
            tokens=tuple(tokens),
            length=len(tokens),
            term_freq=dict(tf),
        )
        self._docs[doc_id] = doc
        self._total_length += doc.length
        for term in tf:
            self._doc_freq[term] += 1

    def remove_document(self, doc_id: str) -> None:
        """Drop a document from the index. No-op if unknown."""
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return
        self._total_length -= doc.length
        for term in doc.term_freq:
            self._doc_freq[term] -= 1
            if self._doc_freq[term] <= 0:
                del self._doc_freq[term]

    def search(self, query: str, *, limit: int = 30) -> list[tuple[str, float]]:
        """Return (doc_id, score) pairs sorted by BM25 score, descending.

        Empty / punctuation-only queries return no matches. Documents
        that share zero terms with the query get filtered out before
        ranking — keeps the result list focused. ``limit`` is capped at
        ``MAX_BM25_LIMIT`` to bound worst-case CPU.
        """
        if not self._docs or limit <= 0:
            return []
        limit = min(limit, MAX_BM25_LIMIT)

        query_terms = self._tokenizer.tokenize(query)
        if not query_terms:
            return []

        n = len(self._docs)
        avg_len = self._avg_length
        # Pre-compute IDF for each query term (BM25+ smoothing avoids
        # negative IDFs on terms in >50% of corpus).
        idf_per_term: dict[str, float] = {}
        for term in set(query_terms):
            df = self._doc_freq.get(term, 0)
            idf_per_term[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

        scores: dict[str, float] = {}
        for doc_id, doc in self._docs.items():
            score = 0.0
            for term in query_terms:
                tf = doc.term_freq.get(term, 0)
                if tf == 0:
                    continue
                idf = idf_per_term.get(term, 0.0)
                norm = 1.0 - _BM25_B + _BM25_B * (doc.length / avg_len if avg_len else 1.0)
                score += idf * (tf * (_BM25_K1 + 1)) / (tf + _BM25_K1 * norm)
            if score > 0:
                scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]
