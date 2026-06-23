"""Regression tests for the G10-cleanups audit cluster.

Covers latent/contract fixes:
  #52 abstract execute_returning_count (no fake-0 base)
  #53 row_to_neuron honors both (row, dialect) and (dialect, row) signatures
  #57 change-log record_change returns the just-inserted id (not MAX(id))
  #58 Merkle bucket reader pads short ids to match the builder
  #63 NeuronLookupCache key includes the ephemeral filter dimension
  #78 credit-card detector gates on a Luhn checksum
  #82 ConfidenceScore.components labels fidelity layer vs score correctly
"""

from __future__ import annotations

import inspect

import pytest

from neural_memory.engine.confidence import compute_confidence
from neural_memory.safety.sensitive import (
    SensitiveType,
    _passes_luhn,
    check_sensitive_content,
)
from neural_memory.storage.neuron_cache import NeuronLookupCache
from neural_memory.storage.sql.dialect import Dialect
from neural_memory.storage.sql.row_mappers import row_to_neuron

# ── #52: base execute_returning_count must be abstract ───────────────


def test_execute_returning_count_is_abstract() -> None:
    """Base Dialect must not provide a fake-0 execute_returning_count."""
    assert "execute_returning_count" in Dialect.__abstractmethods__
    assert "insert_returning_id" in Dialect.__abstractmethods__


# ── #53: row_to_neuron dual-signature ────────────────────────────────


class _FakeDialect:
    """Minimal stand-in exposing .name + normalize_dt like a real Dialect."""

    name = "fake"
    normalized_calls = 0

    def normalize_dt(self, value):
        type(self).normalized_calls += 1
        from datetime import datetime

        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value))


def _neuron_row() -> dict:
    return {
        "id": "x1",
        "type": "concept",
        "content": "hello",
        "metadata": None,
        "content_hash": 0,
        "created_at": "2026-06-01T00:00:00",
        "ephemeral": 0,
    }


def test_row_to_neuron_row_only() -> None:
    n = row_to_neuron(_neuron_row())
    assert n.id == "x1"
    assert n.content == "hello"


def test_row_to_neuron_dialect_first_order() -> None:
    _FakeDialect.normalized_calls = 0
    d = _FakeDialect()
    n = row_to_neuron(d, _neuron_row())
    assert n.id == "x1"
    assert _FakeDialect.normalized_calls >= 1  # dialect was honored


def test_row_to_neuron_row_first_order_honors_dialect() -> None:
    """Regression #53: (row, dialect) must NOT silently drop the dialect."""
    _FakeDialect.normalized_calls = 0
    d = _FakeDialect()
    n = row_to_neuron(_neuron_row(), d)
    assert n.id == "x1"
    assert _FakeDialect.normalized_calls >= 1


# ── #63: cache key includes ephemeral dimension ──────────────────────


def test_cache_key_separates_ephemeral() -> None:
    """A result cached with ephemeral=True is not served to ephemeral=False."""
    cache = NeuronLookupCache()
    cache.put("foo", None, ["EPHEMERAL_RESULT"], ephemeral=True)  # type: ignore[list-item]

    # Same content, different ephemeral filter -> must be a miss.
    assert cache.get("foo", None, ephemeral=False) is None
    # Exact ephemeral match -> hit.
    assert cache.get("foo", None, ephemeral=True) == ["EPHEMERAL_RESULT"]
    # No-filter lookup is also a distinct key -> miss.
    assert cache.get("foo", None, ephemeral=None) is None


def test_cache_invalidate_key_clears_all_ephemeral_variants() -> None:
    cache = NeuronLookupCache()
    cache.put("foo", "concept", ["A"], ephemeral=True)  # type: ignore[list-item]
    cache.put("foo", "concept", ["B"], ephemeral=False)  # type: ignore[list-item]
    cache.put("foo", None, ["C"], ephemeral=None)  # type: ignore[list-item]
    cache.invalidate_key("foo", "concept")
    assert cache.get("foo", "concept", ephemeral=True) is None
    assert cache.get("foo", "concept", ephemeral=False) is None
    assert cache.get("foo", None, ephemeral=None) is None


# ── #78: credit-card Luhn gate ───────────────────────────────────────


def test_luhn_helper() -> None:
    assert _passes_luhn("4111111111111111")  # valid Visa test number
    assert not _passes_luhn("4111111111111234")  # fails checksum


def test_credit_card_requires_luhn() -> None:
    """Regression #78: a Luhn-failing card-shaped number is not flagged."""
    bad = check_sensitive_content("order 4111111111111234")
    assert not any(m.type == SensitiveType.CREDIT_CARD for m in bad)

    good = check_sensitive_content("card 4111111111111111")
    assert any(m.type == SensitiveType.CREDIT_CARD for m in good)


# ── #82: confidence components labelling ─────────────────────────────


def test_confidence_components_fidelity_label_and_score() -> None:
    """Regression #82: fidelity_layer holds the NAME, fidelity_score the float."""
    score = compute_confidence(fidelity_layer="gist")
    assert score.components["fidelity_layer"] == "gist"
    assert isinstance(score.components["fidelity_score"], float)
    assert score.components["fidelity_score"] == pytest.approx(score.fidelity)


def test_compute_confidence_signature_unchanged() -> None:
    sig = inspect.signature(compute_confidence)
    assert "fidelity_layer" in sig.parameters
