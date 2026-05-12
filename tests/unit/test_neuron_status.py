"""Tests for Neuron.status lifecycle field.

Item #2 from plan-tllr-learnings: explicit `_status` metadata field
(active/superseded/expired) so retrieval can filter at query time
instead of only deprioritizing via decay/factor penalties.

Metadata-backed (zero schema migration), follows the `_reflex` /
`_grounded` / `_confidence` extension pattern.
"""

from __future__ import annotations

from neural_memory.core.neuron import Neuron, NeuronStatus, NeuronType


def _make() -> Neuron:
    return Neuron(id="n1", type=NeuronType.CONCEPT, content="x")


def test_default_status_is_active() -> None:
    n = _make()
    assert n.status == NeuronStatus.ACTIVE


def test_with_status_sets_metadata_key() -> None:
    n = _make().with_status(NeuronStatus.SUPERSEDED)
    assert n.status == NeuronStatus.SUPERSEDED
    assert n.metadata["_status"] == "superseded"


def test_with_status_immutable() -> None:
    """`with_status` returns a new Neuron, original unchanged."""
    original = _make()
    updated = original.with_status(NeuronStatus.EXPIRED)
    assert original.status == NeuronStatus.ACTIVE
    assert updated.status == NeuronStatus.EXPIRED
    assert original is not updated


def test_with_status_accepts_string() -> None:
    n = _make().with_status("superseded")
    assert n.status == NeuronStatus.SUPERSEDED


def test_with_status_rejects_unknown() -> None:
    """Unknown status raises ValueError — guards against typos."""
    import pytest

    with pytest.raises(ValueError):
        _make().with_status("garbage")


def test_legacy_superseded_flag_maps_to_status() -> None:
    """Pre-existing neurons with `_superseded=True` flag report status=SUPERSEDED.

    Backward compat: don't lose information from the older `_disputed` /
    `_superseded` boolean flags used by `_deprioritize_disputed`.
    """
    n = Neuron(
        id="n1",
        type=NeuronType.CONCEPT,
        content="x",
        metadata={"_superseded": True},
    )
    assert n.status == NeuronStatus.SUPERSEDED


def test_explicit_status_overrides_legacy_flag() -> None:
    """If `_status` is set, it wins over legacy `_superseded` flag."""
    n = Neuron(
        id="n1",
        type=NeuronType.CONCEPT,
        content="x",
        metadata={"_superseded": True, "_status": "active"},
    )
    assert n.status == NeuronStatus.ACTIVE


def test_with_superseded_by_records_winner() -> None:
    n = _make().with_status(NeuronStatus.SUPERSEDED, superseded_by="winner-id")
    assert n.status == NeuronStatus.SUPERSEDED
    assert n.metadata["_superseded_by"] == "winner-id"


def test_status_string_values() -> None:
    """Enum string values are stable contract for storage."""
    assert str(NeuronStatus.ACTIVE) == "active"
    assert str(NeuronStatus.SUPERSEDED) == "superseded"
    assert str(NeuronStatus.EXPIRED) == "expired"


def test_with_status_does_not_drop_other_metadata() -> None:
    n = Neuron(
        id="n1",
        type=NeuronType.CONCEPT,
        content="x",
        metadata={"_confidence": 0.9, "tags": ["a"]},
    )
    updated = n.with_status(NeuronStatus.SUPERSEDED)
    assert updated.metadata["_confidence"] == 0.9
    assert updated.metadata["tags"] == ["a"]
    assert updated.metadata["_status"] == "superseded"


def test_revive_clears_superseded_by_reference() -> None:
    """Item #2 review M1: revive must drop the stale `_superseded_by` key."""
    n = _make().with_status(NeuronStatus.SUPERSEDED, superseded_by="winner-1")
    assert n.metadata["_superseded_by"] == "winner-1"
    revived = n.with_status(NeuronStatus.ACTIVE)
    assert "_superseded_by" not in revived.metadata
    assert revived.status == NeuronStatus.ACTIVE


def test_revive_preserves_unrelated_metadata() -> None:
    n = Neuron(
        id="n1",
        type=NeuronType.CONCEPT,
        content="x",
        metadata={
            "_status": "superseded",
            "_superseded_by": "winner",
            "_confidence": 0.8,
            "tags": ["x"],
        },
    )
    revived = n.with_status(NeuronStatus.ACTIVE)
    assert revived.metadata["_confidence"] == 0.8
    assert revived.metadata["tags"] == ["x"]
    assert "_superseded_by" not in revived.metadata
