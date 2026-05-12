"""Tests for Neuron validity window (valid_from / valid_until).

Item #3 from plan-tllr-learnings: time-bounded facts get an explicit
cliff at `valid_until`, complementing the gradual decay model with a
hard window for when a memory is supposed to be true.

Metadata-backed (zero migration), follows the `_status` / `_grounded`
extension pattern. ISO-format strings on disk; datetime in memory.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from neural_memory.core.neuron import Neuron, NeuronType


def _make(metadata: dict | None = None) -> Neuron:
    return Neuron(
        id="n1",
        type=NeuronType.CONCEPT,
        content="x",
        metadata=metadata or {},
    )


# ── Property accessors ────────────────────────────────────────────────


def test_default_validity_is_none() -> None:
    n = _make()
    assert n.valid_from is None
    assert n.valid_until is None


def test_valid_from_parses_iso_string() -> None:
    n = _make({"_valid_from": "2026-01-15T10:00:00"})
    assert n.valid_from == datetime(2026, 1, 15, 10, 0, 0)


def test_valid_until_parses_iso_string() -> None:
    n = _make({"_valid_until": "2026-12-31T23:59:59"})
    assert n.valid_until == datetime(2026, 12, 31, 23, 59, 59)


def test_invalid_iso_string_returns_none() -> None:
    """Garbage in metadata should not crash recall — return None and move on."""
    n = _make({"_valid_from": "not-a-date", "_valid_until": "also-bad"})
    assert n.valid_from is None
    assert n.valid_until is None


# ── with_validity helper ──────────────────────────────────────────────


def test_with_validity_sets_both_bounds() -> None:
    start = datetime(2026, 1, 1)
    end = datetime(2026, 12, 31)
    n = _make().with_validity(valid_from=start, valid_until=end)
    assert n.valid_from == start
    assert n.valid_until == end


def test_with_validity_only_until() -> None:
    end = datetime(2026, 6, 30)
    n = _make().with_validity(valid_until=end)
    assert n.valid_from is None
    assert n.valid_until == end


def test_with_validity_only_from() -> None:
    start = datetime(2026, 6, 1)
    n = _make().with_validity(valid_from=start)
    assert n.valid_from == start
    assert n.valid_until is None


def test_with_validity_immutable() -> None:
    original = _make()
    updated = original.with_validity(valid_until=datetime(2026, 12, 31))
    assert original.valid_until is None
    assert updated.valid_until == datetime(2026, 12, 31)
    assert original is not updated


def test_with_validity_clears_via_none() -> None:
    """Passing None explicitly clears a previously-set bound."""
    n = _make({"_valid_until": "2026-12-31T00:00:00"})
    cleared = n.with_validity(valid_until=None, _clear=True)
    assert cleared.valid_until is None


def test_with_validity_rejects_inverted_range() -> None:
    """`valid_from > valid_until` is nonsense — guard at the API boundary."""
    with pytest.raises(ValueError):
        _make().with_validity(
            valid_from=datetime(2026, 12, 31),
            valid_until=datetime(2026, 1, 1),
        )


def test_with_validity_iso_string_input() -> None:
    """Accept ISO strings to ease MCP wiring."""
    n = _make().with_validity(valid_until="2026-09-30T00:00:00")
    assert n.valid_until == datetime(2026, 9, 30, 0, 0, 0)


def test_with_validity_does_not_drop_other_metadata() -> None:
    n = _make({"_confidence": 0.9, "tags": ["a"]})
    updated = n.with_validity(valid_until=datetime(2026, 6, 1))
    assert updated.metadata["_confidence"] == 0.9
    assert updated.metadata["tags"] == ["a"]


# ── is_currently_valid ────────────────────────────────────────────────


def test_is_currently_valid_no_bounds_always_true() -> None:
    n = _make()
    assert n.is_currently_valid(datetime(2030, 1, 1)) is True


def test_is_currently_valid_within_window() -> None:
    n = _make().with_validity(
        valid_from=datetime(2026, 1, 1),
        valid_until=datetime(2026, 12, 31),
    )
    assert n.is_currently_valid(datetime(2026, 6, 1)) is True


def test_is_currently_valid_before_start() -> None:
    n = _make().with_validity(valid_from=datetime(2026, 6, 1))
    assert n.is_currently_valid(datetime(2026, 1, 1)) is False


def test_is_currently_valid_after_end() -> None:
    n = _make().with_validity(valid_until=datetime(2026, 6, 1))
    assert n.is_currently_valid(datetime(2026, 12, 1)) is False


def test_is_currently_valid_at_boundary_inclusive_start() -> None:
    """`valid_from` is inclusive — facts are valid AT the start moment."""
    start = datetime(2026, 6, 1, 0, 0, 0)
    n = _make().with_validity(valid_from=start)
    assert n.is_currently_valid(start) is True


def test_is_currently_valid_at_boundary_inclusive_end() -> None:
    """`valid_until` is inclusive — facts remain valid through the last moment."""
    end = datetime(2026, 6, 1, 23, 59, 59)
    n = _make().with_validity(valid_until=end)
    assert n.is_currently_valid(end) is True


def test_is_currently_valid_one_second_after_end_false() -> None:
    end = datetime(2026, 6, 1, 12, 0, 0)
    n = _make().with_validity(valid_until=end)
    assert n.is_currently_valid(end + timedelta(seconds=1)) is False


# ── Item #3 review fixes (C1, C2, H1, H2) ─────────────────────────────


def test_aware_iso_string_normalized_to_naive_utc() -> None:
    """C1: aware ISO `+00:00` must be stripped to naive UTC, never propagated."""
    n = _make().with_validity(valid_until="2026-09-30T00:00:00+00:00")
    assert n.valid_until is not None
    assert n.valid_until.tzinfo is None
    assert n.valid_until == datetime(2026, 9, 30, 0, 0, 0)


def test_aware_iso_string_with_offset_converted_to_utc() -> None:
    """C1: `+07:00` ISO input is converted to the equivalent UTC moment."""
    n = _make().with_validity(valid_until="2026-09-30T07:00:00+07:00")
    assert n.valid_until == datetime(2026, 9, 30, 0, 0, 0)
    assert n.valid_until.tzinfo is None  # type: ignore[union-attr]


def test_aware_datetime_object_converted_to_utc_not_local() -> None:
    """C2: `astimezone(None)` is local time — must be `astimezone(UTC)`."""
    from datetime import UTC

    aware = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    n = _make().with_validity(valid_until=aware)
    assert n.valid_until == datetime(2026, 6, 1, 12, 0, 0)
    assert n.valid_until.tzinfo is None  # type: ignore[union-attr]


def test_round_trip_aware_iso_then_is_currently_valid() -> None:
    """Pre-fix this raised TypeError comparing aware vs naive."""
    n = _make().with_validity(valid_until="2026-09-30T00:00:00+00:00")
    # The naive `now` from utcnow()-style production calls.
    now = datetime(2026, 6, 1, 0, 0, 0)
    assert n.is_currently_valid(now) is True


def test_coerce_datetime_rejects_int() -> None:
    """H2: ints / lists must not crash with AttributeError on `.tzinfo`."""
    from neural_memory.core.neuron import _coerce_datetime

    assert _coerce_datetime(123) is None  # type: ignore[arg-type]
    assert _coerce_datetime([2026, 1, 1]) is None  # type: ignore[arg-type]
    assert _coerce_datetime({"y": 2026}) is None  # type: ignore[arg-type]
    assert _coerce_datetime(True) is None  # type: ignore[arg-type]


def test_with_validity_partial_update_detects_inversion() -> None:
    """H1: setting only `valid_until` past existing `valid_from` must raise."""
    n = _make().with_validity(valid_from=datetime(2026, 6, 1))
    with pytest.raises(ValueError, match="must be <="):
        n.with_validity(valid_until=datetime(2026, 1, 1))


def test_with_validity_partial_update_preserves_existing_from() -> None:
    """A valid partial update keeps the prior bound intact."""
    n = _make().with_validity(valid_from=datetime(2026, 1, 1))
    updated = n.with_validity(valid_until=datetime(2026, 12, 31))
    assert updated.valid_from == datetime(2026, 1, 1)
    assert updated.valid_until == datetime(2026, 12, 31)
