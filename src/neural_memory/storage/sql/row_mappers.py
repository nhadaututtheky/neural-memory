"""Dialect-agnostic row-to-model conversion functions.

Works with both SQLite (aiosqlite.Row) and PostgreSQL (asyncpg.Record)
rows, since both support dict-style access via row["column"].

All functions accept an optional ``dialect`` parameter for explicit datetime
normalization, but default to auto-detection so callers don't need changes:
  - SQLite  → ISO format strings  (parsed via datetime.fromisoformat)
  - PG      → native datetime objects (tz→UTC normalized)
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from typing import Any

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import (
    Confidence,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
)
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.utils.timeutils import utcnow


def _get(row: Any, key: str, default: Any = None) -> Any:
    """Get value from a row (aiosqlite.Row or asyncpg.Record) with default."""
    try:
        val = row[key]
        return val if val is not None else default
    except (KeyError, TypeError):
        return default


def _normalize_dt(value: Any) -> datetime | None:
    """Normalize a datetime value, auto-detecting the backend format.

    - ``None``               → ``None``
    - ``datetime`` object    → strip/normalize tz to naive UTC
    - Anything else         → ``datetime.fromisoformat(str(value))`` (SQLite ISO strings)
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value
    return datetime.fromisoformat(str(value))


def _parse_json_field(value: Any) -> Any:
    """Parse a JSON field that may be a string (SQLite) or already parsed (PG)."""
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def row_to_neuron(dialect_or_row: Any, row_or_dialect: Any = None) -> Neuron:
    # Support both (dialect, row) and (row, dialect) call signatures.
    # Detect by checking if first arg is subscriptable with string keys.
    if row_or_dialect is None:
        row = dialect_or_row
    elif hasattr(dialect_or_row, "name"):
        # dialect_or_row is a Dialect — (dialect, row) order
        row = row_or_dialect
    else:
        # (row, dialect) order
        row = dialect_or_row
    """Convert database row to Neuron."""
    return Neuron(
        id=str(row["id"]),
        type=NeuronType(row["type"]),
        content=str(row["content"]),
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        content_hash=int(_get(row, "content_hash", 0)),
        created_at=_normalize_dt(row["created_at"]) or utcnow(),
        ephemeral=bool(_get(row, "ephemeral", False)),
    )


def row_to_neuron_state(dialect_or_row: Any, row_or_dialect: Any = None) -> NeuronState:
    if row_or_dialect is None:
        row = dialect_or_row
    elif hasattr(dialect_or_row, "name"):
        row = row_or_dialect
    else:
        row = dialect_or_row
    """Convert database row to NeuronState."""
    return NeuronState(
        neuron_id=str(row["neuron_id"]),
        activation_level=float(_get(row, "activation_level", 0.0)),
        access_frequency=int(_get(row, "access_frequency", 0)),
        last_activated=_normalize_dt(_get(row, "last_activated")),
        decay_rate=float(_get(row, "decay_rate", 0.1)),
        created_at=_normalize_dt(row["created_at"]) or utcnow(),
        firing_threshold=float(_get(row, "firing_threshold", 0.3)),
        refractory_until=_normalize_dt(_get(row, "refractory_until")),
        refractory_period_ms=float(_get(row, "refractory_period_ms", 500.0)),
        homeostatic_target=float(_get(row, "homeostatic_target", 0.5)),
    )


def row_to_synapse(dialect_or_row: Any, row_or_dialect: Any = None) -> Synapse:
    """Convert database row to Synapse."""
    if row_or_dialect is None:
        row = dialect_or_row
    elif hasattr(dialect_or_row, "name"):
        row = row_or_dialect
    else:
        row = dialect_or_row
    return Synapse(
        id=str(row["id"]),
        source_id=str(row["source_id"]),
        target_id=str(row["target_id"]),
        type=SynapseType(row["type"]),
        weight=float(row["weight"]),
        direction=Direction(_get(row, "direction", "uni")),
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        reinforced_count=int(_get(row, "reinforced_count", 0)),
        last_activated=_normalize_dt(_get(row, "last_activated")),
        created_at=_normalize_dt(row["created_at"]) or utcnow(),
    )


def row_to_fiber(dialect: Any, row: Any) -> Fiber:
    """Convert database row to Fiber."""
    neuron_ids = _parse_json_field(row["neuron_ids"]) or []
    synapse_ids = _parse_json_field(row["synapse_ids"]) or []
    pathway = _parse_json_field(row["pathway"]) or []
    tags_raw = set(_parse_json_field(row["tags"]) or [])
    auto_tags = set(_parse_json_field(row["auto_tags"]) or [])
    agent_tags = set(_parse_json_field(row["agent_tags"]) or [])
    if not auto_tags and not agent_tags and tags_raw:
        agent_tags = tags_raw
    metadata = _parse_json_field(row["metadata"]) or {}

    return Fiber(
        id=str(row["id"]),
        neuron_ids=set(neuron_ids),
        synapse_ids=set(synapse_ids),
        anchor_neuron_id=str(row["anchor_neuron_id"]),
        pathway=pathway,
        conductivity=float(_get(row, "conductivity", 1.0)),
        last_conducted=_normalize_dt(_get(row, "last_conducted")),
        time_start=_normalize_dt(_get(row, "time_start")),
        time_end=_normalize_dt(_get(row, "time_end")),
        coherence=float(_get(row, "coherence", 0.0)),
        salience=float(_get(row, "salience", 0.0)),
        frequency=int(_get(row, "frequency", 0)),
        summary=_get(row, "summary"),
        essence=_get(row, "essence"),
        last_ghost_shown_at=_normalize_dt(_get(row, "last_ghost_shown_at")),
        auto_tags=auto_tags,
        agent_tags=agent_tags,
        metadata=metadata,
        compression_tier=int(_get(row, "compression_tier", 0)),
        pinned=bool(_get(row, "pinned", False)),
        created_at=_normalize_dt(row["created_at"]) or utcnow(),
    )


def row_to_brain(dialect_or_row: Any, row_or_dialect: Any = None) -> Brain:
    """Convert database row to Brain.

    Loads every BrainConfig field stored in the row, filtered against the
    current dataclass schema (forward-compat for old saves, backward-compat
    for renamed fields). The previous hard-coded subset dropped any field
    added to BrainConfig after this function was written — see issue #168
    where ``bm25_enabled``/``high_signal_memory_boost`` silently reverted to
    defaults on every load.
    """
    if row_or_dialect is None:
        row = dialect_or_row
    elif hasattr(dialect_or_row, "name"):
        row = row_or_dialect
    else:
        row = dialect_or_row
    config_data = _parse_json_field(row["config"]) or {}
    valid_fields = {f.name for f in dataclasses.fields(BrainConfig)}
    filtered_config = {k: v for k, v in config_data.items() if k in valid_fields}
    config = BrainConfig(**filtered_config)

    shared_with = _parse_json_field(row["shared_with"]) or []
    return Brain(
        id=str(row["id"]),
        name=str(row["name"]),
        config=config,
        owner_id=row["owner_id"],
        is_public=bool(_get(row, "is_public", False)),
        shared_with=shared_with,
        created_at=_normalize_dt(row["created_at"]) or utcnow(),
        updated_at=_normalize_dt(_get(row, "updated_at")) or utcnow(),
    )


def provenance_to_dict(provenance: Provenance) -> dict[str, object]:
    """Serialize Provenance to a JSON-compatible dict."""
    return {
        "source": provenance.source,
        "confidence": provenance.confidence.value,
        "verified": provenance.verified,
        "verified_at": (provenance.verified_at.isoformat() if provenance.verified_at else None),
        "created_by": provenance.created_by,
        "last_confirmed": (
            provenance.last_confirmed.isoformat() if provenance.last_confirmed else None
        ),
    }


def row_to_typed_memory(dialect_or_row: Any, row_or_dialect: Any = None) -> TypedMemory:
    """Convert database row to TypedMemory.

    Works with both SQLite (string provenance) and PostgreSQL (possibly native dict).
    """
    if row_or_dialect is None:
        row = dialect_or_row
    elif hasattr(dialect_or_row, "name"):
        row = row_or_dialect
    else:
        row = dialect_or_row
    prov_raw = row["provenance"]
    prov_data = _parse_json_field(prov_raw) or {}
    provenance = Provenance(
        source=prov_data.get("source", "unknown"),
        confidence=Confidence(prov_data.get("confidence", "medium")),
        verified=prov_data.get("verified", False),
        verified_at=(
            datetime.fromisoformat(prov_data["verified_at"])
            if prov_data.get("verified_at")
            else None
        ),
        created_by=prov_data.get("created_by", "unknown"),
        last_confirmed=(
            datetime.fromisoformat(prov_data["last_confirmed"])
            if prov_data.get("last_confirmed")
            else None
        ),
    )

    # trust_score, source, and tier added in schema v22/v37 — graceful fallback
    trust_score: float | None = None
    try:
        ts = row["trust_score"]
        if ts is not None:
            trust_score = float(ts)
    except (KeyError, TypeError, IndexError):
        pass

    source: str | None = None
    try:
        source = str(row["source"]) if row["source"] is not None else None
    except (KeyError, TypeError):
        pass

    tier: str = "warm"
    try:
        t = row["tier"]
        if t is not None:
            tier = str(t)
    except (KeyError, TypeError):
        pass

    tags_list = _parse_json_field(row["tags"]) or []
    meta_raw = row["metadata"]
    metadata = _parse_json_field(meta_raw) or {}

    return TypedMemory(
        fiber_id=str(row["fiber_id"]),
        memory_type=MemoryType(row["memory_type"]),
        priority=Priority(row["priority"]),
        provenance=provenance,
        expires_at=_normalize_dt(_get(row, "expires_at")),
        project_id=_get(row, "project_id"),
        tags=frozenset(tags_list),
        metadata=metadata,
        created_at=_normalize_dt(row["created_at"]) or utcnow(),
        trust_score=trust_score,
        source=source,
        tier=tier,
    )


def _row_to_joined_synapse(dialect_or_row: Any, row_or_dialect: Any = None) -> Synapse:
    """Convert a joined row (s_ prefixed columns) to a Synapse.

    Used by graph traversal in synapses mixin.
    """
    if row_or_dialect is None:
        row = dialect_or_row
    elif hasattr(dialect_or_row, "name"):
        row = row_or_dialect
    else:
        row = dialect_or_row
    s_created = _get(row, "s_created_at") or _get(row, "s_created")
    created = _normalize_dt(s_created)
    return Synapse(
        id=str(row["s_id"]),
        source_id=str(row["source_id"]),
        target_id=str(row["target_id"]),
        type=SynapseType(row["s_type"]),
        weight=float(row["weight"]),
        direction=Direction(_get(row, "direction", "uni")),
        metadata=_parse_json_field(_get(row, "s_metadata")) or {},
        reinforced_count=int(_get(row, "reinforced_count", 0)),
        last_activated=_normalize_dt(_get(row, "s_last_activated")),
        created_at=created or utcnow(),
    )
