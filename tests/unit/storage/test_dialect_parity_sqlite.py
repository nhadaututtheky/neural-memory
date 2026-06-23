"""Regression tests for the G3-dialect-parity cluster (SQLite + pure-unit).

These cover the findings whose *visible* failure is on the SQLite path or in
pure Python (no Postgres server required):

* #23 — ``get_tool_stats_by_period`` must bucket events by *day*. The old
        ``CAST(created_at AS DATE)`` applied SQLite NUMERIC affinity and
        collapsed every event into a single per-year integer bucket. The
        ``dialect.date_trunc_day`` helper uses ``SUBSTR(created_at,1,10)``.
* #7  — ``_dialect_row_to_project`` must read ``priority`` as a float. The old
        ``int(str(row["priority"]))`` raised ``ValueError`` on ``'1.0'`` for
        every project read.
* #26 — ``row_to_brain`` must reconstruct BrainConfig generically so a
        save→get round-trip preserves every field, and default
        ``embedding_enabled`` to True (the dataclass default).
* #29 — ``SharedStorage.find_neurons`` must forward ``ephemeral`` /
        ``created_before`` so a SHARED brain does not fail open.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Any

import pytest

from neural_memory.storage.sql import SQLStorage
from neural_memory.storage.sql.mixins.projects import ProjectsMixin
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect


@pytest.fixture
async def storage() -> Any:
    from neural_memory.core.brain import Brain

    td = tempfile.mkdtemp()
    s = SQLStorage(SQLiteDialect(db_path=os.path.join(td, "t.db")))
    await s.initialize()
    brain = Brain.create(name="parity-test")
    await s.save_brain(brain)
    s.set_brain(brain.id)
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_tool_stats_buckets_by_day_not_year(storage: Any) -> None:
    """#23: three events on distinct days yield three daily buckets."""
    brain_id = storage.brain_id
    events = [
        {
            "tool_name": "nmem_recall",
            "server_name": "nm",
            "args_summary": "x",
            "success": True,
            "duration_ms": 10,
            "session_id": "s",
            "task_context": "",
            "created_at": f"{day}T10:00:00",
        }
        for day in ("2026-06-01", "2026-06-02", "2026-06-03")
    ]
    await storage.insert_tool_events(brain_id, events)

    stats = await storage.get_tool_stats_by_period(brain_id, days=365)
    days = {row["date"] for row in stats}
    # Must NOT collapse into a single ``2026`` bucket.
    assert days == {"2026-06-01", "2026-06-02", "2026-06-03"}


@pytest.mark.asyncio
async def test_project_priority_read_as_float(storage: Any) -> None:
    """#7: reading a project must not raise on the REAL ``priority`` column."""
    from neural_memory.core.project import Project

    p = Project.create(name="proj", priority=2.5)
    # add_project resolves to a BrainOpsMixin protocol stub on the public class
    # (unrelated MRO quirk); call the real ProjectsMixin implementation.
    await ProjectsMixin.add_project(storage, p)

    got = await storage.get_project(p.id)
    assert got is not None
    assert isinstance(got.priority, float)
    assert got.priority == 2.5

    # Default priority (1.0) must also round-trip without ValueError.
    p2 = Project.create(name="proj2")
    await ProjectsMixin.add_project(storage, p2)
    got2 = await storage.get_project(p2.id)
    assert got2 is not None
    assert got2.priority == 1.0


def test_row_to_brain_reconstructs_all_config_fields() -> None:
    """#26: a save→get round-trip preserves non-default BrainConfig knobs."""
    import dataclasses
    import json

    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.postgres.postgres_row_mappers import row_to_brain

    cfg = BrainConfig(
        activation_strategy="ppr",
        learning_rate=0.2,
        ppr_damping=0.77,
        embedding_enabled=True,
    )
    record = {
        "id": "b1",
        "name": "brain",
        "config": json.dumps(dataclasses.asdict(cfg)),
        "owner_id": None,
        "is_public": 0,
        "shared_with": None,
        "created_at": datetime(2026, 1, 1, 12, 0, 0),
        "updated_at": datetime(2026, 1, 1, 12, 0, 0),
    }
    brain = row_to_brain(record)
    assert brain.config.activation_strategy == "ppr"
    assert brain.config.learning_rate == 0.2
    assert brain.config.ppr_damping == 0.77
    assert brain.config.embedding_enabled is True


def test_row_to_brain_embedding_enabled_defaults_true_on_legacy_row() -> None:
    """#26: a legacy config JSON missing ``embedding_enabled`` defaults to True."""
    import json

    from neural_memory.storage.postgres.postgres_row_mappers import row_to_brain

    record = {
        "id": "b1",
        "name": "brain",
        # Legacy row whose stored config predates the embedding_enabled key.
        "config": json.dumps({"decay_rate": 0.1}),
        "owner_id": None,
        "is_public": 0,
        "shared_with": None,
        "created_at": datetime(2026, 1, 1, 12, 0, 0),
        "updated_at": datetime(2026, 1, 1, 12, 0, 0),
    }
    brain = row_to_brain(record)
    assert brain.config.embedding_enabled is True


def test_row_to_brain_drops_unknown_config_keys() -> None:
    """#26: keys the dataclass no longer knows about are dropped, not crashed."""
    import json

    from neural_memory.storage.postgres.postgres_row_mappers import row_to_brain

    record = {
        "id": "b1",
        "name": "brain",
        "config": json.dumps({"decay_rate": 0.3, "obsolete_removed_knob": 123}),
        "owner_id": None,
        "is_public": 0,
        "shared_with": None,
        "created_at": datetime(2026, 1, 1, 12, 0, 0),
        "updated_at": datetime(2026, 1, 1, 12, 0, 0),
    }
    brain = row_to_brain(record)
    assert brain.config.decay_rate == 0.3
    assert not hasattr(brain.config, "obsolete_removed_knob")


@pytest.mark.asyncio
async def test_shared_storage_forwards_ephemeral_and_created_before() -> None:
    """#29: SharedStorage.find_neurons must forward the new filters as params."""
    from neural_memory.storage.shared_store import SharedStorage

    captured: dict[str, Any] = {}

    store = SharedStorage(server_url="http://example", brain_id="b1")

    async def _fake_request(
        method: str, path: str, *, json_data: Any = None, params: Any = None
    ) -> dict[str, Any]:
        captured["params"] = params
        return {"neurons": []}

    store._request = _fake_request  # type: ignore[assignment]

    cutoff = datetime(2026, 1, 1, 0, 0, 0)
    await store.find_neurons(content_exact="x", ephemeral=False, created_before=cutoff)

    params = captured["params"]
    assert params["ephemeral"] is False
    assert params["created_before"] == cutoff.isoformat()
