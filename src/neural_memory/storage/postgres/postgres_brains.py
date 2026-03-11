"""PostgreSQL brain operations."""

from __future__ import annotations

import json
from dataclasses import asdict

from neural_memory.core.brain import Brain, BrainConfig
from neural_memory.storage.postgres.postgres_base import PostgresBaseMixin
from neural_memory.storage.postgres.postgres_row_mappers import row_to_brain
from neural_memory.utils.timeutils import utcnow


class PostgresBrainMixin(PostgresBaseMixin):
    """PostgreSQL brain CRUD."""

    async def save_brain(self, brain: Brain) -> None:
        config_dict = asdict(brain.config)
        config_json = json.dumps(
            {k: v for k, v in config_dict.items() if v is not None}
        )
        shared_json = json.dumps(brain.shared_with)
        now = utcnow().isoformat()
        await self._query(
            """INSERT INTO brains
               (id, name, config, owner_id, is_public, shared_with, created_at, updated_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
               ON CONFLICT (id) DO UPDATE SET
                 name = EXCLUDED.name,
                 config = EXCLUDED.config,
                 owner_id = EXCLUDED.owner_id,
                 is_public = EXCLUDED.is_public,
                 shared_with = EXCLUDED.shared_with,
                 updated_at = EXCLUDED.updated_at""",
            brain.id,
            brain.name,
            config_json,
            brain.owner_id,
            1 if brain.is_public else 0,
            shared_json,
            brain.created_at.isoformat(),
            now,
        )

    async def get_brain(self, brain_id: str) -> Brain | None:
        row = await self._query_one(
            "SELECT * FROM brains WHERE id = $1", brain_id
        )
        if row is None:
            return None
        return row_to_brain(row)

    async def find_brain_by_name(self, name: str) -> Brain | None:
        row = await self._query_one(
            "SELECT * FROM brains WHERE name = $1", name
        )
        if row is None:
            return None
        return row_to_brain(row)
