"""SQLite project operations mixin."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

from neural_memory.core.project import Project
from neural_memory.storage.sqlite_row_mappers import row_to_project
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite


class SQLiteProjectMixin:
    """Mixin providing project CRUD operations."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def add_project(self, project: Project) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        try:
            await conn.execute(
                """INSERT INTO projects
                   (id, brain_id, name, description, start_date, end_date,
                    tags, priority, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project.id,
                    brain_id,
                    project.name,
                    project.description,
                    project.start_date.isoformat(),
                    project.end_date.isoformat() if project.end_date else None,
                    json.dumps(list(project.tags)),
                    project.priority,
                    json.dumps(project.metadata),
                    project.created_at.isoformat(),
                ),
            )
            await conn.commit()
            return project.id
        except sqlite3.IntegrityError:
            raise ValueError(f"Project {project.id} already exists")

    async def get_project(self, project_id: str) -> Project | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM projects WHERE id = ? AND brain_id = ?",
            (project_id, brain_id),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return row_to_project(row)

    async def get_project_by_name(self, name: str) -> Project | None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            "SELECT * FROM projects WHERE brain_id = ? AND LOWER(name) = LOWER(?)",
            (brain_id, name),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return row_to_project(row)

    async def list_projects(
        self,
        active_only: bool = False,
        tags: set[str] | None = None,
        limit: int = 100,
    ) -> list[Project]:
        limit = min(limit, 1000)
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        query = "SELECT * FROM projects WHERE brain_id = ?"
        params: list[Any] = [brain_id]

        if active_only:
            now = utcnow().isoformat()
            query += " AND start_date <= ? AND (end_date IS NULL OR end_date > ?)"
            params.extend([now, now])

        query += " ORDER BY priority DESC, start_date DESC LIMIT ?"
        params.append(limit)

        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            projects = [row_to_project(row) for row in rows]

        # Filter by tags in Python
        if tags is not None:
            projects = [p for p in projects if tags.intersection(p.tags)]

        return projects

    async def update_project(self, project: Project) -> None:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE projects SET name = ?, description = ?,
               start_date = ?, end_date = ?, tags = ?,
               priority = ?, metadata = ?
               WHERE id = ? AND brain_id = ?""",
            (
                project.name,
                project.description,
                project.start_date.isoformat(),
                project.end_date.isoformat() if project.end_date else None,
                json.dumps(list(project.tags)),
                project.priority,
                json.dumps(project.metadata),
                project.id,
                brain_id,
            ),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Project {project.id} does not exist")

        await conn.commit()

    async def delete_project(self, project_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "DELETE FROM projects WHERE id = ? AND brain_id = ?",
            (project_id, brain_id),
        )
        await conn.commit()

        return cursor.rowcount > 0
