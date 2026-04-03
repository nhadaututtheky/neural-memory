"""Project operations mixin — dialect-agnostic."""

from __future__ import annotations

import json
from typing import Any

from neural_memory.core.project import Project
from neural_memory.storage.sql.dialect import Dialect
from neural_memory.utils.timeutils import utcnow


class ProjectsMixin:
    """Mixin providing project CRUD operations."""

    _dialect: Dialect

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def add_project(self, project: Project) -> str:
        d = self._dialect
        brain_id = self._get_brain_id()

        try:
            await d.execute(
                f"""INSERT INTO projects
                   (id, brain_id, name, description, start_date, end_date,
                    tags, priority, metadata, created_at)
                   VALUES ({d.phs(10)})""",
                [
                    project.id,
                    brain_id,
                    project.name,
                    project.description,
                    d.serialize_dt(project.start_date),
                    d.serialize_dt(project.end_date),
                    json.dumps(list(project.tags)),
                    project.priority,
                    json.dumps(project.metadata),
                    d.serialize_dt(project.created_at),
                ],
            )
            return project.id
        except Exception:
            raise ValueError(f"Project {project.id} already exists")

    async def get_project(self, project_id: str) -> Project | None:
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT * FROM projects WHERE id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            [project_id, brain_id],
        )
        if row is None:
            return None
        return _dialect_row_to_project(row)

    async def get_project_by_name(self, name: str) -> Project | None:
        d = self._dialect
        brain_id = self._get_brain_id()

        row = await d.fetch_one(
            f"SELECT * FROM projects WHERE brain_id = {d.ph(1)} AND LOWER(name) = LOWER({d.ph(2)})",
            [brain_id, name],
        )
        if row is None:
            return None
        return _dialect_row_to_project(row)

    async def list_projects(
        self,
        active_only: bool = False,
        tags: set[str] | None = None,
        limit: int = 100,
    ) -> list[Project]:
        limit = min(limit, 1000)
        d = self._dialect
        brain_id = self._get_brain_id()

        query = f"SELECT * FROM projects WHERE brain_id = {d.ph(1)}"
        params: list[Any] = [brain_id]

        if active_only:
            now = d.serialize_dt(utcnow())
            query += f" AND start_date <= {d.ph(len(params) + 1)} AND (end_date IS NULL OR end_date > {d.ph(len(params) + 2)})"
            params.extend([now, now])

        query += f" ORDER BY priority DESC, start_date DESC LIMIT {d.ph(len(params) + 1)}"
        params.append(limit)

        rows = await d.fetch_all(query, params)
        projects = [_dialect_row_to_project(r) for r in rows]

        # Filter by tags in Python
        if tags is not None:
            projects = [p for p in projects if tags.intersection(p.tags)]

        return projects

    async def update_project(self, project: Project) -> None:
        d = self._dialect
        brain_id = self._get_brain_id()

        count = await d.execute_count(
            f"""UPDATE projects SET name = {d.ph(1)}, description = {d.ph(2)},
               start_date = {d.ph(3)}, end_date = {d.ph(4)}, tags = {d.ph(5)},
               priority = {d.ph(6)}, metadata = {d.ph(7)}
               WHERE id = {d.ph(8)} AND brain_id = {d.ph(9)}""",
            [
                project.name,
                project.description,
                d.serialize_dt(project.start_date),
                d.serialize_dt(project.end_date),
                json.dumps(list(project.tags)),
                project.priority,
                json.dumps(project.metadata),
                project.id,
                brain_id,
            ],
        )

        if count == 0:
            raise ValueError(f"Project {project.id} does not exist")

    async def delete_project(self, project_id: str) -> bool:
        d = self._dialect
        brain_id = self._get_brain_id()

        count = await d.execute_count(
            f"DELETE FROM projects WHERE id = {d.ph(1)} AND brain_id = {d.ph(2)}",
            [project_id, brain_id],
        )
        return count > 0


def _dialect_row_to_project(row: dict[str, object]) -> Project:
    """Convert a dialect row dict to a Project (dialect-agnostic version)."""
    from datetime import datetime

    def _safe_dt(val: object) -> datetime | None:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        return datetime.fromisoformat(str(val))

    return Project(
        id=str(row["id"]),
        name=str(row["name"]),
        description=str(row["description"]) if row["description"] else "",
        start_date=_safe_dt(row["start_date"]) or utcnow(),
        end_date=_safe_dt(row["end_date"]),
        tags=frozenset(json.loads(str(row["tags"]))),
        priority=int(str(row["priority"])),
        metadata=json.loads(str(row["metadata"])) if row["metadata"] else {},
        created_at=_safe_dt(row["created_at"]) or utcnow(),
    )
