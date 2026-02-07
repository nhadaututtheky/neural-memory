"""In-memory fiber, typed memory, and project operations mixin."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from neural_memory.core.fiber import Fiber
from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
from neural_memory.core.project import Project


class InMemoryCollectionsMixin:
    """Mixin providing fiber, typed memory, and project operations."""

    _fibers: dict[str, dict[str, Fiber]]
    _typed_memories: dict[str, dict[str, TypedMemory]]
    _projects: dict[str, dict[str, Project]]

    def _get_brain_id(self) -> str: ...

    # ========== Fiber Operations ==========

    async def add_fiber(self, fiber: Fiber) -> str:
        brain_id = self._get_brain_id()

        if fiber.id in self._fibers[brain_id]:
            raise ValueError(f"Fiber {fiber.id} already exists")

        self._fibers[brain_id][fiber.id] = fiber
        return fiber.id

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        brain_id = self._get_brain_id()
        return self._fibers[brain_id].get(fiber_id)

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        limit: int = 100,
    ) -> list[Fiber]:
        brain_id = self._get_brain_id()
        results: list[Fiber] = []

        for fiber in self._fibers[brain_id].values():
            if contains_neuron is not None and contains_neuron not in fiber.neuron_ids:
                continue
            if time_overlaps is not None:
                start, end = time_overlaps
                if not fiber.overlaps_time(start, end):
                    continue
            if tags is not None and not tags.issubset(fiber.tags):
                continue
            if min_salience is not None and fiber.salience < min_salience:
                continue

            results.append(fiber)
            if len(results) >= limit:
                break

        results.sort(key=lambda f: f.salience, reverse=True)
        return results

    async def find_fibers_batch(
        self,
        neuron_ids: list[str],
        limit_per_neuron: int = 10,
    ) -> list[Fiber]:
        """Find fibers containing any of the given neurons from in-memory store."""
        brain_id = self._get_brain_id()
        nid_set = set(neuron_ids)
        seen: set[str] = set()
        result: list[Fiber] = []

        for fiber in self._fibers[brain_id].values():
            if fiber.id in seen:
                continue
            if fiber.neuron_ids & nid_set:
                seen.add(fiber.id)
                result.append(fiber)

        result.sort(key=lambda f: f.salience, reverse=True)
        return result

    async def update_fiber(self, fiber: Fiber) -> None:
        brain_id = self._get_brain_id()

        if fiber.id not in self._fibers[brain_id]:
            raise ValueError(f"Fiber {fiber.id} does not exist")

        self._fibers[brain_id][fiber.id] = fiber

    async def delete_fiber(self, fiber_id: str) -> bool:
        brain_id = self._get_brain_id()

        if fiber_id not in self._fibers[brain_id]:
            return False

        del self._fibers[brain_id][fiber_id]
        return True

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        brain_id = self._get_brain_id()
        fibers = list(self._fibers[brain_id].values())

        sort_keys = {
            "created_at": lambda f: f.created_at,
            "salience": lambda f: f.salience,
            "frequency": lambda f: f.frequency,
        }
        fibers.sort(key=sort_keys[order_by], reverse=descending)
        return fibers[:limit]

    # ========== TypedMemory Operations ==========

    async def add_typed_memory(self, typed_memory: TypedMemory) -> str:
        brain_id = self._get_brain_id()

        if typed_memory.fiber_id not in self._fibers[brain_id]:
            raise ValueError(f"Fiber {typed_memory.fiber_id} does not exist")

        self._typed_memories[brain_id][typed_memory.fiber_id] = typed_memory
        return typed_memory.fiber_id

    async def get_typed_memory(self, fiber_id: str) -> TypedMemory | None:
        brain_id = self._get_brain_id()
        return self._typed_memories[brain_id].get(fiber_id)

    async def find_typed_memories(
        self,
        memory_type: MemoryType | None = None,
        min_priority: Priority | None = None,
        include_expired: bool = False,
        project_id: str | None = None,
        tags: set[str] | None = None,
        limit: int = 100,
    ) -> list[TypedMemory]:
        brain_id = self._get_brain_id()
        results: list[TypedMemory] = []

        for tm in self._typed_memories[brain_id].values():
            if memory_type is not None and tm.memory_type != memory_type:
                continue
            if min_priority is not None and tm.priority < min_priority:
                continue
            if not include_expired and tm.is_expired:
                continue
            if project_id is not None and tm.project_id != project_id:
                continue
            if tags is not None and not tags.issubset(tm.tags):
                continue

            results.append(tm)
            if len(results) >= limit:
                break

        results.sort(key=lambda t: (t.priority, t.created_at), reverse=True)
        return results

    async def update_typed_memory(self, typed_memory: TypedMemory) -> None:
        brain_id = self._get_brain_id()

        if typed_memory.fiber_id not in self._typed_memories[brain_id]:
            raise ValueError(f"TypedMemory for fiber {typed_memory.fiber_id} does not exist")

        self._typed_memories[brain_id][typed_memory.fiber_id] = typed_memory

    async def delete_typed_memory(self, fiber_id: str) -> bool:
        brain_id = self._get_brain_id()

        if fiber_id not in self._typed_memories[brain_id]:
            return False

        del self._typed_memories[brain_id][fiber_id]
        return True

    async def get_expired_memories(self) -> list[TypedMemory]:
        brain_id = self._get_brain_id()
        return [tm for tm in self._typed_memories[brain_id].values() if tm.is_expired]

    async def get_project_memories(
        self,
        project_id: str,
        include_expired: bool = False,
    ) -> list[TypedMemory]:
        brain_id = self._get_brain_id()
        results: list[TypedMemory] = []

        for tm in self._typed_memories[brain_id].values():
            if tm.project_id != project_id:
                continue
            if not include_expired and tm.is_expired:
                continue
            results.append(tm)

        results.sort(key=lambda t: (t.priority, t.created_at), reverse=True)
        return results

    # ========== Project Operations ==========

    async def add_project(self, project: Project) -> str:
        brain_id = self._get_brain_id()

        if project.id in self._projects[brain_id]:
            raise ValueError(f"Project {project.id} already exists")

        self._projects[brain_id][project.id] = project
        return project.id

    async def get_project(self, project_id: str) -> Project | None:
        brain_id = self._get_brain_id()
        return self._projects[brain_id].get(project_id)

    async def get_project_by_name(self, name: str) -> Project | None:
        brain_id = self._get_brain_id()
        name_lower = name.lower()
        for project in self._projects[brain_id].values():
            if project.name.lower() == name_lower:
                return project
        return None

    async def list_projects(
        self,
        active_only: bool = False,
        tags: set[str] | None = None,
        limit: int = 100,
    ) -> list[Project]:
        brain_id = self._get_brain_id()
        results: list[Project] = []

        for project in self._projects[brain_id].values():
            if active_only and not project.is_active:
                continue
            if tags is not None and not tags.intersection(project.tags):
                continue

            results.append(project)
            if len(results) >= limit:
                break

        results.sort(key=lambda p: (p.priority, p.start_date), reverse=True)
        return results

    async def update_project(self, project: Project) -> None:
        brain_id = self._get_brain_id()

        if project.id not in self._projects[brain_id]:
            raise ValueError(f"Project {project.id} does not exist")

        self._projects[brain_id][project.id] = project

    async def delete_project(self, project_id: str) -> bool:
        brain_id = self._get_brain_id()

        if project_id not in self._projects[brain_id]:
            return False

        del self._projects[brain_id][project_id]
        return True
