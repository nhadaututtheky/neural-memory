"""Shared storage mixin for fiber and brain operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from neural_memory.core.brain import Brain, BrainSnapshot
from neural_memory.core.fiber import Fiber
from neural_memory.storage.shared_store_mappers import dict_to_brain, dict_to_fiber


class SharedStorageError(Exception):
    """Error from shared storage operations."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SharedFiberBrainMixin:
    """Mixin providing fiber and brain operations for SharedStorage."""

    _server_url: str
    _brain_id: str

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    # ========== Fiber Operations ==========

    async def add_fiber(self, fiber: Fiber) -> str:
        """Add a fiber."""
        data = {
            "id": fiber.id,
            "neuron_ids": list(fiber.neuron_ids),
            "synapse_ids": list(fiber.synapse_ids),
            "anchor_neuron_id": fiber.anchor_neuron_id,
            "time_start": fiber.time_start.isoformat() if fiber.time_start else None,
            "time_end": fiber.time_end.isoformat() if fiber.time_end else None,
            "coherence": fiber.coherence,
            "salience": fiber.salience,
            "frequency": fiber.frequency,
            "summary": fiber.summary,
            "tags": list(fiber.tags),
            "created_at": fiber.created_at.isoformat(),
        }
        result = await self._request("POST", "/memory/fibers", json_data=data)
        return result.get("id", fiber.id)

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        """Get a fiber by ID."""
        try:
            result = await self._request("GET", f"/memory/fiber/{fiber_id}")
            return dict_to_fiber(result)
        except SharedStorageError as e:
            if e.status_code == 404:
                return None
            raise

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        limit: int = 100,
    ) -> list[Fiber]:
        """Find fibers matching criteria."""
        params: dict[str, Any] = {"limit": limit}
        if contains_neuron:
            params["contains_neuron"] = contains_neuron
        if time_overlaps:
            params["time_start"] = time_overlaps[0].isoformat()
            params["time_end"] = time_overlaps[1].isoformat()
        if tags:
            params["tags"] = ",".join(tags)
        if min_salience is not None:
            params["min_salience"] = min_salience

        result = await self._request("GET", "/memory/fibers", params=params)
        return [dict_to_fiber(f) for f in result.get("fibers", [])]

    async def update_fiber(self, fiber: Fiber) -> None:
        """Update an existing fiber."""
        data = {
            "neuron_ids": list(fiber.neuron_ids),
            "synapse_ids": list(fiber.synapse_ids),
            "coherence": fiber.coherence,
            "salience": fiber.salience,
            "frequency": fiber.frequency,
            "summary": fiber.summary,
            "tags": list(fiber.tags),
        }
        await self._request("PUT", f"/memory/fibers/{fiber.id}", json_data=data)

    async def delete_fiber(self, fiber_id: str) -> bool:
        """Delete a fiber."""
        try:
            await self._request("DELETE", f"/memory/fibers/{fiber_id}")
            return True
        except SharedStorageError as e:
            if e.status_code == 404:
                return False
            raise

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        """Get fibers with ordering."""
        params = {
            "limit": limit,
            "order_by": order_by,
            "descending": descending,
        }
        result = await self._request("GET", "/memory/fibers", params=params)
        return [dict_to_fiber(f) for f in result.get("fibers", [])]

    # ========== Brain Operations ==========

    async def save_brain(self, brain: Brain) -> None:
        """Save brain metadata."""
        existing = await self.get_brain(brain.id)
        if existing:
            data = {
                "name": brain.name,
                "is_public": brain.is_public,
            }
            await self._request("PUT", f"/brain/{brain.id}", json_data=data)
        else:
            data = {
                "name": brain.name,
                "owner_id": brain.owner_id,
                "is_public": brain.is_public,
                "config": {
                    "decay_rate": brain.config.decay_rate,
                    "reinforcement_delta": brain.config.reinforcement_delta,
                    "activation_threshold": brain.config.activation_threshold,
                    "max_spread_hops": brain.config.max_spread_hops,
                    "max_context_tokens": brain.config.max_context_tokens,
                },
            }
            await self._request("POST", "/brain/create", json_data=data)

    async def get_brain(self, brain_id: str) -> Brain | None:
        """Get brain metadata."""
        try:
            result = await self._request("GET", f"/brain/{brain_id}")
            return dict_to_brain(result)
        except SharedStorageError as e:
            if e.status_code == 404:
                return None
            raise

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        """Export brain as snapshot."""
        result = await self._request("GET", f"/brain/{brain_id}/export")
        return BrainSnapshot(
            brain_id=result["brain_id"],
            brain_name=result["brain_name"],
            exported_at=datetime.fromisoformat(result["exported_at"]),
            version=result["version"],
            neurons=result["neurons"],
            synapses=result["synapses"],
            fibers=result["fibers"],
            config=result["config"],
            metadata=result.get("metadata", {}),
        )

    async def import_brain(
        self,
        snapshot: BrainSnapshot,
        target_brain_id: str | None = None,
    ) -> str:
        """Import a brain snapshot."""
        brain_id = target_brain_id or snapshot.brain_id
        data = {
            "brain_id": snapshot.brain_id,
            "brain_name": snapshot.brain_name,
            "exported_at": snapshot.exported_at.isoformat(),
            "version": snapshot.version,
            "neurons": snapshot.neurons,
            "synapses": snapshot.synapses,
            "fibers": snapshot.fibers,
            "config": snapshot.config,
            "metadata": snapshot.metadata,
        }
        result = await self._request(
            "POST",
            f"/brain/{brain_id}/import",
            json_data=data,
        )
        return result.get("id", brain_id)
