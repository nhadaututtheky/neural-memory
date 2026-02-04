"""Shared storage client for remote brain access via HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import aiohttp

from neural_memory.core.brain import Brain, BrainConfig, BrainSnapshot
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.base import NeuralStorage


class SharedStorageError(Exception):
    """Error from shared storage operations."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SharedStorage(NeuralStorage):
    """
    HTTP-based storage client that connects to a remote NeuralMemory server.

    Enables real-time brain sharing between multiple agents/instances.

    Usage:
        async with SharedStorage("http://localhost:8000", "brain-123") as storage:
            await storage.add_neuron(neuron)
            neurons = await storage.find_neurons(type=NeuronType.CONCEPT)

    Or without context manager:
        storage = SharedStorage("http://localhost:8000", "brain-123")
        await storage.connect()
        try:
            await storage.add_neuron(neuron)
        finally:
            await storage.disconnect()
    """

    def __init__(
        self,
        server_url: str,
        brain_id: str,
        *,
        timeout: float = 30.0,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize shared storage client.

        Args:
            server_url: Base URL of NeuralMemory server (e.g., "http://localhost:8000")
            brain_id: ID of the brain to connect to
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self._server_url = server_url.rstrip("/")
        self._brain_id = brain_id
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None
        self._connected = False

    @property
    def server_url(self) -> str:
        """Get the server URL."""
        return self._server_url

    @property
    def brain_id(self) -> str:
        """Get the current brain ID."""
        return self._brain_id

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected and self._session is not None

    def set_brain(self, brain_id: str) -> None:
        """Set the current brain context."""
        self._brain_id = brain_id

    async def connect(self) -> None:
        """Establish connection to server."""
        if self._session is None:
            headers = {"X-Brain-ID": self._brain_id}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers=headers,
            )
            self._connected = True

    async def disconnect(self) -> None:
        """Close connection to server."""
        if self._session:
            await self._session.close()
            self._session = None
            self._connected = False

    async def __aenter__(self) -> SharedStorage:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect()

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with brain ID."""
        headers = {"X-Brain-ID": self._brain_id, "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request to server."""
        if not self._session:
            await self.connect()

        assert self._session is not None

        url = f"{self._server_url}{path}"
        headers = self._get_headers()

        try:
            async with self._session.request(
                method,
                url,
                json=json_data,
                params=params,
                headers=headers,
            ) as response:
                if response.status >= 400:
                    text = await response.text()
                    raise SharedStorageError(
                        f"Server error: {text}",
                        status_code=response.status,
                    )
                return await response.json()
        except aiohttp.ClientError as e:
            raise SharedStorageError(f"Connection error: {e}") from e

    # ========== Neuron Operations ==========

    async def add_neuron(self, neuron: Neuron) -> str:
        """Add a neuron via API."""
        data = {
            "id": neuron.id,
            "type": neuron.type.value,
            "content": neuron.content,
            "metadata": neuron.metadata,
            "created_at": neuron.created_at.isoformat(),
        }
        result = await self._request("POST", "/memory/neurons", json_data=data)
        return result.get("id", neuron.id)

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        """Get a neuron by ID."""
        try:
            result = await self._request("GET", f"/memory/neurons/{neuron_id}")
            return self._dict_to_neuron(result)
        except SharedStorageError as e:
            if e.status_code == 404:
                return None
            raise

    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
    ) -> list[Neuron]:
        """Find neurons matching criteria."""
        params: dict[str, Any] = {"limit": limit}
        if type:
            params["type"] = type.value
        if content_contains:
            params["content_contains"] = content_contains
        if content_exact:
            params["content_exact"] = content_exact
        if time_range:
            params["time_start"] = time_range[0].isoformat()
            params["time_end"] = time_range[1].isoformat()

        result = await self._request("GET", "/memory/neurons", params=params)
        return [self._dict_to_neuron(n) for n in result.get("neurons", [])]

    async def update_neuron(self, neuron: Neuron) -> None:
        """Update an existing neuron."""
        data = {
            "type": neuron.type.value,
            "content": neuron.content,
            "metadata": neuron.metadata,
        }
        await self._request("PUT", f"/memory/neurons/{neuron.id}", json_data=data)

    async def delete_neuron(self, neuron_id: str) -> bool:
        """Delete a neuron."""
        try:
            await self._request("DELETE", f"/memory/neurons/{neuron_id}")
            return True
        except SharedStorageError as e:
            if e.status_code == 404:
                return False
            raise

    # ========== Neuron State Operations ==========

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        """Get neuron activation state."""
        try:
            result = await self._request("GET", f"/memory/neurons/{neuron_id}/state")
            return self._dict_to_neuron_state(result)
        except SharedStorageError as e:
            if e.status_code == 404:
                return None
            raise

    async def update_neuron_state(self, state: NeuronState) -> None:
        """Update neuron state."""
        data = {
            "neuron_id": state.neuron_id,
            "activation_level": state.activation_level,
            "access_frequency": state.access_frequency,
            "last_activated": state.last_activated.isoformat() if state.last_activated else None,
            "decay_rate": state.decay_rate,
        }
        await self._request(
            "PUT",
            f"/memory/neurons/{state.neuron_id}/state",
            json_data=data,
        )

    # ========== Synapse Operations ==========

    async def add_synapse(self, synapse: Synapse) -> str:
        """Add a synapse."""
        data = {
            "id": synapse.id,
            "source_id": synapse.source_id,
            "target_id": synapse.target_id,
            "type": synapse.type.value,
            "weight": synapse.weight,
            "direction": synapse.direction.value,
            "metadata": synapse.metadata,
            "created_at": synapse.created_at.isoformat(),
        }
        result = await self._request("POST", "/memory/synapses", json_data=data)
        return result.get("id", synapse.id)

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        """Get a synapse by ID."""
        try:
            result = await self._request("GET", f"/memory/synapses/{synapse_id}")
            return self._dict_to_synapse(result)
        except SharedStorageError as e:
            if e.status_code == 404:
                return None
            raise

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
    ) -> list[Synapse]:
        """Find synapses matching criteria."""
        params: dict[str, Any] = {}
        if source_id:
            params["source_id"] = source_id
        if target_id:
            params["target_id"] = target_id
        if type:
            params["type"] = type.value
        if min_weight is not None:
            params["min_weight"] = min_weight

        result = await self._request("GET", "/memory/synapses", params=params)
        return [self._dict_to_synapse(s) for s in result.get("synapses", [])]

    async def update_synapse(self, synapse: Synapse) -> None:
        """Update an existing synapse."""
        data = {
            "weight": synapse.weight,
            "metadata": synapse.metadata,
        }
        await self._request("PUT", f"/memory/synapses/{synapse.id}", json_data=data)

    async def delete_synapse(self, synapse_id: str) -> bool:
        """Delete a synapse."""
        try:
            await self._request("DELETE", f"/memory/synapses/{synapse_id}")
            return True
        except SharedStorageError as e:
            if e.status_code == 404:
                return False
            raise

    # ========== Graph Traversal ==========

    async def get_neighbors(
        self,
        neuron_id: str,
        direction: Literal["out", "in", "both"] = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
    ) -> list[tuple[Neuron, Synapse]]:
        """Get neighboring neurons."""
        params: dict[str, Any] = {"direction": direction}
        if synapse_types:
            params["synapse_types"] = ",".join(t.value for t in synapse_types)
        if min_weight is not None:
            params["min_weight"] = min_weight

        result = await self._request(
            "GET",
            f"/memory/neurons/{neuron_id}/neighbors",
            params=params,
        )

        neighbors = []
        for item in result.get("neighbors", []):
            neuron = self._dict_to_neuron(item["neuron"])
            synapse = self._dict_to_synapse(item["synapse"])
            neighbors.append((neuron, synapse))
        return neighbors

    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
    ) -> list[tuple[Neuron, Synapse]] | None:
        """Find shortest path between neurons."""
        params = {"target_id": target_id, "max_hops": max_hops}
        try:
            result = await self._request(
                "GET",
                f"/memory/neurons/{source_id}/path",
                params=params,
            )
            if not result.get("path"):
                return None

            path = []
            for item in result["path"]:
                neuron = self._dict_to_neuron(item["neuron"])
                synapse = self._dict_to_synapse(item["synapse"])
                path.append((neuron, synapse))
            return path
        except SharedStorageError as e:
            if e.status_code == 404:
                return None
            raise

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
            return self._dict_to_fiber(result)
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
        return [self._dict_to_fiber(f) for f in result.get("fibers", [])]

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
        return [self._dict_to_fiber(f) for f in result.get("fibers", [])]

    # ========== Brain Operations ==========

    async def save_brain(self, brain: Brain) -> None:
        """Save brain metadata."""
        # Check if brain exists
        existing = await self.get_brain(brain.id)
        if existing:
            # Update
            data = {
                "name": brain.name,
                "is_public": brain.is_public,
            }
            await self._request("PUT", f"/brain/{brain.id}", json_data=data)
        else:
            # Create
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
            return self._dict_to_brain(result)
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

    # ========== Statistics ==========

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        """Get brain statistics."""
        result = await self._request("GET", f"/brain/{brain_id}/stats")
        return {
            "neuron_count": result.get("neuron_count", 0),
            "synapse_count": result.get("synapse_count", 0),
            "fiber_count": result.get("fiber_count", 0),
        }

    # ========== Cleanup ==========

    async def clear(self, brain_id: str) -> None:
        """Clear all data for a brain."""
        await self._request("DELETE", f"/brain/{brain_id}")

    # ========== Conversion Helpers ==========

    def _dict_to_neuron(self, data: dict[str, Any]) -> Neuron:
        """Convert API response dict to Neuron."""
        return Neuron(
            id=data["id"],
            type=NeuronType(data["type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def _dict_to_neuron_state(self, data: dict[str, Any]) -> NeuronState:
        """Convert API response dict to NeuronState."""
        return NeuronState(
            neuron_id=data["neuron_id"],
            activation_level=data.get("activation_level", 0.0),
            access_frequency=data.get("access_frequency", 0),
            last_activated=datetime.fromisoformat(data["last_activated"])
            if data.get("last_activated")
            else None,
            decay_rate=data.get("decay_rate", 0.1),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
        )

    def _dict_to_synapse(self, data: dict[str, Any]) -> Synapse:
        """Convert API response dict to Synapse."""
        return Synapse(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=SynapseType(data["type"]),
            weight=data.get("weight", 0.5),
            direction=Direction(data.get("direction", "uni")),
            metadata=data.get("metadata", {}),
            reinforced_count=data.get("reinforced_count", 0),
            last_activated=datetime.fromisoformat(data["last_activated"])
            if data.get("last_activated")
            else None,
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
        )

    def _dict_to_fiber(self, data: dict[str, Any]) -> Fiber:
        """Convert API response dict to Fiber."""
        return Fiber(
            id=data["id"],
            neuron_ids=frozenset(data.get("neuron_ids", [])),
            synapse_ids=frozenset(data.get("synapse_ids", [])),
            anchor_neuron_id=data["anchor_neuron_id"],
            time_start=datetime.fromisoformat(data["time_start"])
            if data.get("time_start")
            else None,
            time_end=datetime.fromisoformat(data["time_end"])
            if data.get("time_end")
            else None,
            coherence=data.get("coherence", 0.0),
            salience=data.get("salience", 0.0),
            frequency=data.get("frequency", 0),
            summary=data.get("summary"),
            tags=frozenset(data.get("tags", [])),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
        )

    def _dict_to_brain(self, data: dict[str, Any]) -> Brain:
        """Convert API response dict to Brain."""
        return Brain(
            id=data["id"],
            name=data["name"],
            config=BrainConfig(),  # Default config, actual config fetched from server
            owner_id=data.get("owner_id"),
            is_public=data.get("is_public", False),
            shared_with=data.get("shared_with", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.now(),
        )
