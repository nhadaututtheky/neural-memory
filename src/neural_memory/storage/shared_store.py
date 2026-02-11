"""Shared storage client for remote brain access via HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import aiohttp

from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.shared_store_collections import SharedFiberBrainMixin, SharedStorageError
from neural_memory.storage.shared_store_mappers import (
    dict_to_neuron,
    dict_to_neuron_state,
    dict_to_synapse,
)


class SharedStorage(SharedFiberBrainMixin, NeuralStorage):
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
                result: dict[str, Any] = await response.json()
                return result
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
        return str(result.get("id", neuron.id))

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        """Get a neuron by ID."""
        try:
            result = await self._request("GET", f"/memory/neurons/{neuron_id}")
            return dict_to_neuron(result)
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
        return [dict_to_neuron(n) for n in result.get("neurons", [])]

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

    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: NeuronType | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get neuron suggestions via API."""
        params: dict[str, Any] = {"prefix": prefix, "limit": limit}
        if type_filter:
            params["type"] = type_filter.value
        result = await self._request("GET", "/memory/suggest", params=params)
        suggestions: list[dict[str, Any]] = result.get("suggestions", [])
        return suggestions

    # ========== Neuron State Operations ==========

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        """Get neuron activation state."""
        try:
            result = await self._request("GET", f"/memory/neurons/{neuron_id}/state")
            return dict_to_neuron_state(result)
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
            "firing_threshold": state.firing_threshold,
            "refractory_until": (
                state.refractory_until.isoformat() if state.refractory_until else None
            ),
            "refractory_period_ms": state.refractory_period_ms,
            "homeostatic_target": state.homeostatic_target,
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
        return str(result.get("id", synapse.id))

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        """Get a synapse by ID."""
        try:
            result = await self._request("GET", f"/memory/synapses/{synapse_id}")
            return dict_to_synapse(result)
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
        return [dict_to_synapse(s) for s in result.get("synapses", [])]

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
            neuron = dict_to_neuron(item["neuron"])
            synapse = dict_to_synapse(item["synapse"])
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
                neuron = dict_to_neuron(item["neuron"])
                synapse = dict_to_synapse(item["synapse"])
                path.append((neuron, synapse))
            return path
        except SharedStorageError as e:
            if e.status_code == 404:
                return None
            raise

    # ========== Statistics ==========

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        """Get brain statistics."""
        result = await self._request("GET", f"/brain/{brain_id}/stats")
        return {
            "neuron_count": result.get("neuron_count", 0),
            "synapse_count": result.get("synapse_count", 0),
            "fiber_count": result.get("fiber_count", 0),
        }

    async def get_enhanced_stats(self, brain_id: str) -> dict[str, Any]:
        """Get enhanced brain statistics via API."""
        result = await self._request("GET", f"/brain/{brain_id}/stats")
        return result

    # ========== Cleanup ==========

    async def clear(self, brain_id: str) -> None:
        """Clear all data for a brain."""
        await self._request("DELETE", f"/brain/{brain_id}")
