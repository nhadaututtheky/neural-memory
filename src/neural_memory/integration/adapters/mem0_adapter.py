"""Mem0 source adapter for importing memories."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

from neural_memory.integration.models import (
    ExternalRecord,
    SourceCapability,
    SourceSystemType,
)
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

# Tag constraints (consistent with _remember handler in server.py)
_MAX_TAGS = 50
_MAX_TAG_LEN = 100


def _parse_mem0_records(
    memories: list[dict[str, Any]] | dict[str, Any],
    *,
    source_system: str,
    user_id: str | None,
    agent_id: str | None,
    limit: int | None,
) -> list[ExternalRecord]:
    """Parse raw Mem0 response into ExternalRecord list.

    Shared by both Platform and Self-hosted adapters.
    """
    records: list[ExternalRecord] = []
    items = memories if isinstance(memories, list) else memories.get("results", [])

    for mem in items:
        if limit is not None and len(records) >= limit:
            break

        mem_id = mem.get("id", "")
        content = mem.get("memory", "") or mem.get("text", "")
        if not content:
            continue

        metadata = mem.get("metadata", {}) or {}

        created_at = utcnow()
        if "created_at" in mem:
            try:
                created_at = datetime.fromisoformat(str(mem["created_at"]).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        updated_at = None
        if "updated_at" in mem:
            try:
                updated_at = datetime.fromisoformat(str(mem["updated_at"]).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        source_type = mem.get("category", metadata.get("type", "memory"))

        tags: set[str] = set()
        if "categories" in mem:
            for cat in mem["categories"]:
                if isinstance(cat, str) and len(cat) <= _MAX_TAG_LEN and len(tags) < _MAX_TAGS:
                    tags.add(cat)
        if user_id:
            tags.add(f"user:{user_id}")
        if agent_id:
            tags.add(f"agent:{agent_id}")

        record = ExternalRecord.create(
            id=str(mem_id),
            source_system=source_system,
            content=content,
            source_collection=user_id or agent_id or "default",
            created_at=created_at,
            updated_at=updated_at,
            source_type=source_type,
            metadata=metadata,
            tags=tags,
        )
        records.append(record)

    return records


class _BaseMem0Adapter:
    """Base class for Mem0 adapters (Platform and Self-hosted).

    Subclasses must override ``_get_client()`` and ``system_name``.
    """

    def __init__(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._user_id = user_id
        self._agent_id = agent_id
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the Mem0 client. Must be overridden."""
        raise NotImplementedError

    @property
    def system_type(self) -> SourceSystemType:
        return SourceSystemType.MEMORY_LAYER

    @property
    def system_name(self) -> str:
        raise NotImplementedError

    @property
    def capabilities(self) -> frozenset[SourceCapability]:
        return frozenset(
            {
                SourceCapability.FETCH_ALL,
                SourceCapability.FETCH_METADATA,
                SourceCapability.HEALTH_CHECK,
            }
        )

    async def fetch_all(
        self,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Fetch all memories from Mem0."""
        client = self._get_client()

        kwargs: dict[str, Any] = {}
        if self._user_id:
            kwargs["user_id"] = self._user_id
        if self._agent_id:
            kwargs["agent_id"] = self._agent_id

        memories = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: client.get_all(**kwargs),
        )

        return _parse_mem0_records(
            memories,
            source_system=self.system_name,
            user_id=self._user_id,
            agent_id=self._agent_id,
            limit=limit,
        )

    async def fetch_since(
        self,
        since: datetime,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Mem0 does not support temporal queries natively."""
        raise NotImplementedError(
            f"{self.system_name} adapter does not support incremental sync. "
            "Use fetch_all() instead."
        )

    async def health_check(self) -> dict[str, Any]:
        """Check Mem0 connectivity."""
        try:
            client = self._get_client()
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.get_all(user_id=self._user_id or "healthcheck", limit=1),
            )
            return {
                "healthy": True,
                "message": f"{self.system_name} connected successfully",
                "system": self.system_name,
            }
        except Exception:
            logger.debug("%s health check failed", self.system_name, exc_info=True)
            return {
                "healthy": False,
                "message": f"{self.system_name} connection failed",
                "system": self.system_name,
            }


class Mem0Adapter(_BaseMem0Adapter):
    """Adapter for Mem0 Platform (cloud API, requires API key).

    Usage:
        adapter = Mem0Adapter(api_key="...", user_id="alice")
        records = await adapter.fetch_all()
    """

    def __init__(
        self,
        api_key: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(user_id=user_id, agent_id=agent_id)
        self._api_key = api_key

    def _get_client(self) -> Any:
        """Lazy-initialize Mem0 Platform client."""
        if self._client is None:
            from mem0 import MemoryClient

            api_key = self._api_key or os.environ.get("MEM0_API_KEY")
            if not api_key:
                msg = (
                    "Mem0 API key required. Provide via api_key parameter "
                    "or MEM0_API_KEY environment variable."
                )
                raise ValueError(msg)

            self._client = MemoryClient(api_key=api_key)

        return self._client

    @property
    def system_name(self) -> str:
        return "mem0"


class Mem0SelfHostedAdapter(_BaseMem0Adapter):
    """Adapter for self-hosted Mem0 using ``from mem0 import Memory`` (no API key).

    Usage:
        adapter = Mem0SelfHostedAdapter(user_id="alice")
        records = await adapter.fetch_all()
    """

    def __init__(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(user_id=user_id, agent_id=agent_id)
        self._mem0_config = config

    def _get_client(self) -> Any:
        """Lazy-initialize self-hosted Mem0 Memory instance."""
        if self._client is None:
            from mem0 import Memory

            if self._mem0_config:
                self._client = Memory.from_config(self._mem0_config)
            else:
                self._client = Memory()

        return self._client

    @property
    def system_name(self) -> str:
        return "mem0_self_hosted"
