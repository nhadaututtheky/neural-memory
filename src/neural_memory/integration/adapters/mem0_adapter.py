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


class Mem0Adapter:
    """Adapter for importing memories from Mem0 memory stores.

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
        self._api_key = api_key
        self._user_id = user_id
        self._agent_id = agent_id
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize Mem0 client."""
        if self._client is None:
            from mem0 import MemoryClient  # type: ignore[import-untyped]

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
    def system_type(self) -> SourceSystemType:
        return SourceSystemType.MEMORY_LAYER

    @property
    def system_name(self) -> str:
        return "mem0"

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

        records: list[ExternalRecord] = []
        items = memories if isinstance(memories, list) else memories.get("results", [])

        for mem in items:
            if limit and len(records) >= limit:
                break

            mem_id = mem.get("id", "")
            content = mem.get("memory", "") or mem.get("text", "")
            if not content:
                continue

            metadata = mem.get("metadata", {}) or {}

            created_at = utcnow()
            if "created_at" in mem:
                try:
                    created_at = datetime.fromisoformat(
                        str(mem["created_at"]).replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            updated_at = None
            if "updated_at" in mem:
                try:
                    updated_at = datetime.fromisoformat(
                        str(mem["updated_at"]).replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            source_type = mem.get("category", metadata.get("type", "memory"))

            tags: set[str] = set()
            if "categories" in mem:
                tags = set(mem["categories"])
            if self._user_id:
                tags.add(f"user:{self._user_id}")
            if self._agent_id:
                tags.add(f"agent:{self._agent_id}")

            record = ExternalRecord.create(
                id=str(mem_id),
                source_system="mem0",
                content=content,
                source_collection=self._user_id or self._agent_id or "default",
                created_at=created_at,
                updated_at=updated_at,
                source_type=source_type,
                metadata=metadata,
                tags=tags,
            )
            records.append(record)

        return records

    async def fetch_since(
        self,
        since: datetime,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Mem0 does not support temporal queries natively."""
        raise NotImplementedError(
            "Mem0 adapter does not support incremental sync. Use fetch_all() instead."
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
                "message": "Mem0 connected successfully",
                "system": "mem0",
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Mem0 connection failed: {e}",
                "system": "mem0",
            }
