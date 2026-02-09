"""ChromaDB source adapter for importing documents and embeddings."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from neural_memory.integration.models import (
    ExternalRecord,
    SourceCapability,
    SourceSystemType,
)
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class ChromaDBAdapter:
    """Adapter for importing memories from ChromaDB collections.

    Supports both persistent (file path) and client-server modes.

    Usage:
        adapter = ChromaDBAdapter(path="/path/to/chroma/persist")
        records = await adapter.fetch_all(collection="my_collection")
    """

    def __init__(
        self,
        path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        collection: str | None = None,
    ) -> None:
        self._path = path
        self._host = host
        self._port = port
        self._default_collection = collection
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize ChromaDB client."""
        if self._client is None:
            import chromadb  # type: ignore[import-untyped]

            if self._host:
                self._client = chromadb.HttpClient(
                    host=self._host,
                    port=self._port or 8000,
                )
            elif self._path:
                self._client = chromadb.PersistentClient(path=self._path)
            else:
                self._client = chromadb.Client()

        return self._client

    @property
    def system_type(self) -> SourceSystemType:
        return SourceSystemType.VECTOR_DB

    @property
    def system_name(self) -> str:
        return "chromadb"

    @property
    def capabilities(self) -> frozenset[SourceCapability]:
        return frozenset(
            {
                SourceCapability.FETCH_ALL,
                SourceCapability.FETCH_EMBEDDINGS,
                SourceCapability.FETCH_METADATA,
                SourceCapability.HEALTH_CHECK,
            }
        )

    async def fetch_all(
        self,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Fetch all documents from a ChromaDB collection."""
        client = self._get_client()
        collection_name = collection or self._default_collection

        if collection_name is None:
            collections = await asyncio.get_running_loop().run_in_executor(
                None,
                client.list_collections,
            )
            records: list[ExternalRecord] = []
            for coll in collections:
                coll_records = await self._fetch_collection(coll.name, limit=limit)
                records.extend(coll_records)
                if limit and len(records) >= limit:
                    return records[:limit]
            return records

        return await self._fetch_collection(collection_name, limit=limit)

    async def _fetch_collection(
        self,
        collection_name: str,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Fetch all documents from a single collection."""
        client = self._get_client()

        try:
            coll = await asyncio.get_running_loop().run_in_executor(
                None,
                client.get_collection,
                collection_name,
            )
        except Exception as e:
            logger.warning("Failed to access collection %s: %s", collection_name, e)
            return []

        kwargs: dict[str, Any] = {"include": ["documents", "metadatas", "embeddings"]}
        if limit:
            kwargs["limit"] = limit

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: coll.get(**kwargs),
        )

        records: list[ExternalRecord] = []
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        embeddings = result.get("embeddings")

        for i, doc_id in enumerate(ids):
            content = documents[i] if i < len(documents) and documents[i] else ""
            if not content:
                continue

            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            embedding = embeddings[i] if embeddings and i < len(embeddings) else None

            created_at = utcnow()
            if "created_at" in metadata:
                try:
                    created_at = datetime.fromisoformat(str(metadata["created_at"]))
                except (ValueError, TypeError):
                    pass

            tags: set[str] = set()
            if "tags" in metadata and isinstance(metadata["tags"], (list, str)):
                if isinstance(metadata["tags"], str):
                    tags = {t.strip() for t in metadata["tags"].split(",")}
                else:
                    tags = set(metadata["tags"])

            record = ExternalRecord.create(
                id=doc_id,
                source_system="chromadb",
                content=content,
                source_collection=collection_name,
                created_at=created_at,
                source_type=metadata.get("type", "document"),
                metadata=metadata,
                embedding=embedding,
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
        """ChromaDB does not natively support temporal queries.

        Falls back to fetching all and filtering client-side.
        """
        all_records = await self.fetch_all(collection=collection, limit=None)
        filtered = [
            r
            for r in all_records
            if r.created_at >= since or (r.updated_at and r.updated_at >= since)
        ]
        if limit:
            filtered = filtered[:limit]
        return filtered

    async def health_check(self) -> dict[str, Any]:
        """Check ChromaDB connectivity."""
        try:
            client = self._get_client()
            heartbeat = await asyncio.get_running_loop().run_in_executor(
                None,
                client.heartbeat,
            )
            return {
                "healthy": True,
                "message": f"ChromaDB connected (heartbeat: {heartbeat})",
                "system": "chromadb",
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"ChromaDB connection failed: {e}",
                "system": "chromadb",
            }
