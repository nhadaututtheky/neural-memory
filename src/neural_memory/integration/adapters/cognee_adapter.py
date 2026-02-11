"""Cognee source adapter for importing knowledge graph data."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from neural_memory.integration.models import (
    ExternalRecord,
    ExternalRelationship,
    SourceCapability,
    SourceSystemType,
)
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class CogneeAdapter:
    """Adapter for importing knowledge from Cognee's concept graph.

    Cognee organizes knowledge into concept nodes with relationship edges.
    This adapter fetches chunks and maps graph edges to ExternalRelationship
    tuples for synapse creation in NeuralMemory.

    Usage:
        adapter = CogneeAdapter(api_key="...")
        records = await adapter.fetch_all()
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client_initialized = False

    def _get_client(self) -> Any:
        """Lazy-initialize Cognee client."""
        if not self._client_initialized:
            import cognee

            api_key = self._api_key or os.environ.get("COGNEE_API_KEY")
            if api_key:
                cognee.config.set_llm_api_key(api_key)

            self._client_initialized = True
            return cognee

        import cognee

        return cognee

    @property
    def system_type(self) -> SourceSystemType:
        return SourceSystemType.GRAPH_STORE

    @property
    def system_name(self) -> str:
        return "cognee"

    @property
    def capabilities(self) -> frozenset[SourceCapability]:
        return frozenset(
            {
                SourceCapability.FETCH_ALL,
                SourceCapability.FETCH_RELATIONSHIPS,
                SourceCapability.FETCH_METADATA,
                SourceCapability.HEALTH_CHECK,
            }
        )

    async def fetch_all(
        self,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Fetch all knowledge chunks and relationships from Cognee."""
        cognee = self._get_client()
        from cognee.api.v1.search import SearchType

        search_results = await cognee.search(query_text="*", query_type=SearchType.CHUNKS)

        if not search_results:
            return []

        records: list[ExternalRecord] = []
        relationships: list[ExternalRelationship] = []

        for i, result in enumerate(search_results):
            if limit and len(records) >= limit:
                break

            # Handle both dict and object results
            if isinstance(result, dict):
                chunk_id = str(result.get("id", f"cognee-chunk-{i}"))
                content = result.get("text", "") or result.get("content", "")
                metadata = {k: v for k, v in result.items() if k not in ("id", "text", "content")}
                created_at_raw = result.get("created_at")
            else:
                chunk_id = str(getattr(result, "id", f"cognee-chunk-{i}"))
                content = getattr(result, "text", "") or getattr(result, "content", "")
                metadata = getattr(result, "metadata", {}) or {}
                created_at_raw = getattr(result, "created_at", None)

            if not content:
                continue

            created_at = utcnow()
            if created_at_raw:
                try:
                    created_at = datetime.fromisoformat(str(created_at_raw).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            tags: set[str] = {"import:cognee"}
            if collection:
                tags.add(f"collection:{collection}")

            # Extract graph edges as relationships
            edges = (
                result.get("edges", [])
                if isinstance(result, dict)
                else getattr(result, "edges", [])
            )
            for edge in edges or []:
                if isinstance(edge, dict):
                    source_id = str(edge.get("source_id", chunk_id))
                    target_id = str(edge.get("target_id", ""))
                    relation = edge.get("relationship_type", "related_to")
                    weight = float(edge.get("weight", 0.5))
                else:
                    source_id = str(getattr(edge, "source_id", chunk_id))
                    target_id = str(getattr(edge, "target_id", ""))
                    relation = getattr(edge, "relationship_type", "related_to")
                    weight = float(getattr(edge, "weight", 0.5))

                if target_id:
                    relationships.append(
                        ExternalRelationship(
                            source_record_id=source_id,
                            target_record_id=target_id,
                            relation_type=relation,
                            weight=weight,
                        )
                    )

            record = ExternalRecord.create(
                id=chunk_id,
                source_system="cognee",
                content=content,
                source_collection=collection or "default",
                created_at=created_at,
                source_type="knowledge",
                metadata=metadata,
                tags=tags,
                relationships=list(relationships) if relationships else None,
            )
            records.append(record)
            relationships = []

        return records

    async def fetch_since(
        self,
        since: datetime,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Cognee does not support temporal queries."""
        raise NotImplementedError(
            "Cognee adapter does not support incremental sync. Use fetch_all() instead."
        )

    async def health_check(self) -> dict[str, Any]:
        """Check Cognee connectivity."""
        try:
            cognee = self._get_client()
            from cognee.api.v1.search import SearchType

            await cognee.search(query_text="test", query_type=SearchType.CHUNKS)
            return {
                "healthy": True,
                "message": "Cognee connected successfully",
                "system": "cognee",
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Cognee connection failed: {e}",
                "system": "cognee",
            }
