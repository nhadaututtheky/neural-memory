"""Graphiti source adapter for importing bi-temporal knowledge graph data."""

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


class GraphitiAdapter:
    """Adapter for importing knowledge from Graphiti (by Zep) bi-temporal graph.

    Graphiti maintains temporal metadata (valid_at/invalid_at) on edges,
    making it ideal for tracking how knowledge evolves over time.

    Usage:
        adapter = GraphitiAdapter(uri="bolt://localhost:7687")
        records = await adapter.fetch_all()
    """

    _ALLOWED_SCHEMES = frozenset({"bolt://", "bolt+s://", "bolt+ssc://"})

    def __init__(
        self,
        uri: str | None = None,
        group_id: str | None = None,
    ) -> None:
        resolved_uri = uri or os.environ.get("GRAPHITI_URI", "bolt://localhost:7687")
        if not any(resolved_uri.startswith(s) for s in self._ALLOWED_SCHEMES):
            raise ValueError(
                f"Invalid Graphiti URI scheme. Must start with one of: "
                f"{', '.join(sorted(self._ALLOWED_SCHEMES))}"
            )
        self._uri = resolved_uri
        self._group_id = group_id or os.environ.get("GRAPHITI_GROUP_ID")
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize Graphiti client."""
        if self._client is None:
            from graphiti_core import Graphiti

            self._client = Graphiti(uri=self._uri)

        return self._client

    @property
    def system_type(self) -> SourceSystemType:
        return SourceSystemType.GRAPH_STORE

    @property
    def system_name(self) -> str:
        return "graphiti"

    @property
    def capabilities(self) -> frozenset[SourceCapability]:
        return frozenset(
            {
                SourceCapability.FETCH_ALL,
                SourceCapability.FETCH_SINCE,
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
        """Fetch all nodes and episodes from Graphiti."""
        client = self._get_client()
        effective_limit = limit or 100

        # Fetch nodes and episodes (Graphiti has async API)
        nodes = await client.retrieve_nodes(query="*", num_results=effective_limit)
        episodes = await client.retrieve_episodes(query="*", num_results=effective_limit)

        records: list[ExternalRecord] = []
        all_relationships: list[ExternalRelationship] = []

        # Process nodes
        for node in nodes or []:
            if limit and len(records) >= limit:
                break

            node_id = str(getattr(node, "id", "") or getattr(node, "uuid", ""))
            name = getattr(node, "name", "")
            summary = getattr(node, "summary", "") or name
            if not summary:
                continue

            created_at = utcnow()
            raw_created = getattr(node, "created_at", None)
            if raw_created:
                try:
                    if isinstance(raw_created, datetime):
                        created_at = raw_created
                    else:
                        created_at = datetime.fromisoformat(str(raw_created).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            metadata: dict[str, Any] = {}
            if name:
                metadata["name"] = name
            group = getattr(node, "group_id", self._group_id)
            if group:
                metadata["group_id"] = group

            tags: set[str] = {"import:graphiti", "entity"}
            if collection:
                tags.add(f"collection:{collection}")

            record = ExternalRecord.create(
                id=node_id or f"graphiti-node-{len(records)}",
                source_system="graphiti",
                content=summary,
                source_collection=collection or "default",
                created_at=created_at,
                source_type="entity",
                metadata=metadata,
                tags=tags,
            )
            records.append(record)

        # Process episodes (edges/facts)
        for episode in episodes or []:
            if limit and len(records) >= limit:
                break

            ep_id = str(getattr(episode, "id", "") or getattr(episode, "uuid", ""))
            content = getattr(episode, "content", "") or getattr(episode, "name", "")
            if not content:
                continue

            created_at = utcnow()
            raw_created = getattr(episode, "created_at", None)
            if raw_created:
                try:
                    if isinstance(raw_created, datetime):
                        created_at = raw_created
                    else:
                        created_at = datetime.fromisoformat(str(raw_created).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            # Compute temporal weight from valid_at/invalid_at
            valid_at = getattr(episode, "valid_at", None)
            invalid_at = getattr(episode, "invalid_at", None)
            weight = self._compute_temporal_weight(valid_at, invalid_at)

            metadata = {}
            if valid_at:
                metadata["valid_at"] = str(valid_at)
            if invalid_at:
                metadata["invalid_at"] = str(invalid_at)

            # Build relationship if source/target available
            source_id = str(getattr(episode, "source_node_id", "") or "")
            target_id = str(getattr(episode, "target_node_id", "") or "")
            relation_type = getattr(episode, "name", "related_to") or "related_to"

            if source_id and target_id:
                all_relationships.append(
                    ExternalRelationship(
                        source_record_id=source_id,
                        target_record_id=target_id,
                        relation_type=relation_type,
                        weight=weight,
                        metadata=metadata,
                    )
                )

            tags = {"import:graphiti", "episode"}
            if collection:
                tags.add(f"collection:{collection}")

            record = ExternalRecord.create(
                id=ep_id or f"graphiti-episode-{len(records)}",
                source_system="graphiti",
                content=content,
                source_collection=collection or "default",
                created_at=created_at,
                source_type="episode",
                metadata=metadata,
                tags=tags,
                relationships=list(all_relationships) if all_relationships else None,
            )
            records.append(record)
            all_relationships = []

        return records

    async def fetch_since(
        self,
        since: datetime,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """Fetch records created or updated since a given timestamp."""
        client = self._get_client()
        effective_limit = limit or 100

        # Use reference_time for episodes
        episodes = await client.retrieve_episodes(
            query="*",
            num_results=effective_limit,
            reference_time=since,
        )

        # Fetch all nodes, then filter client-side
        nodes = await client.retrieve_nodes(query="*", num_results=effective_limit)

        filtered_nodes = []
        for node in nodes or []:
            raw_created = getattr(node, "created_at", None)
            if raw_created:
                try:
                    node_time = (
                        raw_created
                        if isinstance(raw_created, datetime)
                        else datetime.fromisoformat(str(raw_created).replace("Z", "+00:00"))
                    )
                    if node_time >= since:
                        filtered_nodes.append(node)
                except (ValueError, TypeError):
                    pass

        records: list[ExternalRecord] = []

        for node in filtered_nodes:
            if limit and len(records) >= limit:
                break

            node_id = str(getattr(node, "id", "") or getattr(node, "uuid", ""))
            summary = getattr(node, "summary", "") or getattr(node, "name", "")
            if not summary:
                continue

            created_at = utcnow()
            raw_created = getattr(node, "created_at", None)
            if raw_created:
                try:
                    created_at = (
                        raw_created
                        if isinstance(raw_created, datetime)
                        else datetime.fromisoformat(str(raw_created).replace("Z", "+00:00"))
                    )
                except (ValueError, TypeError):
                    pass

            record = ExternalRecord.create(
                id=node_id or f"graphiti-node-{len(records)}",
                source_system="graphiti",
                content=summary,
                source_collection=collection or "default",
                created_at=created_at,
                source_type="entity",
                tags={"import:graphiti", "entity"},
            )
            records.append(record)

        for episode in episodes or []:
            if limit and len(records) >= limit:
                break

            ep_id = str(getattr(episode, "id", "") or getattr(episode, "uuid", ""))
            content = getattr(episode, "content", "") or getattr(episode, "name", "")
            if not content:
                continue

            created_at = utcnow()
            raw_created = getattr(episode, "created_at", None)
            if raw_created:
                try:
                    created_at = (
                        raw_created
                        if isinstance(raw_created, datetime)
                        else datetime.fromisoformat(str(raw_created).replace("Z", "+00:00"))
                    )
                except (ValueError, TypeError):
                    pass

            record = ExternalRecord.create(
                id=ep_id or f"graphiti-episode-{len(records)}",
                source_system="graphiti",
                content=content,
                source_collection=collection or "default",
                created_at=created_at,
                source_type="episode",
                tags={"import:graphiti", "episode"},
            )
            records.append(record)

        return records

    async def health_check(self) -> dict[str, Any]:
        """Check Graphiti connectivity."""
        try:
            client = self._get_client()
            await client.retrieve_nodes(query="test", num_results=1)
            return {
                "healthy": True,
                "message": "Graphiti connected successfully",
                "system": "graphiti",
            }
        except Exception as e:
            logger.warning("Graphiti health check failed: %s", e)
            return {
                "healthy": False,
                "message": "Graphiti connection failed",
                "system": "graphiti",
            }

    @staticmethod
    def _compute_temporal_weight(
        valid_at: Any | None,
        invalid_at: Any | None,
    ) -> float:
        """Compute edge weight based on temporal validity.

        Currently valid edges (no invalid_at) get higher weight.
        Expired edges get reduced weight.
        """
        if invalid_at is not None:
            return 0.3
        if valid_at is not None:
            return 0.8
        return 0.5
