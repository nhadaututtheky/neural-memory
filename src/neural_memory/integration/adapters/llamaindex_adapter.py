"""LlamaIndex source adapter for importing index nodes and embeddings."""

from __future__ import annotations

import asyncio
import logging
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


class LlamaIndexAdapter:
    """Adapter for importing nodes from a LlamaIndex index.

    Supports loading from a persisted index directory or accepting
    a live index object directly. Extracts text nodes, embeddings,
    and parent/child relationships.

    Usage:
        adapter = LlamaIndexAdapter(persist_dir="/path/to/index")
        records = await adapter.fetch_all()

        # Or with a live index:
        adapter = LlamaIndexAdapter(index=my_index)
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        index: Any | None = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._index = index

    def _get_index(self) -> Any:
        """Lazy-load or return the LlamaIndex index."""
        if self._index is not None:
            return self._index

        if self._persist_dir is None:
            msg = (
                "LlamaIndex adapter requires either 'persist_dir' or 'index' parameter. "
                "Provide a path to a persisted index or a live index object."
            )
            raise ValueError(msg)

        from llama_index.core import (
            StorageContext,
            load_index_from_storage,
        )

        storage_context = StorageContext.from_defaults(persist_dir=self._persist_dir)
        self._index = load_index_from_storage(storage_context)
        return self._index

    @property
    def system_type(self) -> SourceSystemType:
        return SourceSystemType.INDEX_STORE

    @property
    def system_name(self) -> str:
        return "llamaindex"

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
        """Fetch all nodes from the LlamaIndex index."""
        index = await asyncio.get_running_loop().run_in_executor(None, self._get_index)

        docstore = index.docstore
        all_docs = docstore.docs

        records: list[ExternalRecord] = []

        for node_id, node in all_docs.items():
            if limit and len(records) >= limit:
                break

            content = getattr(node, "text", "") or getattr(node, "content", "")
            if not content:
                continue

            metadata = getattr(node, "metadata", {}) or {}
            embedding = getattr(node, "embedding", None)

            created_at = utcnow()
            if "created_at" in metadata:
                try:
                    created_at = datetime.fromisoformat(str(metadata["created_at"]))
                except (ValueError, TypeError):
                    pass

            # Extract parent/child relationships
            relationships: list[ExternalRelationship] = []
            node_relationships = getattr(node, "relationships", {}) or {}
            for rel_type, related_info in node_relationships.items():
                rel_type_str = str(rel_type)
                if "PARENT" in rel_type_str.upper():
                    relation = "parent_of"
                elif "CHILD" in rel_type_str.upper():
                    relation = "child_of"
                elif "NEXT" in rel_type_str.upper():
                    relation = "leads_to"
                elif "PREVIOUS" in rel_type_str.upper():
                    relation = "preceded_by"
                else:
                    relation = "related_to"

                # related_info can be a single RelatedNodeInfo or a list
                related_items = related_info if isinstance(related_info, list) else [related_info]
                for item in related_items:
                    target_id = str(getattr(item, "node_id", "") or item) if item else ""
                    if target_id:
                        relationships.append(
                            ExternalRelationship(
                                source_record_id=str(node_id),
                                target_record_id=target_id,
                                relation_type=relation,
                            )
                        )

            tags: set[str] = {"import:llamaindex"}
            if collection:
                tags.add(f"collection:{collection}")

            node_type = type(node).__name__
            source_type = "text_node" if "Text" in node_type else "document"

            record = ExternalRecord.create(
                id=str(node_id),
                source_system="llamaindex",
                content=content,
                source_collection=collection or "default",
                created_at=created_at,
                source_type=source_type,
                metadata=metadata,
                embedding=embedding,
                tags=tags,
                relationships=relationships if relationships else None,
            )
            records.append(record)

        return records

    async def fetch_since(
        self,
        since: datetime,
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[ExternalRecord]:
        """LlamaIndex does not support temporal queries."""
        raise NotImplementedError(
            "LlamaIndex adapter does not support incremental sync. Use fetch_all() instead."
        )

    async def health_check(self) -> dict[str, Any]:
        """Check LlamaIndex index accessibility."""
        try:
            index = await asyncio.get_running_loop().run_in_executor(None, self._get_index)
            doc_count = len(index.docstore.docs)
            return {
                "healthy": True,
                "message": f"LlamaIndex index loaded ({doc_count} documents)",
                "system": "llamaindex",
            }
        except Exception as e:
            logger.warning("LlamaIndex health check failed: %s", e)
            return {
                "healthy": False,
                "message": "LlamaIndex index failed",
                "system": "llamaindex",
            }
