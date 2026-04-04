"""MCP handler mixin for source registry, provenance tracking, and memory show."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.mcp.tool_handler_utils import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class ProvenanceHandler:
    """Mixin providing source, provenance, and show handler implementations."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    # ========== Source Registry ==========

    async def _source(self, args: dict[str, Any]) -> dict[str, Any]:
        """Manage memory sources (provenance registry)."""
        action = args.get("action", "")
        if not action:
            return {"error": "action is required"}

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for source action '%s'", action)
            return {"error": "No brain configured"}

        if action == "register":
            name = args.get("name")
            if not name or not isinstance(name, str):
                return {"error": "name is required for register"}

            from neural_memory.core.source import Source

            version = str(args.get("version", ""))[:255]
            file_hash = str(args.get("file_hash", ""))[:255]
            raw_metadata = args.get("metadata")
            metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

            try:
                source = Source.create(
                    brain_id=brain_id,
                    name=name,
                    source_type=args.get("source_type", "document"),
                    version=version,
                    file_hash=file_hash,
                    metadata=metadata,
                )
            except ValueError:
                return {"error": f"Invalid source_type: {args.get('source_type')}"}
            source_id = await storage.add_source(source)
            return {
                "source_id": source_id,
                "name": source.name,
                "source_type": source.source_type.value,
                "status": source.status.value,
            }

        if action == "list":
            sources = await storage.list_sources(
                source_type=args.get("source_type"),
                status=args.get("status"),
            )
            return {
                "sources": [
                    {
                        "source_id": s.id,
                        "name": s.name,
                        "source_type": s.source_type.value,
                        "version": s.version,
                        "status": s.status.value,
                        "created_at": s.created_at.isoformat(),
                    }
                    for s in sources
                ],
                "count": len(sources),
            }

        if action == "get":
            source_id = str(args.get("source_id") or "")
            if not source_id:
                return {"error": "source_id is required for get"}
            source = await storage.get_source(source_id)
            if source is None:
                return {"error": f"Source '{source_id}' not found"}
            neuron_count = await storage.count_neurons_for_source(source_id)
            return {
                "source_id": source.id,
                "name": source.name,
                "source_type": source.source_type.value,
                "version": source.version,
                "status": source.status.value,
                "file_hash": source.file_hash,
                "metadata": source.metadata,
                "linked_neuron_count": neuron_count,
                "created_at": source.created_at.isoformat(),
                "updated_at": source.updated_at.isoformat(),
            }

        if action == "update":
            source_id = str(args.get("source_id") or "")
            if not source_id:
                return {"error": "source_id is required for update"}
            updated = await storage.update_source(
                source_id,
                status=args.get("status"),
                version=args.get("version"),
                metadata=args.get("metadata"),
            )
            if not updated:
                return {"error": f"Source '{source_id}' not found"}
            return {"updated": True, "source_id": source_id}

        if action == "delete":
            source_id = str(args.get("source_id") or "")
            if not source_id:
                return {"error": "source_id is required for delete"}
            # Warn about linked neurons
            neuron_count = await storage.count_neurons_for_source(source_id)
            if neuron_count > 0:
                # Soft-delete: mark superseded instead of hard delete
                await storage.update_source(source_id, status="superseded")
                return {
                    "deleted": False,
                    "superseded": True,
                    "source_id": source_id,
                    "warning": f"Source has {neuron_count} linked neurons. "
                    "Marked as superseded instead of deleted.",
                }
            deleted = await storage.delete_source(source_id)
            if not deleted:
                return {"error": f"Source '{source_id}' not found"}
            return {"deleted": True, "source_id": source_id}

        return {"error": f"Unknown action: {action}"}

    # ========== Provenance ==========

    async def _provenance(self, args: dict[str, Any]) -> dict[str, Any]:
        """Trace provenance, verify, or approve a neuron."""
        action = args.get("action", "")
        if not action:
            return {"error": "action is required (trace, verify, approve)"}

        neuron_id = args.get("neuron_id")
        if not neuron_id or not isinstance(neuron_id, str):
            return {"error": "neuron_id is required"}

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for provenance")
            return {"error": "No brain configured"}

        # Verify neuron exists
        neuron = await storage.get_neuron(neuron_id)
        if neuron is None:
            return {"error": f"Neuron '{neuron_id}' not found"}

        if action == "trace":
            return await self._provenance_trace(storage, neuron_id)

        if action == "verify":
            actor = args.get("actor", "mcp_agent")
            return await self._provenance_add_audit(
                storage, neuron_id, SynapseType.VERIFIED_AT, actor
            )

        if action == "approve":
            actor = args.get("actor", "mcp_agent")
            return await self._provenance_add_audit(
                storage, neuron_id, SynapseType.APPROVED_BY, actor
            )

        return {"error": f"Unknown action: {action}. Use trace, verify, or approve."}

    async def _provenance_trace(self, storage: NeuralStorage, neuron_id: str) -> dict[str, Any]:
        """Trace full provenance chain for a neuron."""
        synapses = await storage.get_synapses(target_id=neuron_id)

        chain: list[dict[str, Any]] = []

        for syn in synapses:
            if syn.type == SynapseType.SOURCE_OF:
                source_obj = await storage.get_source(syn.source_id)
                chain.append(
                    {
                        "type": "source",
                        "source_id": syn.source_id,
                        "source_name": source_obj.name if source_obj else None,
                        "source_type": source_obj.source_type.value if source_obj else None,
                        "timestamp": syn.created_at.isoformat() if syn.created_at else None,
                    }
                )
            elif syn.type == SynapseType.STORED_BY:
                chain.append(
                    {
                        "type": "stored_by",
                        "actor": syn.metadata.get("actor", syn.source_id),
                        "tool": syn.metadata.get("tool"),
                        "timestamp": syn.created_at.isoformat() if syn.created_at else None,
                    }
                )
            elif syn.type == SynapseType.VERIFIED_AT:
                chain.append(
                    {
                        "type": "verified",
                        "actor": syn.metadata.get("actor", syn.source_id),
                        "timestamp": syn.created_at.isoformat() if syn.created_at else None,
                    }
                )
            elif syn.type == SynapseType.APPROVED_BY:
                chain.append(
                    {
                        "type": "approved",
                        "actor": syn.metadata.get("actor", syn.source_id),
                        "timestamp": syn.created_at.isoformat() if syn.created_at else None,
                    }
                )

        return {
            "neuron_id": neuron_id,
            "provenance": chain,
            "has_source": any(e["type"] == "source" for e in chain),
            "is_verified": any(e["type"] == "verified" for e in chain),
            "is_approved": any(e["type"] == "approved" for e in chain),
        }

    async def _provenance_add_audit(
        self,
        storage: NeuralStorage,
        neuron_id: str,
        synapse_type: SynapseType,
        actor: str,
    ) -> dict[str, Any]:
        """Add a VERIFIED_AT or APPROVED_BY audit synapse."""
        syn = Synapse.create(
            source_id=neuron_id,
            target_id=neuron_id,  # self-referencing audit
            type=synapse_type,
            weight=1.0,
            metadata={"actor": actor, "tool": "nmem_provenance"},
        )
        await storage.add_synapse(syn)
        return {
            "success": True,
            "neuron_id": neuron_id,
            "action": synapse_type.value,
            "actor": actor,
            "synapse_id": syn.id,
        }

    # ========== Show ==========

    async def _show(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get full verbatim content + metadata + synapses for a memory by ID."""
        memory_id = args.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            return {"error": "memory_id is required"}

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for show")
            return {"error": "No brain configured"}

        # Try as fiber_id first (typed memory), then as neuron_id
        typed_mem = await storage.get_typed_memory(memory_id)
        fiber = await storage.get_fiber(memory_id) if typed_mem else None

        if typed_mem and fiber:
            anchor = await storage.get_neuron(fiber.anchor_neuron_id)
            content = anchor.content if anchor else ""

            # Decrypt if needed
            if anchor and fiber.metadata.get("encrypted"):
                try:
                    from pathlib import Path

                    from neural_memory.safety.encryption import MemoryEncryptor

                    keys_dir_str = getattr(self.config.encryption, "keys_dir", "")
                    keys_dir = (
                        Path(keys_dir_str) if keys_dir_str else (self.config.data_dir / "keys")
                    )
                    encryptor = MemoryEncryptor(keys_dir=keys_dir)
                    bid = storage.brain_id or ""
                    content = encryptor.decrypt(content, bid)
                except Exception:
                    logger.debug("Decryption failed in show", exc_info=True)

            # Get connected synapses
            synapses_out = await storage.get_synapses(source_id=fiber.anchor_neuron_id)
            synapses_in = await storage.get_synapses(target_id=fiber.anchor_neuron_id)
            synapse_list = [
                {
                    "type": s.type.value if hasattr(s.type, "value") else str(s.type),
                    "target_id": s.target_id,
                    "source_id": s.source_id,
                    "weight": s.weight,
                }
                for s in [*synapses_out, *synapses_in]
            ]

            return {
                "memory_id": memory_id,
                "content": content,
                "memory_type": typed_mem.memory_type.value,
                "priority": typed_mem.priority.value,
                "tags": list(typed_mem.tags) if typed_mem.tags else [],
                "created_at": fiber.created_at.isoformat() if fiber.created_at else None,
                "anchor_neuron_id": fiber.anchor_neuron_id,
                "neuron_count": fiber.neuron_count,
                "summary": fiber.summary,
                "metadata": fiber.metadata,
                "synapses": synapse_list,
                "trust_score": typed_mem.trust_score,
                "expires_at": typed_mem.expires_at.isoformat() if typed_mem.expires_at else None,
            }

        # Try as direct neuron_id
        neuron = await storage.get_neuron(memory_id)
        if neuron:
            synapses_out = await storage.get_synapses(source_id=memory_id)
            synapses_in = await storage.get_synapses(target_id=memory_id)
            synapse_list = [
                {
                    "type": s.type.value if hasattr(s.type, "value") else str(s.type),
                    "target_id": s.target_id,
                    "source_id": s.source_id,
                    "weight": s.weight,
                }
                for s in [*synapses_out, *synapses_in]
            ]

            return {
                "memory_id": memory_id,
                "content": neuron.content,
                "neuron_type": neuron.type.value
                if hasattr(neuron.type, "value")
                else str(neuron.type),
                "created_at": neuron.created_at.isoformat() if neuron.created_at else None,
                "metadata": neuron.metadata,
                "synapses": synapse_list,
            }

        return {"error": "Memory not found"}
