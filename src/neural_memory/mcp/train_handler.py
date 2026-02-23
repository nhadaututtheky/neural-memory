"""Doc-to-brain training handler for MCP server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from neural_memory.mcp.tool_handlers import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Validation limits
_MAX_DOMAIN_TAG_LEN = 100
_MAX_BRAIN_NAME_LEN = 64
_ALLOWED_EXTENSIONS = frozenset({".md", ".mdx", ".txt", ".rst"})


class TrainHandler:
    """Mixin: doc-to-brain training tool handlers."""

    async def get_storage(self) -> NeuralStorage:
        raise NotImplementedError

    async def _train(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle doc-to-brain training actions."""
        action = args.get("action", "train")

        if action == "train":
            return await self._train_docs(args)
        elif action == "status":
            return await self._train_status()
        return {"error": f"Unknown train action: {action}"}

    async def _train_docs(self, args: dict[str, Any]) -> dict[str, Any]:
        """Train a brain from documentation files."""
        from pathlib import Path

        from neural_memory.engine.doc_trainer import DocTrainer, TrainingConfig

        storage = await self.get_storage()
        brain = await storage.get_brain(_require_brain_id(storage))
        if not brain:
            return {"error": "No brain configured"}

        # Validate domain_tag length
        domain_tag = args.get("domain_tag", "")
        if len(domain_tag) > _MAX_DOMAIN_TAG_LEN:
            return {"error": f"domain_tag too long (max {_MAX_DOMAIN_TAG_LEN} chars)"}

        # Validate brain_name length
        brain_name = args.get("brain_name", "")
        if len(brain_name) > _MAX_BRAIN_NAME_LEN:
            return {"error": f"brain_name too long (max {_MAX_BRAIN_NAME_LEN} chars)"}

        # Validate extensions
        extensions_raw = args.get("extensions", [".md"])
        for ext in extensions_raw:
            if ext not in _ALLOWED_EXTENSIONS:
                return {
                    "error": f"Extension '{ext}' not allowed. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
                }

        # Validate path (security: must be within CWD)
        cwd = Path(".").resolve()
        path = Path(args.get("path", ".")).resolve()

        if not path.exists():
            return {"error": "Path not found"}
        if not path.is_relative_to(cwd):
            return {"error": "Path must be within working directory"}

        # Build training config
        tc = TrainingConfig(
            domain_tag=domain_tag,
            brain_name=brain_name,
            extensions=tuple(extensions_raw),
            consolidate=args.get("consolidate", True),
        )

        trainer = DocTrainer(storage, brain.config)
        storage.disable_auto_save()

        try:
            if path.is_file():
                result = await trainer.train_file(path, tc)
            else:
                result = await trainer.train_directory(path, tc)
            await storage.batch_save()
        except Exception as exc:
            logger.error("Training failed: %s", exc, exc_info=True)
            return {"error": "Training failed unexpectedly"}
        finally:
            storage.enable_auto_save()

        response: dict[str, Any] = {
            "files_processed": result.files_processed,
            "chunks_encoded": result.chunks_encoded,
            "chunks_skipped": result.chunks_skipped,
            "neurons_created": result.neurons_created,
            "synapses_created": result.synapses_created,
            "hierarchy_synapses": result.hierarchy_synapses,
            "session_synapses": result.session_synapses,
            "enrichment_synapses": result.enrichment_synapses,
            "brain_name": result.brain_name,
            "message": (
                f"Trained {result.chunks_encoded} chunks from {result.files_processed} files"
                if result.chunks_encoded > 0
                else "No documentation chunks found"
            ),
        }
        if result.chunks_failed > 0:
            response["chunks_failed"] = result.chunks_failed
        return response

    async def _train_status(self) -> dict[str, Any]:
        """Show doc-to-brain training status."""
        storage = await self.get_storage()

        from neural_memory.core.neuron import NeuronType

        # Query with type filter for efficiency, then filter by metadata
        doc_neurons = await storage.find_neurons(
            type=NeuronType.CONCEPT,
            limit=1000,
        )
        trained_count = sum(1 for n in doc_neurons if n.metadata.get("doc_train"))

        return {
            "trained_chunks": trained_count,
            "has_training_data": trained_count > 0,
            "message": (
                f"Brain has {trained_count}+ trained document chunks"
                if trained_count > 0
                else "No training data. Use nmem_train(action='train', path='docs/') to train."
            ),
        }
