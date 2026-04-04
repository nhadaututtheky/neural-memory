"""MCP handler mixin for adaptive instruction refinement and outcome reporting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.constants import MAX_CONTENT_LENGTH
from neural_memory.mcp.tool_handler_utils import _require_brain_id
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class InstructionHandler:
    """Mixin providing instruction refinement and outcome reporting handlers."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    def _ensure_instruction_meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Backfill missing instruction metadata fields (non-destructive).

        Returns a new dict with all required instruction fields present.
        Existing keys in *meta* are preserved unchanged.
        """
        defaults: dict[str, Any] = {
            "version": 1,
            "execution_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": None,
            "last_executed_at": None,
            "failure_modes": [],
            "trigger_patterns": [],
            "refinement_history": [],
        }
        return {**defaults, **meta}

    async def _refine(self, args: dict[str, Any]) -> dict[str, Any]:
        """Refine an instruction or workflow memory.

        Increments the version counter, stores a snapshot in refinement_history,
        appends failure modes / trigger patterns, and persists the updated metadata.
        """
        neuron_id = args.get("neuron_id")
        if not neuron_id or not isinstance(neuron_id, str):
            return {"error": "neuron_id is required"}

        new_content = args.get("new_content")
        reason = args.get("reason", "")
        add_failure_mode = args.get("add_failure_mode")
        add_trigger = args.get("add_trigger")

        if new_content is None and not add_failure_mode and not add_trigger:
            return {
                "error": "At least one of new_content, add_failure_mode, or add_trigger is required"
            }

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for refine")
            return {"error": "No brain configured"}

        # Resolve fiber by ID
        typed_mem = await storage.get_typed_memory(neuron_id)
        fiber = await storage.get_fiber(neuron_id) if typed_mem else None

        if not typed_mem or not fiber:
            return {"error": "Memory not found"}

        # Validate instruction/workflow type
        mem_type_val = typed_mem.memory_type.value
        if mem_type_val not in ("instruction", "workflow"):
            return {
                "error": f"nmem_refine only supports instruction/workflow memories, got '{mem_type_val}'"
            }

        # Backfill metadata if old neuron lacks instruction fields
        meta = self._ensure_instruction_meta(dict(fiber.metadata))

        changes: list[str] = []

        if new_content is not None:
            if len(new_content) > MAX_CONTENT_LENGTH:
                return {
                    "error": f"Content too long ({len(new_content)} chars). Max: {MAX_CONTENT_LENGTH}."
                }
            # Fetch anchor to snapshot old content
            anchor = await storage.get_neuron(fiber.anchor_neuron_id)
            old_content = anchor.content if anchor else ""

            # Increment version and record history
            old_version = meta["version"]
            new_version = old_version + 1
            history_entry: dict[str, Any] = {
                "version": old_version,
                "changed_at": utcnow().isoformat(),
                "reason": reason,
                "old_content": old_content[:100],
            }
            refinement_history: list[dict[str, Any]] = list(meta["refinement_history"])
            if len(refinement_history) >= 10:
                refinement_history = refinement_history[-9:]
            refinement_history.append(history_entry)

            meta = {
                **meta,
                "version": new_version,
                "refinement_history": refinement_history,
            }

            # Update anchor neuron content
            if anchor:
                from dataclasses import replace as dc_replace

                updated_neuron = dc_replace(anchor, content=new_content)
                await storage.update_neuron(updated_neuron)
                changes.append(f"content updated (v{old_version}→v{new_version})")

        if add_failure_mode:
            failure_modes: list[str] = list(meta["failure_modes"])
            if add_failure_mode not in failure_modes:
                failure_modes = [*failure_modes, add_failure_mode]
            if len(failure_modes) > 20:
                failure_modes = failure_modes[-20:]
            meta = {**meta, "failure_modes": failure_modes}
            changes.append(f"failure_mode added: {add_failure_mode[:60]}")

        if add_trigger:
            trigger_patterns: list[str] = list(meta["trigger_patterns"])
            normalized_trigger = add_trigger.lower().strip()
            if normalized_trigger and normalized_trigger not in trigger_patterns:
                trigger_patterns = [*trigger_patterns, normalized_trigger]
            if len(trigger_patterns) > 10:
                trigger_patterns = trigger_patterns[-10:]
            meta = {**meta, "trigger_patterns": trigger_patterns}
            changes.append(f"trigger added: {normalized_trigger}")

        # Persist updated fiber metadata
        await storage.update_fiber_metadata(fiber.id, meta)

        return {
            "status": "refined",
            "memory_id": neuron_id,
            "changes": changes,
            "metadata": {
                "version": meta["version"],
                "execution_count": meta["execution_count"],
                "success_count": meta["success_count"],
                "failure_count": meta["failure_count"],
                "success_rate": meta["success_rate"],
                "last_executed_at": meta["last_executed_at"],
                "failure_modes": meta["failure_modes"],
                "trigger_patterns": meta["trigger_patterns"],
                "refinement_history": meta["refinement_history"],
            },
        }

    async def _report_outcome(self, args: dict[str, Any]) -> dict[str, Any]:
        """Report execution outcome for an instruction or workflow memory.

        Increments execution_count, success_count or failure_count, recomputes
        success_rate, updates last_executed_at, and optionally records failure modes.
        """
        neuron_id = args.get("neuron_id")
        if not neuron_id or not isinstance(neuron_id, str):
            return {"error": "neuron_id is required"}

        raw_success = args.get("success")
        if raw_success is None:
            return {"error": "success (bool) is required"}
        success = bool(raw_success)

        failure_description = args.get("failure_description")
        # context arg accepted but used only for logging / future linking
        _context = args.get("context")

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            logger.error("No brain configured for report_outcome")
            return {"error": "No brain configured"}

        typed_mem = await storage.get_typed_memory(neuron_id)
        fiber = await storage.get_fiber(neuron_id) if typed_mem else None

        if not typed_mem or not fiber:
            return {"error": "Memory not found"}

        mem_type_val = typed_mem.memory_type.value
        if mem_type_val not in ("instruction", "workflow"):
            return {
                "error": f"nmem_report_outcome only supports instruction/workflow memories, got '{mem_type_val}'"
            }

        meta = self._ensure_instruction_meta(dict(fiber.metadata))

        # Increment counters
        new_exec_count = meta["execution_count"] + 1
        new_success_count = meta["success_count"] + (1 if success else 0)
        new_failure_count = meta["failure_count"] + (0 if success else 1)
        new_success_rate = new_success_count / new_exec_count
        new_last_executed = utcnow().isoformat()

        # Append failure mode if provided
        failure_modes: list[str] = list(meta["failure_modes"])
        if not success and failure_description:
            if failure_description not in failure_modes:
                failure_modes = [*failure_modes, failure_description]
            if len(failure_modes) > 20:
                failure_modes = failure_modes[-20:]

        updated_meta = {
            **meta,
            "execution_count": new_exec_count,
            "success_count": new_success_count,
            "failure_count": new_failure_count,
            "success_rate": round(new_success_rate, 4),
            "last_executed_at": new_last_executed,
            "failure_modes": failure_modes,
        }

        await storage.update_fiber_metadata(fiber.id, updated_meta)

        return {
            "status": "recorded",
            "memory_id": neuron_id,
            "success": success,
            "execution_count": new_exec_count,
            "success_count": new_success_count,
            "failure_count": new_failure_count,
            "success_rate": round(new_success_rate, 4),
            "last_executed_at": new_last_executed,
            "failure_modes": failure_modes,
        }
