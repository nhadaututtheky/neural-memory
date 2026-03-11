"""MCP handler mixin for cognitive layer tools (hypothesis, evidence, prediction)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from neural_memory.mcp.tool_handlers import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

# Valid hypothesis statuses (must match CHECK constraint in schema)
_VALID_STATUSES = frozenset({"active", "confirmed", "refuted", "superseded", "pending", "expired"})


class CognitiveHandler:
    """Mixin providing hypothesis, evidence, prediction, and verify tool handlers for MCPServer."""

    if TYPE_CHECKING:
        config: UnifiedConfig

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

    async def _hypothesize(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_hypothesize tool calls.

        Create or list hypotheses — evolving beliefs tracked with
        Bayesian confidence updates and auto-resolution.
        """
        action = args.get("action", "create")
        if action not in ("create", "list", "get"):
            return {"error": f"Invalid action: {action}. Must be 'create', 'list', or 'get'."}

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}

        if action == "create":
            return await self._hypothesize_create(storage, args)
        elif action == "list":
            return await self._hypothesize_list(storage, args)
        else:  # get
            return await self._hypothesize_get(storage, args)

    async def _hypothesize_create(
        self, storage: NeuralStorage, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a new hypothesis neuron with cognitive state."""
        content = args.get("content", "").strip()
        if not content:
            return {"error": "content is required"}
        if len(content) > 100_000:
            return {"error": f"Content too long ({len(content)} chars). Max: 100,000."}

        initial_confidence = args.get("confidence", 0.5)
        try:
            initial_confidence = max(0.01, min(0.99, float(initial_confidence)))
        except (TypeError, ValueError):
            initial_confidence = 0.5

        tags = set()
        for t in args.get("tags", []):
            if isinstance(t, str) and len(t) <= 100:
                tags.add(t)

        # Create hypothesis neuron via encoder
        from neural_memory.core.memory_types import (
            MemoryType,
            Priority,
            TypedMemory,
            get_decay_rate,
        )
        from neural_memory.core.neuron import NeuronState
        from neural_memory.engine.encoder import MemoryEncoder
        from neural_memory.utils.timeutils import utcnow

        brain_id = storage.current_brain_id
        assert brain_id is not None
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "Brain not found"}

        encoder = MemoryEncoder(storage, brain.config)

        try:
            storage.disable_auto_save()

            result = await encoder.encode(
                content=content, timestamp=utcnow(), tags=tags if tags else None
            )

            # Create TypedMemory as hypothesis type
            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=MemoryType.HYPOTHESIS,
                priority=Priority.from_int(args.get("priority", 6)),
                source="mcp_cognitive",
                expires_in_days=180,
                tags=tags if tags else None,
            )
            await storage.add_typed_memory(typed_mem)

            # Set hypothesis decay rate on neuron states
            decay_rate = get_decay_rate("hypothesis")
            for neuron in result.neurons_created:
                state = await storage.get_neuron_state(neuron.id)
                if state and state.decay_rate != decay_rate:
                    updated_state = NeuronState(
                        neuron_id=state.neuron_id,
                        activation_level=state.activation_level,
                        access_frequency=state.access_frequency,
                        last_activated=state.last_activated,
                        decay_rate=decay_rate,
                        created_at=state.created_at,
                    )
                    await storage.update_neuron_state(updated_state)

            # Create cognitive state record (check for existing to avoid overwrite)
            anchor_id = result.fiber.anchor_neuron_id
            existing = await storage.get_cognitive_state(anchor_id)
            if existing:
                return {
                    "status": "existing",
                    "hypothesis_id": anchor_id,
                    "confidence": existing["confidence"],
                    "message": "Hypothesis with identical content already exists",
                }

            await storage.upsert_cognitive_state(
                anchor_id,
                confidence=initial_confidence,
                status="active",
            )

            await storage.batch_save()
        except Exception:
            logger.error("Hypothesis create failed", exc_info=True)
            return {"error": "Failed to create hypothesis"}
        finally:
            storage.enable_auto_save()

        return {
            "status": "created",
            "hypothesis_id": anchor_id,
            "fiber_id": result.fiber.id,
            "confidence": initial_confidence,
            "content_preview": content[:120],
            "neurons_created": len(result.neurons_created),
            "synapses_created": len(result.synapses_created),
        }

    async def _hypothesize_list(
        self, storage: NeuralStorage, args: dict[str, Any]
    ) -> dict[str, Any]:
        """List hypotheses with their cognitive state."""
        status_filter = args.get("status")
        if status_filter and status_filter not in _VALID_STATUSES:
            return {"error": f"Invalid status: {status_filter}. Valid: {sorted(_VALID_STATUSES)}"}

        try:
            limit = min(int(args.get("limit", 20)), 100)
        except (TypeError, ValueError):
            limit = 20

        try:
            states = await storage.list_cognitive_states(
                status=status_filter,
                limit=limit,
            )

            # Enrich with neuron content
            hypotheses = []
            for state in states:
                neuron = await storage.get_neuron(state["neuron_id"])
                content = neuron.content if neuron else "(deleted)"
                hypotheses.append(
                    {
                        "hypothesis_id": state["neuron_id"],
                        "content": content[:200] if content else "",
                        "confidence": state["confidence"],
                        "evidence_for": state["evidence_for_count"],
                        "evidence_against": state["evidence_against_count"],
                        "status": state["status"],
                        "last_evidence_at": state.get("last_evidence_at"),
                        "created_at": state["created_at"],
                    }
                )

            return {
                "count": len(hypotheses),
                "hypotheses": hypotheses,
            }
        except Exception:
            logger.error("Hypothesis list failed", exc_info=True)
            return {"error": "Failed to list hypotheses"}

    async def _hypothesize_get(
        self, storage: NeuralStorage, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Get a single hypothesis with full details."""
        hypothesis_id = args.get("hypothesis_id", "").strip()
        if not hypothesis_id:
            return {"error": "hypothesis_id is required"}

        try:
            state = await storage.get_cognitive_state(hypothesis_id)
            if not state:
                return {"error": f"No hypothesis found with id: {hypothesis_id}"}

            neuron = await storage.get_neuron(hypothesis_id)
            content = neuron.content if neuron else "(deleted)"

            # Get evidence synapses (capped at 50 to avoid unbounded fetch)
            from neural_memory.core.synapse import SynapseType

            evidence_for: list[dict[str, Any]] = []
            evidence_against: list[dict[str, Any]] = []

            synapses = await storage.get_synapses(target_id=hypothesis_id)
            for syn in synapses[:50]:
                if syn.type == SynapseType.EVIDENCE_FOR:
                    src_neuron = await storage.get_neuron(syn.source_id)
                    evidence_for.append(
                        {
                            "neuron_id": syn.source_id,
                            "content": (src_neuron.content[:120] if src_neuron else "(deleted)"),
                            "weight": syn.weight,
                        }
                    )
                elif syn.type == SynapseType.EVIDENCE_AGAINST:
                    src_neuron = await storage.get_neuron(syn.source_id)
                    evidence_against.append(
                        {
                            "neuron_id": syn.source_id,
                            "content": (src_neuron.content[:120] if src_neuron else "(deleted)"),
                            "weight": syn.weight,
                        }
                    )

            return {
                "hypothesis_id": hypothesis_id,
                "content": content,
                "confidence": state["confidence"],
                "status": state["status"],
                "evidence_for_count": state["evidence_for_count"],
                "evidence_against_count": state["evidence_against_count"],
                "evidence_for": evidence_for,
                "evidence_against": evidence_against,
                "last_evidence_at": state.get("last_evidence_at"),
                "created_at": state["created_at"],
            }
        except Exception:
            logger.error("Hypothesis get failed", exc_info=True)
            return {"error": "Failed to get hypothesis details"}

    async def _evidence(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_evidence tool calls.

        Add evidence for or against a hypothesis, updating its
        confidence via Bayesian update with auto-resolution check.
        """
        hypothesis_id = args.get("hypothesis_id", "").strip()
        if not hypothesis_id:
            return {"error": "hypothesis_id is required"}

        content = args.get("content", "").strip()
        if not content:
            return {"error": "content is required"}
        if len(content) > 100_000:
            return {"error": f"Content too long ({len(content)} chars). Max: 100,000."}

        evidence_type = args.get("type", "for")
        if evidence_type not in ("for", "against"):
            return {"error": f"type must be 'for' or 'against', got: {evidence_type}"}

        weight = args.get("weight", 0.5)
        try:
            weight = max(0.1, min(1.0, float(weight)))
        except (TypeError, ValueError):
            weight = 0.5

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}

        # Verify hypothesis exists
        state = await storage.get_cognitive_state(hypothesis_id)
        if not state:
            return {"error": f"No hypothesis found with id: {hypothesis_id}"}

        if state["status"] not in ("active", "pending"):
            return {
                "error": f"Hypothesis is already {state['status']}. "
                "Cannot add evidence to resolved hypotheses."
            }

        # Create evidence neuron and link to hypothesis
        from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
        from neural_memory.core.synapse import Synapse, SynapseType
        from neural_memory.engine.cognitive import detect_auto_resolution, update_confidence
        from neural_memory.engine.encoder import MemoryEncoder
        from neural_memory.utils.timeutils import utcnow

        brain_id = storage.current_brain_id
        assert brain_id is not None
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "Brain not found"}

        encoder = MemoryEncoder(storage, brain.config)

        tags = set()
        for t in args.get("tags", []):
            if isinstance(t, str) and len(t) <= 100:
                tags.add(t)

        try:
            storage.disable_auto_save()

            # Encode evidence as a regular memory
            result = await encoder.encode(
                content=content, timestamp=utcnow(), tags=tags if tags else None
            )

            # Create typed memory for evidence
            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=MemoryType.INSIGHT,
                priority=Priority.from_int(args.get("priority", 5)),
                source="mcp_cognitive",
                tags=tags if tags else None,
            )
            await storage.add_typed_memory(typed_mem)

            # Create evidence synapse: evidence_neuron -> hypothesis_neuron
            synapse_type = (
                SynapseType.EVIDENCE_FOR if evidence_type == "for" else SynapseType.EVIDENCE_AGAINST
            )
            evidence_synapse = Synapse.create(
                source_id=result.fiber.anchor_neuron_id,
                target_id=hypothesis_id,
                type=synapse_type,
                weight=weight,
            )
            await storage.add_synapse(evidence_synapse)

            # Update confidence via Bayesian update
            old_confidence = state["confidence"]
            for_count = state["evidence_for_count"]
            against_count = state["evidence_against_count"]

            new_confidence = update_confidence(
                current=old_confidence,
                evidence_type=evidence_type,
                weight=weight,
                for_count=for_count,
                against_count=against_count,
            )

            new_for = for_count + (1 if evidence_type == "for" else 0)
            new_against = against_count + (1 if evidence_type == "against" else 0)

            # Check for auto-resolution
            resolution = detect_auto_resolution(new_confidence, new_for, new_against)
            new_status = resolution if resolution else state["status"]
            resolved_at = utcnow().isoformat() if resolution else state.get("resolved_at")

            # Targeted update — preserves predicted_at, schema_version, parent_schema_id
            await storage.update_cognitive_evidence(
                hypothesis_id,
                confidence=new_confidence,
                evidence_for_count=new_for,
                evidence_against_count=new_against,
                status=new_status,
                resolved_at=resolved_at,
                last_evidence_at=utcnow().isoformat(),
            )

            await storage.batch_save()
        except Exception:
            logger.error("Evidence addition failed", exc_info=True)
            return {"error": "Failed to add evidence"}
        finally:
            storage.enable_auto_save()

        result_dict: dict[str, Any] = {
            "status": "evidence_added",
            "hypothesis_id": hypothesis_id,
            "evidence_type": evidence_type,
            "evidence_neuron_id": result.fiber.anchor_neuron_id,
            "weight": weight,
            "confidence_before": round(old_confidence, 4),
            "confidence_after": round(new_confidence, 4),
            "confidence_delta": round(new_confidence - old_confidence, 4),
            "evidence_for_count": new_for,
            "evidence_against_count": new_against,
            "hypothesis_status": new_status,
        }

        if resolution:
            result_dict["auto_resolved"] = resolution
            result_dict["resolved_at"] = resolved_at

        return result_dict

    # ──────────────────── Prediction ────────────────────

    async def _predict(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_predict tool calls.

        Create, list, or inspect predictions — falsifiable claims about
        future observations with optional deadline and hypothesis linkage.
        """
        action = args.get("action", "create")
        if action not in ("create", "list", "get"):
            return {"error": f"Invalid action: {action}. Must be 'create', 'list', or 'get'."}

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}

        if action == "create":
            return await self._predict_create(storage, args)
        elif action == "list":
            return await self._predict_list(storage, args)
        else:  # get
            return await self._predict_get(storage, args)

    async def _predict_create(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        """Create a new prediction with optional deadline and hypothesis link."""
        content = args.get("content", "").strip()
        if not content:
            return {"error": "content is required"}
        if len(content) > 100_000:
            return {"error": f"Content too long ({len(content)} chars). Max: 100,000."}

        confidence = args.get("confidence", 0.7)
        try:
            confidence = max(0.01, min(0.99, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.7

        from neural_memory.core.memory_types import (
            MemoryType,
            Priority,
            TypedMemory,
            get_decay_rate,
        )
        from neural_memory.core.neuron import NeuronState
        from neural_memory.engine.encoder import MemoryEncoder
        from neural_memory.utils.timeutils import utcnow

        # Parse deadline
        deadline = args.get("deadline")
        if deadline:
            try:
                from datetime import datetime

                parsed = datetime.fromisoformat(str(deadline))
                # Strip tzinfo for comparison with naive UTC utcnow()
                naive_parsed = parsed.replace(tzinfo=None)
                if naive_parsed <= utcnow():
                    return {"error": "Deadline must be in the future"}
                deadline = naive_parsed.isoformat()
            except (TypeError, ValueError):
                return {"error": f"Invalid deadline format: {deadline}. Use ISO format."}

        raw_hyp = args.get("hypothesis_id")
        hypothesis_id = str(raw_hyp).strip() if raw_hyp else None

        tags = set()
        for t in args.get("tags", []):
            if isinstance(t, str) and len(t) <= 100:
                tags.add(t)

        brain_id = storage.current_brain_id
        assert brain_id is not None
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "Brain not found"}

        # If hypothesis_id given, verify it exists
        if hypothesis_id:
            hyp_state = await storage.get_cognitive_state(hypothesis_id)
            if not hyp_state:
                return {"error": f"No hypothesis found with id: {hypothesis_id}"}

        encoder = MemoryEncoder(storage, brain.config)

        try:
            storage.disable_auto_save()

            result = await encoder.encode(
                content=content, timestamp=utcnow(), tags=tags if tags else None
            )

            # TypedMemory as PREDICTION type
            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=MemoryType.PREDICTION,
                priority=Priority.from_int(args.get("priority", 5)),
                source="mcp_cognitive",
                expires_in_days=30,
                tags=tags if tags else None,
            )
            await storage.add_typed_memory(typed_mem)

            # Set prediction decay rate on neuron states
            decay_rate = get_decay_rate("prediction")
            for neuron in result.neurons_created:
                state = await storage.get_neuron_state(neuron.id)
                if state and state.decay_rate != decay_rate:
                    updated_state = NeuronState(
                        neuron_id=state.neuron_id,
                        activation_level=state.activation_level,
                        access_frequency=state.access_frequency,
                        last_activated=state.last_activated,
                        decay_rate=decay_rate,
                        created_at=state.created_at,
                    )
                    await storage.update_neuron_state(updated_state)

            # Create cognitive state with predicted_at (marks it as prediction)
            anchor_id = result.fiber.anchor_neuron_id
            existing = await storage.get_cognitive_state(anchor_id)
            if existing:
                return {
                    "status": "existing",
                    "prediction_id": anchor_id,
                    "confidence": existing["confidence"],
                    "message": "Prediction with identical content already exists",
                }

            await storage.upsert_cognitive_state(
                anchor_id,
                confidence=confidence,
                status="pending",
                predicted_at=deadline or utcnow().isoformat(),
            )

            # If linked to hypothesis, create PREDICTED synapse
            if hypothesis_id:
                from neural_memory.core.synapse import Synapse, SynapseType

                predicted_synapse = Synapse.create(
                    source_id=anchor_id,
                    target_id=hypothesis_id,
                    type=SynapseType.PREDICTED,
                    weight=confidence,
                )
                await storage.add_synapse(predicted_synapse)

            await storage.batch_save()
        except Exception:
            logger.error("Prediction create failed", exc_info=True)
            return {"error": "Failed to create prediction"}
        finally:
            storage.enable_auto_save()

        result_dict: dict[str, Any] = {
            "status": "created",
            "prediction_id": anchor_id,
            "fiber_id": result.fiber.id,
            "confidence": confidence,
            "deadline": deadline,
            "content_preview": content[:120],
            "neurons_created": len(result.neurons_created),
            "synapses_created": len(result.synapses_created),
        }
        if hypothesis_id:
            result_dict["linked_hypothesis_id"] = hypothesis_id
        return result_dict

    async def _predict_list(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        """List predictions with their cognitive state."""
        status_filter = args.get("status")
        if status_filter and status_filter not in _VALID_STATUSES:
            return {"error": f"Invalid status: {status_filter}. Valid: {sorted(_VALID_STATUSES)}"}

        try:
            limit = min(int(args.get("limit", 20)), 100)
        except (TypeError, ValueError):
            limit = 20

        try:
            states = await storage.list_predictions(
                status=status_filter,
                limit=limit,
            )

            predictions: list[dict[str, Any]] = []
            for state in states:
                neuron = await storage.get_neuron(state["neuron_id"])
                content = neuron.content if neuron else "(deleted)"
                predictions.append(
                    {
                        "prediction_id": state["neuron_id"],
                        "content": content[:200] if content else "",
                        "confidence": state["confidence"],
                        "status": state["status"],
                        "deadline": state["predicted_at"],
                        "resolved_at": state.get("resolved_at"),
                        "created_at": state["created_at"],
                    }
                )

            # Include calibration stats
            cal = await storage.get_calibration_stats()
            from neural_memory.engine.cognitive import compute_calibration

            calibration = compute_calibration(cal["correct_count"], cal["total_resolved"])

            return {
                "count": len(predictions),
                "predictions": predictions,
                "calibration": {
                    "score": round(calibration, 4),
                    "correct": cal["correct_count"],
                    "wrong": cal["wrong_count"],
                    "total_resolved": cal["total_resolved"],
                    "pending": cal["pending_count"],
                },
            }
        except Exception:
            logger.error("Prediction list failed", exc_info=True)
            return {"error": "Failed to list predictions"}

    async def _predict_get(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        """Get a single prediction with full details."""
        prediction_id = args.get("prediction_id", "").strip()
        if not prediction_id:
            return {"error": "prediction_id is required"}

        try:
            state = await storage.get_cognitive_state(prediction_id)
            if not state:
                return {"error": f"No prediction found with id: {prediction_id}"}
            if not state.get("predicted_at"):
                return {"error": f"Neuron {prediction_id} is not a prediction (no deadline)"}

            neuron = await storage.get_neuron(prediction_id)
            content = neuron.content if neuron else "(deleted)"

            # Find linked hypothesis via PREDICTED synapse
            from neural_memory.core.synapse import SynapseType

            linked_hypothesis = None
            verification = None

            synapses = await storage.get_synapses(source_id=prediction_id)
            cognitive_types = frozenset(
                {
                    SynapseType.PREDICTED,
                    SynapseType.VERIFIED_BY,
                    SynapseType.FALSIFIED_BY,
                }
            )
            for syn in synapses:
                if syn.type not in cognitive_types:
                    continue
                if syn.type == SynapseType.PREDICTED and not linked_hypothesis:
                    hyp_neuron = await storage.get_neuron(syn.target_id)
                    linked_hypothesis = {
                        "hypothesis_id": syn.target_id,
                        "content": (hyp_neuron.content[:120] if hyp_neuron else "(deleted)"),
                    }
                elif syn.type in (SynapseType.VERIFIED_BY, SynapseType.FALSIFIED_BY):
                    obs_neuron = await storage.get_neuron(syn.target_id)
                    verification = {
                        "outcome": "correct" if syn.type == SynapseType.VERIFIED_BY else "wrong",
                        "observation_id": syn.target_id,
                        "content": (obs_neuron.content[:120] if obs_neuron else "(deleted)"),
                    }
                if linked_hypothesis and verification:
                    break

            result_dict: dict[str, Any] = {
                "prediction_id": prediction_id,
                "content": content,
                "confidence": state["confidence"],
                "status": state["status"],
                "deadline": state["predicted_at"],
                "resolved_at": state.get("resolved_at"),
                "created_at": state["created_at"],
            }
            if linked_hypothesis:
                result_dict["linked_hypothesis"] = linked_hypothesis
            if verification:
                result_dict["verification"] = verification
            return result_dict
        except Exception:
            logger.error("Prediction get failed", exc_info=True)
            return {"error": "Failed to get prediction details"}

    # ──────────────────── Verify ────────────────────

    async def _verify(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_verify tool calls.

        Verify a prediction as correct or wrong. Optionally creates an
        observation neuron, links via VERIFIED_BY/FALSIFIED_BY synapse,
        and propagates to linked hypothesis.
        """
        prediction_id = args.get("prediction_id", "").strip()
        if not prediction_id:
            return {"error": "prediction_id is required"}

        outcome = args.get("outcome", "").strip()
        if outcome not in ("correct", "wrong"):
            return {"error": f"outcome must be 'correct' or 'wrong', got: {outcome}"}

        content = args.get("content", "").strip() if args.get("content") else None
        if content and len(content) > 100_000:
            return {"error": f"Content too long ({len(content)} chars). Max: 100,000."}

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}

        # Verify prediction exists and is pending
        state = await storage.get_cognitive_state(prediction_id)
        if not state:
            return {"error": f"No prediction found with id: {prediction_id}"}
        if not state.get("predicted_at"):
            return {"error": f"Neuron {prediction_id} is not a prediction"}
        if state["status"] not in ("pending", "active"):
            return {
                "error": f"Prediction is already {state['status']}. "
                "Cannot verify resolved predictions."
            }

        from neural_memory.core.synapse import Synapse, SynapseType
        from neural_memory.engine.cognitive import compute_calibration
        from neural_memory.utils.timeutils import utcnow

        brain_id = storage.current_brain_id
        assert brain_id is not None
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "Brain not found"}

        tags = set()
        for t in args.get("tags", []):
            if isinstance(t, str) and len(t) <= 100:
                tags.add(t)

        try:
            storage.disable_auto_save()

            observation_id = None

            # Optionally encode observation content
            if content:
                from neural_memory.core.memory_types import MemoryType, Priority, TypedMemory
                from neural_memory.engine.encoder import MemoryEncoder

                encoder = MemoryEncoder(storage, brain.config)
                result = await encoder.encode(
                    content=content, timestamp=utcnow(), tags=tags if tags else None
                )
                observation_id = result.fiber.anchor_neuron_id

                typed_mem = TypedMemory.create(
                    fiber_id=result.fiber.id,
                    memory_type=MemoryType.INSIGHT,
                    priority=Priority.from_int(args.get("priority", 5)),
                    source="mcp_cognitive",
                    tags=tags if tags else None,
                )
                await storage.add_typed_memory(typed_mem)

                # Link prediction -> observation
                synapse_type = (
                    SynapseType.VERIFIED_BY if outcome == "correct" else SynapseType.FALSIFIED_BY
                )
                verify_synapse = Synapse.create(
                    source_id=prediction_id,
                    target_id=observation_id,
                    type=synapse_type,
                    weight=0.8,
                )
                await storage.add_synapse(verify_synapse)

            # Update prediction status
            new_status = "confirmed" if outcome == "correct" else "refuted"
            now = utcnow().isoformat()
            await storage.update_cognitive_evidence(
                prediction_id,
                confidence=state["confidence"],
                evidence_for_count=state["evidence_for_count"],
                evidence_against_count=state["evidence_against_count"],
                status=new_status,
                resolved_at=now,
                last_evidence_at=now,
            )

            # Propagate to linked hypothesis if exists
            propagated_to = None
            synapses = await storage.get_synapses(source_id=prediction_id)
            predicted_synapses = [s for s in synapses if s.type == SynapseType.PREDICTED]
            if predicted_synapses:
                syn = predicted_synapses[0]
                hyp_state = await storage.get_cognitive_state(syn.target_id)
                if hyp_state and hyp_state["status"] in ("active", "pending"):
                    from neural_memory.engine.cognitive import (
                        detect_auto_resolution,
                        update_confidence,
                    )

                    evidence_type: Literal["for", "against"] = (
                        "for" if outcome == "correct" else "against"
                    )
                    new_conf = update_confidence(
                        current=hyp_state["confidence"],
                        evidence_type=evidence_type,
                        weight=0.6,
                        for_count=hyp_state["evidence_for_count"],
                        against_count=hyp_state["evidence_against_count"],
                    )
                    new_for = hyp_state["evidence_for_count"] + (1 if evidence_type == "for" else 0)
                    new_against = hyp_state["evidence_against_count"] + (
                        1 if evidence_type == "against" else 0
                    )
                    resolution = detect_auto_resolution(new_conf, new_for, new_against)
                    hyp_status = resolution if resolution else hyp_state["status"]
                    hyp_resolved = now if resolution else hyp_state.get("resolved_at")

                    await storage.update_cognitive_evidence(
                        syn.target_id,
                        confidence=new_conf,
                        evidence_for_count=new_for,
                        evidence_against_count=new_against,
                        status=hyp_status,
                        resolved_at=hyp_resolved,
                        last_evidence_at=now,
                    )
                    propagated_to = {
                        "hypothesis_id": syn.target_id,
                        "evidence_type": evidence_type,
                        "confidence_after": round(new_conf, 4),
                        "hypothesis_status": hyp_status,
                    }

            await storage.batch_save()
        except Exception:
            logger.error("Prediction verification failed", exc_info=True)
            return {"error": "Failed to verify prediction"}
        finally:
            storage.enable_auto_save()

        # Compute calibration (separate try to avoid losing verification result)
        try:
            cal = await storage.get_calibration_stats()
            calibration = compute_calibration(cal["correct_count"], cal["total_resolved"])
        except Exception:
            logger.error("Calibration stats fetch failed", exc_info=True)
            cal = {"correct_count": 0, "wrong_count": 0, "total_resolved": 0, "pending_count": 0}
            calibration = 0.5

        result_dict: dict[str, Any] = {
            "status": "verified",
            "prediction_id": prediction_id,
            "outcome": outcome,
            "prediction_status": new_status,
            "resolved_at": now,
            "calibration_score": round(calibration, 4),
            "calibration_stats": {
                "correct": cal["correct_count"],
                "wrong": cal["wrong_count"],
                "total_resolved": cal["total_resolved"],
                "pending": cal["pending_count"],
            },
        }
        if observation_id:
            result_dict["observation_id"] = observation_id
        if propagated_to:
            result_dict["propagated_to_hypothesis"] = propagated_to

        return result_dict

    # ──────────────────── Hot Index (nmem_cognitive) ────────────────────

    async def _cognitive(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_cognitive tool calls.

        Provides O(1) cognitive summary — top hypotheses, predictions,
        and knowledge gaps ranked by relevance/urgency.
        """
        action = args.get("action", "summary")
        if action not in ("summary", "refresh"):
            return {"error": f"Invalid action: {action}. Must be 'summary' or 'refresh'."}

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}

        if action == "summary":
            return await self._cognitive_summary(storage, args)
        else:
            return await self._cognitive_refresh(storage)

    async def _cognitive_summary(
        self, storage: NeuralStorage, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the current hot index — precomputed cognitive overview."""
        try:
            limit = min(int(args.get("limit", 10)), 20)
        except (TypeError, ValueError):
            limit = 10

        try:
            items = await storage.get_hot_index(limit=limit)

            # Also include calibration and gap counts
            cal = await storage.get_calibration_stats()
            gaps = await storage.list_knowledge_gaps(include_resolved=False, limit=5)

            from neural_memory.engine.cognitive import compute_calibration

            calibration = compute_calibration(cal["correct_count"], cal["total_resolved"])

            return {
                "hot_items": items,
                "hot_count": len(items),
                "calibration": {
                    "score": round(calibration, 4),
                    "correct": cal["correct_count"],
                    "wrong": cal["wrong_count"],
                    "total_resolved": cal["total_resolved"],
                    "pending": cal["pending_count"],
                },
                "top_gaps_count": len(gaps),
                "top_gaps_note": "showing top 5 only",
                "top_gaps": [
                    {"id": g["id"], "topic": g["topic"], "priority": g["priority"]}
                    for g in gaps[:3]
                ],
            }
        except Exception:
            logger.error("Cognitive summary failed", exc_info=True)
            return {"error": "Failed to get cognitive summary"}

    async def _cognitive_refresh(self, storage: NeuralStorage) -> dict[str, Any]:
        """Recompute the hot index from current cognitive state."""
        from datetime import datetime

        from neural_memory.engine.cognitive import (
            score_hypothesis,
            score_prediction,
            truncate_summary,
        )
        from neural_memory.utils.timeutils import utcnow

        try:
            now = utcnow()

            # Gather active hypotheses and pending predictions
            hypotheses = await storage.list_cognitive_states(status="active", limit=50)
            predictions = await storage.list_predictions(status="pending", limit=50)

            # Batch-fetch neurons to avoid N+1 queries
            all_neuron_ids = [h["neuron_id"] for h in hypotheses] + [
                p["neuron_id"] for p in predictions
            ]
            neurons_map = await storage.get_neurons_batch(all_neuron_ids)

            scored_items: list[dict[str, Any]] = []

            for hyp in hypotheses:
                neuron = neurons_map.get(hyp["neuron_id"])
                content = neuron.content if neuron else ""

                try:
                    created = datetime.fromisoformat(hyp["created_at"])
                    age_days = (now - created).total_seconds() / 86400
                except (TypeError, ValueError):
                    age_days = 0.0

                evidence_count = hyp["evidence_for_count"] + hyp["evidence_against_count"]
                score = score_hypothesis(hyp["confidence"], evidence_count, age_days)
                scored_items.append(
                    {
                        "category": "hypothesis",
                        "neuron_id": hyp["neuron_id"],
                        "summary": truncate_summary(content),
                        "confidence": hyp["confidence"],
                        "score": round(score, 4),
                    }
                )

            for pred in predictions:
                neuron = neurons_map.get(pred["neuron_id"])
                content = neuron.content if neuron else ""

                try:
                    deadline = datetime.fromisoformat(pred["predicted_at"])
                    days_until = (deadline - now).total_seconds() / 86400
                except (TypeError, ValueError):
                    days_until = 30.0

                score = score_prediction(days_until)
                scored_items.append(
                    {
                        "category": "prediction",
                        "neuron_id": pred["neuron_id"],
                        "summary": truncate_summary(content),
                        "confidence": pred["confidence"],
                        "score": round(score, 4),
                    }
                )

            # Sort by score descending, assign slots
            scored_items.sort(key=lambda x: x["score"], reverse=True)
            for i, item in enumerate(scored_items[:20]):
                item["slot"] = i

            count = await storage.refresh_hot_index(scored_items[:20])

            return {
                "status": "refreshed",
                "items_indexed": count,
                "hypotheses_scored": len(hypotheses),
                "predictions_scored": len(predictions),
            }
        except Exception:
            logger.error("Hot index refresh failed", exc_info=True)
            return {"error": "Failed to refresh hot index"}

    # ──────────────────── Knowledge Gaps (nmem_gaps) ────────────────────

    _VALID_DETECTION_SOURCES = frozenset(
        {
            "contradicting_evidence",
            "low_confidence_hypothesis",
            "user_flagged",
            "recall_miss",
            "stale_schema",
        }
    )

    async def _gaps(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_gaps tool calls.

        Metacognition — track what the brain doesn't know.
        """
        action = args.get("action", "list")
        if action not in ("detect", "list", "resolve", "get"):
            return {
                "error": f"Invalid action: {action}. Must be 'detect', 'list', 'resolve', or 'get'."
            }

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}

        if action == "detect":
            return await self._gaps_detect(storage, args)
        elif action == "list":
            return await self._gaps_list(storage, args)
        elif action == "resolve":
            return await self._gaps_resolve(storage, args)
        else:
            return await self._gaps_get(storage, args)

    async def _gaps_detect(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        """Register a new knowledge gap."""
        topic = args.get("topic", "").strip()
        if not topic:
            return {"error": "topic is required"}
        if len(topic) > 500:
            return {"error": f"Topic too long ({len(topic)} chars). Max: 500."}

        source = args.get("source", "user_flagged")
        if source not in self._VALID_DETECTION_SOURCES:
            return {
                "error": f"Invalid source: {source}. Valid: {sorted(self._VALID_DETECTION_SOURCES)}"
            }

        related = args.get("related_neuron_ids", [])
        if not isinstance(related, list):
            related = []
        # Cast all items to string, cap at 10
        related = [str(r) for r in related][:10]

        from neural_memory.engine.cognitive import gap_priority

        priority = args.get("priority")
        if priority is not None:
            try:
                priority = max(0.0, min(1.0, float(priority)))
            except (TypeError, ValueError):
                priority = gap_priority(source)
        else:
            priority = gap_priority(source)

        try:
            gap_id = await storage.add_knowledge_gap(
                topic=topic,
                detection_source=source,
                priority=priority,
                related_neuron_ids=related,
            )
            return {
                "status": "detected",
                "gap_id": gap_id,
                "topic": topic,
                "source": source,
                "priority": round(priority, 4),
            }
        except Exception:
            logger.error("Gap detection failed", exc_info=True)
            return {"error": "Failed to detect knowledge gap"}

    async def _gaps_list(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        """List knowledge gaps."""
        include_resolved = bool(args.get("include_resolved", False))

        try:
            limit = min(int(args.get("limit", 20)), 100)
        except (TypeError, ValueError):
            limit = 20

        try:
            gaps = await storage.list_knowledge_gaps(
                include_resolved=include_resolved,
                limit=limit,
            )
            return {
                "count": len(gaps),
                "gaps": gaps,
            }
        except Exception:
            logger.error("Gap list failed", exc_info=True)
            return {"error": "Failed to list knowledge gaps"}

    async def _gaps_resolve(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve a knowledge gap."""
        gap_id = args.get("gap_id", "").strip()
        if not gap_id:
            return {"error": "gap_id is required"}

        resolved_by = (args.get("resolved_by_neuron_id") or "").strip() or None

        try:
            success = await storage.resolve_knowledge_gap(
                gap_id,
                resolved_by_neuron_id=resolved_by,
            )
            if not success:
                return {"error": "Gap not found or already resolved"}
            return {
                "status": "resolved",
                "gap_id": gap_id,
                "resolved_by_neuron_id": resolved_by,
            }
        except Exception:
            logger.error("Gap resolve failed", exc_info=True)
            return {"error": "Failed to resolve knowledge gap"}

    async def _gaps_get(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        """Get details of a specific knowledge gap."""
        gap_id = args.get("gap_id", "").strip()
        if not gap_id:
            return {"error": "gap_id is required"}

        try:
            gap = await storage.get_knowledge_gap(gap_id)
            if not gap:
                return {"error": "No gap found with the given id"}
            return gap
        except Exception:
            logger.error("Gap get failed", exc_info=True)
            return {"error": "Failed to get knowledge gap details"}

    # ──────────────────── Schema Evolution (nmem_schema) ────────────────────

    async def _schema(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_schema tool calls.

        Schema evolution — evolve hypotheses into new versions linked by
        SUPERSEDES synapses. Tracks how beliefs change over time.
        """
        action = args.get("action", "history")
        if action not in ("evolve", "history", "compare"):
            return {
                "error": f"Invalid action: {action}. Must be 'evolve', 'history', or 'compare'."
            }

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}

        if action == "evolve":
            return await self._schema_evolve(storage, args)
        elif action == "history":
            return await self._schema_history(storage, args)
        else:
            return await self._schema_compare(storage, args)

    async def _schema_evolve(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        """Create a new version of a hypothesis with SUPERSEDES link."""
        hypothesis_id = (args.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            return {"error": "hypothesis_id is required"}

        content = (args.get("content") or "").strip()
        if not content:
            return {"error": "content is required for evolve"}
        if len(content) > 100_000:
            return {"error": f"Content too long ({len(content)} chars). Max: 100,000."}

        # Verify the old hypothesis exists and is a hypothesis (not a prediction)
        old_state = await storage.get_cognitive_state(hypothesis_id)
        if not old_state:
            return {"error": "Hypothesis not found"}
        if old_state.get("predicted_at"):
            return {"error": "Cannot evolve a prediction — only hypotheses can be evolved"}
        if old_state["status"] == "superseded":
            return {"error": "Already superseded — evolve the latest version instead"}

        # Resolve confidence
        new_confidence = args.get("confidence")
        if new_confidence is not None:
            try:
                new_confidence = max(0.01, min(0.99, float(new_confidence)))
            except (TypeError, ValueError):
                new_confidence = old_state["confidence"]
        else:
            new_confidence = old_state["confidence"]

        reason = (args.get("reason") or "").strip()[:500]

        tags = set()
        for t in args.get("tags", []):
            if isinstance(t, str) and len(t) <= 100:
                tags.add(t)

        # Create new hypothesis neuron
        from neural_memory.core.memory_types import (
            MemoryType,
            Priority,
            TypedMemory,
            get_decay_rate,
        )
        from neural_memory.core.neuron import NeuronState
        from neural_memory.core.synapse import Synapse, SynapseType
        from neural_memory.engine.encoder import MemoryEncoder
        from neural_memory.utils.timeutils import utcnow

        brain_id = storage.current_brain_id
        assert brain_id is not None
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "Brain not found"}

        encoder = MemoryEncoder(storage, brain.config)
        old_version = old_state.get("schema_version", 1)

        try:
            storage.disable_auto_save()

            # Encode new content
            result = await encoder.encode(
                content=content, timestamp=utcnow(), tags=tags if tags else None
            )

            # Typed memory
            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=MemoryType.HYPOTHESIS,
                priority=Priority.from_int(min(10, max(0, int(args.get("priority", 6))))),
                source="mcp_cognitive_evolve",
                expires_in_days=180,
                tags=tags if tags else None,
            )
            await storage.add_typed_memory(typed_mem)

            # Set decay rate
            decay_rate = get_decay_rate("hypothesis")
            for neuron in result.neurons_created:
                state = await storage.get_neuron_state(neuron.id)
                if state and state.decay_rate != decay_rate:
                    updated = NeuronState(
                        neuron_id=state.neuron_id,
                        activation_level=state.activation_level,
                        access_frequency=state.access_frequency,
                        last_activated=state.last_activated,
                        decay_rate=decay_rate,
                        created_at=state.created_at,
                    )
                    await storage.update_neuron_state(updated)

            new_anchor = result.fiber.anchor_neuron_id

            # Create cognitive state with bumped version
            await storage.upsert_cognitive_state(
                new_anchor,
                confidence=new_confidence,
                status="active",
                schema_version=old_version + 1,
                parent_schema_id=hypothesis_id,
            )

            # Create SUPERSEDES synapse: new → old
            metadata = {"reason": reason} if reason else None
            supersedes_syn = Synapse.create(
                source_id=new_anchor,
                target_id=hypothesis_id,
                type=SynapseType.SUPERSEDES,
                weight=1.0,
                metadata=metadata,
            )
            await storage.add_synapse(supersedes_syn)

            # Mark old hypothesis as superseded
            await storage.update_cognitive_evidence(
                hypothesis_id,
                confidence=old_state["confidence"],
                evidence_for_count=old_state["evidence_for_count"],
                evidence_against_count=old_state["evidence_against_count"],
                status="superseded",
            )

            await storage.batch_save()

            evolve_result = {
                "status": "evolved",
                "new_hypothesis_id": new_anchor,
                "old_hypothesis_id": hypothesis_id,
                "schema_version": old_version + 1,
                "confidence": round(new_confidence, 4),
                "reason": reason or None,
            }
        except Exception:
            logger.error("Schema evolve failed", exc_info=True)
            return {"error": "Failed to evolve hypothesis"}
        finally:
            storage.enable_auto_save()

        return evolve_result

    async def _schema_history(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        """Get version chain for a hypothesis."""
        hypothesis_id = (args.get("hypothesis_id") or "").strip()
        if not hypothesis_id:
            return {"error": "hypothesis_id is required"}

        try:
            history = await storage.get_schema_history(hypothesis_id, max_depth=20)
            if not history:
                return {"error": "Hypothesis not found or no cognitive state"}

            # Enrich with content summaries
            from neural_memory.engine.cognitive import truncate_summary

            enriched: list[dict[str, Any]] = []
            for entry in history:
                neuron = await storage.get_neuron(entry["neuron_id"])
                enriched.append(
                    {
                        "neuron_id": entry["neuron_id"],
                        "version": entry.get("schema_version", 1),
                        "confidence": entry["confidence"],
                        "status": entry["status"],
                        "summary": truncate_summary(neuron.content) if neuron else "",
                        "created_at": entry.get("created_at"),
                    }
                )

            return {
                "hypothesis_id": hypothesis_id,
                "version_count": len(enriched),
                "versions": enriched,
            }
        except Exception:
            logger.error("Schema history failed", exc_info=True)
            return {"error": "Failed to get schema history"}

    async def _schema_compare(self, storage: NeuralStorage, args: dict[str, Any]) -> dict[str, Any]:
        """Compare two hypothesis versions side by side."""
        id_a = (args.get("hypothesis_id") or "").strip()
        id_b = (args.get("other_id") or "").strip()
        if not id_a or not id_b:
            return {"error": "Both hypothesis_id and other_id are required for compare"}

        try:
            from neural_memory.engine.cognitive import truncate_summary

            state_a = await storage.get_cognitive_state(id_a)
            state_b = await storage.get_cognitive_state(id_b)
            if not state_a or not state_b:
                return {"error": "One or both hypotheses not found"}

            neuron_a = await storage.get_neuron(id_a)
            neuron_b = await storage.get_neuron(id_b)

            return {
                "version_a": {
                    "neuron_id": id_a,
                    "version": state_a.get("schema_version", 1),
                    "confidence": state_a["confidence"],
                    "status": state_a["status"],
                    "evidence_for": state_a["evidence_for_count"],
                    "evidence_against": state_a["evidence_against_count"],
                    "summary": truncate_summary(neuron_a.content) if neuron_a else "",
                },
                "version_b": {
                    "neuron_id": id_b,
                    "version": state_b.get("schema_version", 1),
                    "confidence": state_b["confidence"],
                    "status": state_b["status"],
                    "evidence_for": state_b["evidence_for_count"],
                    "evidence_against": state_b["evidence_against_count"],
                    "summary": truncate_summary(neuron_b.content) if neuron_b else "",
                },
                "confidence_delta": round(state_b["confidence"] - state_a["confidence"], 4),
            }
        except Exception:
            logger.error("Schema compare failed", exc_info=True)
            return {"error": "Failed to compare hypothesis versions"}
