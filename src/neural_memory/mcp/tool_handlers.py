"""MCP tool handler implementations.

Extracted from server.py to keep file sizes manageable.
Each method handles one MCP tool call (nmem_*).

The ToolHandler mixin is inherited by MCPServer in server.py.
All methods access storage/config via self.get_storage() and self.config
from the MCPServer base class.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory import __version__
from neural_memory.core.memory_types import (
    MemoryType,
    Priority,
    TypedMemory,
    get_decay_rate,
    suggest_memory_type,
)
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.hooks import HookEvent
from neural_memory.engine.retrieval import DepthLevel, ReflexPipeline
from neural_memory.mcp.constants import MAX_CONTENT_LENGTH
from neural_memory.utils.timeutils import utcnow

# Max tags per recall query (remember allows 50 for storage, recall caps at 20 for filtering)
_MAX_RECALL_TAGS = 20
_MAX_TAG_LENGTH = 100


def _parse_tags(args: dict[str, Any], *, max_items: int = _MAX_RECALL_TAGS) -> set[str] | None:
    """Parse and validate tags from MCP tool arguments.

    Returns a set of valid tag strings, or None if no valid tags provided.
    """
    raw_tags = args.get("tags")
    if not raw_tags or not isinstance(raw_tags, list):
        return None
    tags = {t for t in raw_tags[:max_items] if isinstance(t, str) and 0 < len(t) <= _MAX_TAG_LENGTH}
    return tags or None


if TYPE_CHECKING:
    from neural_memory.engine.hooks import HookRegistry
    from neural_memory.mcp.maintenance_handler import HealthPulse
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


def _require_brain_id(storage: NeuralStorage) -> str:
    """Return the current brain ID or raise ValueError if not set."""
    brain_id = storage.brain_id
    if not brain_id:
        raise ValueError("No brain context set")
    return brain_id


async def _get_brain_or_error(
    storage: NeuralStorage,
) -> tuple[Any, dict[str, Any] | None]:
    """Get brain object or return (None, error_dict)."""
    try:
        brain_id = _require_brain_id(storage)
    except ValueError:
        return None, {"error": "No brain configured"}
    brain = await storage.get_brain(brain_id)
    if not brain:
        return None, {"error": "No brain configured"}
    return brain, None


async def _build_citation_audit(
    storage: NeuralStorage,
    neuron_id: str,
    include_citations: bool = True,
) -> dict[str, Any]:
    """Build citation and audit trail for a neuron from its synapses.

    Looks up SOURCE_OF, STORED_BY, VERIFIED_AT, APPROVED_BY synapses
    connected to the neuron and returns citation + audit dicts.
    """
    from neural_memory.core.synapse import SynapseType

    result: dict[str, Any] = {}

    # Fetch incoming synapses for this neuron
    synapses = await storage.get_synapses(target_id=neuron_id)

    # Build citation from SOURCE_OF synapse
    if include_citations:
        source_synapses = [s for s in synapses if s.type == SynapseType.SOURCE_OF]
        if source_synapses:
            source_syn = source_synapses[0]
            try:
                source_obj = await storage.get_source(source_syn.source_id)
                if source_obj:
                    from neural_memory.engine.citation import (
                        CitationFormat,
                        CitationInput,
                        format_citation,
                    )

                    citation_input = CitationInput(
                        source_name=source_obj.name,
                        source_type=source_obj.source_type.value,
                        source_version=source_obj.version,
                        effective_date=(
                            source_obj.effective_date.isoformat()
                            if source_obj.effective_date
                            else None
                        ),
                        neuron_id=neuron_id,
                        metadata=source_obj.metadata,
                    )
                    result["citation"] = {
                        "inline": format_citation(citation_input, CitationFormat.INLINE),
                        "footnote": format_citation(citation_input, CitationFormat.FOOTNOTE),
                        "source_id": source_obj.id,
                        "source_name": source_obj.name,
                        "source_type": source_obj.source_type.value,
                    }
            except Exception:
                logger.debug("Citation generation failed", exc_info=True)

    # Build audit trail from STORED_BY, VERIFIED_AT, APPROVED_BY synapses
    stored_by_syns = [s for s in synapses if s.type == SynapseType.STORED_BY]
    verified_syns = [s for s in synapses if s.type == SynapseType.VERIFIED_AT]
    approved_syns = [s for s in synapses if s.type == SynapseType.APPROVED_BY]

    if stored_by_syns or verified_syns or approved_syns:
        audit: dict[str, Any] = {}
        if stored_by_syns:
            syn = stored_by_syns[0]
            audit["stored_by"] = syn.metadata.get("actor", syn.source_id)
            audit["stored_at"] = syn.created_at.isoformat() if syn.created_at else None
        if verified_syns:
            syn = verified_syns[0]
            audit["verified"] = True
            audit["verified_by"] = syn.metadata.get("actor", syn.source_id)
            audit["verified_at"] = syn.created_at.isoformat() if syn.created_at else None
        else:
            audit["verified"] = False
        if approved_syns:
            syn = approved_syns[0]
            audit["approved_by"] = syn.metadata.get("actor", syn.source_id)
            audit["approved_at"] = syn.created_at.isoformat() if syn.created_at else None
        result["audit"] = audit

    return result


class ToolHandler:
    """Mixin providing all MCP tool handler implementations.

    Protocol stubs for attributes/methods used from MCPServer.
    """

    if TYPE_CHECKING:
        config: UnifiedConfig
        hooks: HookRegistry

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

        def _fire_eternal_trigger(self, content: str) -> None:
            raise NotImplementedError

        async def _check_maintenance(self) -> HealthPulse | None:
            raise NotImplementedError

        def _get_maintenance_hint(self, pulse: HealthPulse | None) -> str | None:
            raise NotImplementedError

        async def _passive_capture(self, text: str) -> None:
            raise NotImplementedError

        async def _get_active_session(self, storage: NeuralStorage) -> dict[str, Any] | None:
            raise NotImplementedError

        async def _check_onboarding(self) -> dict[str, Any] | None:
            raise NotImplementedError

        def get_update_hint(self) -> dict[str, Any] | None:
            raise NotImplementedError

    # ──────────────────── Helpers ────────────────────

    async def _check_cross_language_hint(
        self,
        query: str,
        result: Any,
        config: Any,
    ) -> str | None:
        """Return a hint if recall likely missed due to cross-language mismatch.

        Conditions (all must be true):
        1. Recall returned 0 fibers or very low confidence
        2. Embedding is NOT enabled
        3. Query language differs from brain majority language
        """
        # Only hint when results are poor
        if result.fibers_matched and result.confidence >= 0.3:
            return None

        # No hint needed if embedding is already enabled
        if getattr(config, "embedding_enabled", False):
            return None

        from neural_memory.extraction.parser import detect_language

        query_lang = detect_language(query)

        # Sample recent neurons to detect brain majority language
        try:
            storage = await self.get_storage()
            sample_neurons = await storage.find_neurons(limit=20)
            if len(sample_neurons) < 3:
                return None  # Too few memories to determine majority

            lang_counts: dict[str, int] = {}
            for neuron in sample_neurons:
                if neuron.content.strip():
                    lang = detect_language(neuron.content)
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

            if not lang_counts:
                return None

            majority_lang = max(lang_counts, key=lambda k: lang_counts[k])

            if query_lang == majority_lang:
                return None  # Same language — not a cross-language issue

            # Language mismatch detected — build hint
            try:
                import sentence_transformers as _st  # noqa: F401

                return (
                    f"Your query is in {'Vietnamese' if query_lang == 'vi' else 'English'} "
                    f"but most memories are in {'Vietnamese' if majority_lang == 'vi' else 'English'}. "
                    "Enable cross-language recall: add [embedding] section to "
                    "~/.neuralmemory/config.toml with enabled=true, "
                    "provider='sentence_transformer', "
                    "model='paraphrase-multilingual-MiniLM-L12-v2'."
                )
            except ImportError:
                return (
                    f"Your query is in {'Vietnamese' if query_lang == 'vi' else 'English'} "
                    f"but most memories are in {'Vietnamese' if majority_lang == 'vi' else 'English'}. "
                    "Enable cross-language recall: "
                    "pip install neural-memory[embeddings], then add [embedding] section to "
                    "~/.neuralmemory/config.toml with enabled=true, "
                    "provider='sentence_transformer', "
                    "model='paraphrase-multilingual-MiniLM-L12-v2'."
                )
        except Exception:
            logger.debug("Cross-language hint check failed (non-critical)", exc_info=True)
            return None

    # ──────────────────── Core tool handlers ────────────────────

    async def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
        """Store a memory in the neural graph."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        content = args.get("content")
        if not content or not isinstance(content, str):
            return {"error": "content is required and must be a string"}
        if len(content) > MAX_CONTENT_LENGTH:
            return {"error": f"Content too long ({len(content)} chars). Max: {MAX_CONTENT_LENGTH}."}

        # Check for sensitive content with selective auto-redaction
        from neural_memory.safety.sensitive import auto_redact_content, check_sensitive_content

        try:
            auto_redact_severity = int(self.config.safety.auto_redact_min_severity)
        except (TypeError, ValueError, AttributeError):
            auto_redact_severity = 3
        redacted_content, redacted_matches, content_hash = auto_redact_content(
            content, min_severity=auto_redact_severity
        )

        if redacted_matches:
            # Content was auto-redacted — use redacted version
            content = redacted_content
            logger.info(
                "Auto-redacted %d sensitive matches (severity >= %d)",
                len(redacted_matches),
                auto_redact_severity,
            )

        # Check for remaining sensitive content below auto-redact threshold
        remaining_matches = check_sensitive_content(content, min_severity=2)
        # Filter out matches that were already redacted
        remaining_matches = [m for m in remaining_matches if m.severity < auto_redact_severity]
        sensitive_detected = bool(remaining_matches)

        # Determine if content should be encrypted
        should_encrypt = args.get("encrypted", False)
        encrypted_content: str | None = None
        encryption_meta: dict[str, Any] = {}

        try:
            encryption_cfg = self.config.encryption
            encryption_enabled = encryption_cfg.enabled
        except AttributeError:
            encryption_enabled = False

        # Auto-encrypt sensitive content instead of blocking
        if sensitive_detected and encryption_enabled:
            should_encrypt = True
            logger.info(
                "Sensitive content detected (types: %s) — auto-encrypting instead of blocking",
                ", ".join(sorted({m.type.value for m in remaining_matches})),
            )
        elif sensitive_detected and not encryption_enabled:
            # Encryption not available — reject as before
            types_found = sorted({m.type.value for m in remaining_matches})
            return {
                "error": "Sensitive content detected",
                "sensitive_types": types_found,
                "message": "Content contains potentially sensitive information. "
                "Enable encryption (config.toml [encryption] enabled=true) to "
                "auto-encrypt sensitive memories, or remove secrets before storing.",
            }

        if encryption_enabled:
            # Auto-encrypt if sensitive content was detected in original input
            if not should_encrypt and getattr(encryption_cfg, "auto_encrypt_sensitive", True):
                from neural_memory.safety.sensitive import (
                    check_sensitive_content as _check_sensitive,
                )

                original_matches = _check_sensitive(args["content"], min_severity=2)
                if original_matches:
                    should_encrypt = True

            if should_encrypt:
                try:
                    from pathlib import Path

                    from neural_memory.safety.encryption import MemoryEncryptor

                    brain_id = _require_brain_id(storage)
                    keys_dir_str = getattr(encryption_cfg, "keys_dir", "")
                    keys_dir = (
                        Path(keys_dir_str) if keys_dir_str else (self.config.data_dir / "keys")
                    )

                    encryptor = MemoryEncryptor(keys_dir=keys_dir)
                    enc_result = encryptor.encrypt(content, brain_id)
                    encrypted_content = enc_result.ciphertext
                    encryption_meta = {
                        "encrypted": True,
                        "key_id": enc_result.key_id,
                        "algorithm": enc_result.algorithm,
                    }
                    logger.info("Encrypted memory content for brain %s", brain_id)
                except Exception:
                    logger.error("Encryption failed, refusing to store plaintext", exc_info=True)
                    return {"error": "Encryption failed — memory not stored. Check encryption key."}

        # Determine memory type
        if "type" in args:
            try:
                mem_type = MemoryType(args["type"])
            except ValueError:
                return {"error": f"Invalid memory type: {args['type']}"}
        else:
            mem_type = suggest_memory_type(content)

        priority = Priority.from_int(args.get("priority", 5))

        # Build dedup pipeline if enabled
        dedup_pipeline = None
        try:
            dedup_settings = self.config.dedup
            if isinstance(dedup_settings.enabled, bool) and dedup_settings.enabled:
                from neural_memory.engine.dedup.config import DedupConfig
                from neural_memory.engine.dedup.pipeline import DedupPipeline

                dedup_cfg = DedupConfig(
                    enabled=True,
                    simhash_threshold=int(dedup_settings.simhash_threshold),
                    embedding_threshold=float(dedup_settings.embedding_threshold),
                    embedding_ambiguous_low=float(dedup_settings.embedding_ambiguous_low),
                    llm_enabled=bool(dedup_settings.llm_enabled),
                    llm_provider=str(dedup_settings.llm_provider),
                    llm_model=str(dedup_settings.llm_model),
                    llm_max_pairs_per_encode=int(dedup_settings.llm_max_pairs_per_encode),
                    merge_strategy=str(dedup_settings.merge_strategy),
                )

                # Create LLM judge if enabled
                llm_judge = None
                if dedup_cfg.llm_enabled and dedup_cfg.llm_provider != "none":
                    from neural_memory.engine.dedup.llm_judge import create_judge

                    llm_judge = create_judge(dedup_cfg.llm_provider, dedup_cfg.llm_model)

                dedup_pipeline = DedupPipeline(
                    config=dedup_cfg,
                    storage=storage,
                    llm_judge=llm_judge,
                )
        except (AttributeError, TypeError, ValueError):
            dedup_pipeline = None

        encoder = MemoryEncoder(storage, brain.config, dedup_pipeline=dedup_pipeline)

        await self.hooks.emit(HookEvent.PRE_REMEMBER, {"content": content, "type": mem_type.value})

        try:
            storage.disable_auto_save()
            raw_tags = args.get("tags", [])
            if len(raw_tags) > 50:
                return {"error": f"Too many tags ({len(raw_tags)}). Max: 50."}
            tags = set()
            for t in raw_tags:
                if isinstance(t, str) and len(t) <= 100:
                    tags.add(t)
            # Parse event_at for original event timestamp
            event_timestamp = utcnow()
            raw_event_at = args.get("event_at")
            if raw_event_at:
                try:
                    event_timestamp = datetime.fromisoformat(raw_event_at)
                    # Convert to UTC before stripping timezone
                    if event_timestamp.tzinfo is not None:
                        from datetime import UTC

                        event_timestamp = event_timestamp.astimezone(UTC).replace(tzinfo=None)
                except (ValueError, TypeError):
                    return {
                        "error": f"Invalid event_at format: {raw_event_at}. Use ISO format (e.g. '2026-03-02T08:00:00')."
                    }

            encode_content = encrypted_content if encrypted_content is not None else content
            result = await encoder.encode(
                content=encode_content, timestamp=event_timestamp, tags=tags if tags else None
            )

            # Attach encryption metadata to fiber
            if encryption_meta:
                from dataclasses import replace as dc_replace

                updated_meta = {**result.fiber.metadata, **encryption_meta}
                updated_fiber = dc_replace(result.fiber, metadata=updated_meta)
                result = dc_replace(result, fiber=updated_fiber)

            import os

            _source = os.environ.get("NEURALMEMORY_SOURCE", "")[:256]
            mcp_source = f"mcp:{_source}" if _source else "mcp_tool"

            expiry_days = args.get("expires_days")
            raw_trust = args.get("trust_score")
            trust_score: float | None = None
            if raw_trust is not None:
                try:
                    trust_score = float(raw_trust)
                    if not (0.0 <= trust_score <= 1.0):
                        return {"error": f"trust_score must be 0.0-1.0, got {raw_trust}"}
                except (TypeError, ValueError):
                    return {"error": f"Invalid trust_score: {raw_trust}"}

            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=mem_type,
                priority=priority,
                source=mcp_source,
                expires_in_days=expiry_days,
                tags=tags if tags else None,
                trust_score=trust_score,
            )
            await storage.add_typed_memory(typed_mem)

            # Set type-specific decay rate on neuron states
            type_decay_rate = get_decay_rate(mem_type.value)
            for neuron in result.neurons_created:
                state = await storage.get_neuron_state(neuron.id)
                if state and state.decay_rate != type_decay_rate:
                    from neural_memory.core.neuron import NeuronState

                    updated_state = NeuronState(
                        neuron_id=state.neuron_id,
                        activation_level=state.activation_level,
                        access_frequency=state.access_frequency,
                        last_activated=state.last_activated,
                        decay_rate=type_decay_rate,
                        created_at=state.created_at,
                    )
                    await storage.update_neuron_state(updated_state)

            # Link to registered source if source_id provided
            source_id = args.get("source_id")
            if source_id and isinstance(source_id, str):
                source_obj = await storage.get_source(source_id)
                if source_obj is not None:
                    # Store source_id in typed_memory's source field
                    await storage.update_typed_memory_source(result.fiber.id, f"source:{source_id}")
                    # Create SOURCE_OF synapse: source → anchor neuron
                    from neural_memory.core.synapse import Synapse, SynapseType

                    source_syn = Synapse.create(
                        source_id=source_id,
                        target_id=result.fiber.anchor_neuron_id,
                        type=SynapseType.SOURCE_OF,
                        weight=1.0,
                    )
                    await storage.add_synapse(source_syn)
                else:
                    logger.warning("source_id '%s' not found, skipping source link", source_id)

            # Create STORED_BY audit synapse: anchor → agent identifier
            try:
                from neural_memory.core.synapse import Synapse, SynapseType

                stored_by_actor = args.get("stored_by", "mcp_agent")
                stored_by_syn = Synapse.create(
                    source_id=result.fiber.anchor_neuron_id,
                    target_id=result.fiber.anchor_neuron_id,  # self-referencing audit
                    type=SynapseType.STORED_BY,
                    weight=1.0,
                    metadata={"actor": stored_by_actor, "tool": "nmem_remember"},
                )
                await storage.add_synapse(stored_by_syn)
            except Exception:
                logger.debug("STORED_BY synapse creation failed (non-critical)", exc_info=True)

            await storage.batch_save()
        finally:
            storage.enable_auto_save()

        # Auto-schedule high-priority fibers for spaced repetition
        if priority.value >= 7:
            try:
                from neural_memory.engine.spaced_repetition import SpacedRepetitionEngine

                sr_engine = SpacedRepetitionEngine(storage, brain.config)
                await sr_engine.auto_schedule_fiber(result.fiber.id, brain.id)
            except Exception:
                logger.debug("Auto-schedule for review failed (non-critical)", exc_info=True)

        self._fire_eternal_trigger(content)

        await self._record_tool_action("remember", content[:100])

        pulse = await self._check_maintenance()

        await self.hooks.emit(
            HookEvent.POST_REMEMBER,
            {
                "fiber_id": result.fiber.id,
                "content": content,
                "type": mem_type.value,
                "neurons_created": len(result.neurons_created),
                "conflicts_detected": result.conflicts_detected,
            },
        )

        response: dict[str, Any] = {
            "success": True,
            "fiber_id": result.fiber.id,
            "memory_type": mem_type.value,
            "neurons_created": len(result.neurons_created),
            "message": f"Remembered: {content[:50]}{'...' if len(content) > 50 else ''}",
        }

        if source_id and isinstance(source_id, str):
            response["source_id"] = source_id

        if redacted_matches:
            response["auto_redacted"] = True
            response["auto_redacted_count"] = len(redacted_matches)

        if encryption_meta:
            response["encrypted"] = True
            if sensitive_detected:
                response["auto_encrypted_sensitive"] = True
                response["sensitive_types_encrypted"] = sorted(
                    {m.type.value for m in remaining_matches}
                )

        if expiry_days is not None:
            response["expires_in_days"] = expiry_days

        # Surface dedup hint when duplicate anchor was reused
        dedup_alias_of = result.fiber.metadata.get("_dedup_alias_of")
        if dedup_alias_of is None and result.neurons_created:
            for neuron in result.neurons_created:
                dedup_alias_of = neuron.metadata.get("_dedup_alias_of")
                if dedup_alias_of:
                    break
        if dedup_alias_of:
            response["dedup_hint"] = {
                "similar_existing": dedup_alias_of,
                "message": "Similar memory already exists. Created alias link.",
            }

        try:
            conflicts_detected = int(result.conflicts_detected)
        except (TypeError, ValueError, AttributeError):
            conflicts_detected = 0
        if conflicts_detected > 0:
            response["conflicts_detected"] = conflicts_detected
            response["message"] += f" ({conflicts_detected} conflict(s) detected)"

        hint = self._get_maintenance_hint(pulse)
        if hint:
            response["maintenance_hint"] = hint

        update_hint = self.get_update_hint()
        if update_hint:
            response["update_hint"] = update_hint

        # Related memory discovery via 2-hop spreading activation
        try:
            anchor_id = result.fiber.anchor_neuron_id
            if anchor_id:
                from neural_memory.engine.activation import SpreadingActivation

                activator = SpreadingActivation(storage, brain.config)
                activations = await activator.activate(
                    anchor_neurons=[anchor_id],
                    max_hops=2,
                    min_activation=0.05,
                )

                # Pre-filter: only keep hop>0 candidates, sort by activation
                # descending, cap to top 20 to limit I/O from get_neurons_batch
                candidates = sorted(
                    (
                        ar
                        for ar in activations.values()
                        if ar.hop_distance > 0 and ar.neuron_id != anchor_id
                    ),
                    key=lambda ar: ar.activation_level,
                    reverse=True,
                )[:20]

                candidate_ids = [c.neuron_id for c in candidates]

                if candidate_ids:
                    related_neurons = await storage.get_neurons_batch(candidate_ids)
                    anchor_neurons = {
                        nid: n for nid, n in related_neurons.items() if n.metadata.get("is_anchor")
                    }

                    if anchor_neurons:
                        # Take top 3 anchor neurons by activation level
                        sorted_anchors = sorted(
                            anchor_neurons.keys(),
                            key=lambda nid: activations[nid].activation_level,
                            reverse=True,
                        )[:3]

                        # Map anchor neurons to their fibers
                        fibers = await storage.find_fibers_batch(sorted_anchors)
                        fiber_by_anchor: dict[str, Any] = {}
                        for fiber in fibers:
                            if (
                                fiber.anchor_neuron_id in anchor_neurons
                                and fiber.id != result.fiber.id
                            ):
                                fiber_by_anchor.setdefault(fiber.anchor_neuron_id, fiber)

                        related_memories = []
                        for nid in sorted_anchors:
                            related_fiber = fiber_by_anchor.get(nid)
                            if related_fiber:
                                preview = (
                                    related_fiber.summary or anchor_neurons[nid].content or ""
                                )[:100]
                                related_memories.append(
                                    {
                                        "fiber_id": related_fiber.id,
                                        "preview": preview,
                                        "similarity": round(activations[nid].activation_level, 2),
                                    }
                                )

                        if related_memories:
                            response["related_memories"] = related_memories
        except Exception:
            logger.warning("Related memory discovery failed (non-critical)", exc_info=True)

        # Onboarding hint for fresh brains
        onboarding = await self._check_onboarding()
        if onboarding:
            response["onboarding"] = onboarding

        # Surface pending alerts count
        alert_info = await self._surface_pending_alerts()  # type: ignore[attr-defined]
        if alert_info:
            response.update(alert_info)

        return response

    async def _remember_batch(self, args: dict[str, Any]) -> dict[str, Any]:
        """Store multiple memories in a single call."""
        from neural_memory.mcp.constants import MAX_BATCH_SIZE, MAX_BATCH_TOTAL_CHARS

        memories = args.get("memories")
        if not memories or not isinstance(memories, list):
            return {"error": "memories is required and must be an array"}
        if len(memories) > MAX_BATCH_SIZE:
            return {"error": f"Too many items ({len(memories)}). Max: {MAX_BATCH_SIZE}."}
        if len(memories) == 0:
            return {"error": "memories array must not be empty"}

        # Validate total content size to prevent memory pressure
        total_chars = sum(len(m.get("content", "")) for m in memories if isinstance(m, dict))
        if total_chars > MAX_BATCH_TOTAL_CHARS:
            return {
                "error": f"Total content too large ({total_chars} chars). Max: {MAX_BATCH_TOTAL_CHARS}."
            }

        results: list[dict[str, Any]] = []
        saved = 0
        failed = 0

        for idx, item in enumerate(memories):
            if not isinstance(item, dict):
                results.append(
                    {"index": idx, "status": "error", "reason": "item must be an object"}
                )
                failed += 1
                continue

            # Build args for single _remember, preserving all supported fields
            single_args: dict[str, Any] = {}
            for key in (
                "content",
                "type",
                "priority",
                "tags",
                "expires_days",
                "encrypted",
                "event_at",
            ):
                if key in item:
                    single_args[key] = item[key]

            try:
                result = await self._remember(single_args)
                if result.get("success"):
                    results.append(
                        {
                            "index": idx,
                            "status": "ok",
                            "fiber_id": result.get("fiber_id"),
                            "memory_type": result.get("memory_type"),
                        }
                    )
                    saved += 1
                else:
                    results.append(
                        {
                            "index": idx,
                            "status": "error",
                            "reason": result.get("error", "unknown error"),
                        }
                    )
                    failed += 1
            except Exception as e:
                logger.error("Batch remember item %d failed: %s", idx, e)
                results.append({"index": idx, "status": "error", "reason": str(e)})
                failed += 1

        return {
            "success": saved > 0,
            "saved": saved,
            "failed": failed,
            "total": len(memories),
            "results": results,
        }

    async def _recall(self, args: dict[str, Any]) -> dict[str, Any]:
        """Query memories via spreading activation."""
        # Cross-brain recall: early return if brains parameter is provided
        brain_names = args.get("brains")
        if brain_names and isinstance(brain_names, list) and len(brain_names) > 0:
            return await self._cross_brain_recall(args, brain_names)

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        query = args.get("query")
        if not query or not isinstance(query, str):
            return {"error": "query is required and must be a string"}
        try:
            depth = DepthLevel(args.get("depth", 1))
        except ValueError:
            return {"error": f"Invalid depth level: {args.get('depth')}. Must be 0-3."}
        max_tokens = min(args.get("max_tokens", 500), 10_000)
        min_confidence = args.get("min_confidence", 0.0)
        recall_mode = args.get("mode", "associative")
        if recall_mode not in ("associative", "exact"):
            return {"error": f"Invalid mode: {recall_mode}. Must be 'associative' or 'exact'."}
        tags = _parse_tags(args)
        include_citations = args.get("include_citations", True)
        min_trust: float | None = None
        raw_min_trust = args.get("min_trust")
        if raw_min_trust is not None:
            try:
                min_trust = float(raw_min_trust)
            except (TypeError, ValueError):
                return {"error": f"Invalid min_trust: {raw_min_trust}"}

        # Inject session context for richer recall on vague queries
        effective_query = query
        try:
            session = await self._get_active_session(storage)
            if session and isinstance(session, dict):
                session_terms: list[str] = []
                feature = session.get("feature", "")
                task = session.get("task", "")
                if isinstance(feature, str) and feature:
                    session_terms.append(feature)
                if isinstance(task, str) and task:
                    session_terms.append(task)
                if session_terms and len(query.split()) < 8:
                    effective_query = f"{query} [context: {', '.join(session_terms)}]"
        except Exception:
            logger.debug("Session context injection failed", exc_info=True)

        # Parse optional temporal filter
        valid_at = None
        if "valid_at" in args:
            try:
                valid_at = datetime.fromisoformat(args["valid_at"])
                # Convert to UTC before stripping timezone
                if valid_at.tzinfo is not None:
                    from datetime import UTC

                    valid_at = valid_at.astimezone(UTC).replace(tzinfo=None)
            except (ValueError, TypeError):
                return {"error": f"Invalid valid_at datetime: {args['valid_at']}"}

        await self.hooks.emit(HookEvent.PRE_RECALL, {"query": query, "depth": depth.value})

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=effective_query,
            depth=depth,
            max_tokens=max_tokens,
            reference_time=utcnow(),
            valid_at=valid_at,
            tags=tags,
        )

        # Passive auto-capture on long queries
        if self.config.auto.enabled and len(query) >= 50:
            await self._passive_capture(query)

        self._fire_eternal_trigger(query)

        if result.confidence < min_confidence:
            return {
                "answer": None,
                "message": f"No memories found with confidence >= {min_confidence}",
                "confidence": result.confidence,
            }

        # Post-filter by trust_score if min_trust is specified
        if min_trust is not None and result.fibers_matched:
            try:
                trusted_fiber_ids: set[str] = set()
                for fid in result.fibers_matched:
                    tm = await storage.get_typed_memory(fid)
                    if tm is None:
                        trusted_fiber_ids.add(fid)  # No typed_memory = include by default
                    elif tm.trust_score is None:
                        trusted_fiber_ids.add(fid)  # Unscored = include by default
                    elif tm.trust_score >= min_trust:
                        trusted_fiber_ids.add(fid)
                filtered_fibers = [f for f in result.fibers_matched if f in trusted_fiber_ids]
                result = (
                    result._replace(fibers_matched=filtered_fibers)
                    if hasattr(result, "_replace")
                    else result
                )
            except Exception:
                logger.debug("Trust filter failed (non-critical)", exc_info=True)

        # Exact mode: return raw neuron contents without truncation
        if recall_mode == "exact" and result.fibers_matched:
            exact_items: list[dict[str, Any]] = []
            for fid in result.fibers_matched:
                fiber = await storage.get_fiber(fid)
                if not fiber:
                    continue
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if not anchor:
                    continue
                content = anchor.content
                # Decrypt if needed
                if fiber.metadata.get("encrypted"):
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
                        logger.debug("Decryption failed in exact recall", exc_info=True)
                tm = await storage.get_typed_memory(fid)
                item: dict[str, Any] = {
                    "fiber_id": fid,
                    "content": content,
                    "memory_type": tm.memory_type.value if tm else None,
                    "priority": tm.priority.value if tm else None,
                    "tags": list(tm.tags) if tm and tm.tags else [],
                    "created_at": fiber.created_at.isoformat() if fiber.created_at else None,
                }
                # Include structure metadata if present
                structure = anchor.metadata.get("_structure") if anchor.metadata else None
                if structure:
                    item["structure"] = structure

                # Citation + audit trail (Phase 4)
                citation_audit = await _build_citation_audit(storage, anchor.id, include_citations)
                if citation_audit.get("citation"):
                    item["citation"] = citation_audit["citation"]
                if citation_audit.get("audit"):
                    item["audit"] = citation_audit["audit"]

                exact_items.append(item)

            response: dict[str, Any] = {
                "mode": "exact",
                "memories": exact_items,
                "confidence": result.confidence,
                "neurons_activated": result.neurons_activated,
                "fibers_matched": result.fibers_matched,
                "depth_used": result.depth_used.value,
            }
        else:
            response = {
                "answer": result.context or "No relevant memories found.",
                "confidence": result.confidence,
                "neurons_activated": result.neurons_activated,
                "fibers_matched": result.fibers_matched,
                "depth_used": result.depth_used.value,
                "tokens_used": result.tokens_used,
            }

        if result.score_breakdown is not None:
            response["score_breakdown"] = {
                "base_activation": round(result.score_breakdown.base_activation, 4),
                "intersection_boost": round(result.score_breakdown.intersection_boost, 4),
                "freshness_boost": round(result.score_breakdown.freshness_boost, 4),
                "frequency_boost": round(result.score_breakdown.frequency_boost, 4),
            }

        # Surface conflict info from retrieval
        disputed_ids: list[str] = (result.metadata or {}).get("disputed_ids", [])
        if disputed_ids:
            response["has_conflicts"] = True
            response["conflict_count"] = len(disputed_ids)

            # Full conflict details only when opt-in
            if args.get("include_conflicts"):
                neurons_map = await storage.get_neurons_batch(disputed_ids)
                response["conflicts"] = [
                    {
                        "existing_neuron_id": nid,
                        "content": n.content[:200] if n else "",
                        "status": "superseded"
                        if n and n.metadata.get("_superseded")
                        else "disputed",
                    }
                    for nid, n in neurons_map.items()
                    if n is not None
                ]

        # Expiry warnings (opt-in)
        warn_expiry_days = args.get("warn_expiry_days")
        if warn_expiry_days is not None and result.fibers_matched:
            try:
                expiring = await storage.get_expiring_memories_for_fibers(
                    fiber_ids=result.fibers_matched,
                    within_days=int(warn_expiry_days),
                )
                if expiring:
                    response["expiry_warnings"] = [
                        {
                            "fiber_id": tm.fiber_id,
                            "memory_type": tm.memory_type.value,
                            "days_until_expiry": tm.days_until_expiry,
                            "priority": tm.priority.value,
                            "suggestion": "Re-store this memory if still relevant, or set a new expires_days.",
                        }
                        for tm in expiring
                    ]
            except Exception:
                logger.debug("Expiry warning check failed", exc_info=True)

        # Enrich results with source metadata from typed_memory.source
        try:
            if result.fibers_matched:
                source_map: dict[str, dict[str, Any]] = {}
                for fid in result.fibers_matched:
                    tm = await storage.get_typed_memory(fid)
                    if not tm or not tm.source or not tm.source.startswith("source:"):
                        continue
                    src_id = tm.source[len("source:") :]
                    src = await storage.get_source(src_id)
                    if src:
                        source_map[fid] = {
                            "source_id": src.id,
                            "name": src.name,
                            "source_type": src.source_type.value,
                            "version": src.version,
                            "status": src.status.value,
                        }
                if source_map:
                    response["sources"] = source_map
        except Exception:
            logger.debug("Source enrichment failed (non-critical)", exc_info=True)

        await self._record_tool_action("recall", query[:100])

        pulse = await self._check_maintenance()
        hint = self._get_maintenance_hint(pulse)
        if hint:
            response["maintenance_hint"] = hint

        update_hint = self.get_update_hint()
        if update_hint:
            response["update_hint"] = update_hint

        await self.hooks.emit(
            HookEvent.POST_RECALL,
            {
                "query": query,
                "confidence": result.confidence,
                "neurons_activated": result.neurons_activated,
                "fibers_matched": result.fibers_matched,
            },
        )

        # Suggest related queries from learned patterns
        try:
            from neural_memory.engine.query_pattern_mining import (
                extract_topics,
                suggest_follow_up_queries,
            )

            topics = extract_topics(query)
            if topics:
                related = await suggest_follow_up_queries(storage, topics, brain.config)
                if related:
                    response["related_queries"] = related
        except Exception:
            logger.debug("Query pattern suggestion failed", exc_info=True)

        # Onboarding hint for fresh brains
        onboarding = await self._check_onboarding()
        if onboarding:
            response["onboarding"] = onboarding

        # Cross-language hint: suggest embedding when recall misses due to language mismatch
        cross_lang_hint = await self._check_cross_language_hint(
            query,
            result,
            brain.config,
        )
        if cross_lang_hint:
            response["cross_language_hint"] = cross_lang_hint

        # Surface pending alerts count
        alert_info = await self._surface_pending_alerts()  # type: ignore[attr-defined]
        if alert_info:
            response.update(alert_info)

        return response

    async def _cross_brain_recall(
        self, args: dict[str, Any], brain_names: list[str]
    ) -> dict[str, Any]:
        """Handle cross-brain recall by querying multiple brains in parallel."""
        from neural_memory.engine.cross_brain import cross_brain_recall

        query = args.get("query", "")
        if not query:
            return {"error": "query is required"}

        # Validate and cap at 5 brains
        import re

        _brain_pattern = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
        brain_names = [n for n in brain_names[:5] if isinstance(n, str) and _brain_pattern.match(n)]
        if not brain_names:
            return {"error": "No valid brain names provided"}
        try:
            depth = int(args.get("depth", 1))
            depth = max(0, min(depth, 3))
        except (TypeError, ValueError):
            depth = 1
        max_tokens = min(int(args.get("max_tokens", 500)), 10_000)

        tags = _parse_tags(args)

        try:
            result = await cross_brain_recall(
                config=self.config,
                brain_names=brain_names,
                query=query,
                depth=depth,
                max_tokens=max_tokens,
                tags=tags,
            )
        except Exception:
            logger.error("Cross-brain recall failed", exc_info=True)
            return {"error": "Cross-brain recall failed"}

        fibers_out = [
            {
                "fiber_id": f.fiber_id,
                "source_brain": f.source_brain,
                "summary": f.summary,
                "confidence": f.confidence,
            }
            for f in result.fibers
        ]

        return {
            "answer": result.merged_context,
            "brains_queried": result.brains_queried,
            "total_neurons_activated": result.total_neurons_activated,
            "fibers": fibers_out,
            "cross_brain": True,
        }

    async def _context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get recent context."""
        storage = await self.get_storage()

        limit = min(args.get("limit", 10), 200)
        fresh_only = args.get("fresh_only", False)

        fibers = await storage.get_fibers(limit=limit * 2 if fresh_only else limit)
        if not fibers:
            result: dict[str, Any] = {"context": "No memories stored yet.", "count": 0}
            onboarding = await self._check_onboarding()
            if onboarding:
                result["onboarding"] = onboarding
            return result

        if fresh_only:
            from neural_memory.safety.freshness import FreshnessLevel, evaluate_freshness

            now = utcnow()
            fresh_fibers = [
                f
                for f in fibers
                if evaluate_freshness(f.created_at, now).level
                in (FreshnessLevel.FRESH, FreshnessLevel.RECENT)
            ]
            fibers = fresh_fibers[:limit]

        # Smart context optimization: score, dedup, budget
        from neural_memory.engine.context_optimizer import optimize_context

        try:
            max_tokens = int(self.config.brain.max_context_tokens)
            if max_tokens < 100:
                max_tokens = 4000
        except (TypeError, ValueError, AttributeError):
            max_tokens = 4000
        plan = await optimize_context(storage, fibers, max_tokens)

        if plan.items:
            context_parts = [f"- {item.content}" for item in plan.items]
            context_text = "\n".join(context_parts)
        else:
            context_text = "No context available."

        await self._record_tool_action("context")

        response: dict[str, Any] = {
            "context": context_text,
            "count": len(plan.items),
            "tokens_used": plan.total_tokens,
        }

        if plan.dropped_count > 0:
            response["optimization_stats"] = {
                "items_dropped": plan.dropped_count,
                "top_score": round(plan.items[0].score, 4) if plan.items else 0.0,
            }

        # Expiry warnings (opt-in)
        warn_expiry_days = args.get("warn_expiry_days")
        if warn_expiry_days is not None and fibers:
            try:
                fiber_ids = [f.id for f in fibers]
                expiring = await storage.get_expiring_memories_for_fibers(
                    fiber_ids=fiber_ids,
                    within_days=int(warn_expiry_days),
                )
                if expiring:
                    response["expiry_warnings"] = [
                        {
                            "fiber_id": tm.fiber_id,
                            "memory_type": tm.memory_type.value,
                            "days_until_expiry": tm.days_until_expiry,
                            "priority": tm.priority.value,
                            "suggestion": "Re-store this memory if still relevant, or set a new expires_days.",
                        }
                        for tm in expiring
                    ]
            except Exception:
                logger.debug("Expiry warning check failed", exc_info=True)

        # Surface pending alerts count
        alert_info = await self._surface_pending_alerts()  # type: ignore[attr-defined]
        if alert_info:
            response.update(alert_info)

        return response

    async def _todo(self, args: dict[str, Any]) -> dict[str, Any]:
        """Add a TODO."""
        task = args.get("task")
        if not task or not isinstance(task, str):
            return {"error": "task is required and must be a string"}
        return await self._remember(
            {
                "content": task,
                "type": "todo",
                "priority": args.get("priority", 5),
                "expires_days": 30,
            }
        )

    async def _stats(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get brain statistics."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        stats = await storage.get_enhanced_stats(brain.id)

        # Count active conflicts (unresolved CONTRADICTS synapses)
        conflicts_active = 0
        try:
            from neural_memory.core.synapse import SynapseType

            contradicts_synapses = await storage.get_synapses(type=SynapseType.CONTRADICTS)
            conflicts_active = sum(
                1 for s in contradicts_synapses if not s.metadata.get("_resolved")
            )
        except Exception:
            logger.debug("Conflict count failed (non-critical)", exc_info=True)

        response = {
            "version": __version__,
            "brain": brain.name,
            "neuron_count": stats["neuron_count"],
            "synapse_count": stats["synapse_count"],
            "fiber_count": stats["fiber_count"],
            "db_size_bytes": stats.get("db_size_bytes", 0),
            "today_fibers_count": stats.get("today_fibers_count", 0),
            "hot_neurons": stats.get("hot_neurons", []),
            "newest_memory": stats.get("newest_memory"),
            "conflicts_active": conflicts_active,
        }

        # Actionable hints based on brain state
        hints = await self._generate_stats_hints(storage, brain.id, stats)
        if hints:
            response["hints"] = hints

        # Onboarding hint for fresh brains
        onboarding = await self._check_onboarding()
        if onboarding:
            response["onboarding"] = onboarding

        update_hint = self.get_update_hint()
        if update_hint:
            response["update_hint"] = update_hint

        return response

    async def _generate_stats_hints(
        self,
        storage: Any,
        brain_id: str,
        stats: dict[str, Any],
    ) -> list[str]:
        """Generate actionable hints based on brain state.

        Hints appear in stats output to guide users on what to do next.
        """
        hints: list[str] = []
        fiber_count = stats.get("fiber_count", 0)
        neuron_count = stats.get("neuron_count", 0)
        synapse_count = stats.get("synapse_count", 0)

        if fiber_count == 0:
            return hints

        # Consolidation hint: many memories but 0% consolidated
        try:
            from neural_memory.engine.memory_stages import MemoryStage

            semantic_records = await storage.find_maturations(stage=MemoryStage.SEMANTIC)
            semantic_count = len(semantic_records)
            consolidation_pct = (semantic_count / fiber_count * 100) if fiber_count else 0

            if fiber_count >= 50 and consolidation_pct == 0:
                hints.append(
                    f"You have {fiber_count} memories but 0% consolidated. "
                    "Run: nmem_auto action='process' or nmem consolidate --strategy mature "
                    "to advance memories from episodic to semantic stage."
                )
            elif fiber_count >= 100 and consolidation_pct < 10:
                hints.append(
                    f"{fiber_count} memories, only {consolidation_pct:.0f}% consolidated. "
                    "Recall topics you've stored to help memories mature, "
                    "then run consolidation."
                )
        except Exception:
            logger.debug("Maturation check failed (non-critical)", exc_info=True)

        # Low activation hint: many neurons but few activated
        try:
            states = await storage.get_all_neuron_states()
            activated = sum(1 for s in states if s.access_frequency > 0)
            activation_pct = (activated / neuron_count * 100) if neuron_count else 0

            if neuron_count >= 50 and activation_pct < 20:
                idle_count = neuron_count - activated
                hints.append(
                    f"{idle_count} neurons ({100 - activation_pct:.0f}%) never accessed. "
                    "Use nmem_recall with topics you've stored to activate them "
                    "and strengthen recall pathways."
                )
        except Exception:
            logger.debug("Activation check failed (non-critical)", exc_info=True)

        # Low connectivity hint
        if neuron_count > 0:
            connectivity = synapse_count / neuron_count
            if connectivity < 2.0 and neuron_count >= 20:
                hints.append(
                    f"Low connectivity ({connectivity:.1f} synapses/neuron, target: 3+). "
                    "Store memories with context like 'X because Y' to build richer links."
                )

        # Spaced repetition hint: if review system has due items
        try:
            from neural_memory.engine.spaced_repetition import SpacedRepetitionEngine

            brain = await storage.get_brain(brain_id)
            if brain:
                review_engine = SpacedRepetitionEngine(storage, brain.config)
                review_stats = await review_engine.get_stats()
                due_count = review_stats.get("due", 0)
                if due_count > 0:
                    hints.append(
                        f"{due_count} memories due for review. "
                        "Run nmem_review action='queue' to strengthen retention."
                    )
        except Exception:
            logger.debug("Review check failed (non-critical)", exc_info=True)

        return hints

    async def _health(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run brain health diagnostics."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return {"error": "No brain configured"}

        from neural_memory.engine.diagnostics import DiagnosticsEngine

        engine = DiagnosticsEngine(storage)
        report = await engine.analyze(brain.id)

        return {
            "brain": brain.name,
            "grade": report.grade,
            "purity_score": report.purity_score,
            "connectivity": report.connectivity,
            "diversity": report.diversity,
            "freshness": report.freshness,
            "consolidation_ratio": report.consolidation_ratio,
            "orphan_rate": report.orphan_rate,
            "activation_efficiency": report.activation_efficiency,
            "recall_confidence": report.recall_confidence,
            "neuron_count": report.neuron_count,
            "synapse_count": report.synapse_count,
            "fiber_count": report.fiber_count,
            "warnings": [
                {"severity": w.severity.value, "code": w.code, "message": w.message}
                for w in report.warnings
            ],
            "recommendations": list(report.recommendations),
            "top_penalties": [
                {
                    "component": p.component,
                    "current_score": p.current_score,
                    "weight": p.weight,
                    "penalty_points": p.penalty_points,
                    "estimated_gain": p.estimated_gain,
                    "action": p.action,
                }
                for p in report.top_penalties
            ],
            "roadmap": self._build_health_roadmap(report),
        }

    @staticmethod
    def _build_health_roadmap(report: Any) -> dict[str, Any]:
        """Build an actionable roadmap from current grade to next grade.

        Shows prioritized steps sorted by estimated_gain (biggest impact first),
        the points needed to reach the next grade, and specific commands to run.
        """
        grade_thresholds = {"F": 40, "D": 60, "C": 75, "B": 90, "A": 100}
        next_grade_map = {"F": "D", "D": "C", "C": "B", "B": "A", "A": "A"}

        current_grade = report.grade
        next_grade = next_grade_map.get(current_grade, "A")
        target_score = grade_thresholds.get(next_grade, 100)
        points_needed = max(0, target_score - report.purity_score)

        # Sort penalties by estimated gain (most impactful first)
        steps: list[dict[str, Any]] = []
        cumulative_gain = 0.0
        for p in sorted(report.top_penalties, key=lambda x: x.estimated_gain, reverse=True):
            if p.estimated_gain <= 0:
                continue
            cumulative_gain += p.estimated_gain
            steps.append(
                {
                    "priority": len(steps) + 1,
                    "component": p.component,
                    "current": f"{p.current_score:.0%}",
                    "action": p.action,
                    "estimated_gain": f"+{p.estimated_gain:.1f} pts",
                    "sufficient": cumulative_gain >= points_needed,
                }
            )

        # Estimate timeframe based on points needed
        if points_needed <= 0:
            timeframe = "Already achieved"
        elif points_needed <= 5:
            timeframe = "~1 week with daily use"
        elif points_needed <= 15:
            timeframe = "~2 weeks with regular use"
        elif points_needed <= 30:
            timeframe = "~1 month with consistent use"
        else:
            timeframe = "~2 months with dedicated effort"

        roadmap: dict[str, Any] = {
            "current_grade": current_grade,
            "current_score": report.purity_score,
            "next_grade": next_grade,
            "points_needed": round(points_needed, 1),
            "timeframe": timeframe,
            "steps": steps,
        }

        if current_grade == "A":
            roadmap["message"] = (
                "Excellent! Brain is at top grade. Maintain regular recall and storage."
            )
        elif points_needed <= sum(p.estimated_gain for p in report.top_penalties):
            roadmap["message"] = (
                f"Grade {current_grade} → {next_grade} is achievable in {timeframe} "
                f"by addressing the top {min(len(steps), 3)} actions below."
            )
        else:
            roadmap["message"] = (
                f"Grade {next_grade} requires {points_needed:.1f} more points ({timeframe}). "
                "Focus on the highest-impact actions and give the brain time to mature."
            )

        return roadmap

    async def _evolution(self, args: dict[str, Any]) -> dict[str, Any]:
        """Measure brain evolution dynamics."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        from neural_memory.engine.brain_evolution import EvolutionEngine

        try:
            engine = EvolutionEngine(storage)
            evo = await engine.analyze(brain.id)
        except Exception:
            logger.error("Evolution analysis failed", exc_info=True)
            return {"error": "Evolution analysis failed"}

        result: dict[str, Any] = {
            "brain": evo.brain_name,
            "proficiency_level": evo.proficiency_level.value,
            "proficiency_index": evo.proficiency_index,
            "maturity_level": evo.maturity_level,
            "plasticity": evo.plasticity,
            "density": evo.density,
            "activity_score": evo.activity_score,
            "semantic_ratio": evo.semantic_ratio,
            "reinforcement_days": evo.reinforcement_days,
            "topology_coherence": evo.topology_coherence,
            "plasticity_index": evo.plasticity_index,
            "knowledge_density": evo.knowledge_density,
            "total_neurons": evo.total_neurons,
            "total_synapses": evo.total_synapses,
            "total_fibers": evo.total_fibers,
            "fibers_at_semantic": evo.fibers_at_semantic,
            "fibers_at_episodic": evo.fibers_at_episodic,
        }

        if evo.stage_distribution is not None:
            result["stage_distribution"] = {
                "short_term": evo.stage_distribution.short_term,
                "working": evo.stage_distribution.working,
                "episodic": evo.stage_distribution.episodic,
                "semantic": evo.stage_distribution.semantic,
                "total": evo.stage_distribution.total,
            }

        if evo.closest_to_semantic:
            result["closest_to_semantic"] = [
                {
                    "fiber_id": p.fiber_id,
                    "stage": p.stage,
                    "days_in_stage": p.days_in_stage,
                    "days_required": p.days_required,
                    "reinforcement_days": p.reinforcement_days,
                    "reinforcement_required": p.reinforcement_required,
                    "progress_pct": p.progress_pct,
                    "next_step": p.next_step,
                }
                for p in evo.closest_to_semantic
            ]

        return result

    async def _suggest(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get prefix-based autocomplete suggestions or idle neuron reinforcement hints."""
        storage = await self.get_storage()
        prefix = args.get("prefix", "")
        limit = min(args.get("limit", 5), 20)

        # When no prefix: return idle neurons that need reinforcement
        if not prefix.strip():
            return await self._suggest_idle_neurons(storage, limit)

        type_filter = None
        if "type_filter" in args:
            from neural_memory.core.neuron import NeuronType

            try:
                type_filter = NeuronType(args["type_filter"])
            except ValueError:
                return {"error": "Invalid type_filter value"}

        suggestions = await storage.suggest_neurons(
            prefix=prefix, type_filter=type_filter, limit=limit
        )
        formatted = [
            {
                "content": s["content"],
                "type": s["type"],
                "neuron_id": s["neuron_id"],
                "score": s["score"],
            }
            for s in suggestions
        ]
        return {
            "suggestions": formatted,
            "count": len(formatted),
            "tokens_used": sum(len(s["content"].split()) for s in formatted),
        }

    async def _suggest_idle_neurons(self, storage: Any, limit: int) -> dict[str, Any]:
        """Return neurons that have never been accessed — candidates for reinforcement.

        Sorted by creation age (oldest idle neurons first) to prioritize
        long-neglected knowledge.
        """
        try:
            states = await storage.get_all_neuron_states()
            idle_states = [s for s in states if s.access_frequency == 0]

            # Sort by creation time ascending (oldest first)
            idle_states.sort(key=lambda s: s.created_at or "")

            suggestions = []
            for state in idle_states[:limit]:
                neuron = await storage.get_neuron(state.neuron_id)
                if neuron is None:
                    continue
                content_preview = neuron.content[:200] if neuron.content else ""
                suggestions.append(
                    {
                        "content": content_preview,
                        "type": neuron.type.value if neuron.type else "unknown",
                        "neuron_id": neuron.id,
                        "score": 0.0,
                        "idle": True,
                    }
                )

            total_idle = len(idle_states)
            hint = ""
            if total_idle > 0:
                hint = (
                    f"{total_idle} neurons never accessed. "
                    "Recall these topics with nmem_recall to activate them "
                    "and strengthen your memory graph."
                )

            return {
                "suggestions": suggestions,
                "count": len(suggestions),
                "total_idle": total_idle,
                "mode": "idle_reinforcement",
                "hint": hint,
                "tokens_used": sum(len(s["content"].split()) for s in suggestions),
            }
        except Exception:
            logger.debug("Idle neuron suggestion failed", exc_info=True)
            return {"suggestions": [], "count": 0}

    async def _habits(self, args: dict[str, Any]) -> dict[str, Any]:
        """Manage learned workflow habits."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        action = args.get("action", "list")

        if action == "suggest":
            current_action = args.get("current_action", "")
            if not current_action:
                return {"error": "current_action is required for suggest"}

            from neural_memory.engine.workflow_suggest import suggest_next_action

            suggestions = await suggest_next_action(storage, current_action, brain.config)
            return {
                "suggestions": [
                    {
                        "action": s.action_type,
                        "confidence": round(s.confidence, 4),
                        "source_habit": s.source_habit,
                        "sequential_count": s.sequential_count,
                    }
                    for s in suggestions
                ],
                "count": len(suggestions),
            }

        elif action == "list":
            habits = await storage.find_fibers(metadata_key="_habit_pattern", limit=1000)
            return {
                "habits": [
                    {
                        "name": h.summary or "unnamed",
                        "steps": h.metadata.get("_workflow_actions", []),
                        "frequency": h.metadata.get("_habit_frequency", 0),
                        "confidence": h.metadata.get("_habit_confidence", 0.0),
                        "fiber_id": h.id,
                    }
                    for h in habits
                ],
                "count": len(habits),
            }

        elif action == "clear":
            habits = await storage.find_fibers(metadata_key="_habit_pattern", limit=1000)
            cleared = 0
            # Delete sequentially to avoid overwhelming SQLite with concurrent writes
            for h in habits:
                await storage.delete_fiber(h.id)
                cleared += 1
            return {"cleared": cleared, "message": f"Cleared {cleared} learned habits"}

        return {"error": f"Unknown action: {action}"}

    async def _version(self, args: dict[str, Any]) -> dict[str, Any]:
        """Brain version control operations."""
        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        from neural_memory.engine.brain_versioning import VersioningEngine

        engine = VersioningEngine(storage)
        action = args.get("action", "list")

        if action == "create":
            name = args.get("name")
            if not name:
                return {"error": "Version name is required for create"}
            description = args.get("description", "")
            try:
                version = await engine.create_version(brain.id, name, description)
            except ValueError:
                return {"error": "Failed to create version: invalid parameters"}
            return {
                "success": True,
                "version_id": version.id,
                "version_name": version.version_name,
                "version_number": version.version_number,
                "neuron_count": version.neuron_count,
                "synapse_count": version.synapse_count,
                "fiber_count": version.fiber_count,
                "message": f"Created version '{name}' (#{version.version_number})",
            }

        elif action == "list":
            limit = min(args.get("limit", 20), 100)
            versions = await engine.list_versions(brain.id, limit=limit)
            return {
                "versions": [
                    {
                        "id": v.id,
                        "name": v.version_name,
                        "number": v.version_number,
                        "description": v.description,
                        "neuron_count": v.neuron_count,
                        "synapse_count": v.synapse_count,
                        "fiber_count": v.fiber_count,
                        "created_at": v.created_at.isoformat(),
                    }
                    for v in versions
                ],
                "count": len(versions),
            }

        elif action == "rollback":
            version_id = args.get("version_id")
            if not version_id:
                return {"error": "version_id is required for rollback"}
            try:
                rollback_v = await engine.rollback(brain.id, version_id)
            except ValueError:
                return {"error": "Rollback failed: version not found or invalid"}
            return {
                "success": True,
                "rollback_version_id": rollback_v.id,
                "rollback_version_name": rollback_v.version_name,
                "neuron_count": rollback_v.neuron_count,
                "synapse_count": rollback_v.synapse_count,
                "fiber_count": rollback_v.fiber_count,
                "message": f"Rolled back to '{rollback_v.version_name}'",
            }

        elif action == "diff":
            from_id = args.get("from_version")
            to_id = args.get("to_version")
            if not from_id or not to_id:
                return {"error": "from_version and to_version are required for diff"}
            try:
                diff = await engine.diff(brain.id, from_id, to_id)
            except ValueError:
                return {"error": "Diff failed: one or both versions not found"}
            return {
                "summary": diff.summary,
                "neurons_added": len(diff.neurons_added),
                "neurons_removed": len(diff.neurons_removed),
                "neurons_modified": len(diff.neurons_modified),
                "synapses_added": len(diff.synapses_added),
                "synapses_removed": len(diff.synapses_removed),
                "synapses_weight_changed": len(diff.synapses_weight_changed),
                "fibers_added": len(diff.fibers_added),
                "fibers_removed": len(diff.fibers_removed),
            }

        return {"error": f"Unknown action: {action}"}

    async def _transplant(self, args: dict[str, Any]) -> dict[str, Any]:
        """Transplant memories from another brain."""
        from neural_memory.unified_config import get_shared_storage

        target_storage = await self.get_storage()
        target_brain_id = target_storage.brain_id
        if not target_brain_id:
            return {"error": "No brain configured"}

        target_brain = await target_storage.get_brain(target_brain_id)
        if not target_brain:
            return {"error": "No brain configured"}

        source_brain_name = args.get("source_brain")
        if not source_brain_name:
            return {"error": "source_brain is required"}

        if source_brain_name == target_brain.name:
            return {
                "error": "Source brain and target brain are the same. "
                "Transplanting a brain into itself is a destructive no-op."
            }

        # Open a separate storage for the source brain (.db file)
        try:
            source_storage = await get_shared_storage(brain_name=source_brain_name)
        except Exception:
            logger.error("Failed to open source brain storage", exc_info=True)
            return {"error": "Source brain not found"}

        source_brain_id = source_storage.brain_id
        if not source_brain_id:
            return {"error": "Source brain not found"}

        source_brain = await source_storage.get_brain(source_brain_id)
        if source_brain is None:
            return {"error": "Source brain not found"}

        from neural_memory.engine.brain_transplant import TransplantFilter, transplant
        from neural_memory.engine.merge import ConflictStrategy

        tags = args.get("tags")
        memory_types = args.get("memory_types")
        strategy_str = args.get("strategy", "prefer_local")

        try:
            strategy = ConflictStrategy(strategy_str)
        except ValueError:
            return {"error": f"Invalid strategy: {strategy_str}"}

        filt = TransplantFilter(
            tags=frozenset(tags) if tags else None,
            memory_types=frozenset(memory_types) if memory_types else None,
        )

        try:
            result = await transplant(
                source_storage=source_storage,
                target_storage=target_storage,
                source_brain_id=source_brain_id,
                target_brain_id=target_brain_id,
                filt=filt,
                strategy=strategy,
            )
        except ValueError as exc:
            logger.error("Transplant failed: %s", exc)
            return {"error": "Transplant failed"}

        return {
            "success": True,
            "fibers_transplanted": result.fibers_transplanted,
            "neurons_transplanted": result.neurons_transplanted,
            "synapses_transplanted": result.synapses_transplanted,
            "merge_summary": result.merge_report.summary(),
            "message": f"Transplanted from '{source_brain_name}': {result.fibers_transplanted} fibers",
        }

    async def _record_tool_action(self, action_type: str, context: str = "") -> None:
        """Record an action event for habit learning (fire-and-forget)."""
        try:
            import os

            source = os.environ.get("NEURALMEMORY_SOURCE", "mcp")[:256]
            storage = await self.get_storage()
            await storage.record_action(
                action_type=action_type,
                action_context=context[:200] if context else "",
                session_id=f"{source}-{id(self)}",
            )
        except Exception:
            logger.debug("Action recording failed (non-critical)", exc_info=True)

    # ========== Source Registry ==========

    async def _source(self, args: dict[str, Any]) -> dict[str, Any]:
        """Manage memory sources (provenance registry)."""
        action = args.get("action", "")
        if not action:
            return {"error": "action is required"}

        storage = await self.get_storage()
        brain_id = _require_brain_id(storage)

        if action == "register":
            name = args.get("name")
            if not name or not isinstance(name, str):
                return {"error": "name is required for register"}

            from neural_memory.core.source import Source

            source = Source.create(
                brain_id=brain_id,
                name=name,
                source_type=args.get("source_type", "document"),
                version=args.get("version", ""),
                file_hash=args.get("file_hash", ""),
                metadata=args.get("metadata") or {},
            )
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
        _require_brain_id(storage)

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

    # ========== Show, Edit & Forget ==========

    async def _show(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get full verbatim content + metadata + synapses for a memory by ID."""
        memory_id = args.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            return {"error": "memory_id is required"}

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
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

        return {"error": f"Memory not found: {memory_id}"}

    async def _edit(self, args: dict[str, Any]) -> dict[str, Any]:
        """Edit an existing memory's type, content, or priority."""
        memory_id = args.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            return {"error": "memory_id is required"}

        new_type = args.get("type")
        new_content = args.get("content")
        new_priority = args.get("priority")

        if new_type is None and new_content is None and new_priority is None:
            return {"error": "At least one of type, content, or priority must be provided"}

        if new_type is not None:
            try:
                MemoryType(new_type)
            except ValueError:
                return {"error": f"Invalid memory type: {new_type}"}

        if new_content is not None and len(new_content) > MAX_CONTENT_LENGTH:
            return {
                "error": f"Content too long ({len(new_content)} chars). Max: {MAX_CONTENT_LENGTH}."
            }

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}

        # Try as fiber_id first, then as neuron_id
        typed_mem = await storage.get_typed_memory(memory_id)
        fiber = await storage.get_fiber(memory_id) if typed_mem else None

        if typed_mem and fiber:
            # Edit via fiber path
            changes: list[str] = []

            # Update typed_memory (type, priority)
            if new_type is not None or new_priority is not None:
                from dataclasses import replace as dc_replace

                updated_tm = typed_mem
                if new_type is not None:
                    updated_tm = dc_replace(updated_tm, memory_type=MemoryType(new_type))
                    changes.append(f"type: {typed_mem.memory_type.value} → {new_type}")
                if new_priority is not None:
                    updated_tm = dc_replace(updated_tm, priority=Priority.from_int(new_priority))
                    changes.append(f"priority: {typed_mem.priority.value} → {new_priority}")
                await storage.update_typed_memory(updated_tm)

            # Update anchor neuron content
            if new_content is not None:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    from dataclasses import replace as dc_replace

                    updated_neuron = dc_replace(anchor, content=new_content)
                    await storage.update_neuron(updated_neuron)
                    changes.append(f"content updated ({len(new_content)} chars)")

            return {
                "status": "edited",
                "memory_id": memory_id,
                "changes": changes,
            }

        # Try as direct neuron_id
        neuron = await storage.get_neuron(memory_id)
        if neuron:
            from dataclasses import replace as dc_replace

            changes = []
            if new_content is not None:
                neuron = dc_replace(neuron, content=new_content)
                changes.append(f"content updated ({len(new_content)} chars)")
            if new_type is not None:
                from neural_memory.core.neuron import NeuronType

                try:
                    neuron = dc_replace(neuron, type=NeuronType(new_type))
                    changes.append(f"neuron type → {new_type}")
                except ValueError:
                    pass  # NeuronType doesn't map 1:1 to MemoryType
            await storage.update_neuron(neuron)
            return {
                "status": "edited",
                "memory_id": memory_id,
                "changes": changes,
            }

        return {"error": f"Memory not found: {memory_id}"}

    async def _forget(self, args: dict[str, Any]) -> dict[str, Any]:
        """Explicitly delete or close a specific memory."""
        memory_id = args.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            return {"error": "memory_id is required"}

        hard = args.get("hard", False)
        reason = args.get("reason", "")

        storage = await self.get_storage()
        try:
            _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}

        # Look up the memory
        typed_mem = await storage.get_typed_memory(memory_id)
        fiber = await storage.get_fiber(memory_id) if typed_mem else None

        if not typed_mem and not fiber:
            # Try as neuron_id — find its fiber
            neuron = await storage.get_neuron(memory_id)
            if not neuron:
                return {"error": f"Memory not found: {memory_id}"}
            # For neuron-only delete in hard mode
            if hard:
                await storage.delete_neuron(memory_id)
                return {
                    "status": "hard_deleted",
                    "memory_id": memory_id,
                    "message": "Neuron permanently deleted",
                }
            return {
                "error": f"No typed memory found for neuron {memory_id}. Use hard=true for neuron deletion."
            }

        if hard:
            # Permanent deletion: fiber + typed_memory + neurons
            storage.disable_auto_save()
            try:
                # Delete typed memory
                await storage.delete_typed_memory(memory_id)

                # Delete fiber (CASCADE handles fiber_neurons junction)
                if fiber:
                    await storage.delete_fiber(memory_id)

                await storage.batch_save()
            finally:
                storage.enable_auto_save()

            logger.info("Hard-deleted memory %s (reason: %s)", memory_id, reason or "none")
            return {
                "status": "hard_deleted",
                "memory_id": memory_id,
                "message": "Memory permanently deleted with cascade cleanup",
            }
        else:
            # Soft delete: expire immediately
            from dataclasses import replace as dc_replace

            assert typed_mem is not None  # guaranteed by early return above
            expired_tm = dc_replace(typed_mem, expires_at=utcnow())
            await storage.update_typed_memory(expired_tm)

            logger.info("Soft-deleted memory %s (reason: %s)", memory_id, reason or "none")
            return {
                "status": "soft_deleted",
                "memory_id": memory_id,
                "message": "Memory marked as expired (will be cleaned up on next consolidation)",
            }

    async def _consolidate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run memory consolidation on the current brain."""
        from neural_memory.engine.consolidation import (
            ConsolidationConfig,
            ConsolidationStrategy,
        )
        from neural_memory.engine.consolidation_delta import run_with_delta

        storage = await self.get_storage()
        try:
            brain_id = _require_brain_id(storage)
        except ValueError:
            return {"error": "No brain configured"}

        # Parse strategy
        strategy_str = args.get("strategy", "all")
        try:
            strategy = ConsolidationStrategy(strategy_str)
        except ValueError:
            valid = [s.value for s in ConsolidationStrategy]
            return {"error": f"Invalid strategy: {strategy_str}. Valid: {valid}"}

        strategies = [strategy]
        dry_run = bool(args.get("dry_run", False))

        # Build config with optional overrides
        config_kwargs: dict[str, Any] = {}
        if "prune_weight_threshold" in args:
            val = args["prune_weight_threshold"]
            if isinstance(val, (int, float)):
                config_kwargs["prune_weight_threshold"] = float(val)
        if "merge_overlap_threshold" in args:
            val = args["merge_overlap_threshold"]
            if isinstance(val, (int, float)):
                config_kwargs["merge_overlap_threshold"] = float(val)
        if "prune_min_inactive_days" in args:
            val = args["prune_min_inactive_days"]
            if isinstance(val, (int, float)):
                config_kwargs["prune_min_inactive_days"] = float(val)

        config = ConsolidationConfig(**config_kwargs) if config_kwargs else None

        try:
            delta = await run_with_delta(
                storage,
                brain_id,
                strategies=strategies,
                dry_run=dry_run,
                config=config,
            )
        except Exception:
            logger.error("Consolidation failed", exc_info=True)
            return {"error": "Consolidation failed unexpectedly"}

        result = delta.to_dict()
        result["strategy"] = strategy_str
        result["dry_run"] = dry_run
        result["summary"] = delta.report.summary()
        return result
