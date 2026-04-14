"""MCP handler mixin for remember tools."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.memory_types import (
    MemoryType,
    Priority,
    TypedMemory,
    get_decay_rate,
    suggest_memory_type,
)
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.engine.hooks import HookEvent
from neural_memory.mcp.constants import MAX_CONTENT_LENGTH
from neural_memory.mcp.tool_handler_utils import _get_brain_or_error, _require_brain_id
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.engine.hooks import HookRegistry
    from neural_memory.mcp.maintenance_handler import HealthPulse
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class RememberHandler:
    """Mixin providing remember-related MCP tool handlers."""

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

        async def _check_onboarding(self) -> dict[str, Any] | None:
            raise NotImplementedError

        def get_update_hint(self) -> dict[str, Any] | None:
            raise NotImplementedError

        async def _surface_pending_alerts(self) -> dict[str, int] | None:
            raise NotImplementedError

        async def _record_tool_action(self, action_type: str, context: str = "") -> None:
            raise NotImplementedError

    # ──────────────────── Helpers ────────────────────

    async def _get_global_storage(self) -> NeuralStorage | None:
        """Get storage for the global brain (cross-project layer).

        Creates the global brain if it doesn't exist. Returns None if
        global brain cannot be initialized. Caller must close the returned
        storage when done.
        """
        from neural_memory.unified_config import GLOBAL_BRAIN_NAME

        storage = None
        try:
            await self.config.ensure_global_brain()

            from neural_memory.storage.sqlite_store import SQLiteStorage

            db_path = self.config.get_brain_db_path(GLOBAL_BRAIN_NAME)
            storage = SQLiteStorage(db_path)
            await storage.initialize()

            brain = await storage.find_brain_by_name(GLOBAL_BRAIN_NAME)
            if brain:
                storage.set_brain(brain.id)
                return storage

            await storage.close()
            return None
        except Exception:
            logger.debug("Failed to get global storage", exc_info=True)
            if storage is not None:
                try:
                    await storage.close()
                except Exception:
                    pass
            return None

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
                import sentence_transformers as _st

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
        # Extract internal-only flags (not user-settable via MCP args)
        is_auto_capture = bool(args.pop("_auto_capture", False))

        storage = await self.get_storage()
        brain, err = await _get_brain_or_error(storage)
        if err:
            return err

        content = args.get("content")
        if not content or not isinstance(content, str):
            return {"error": "content is required and must be a string"}
        if len(content) > MAX_CONTENT_LENGTH:
            return {"error": f"Content too long ({len(content)} chars). Max: {MAX_CONTENT_LENGTH}."}

        # Light sanitization: strip control sequences / fake role tags
        # (non-blocking — see ADR-001)
        from neural_memory.safety.input_firewall import sanitize_explicit_content

        content = sanitize_explicit_content(content)

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

                original_matches = _check_sensitive(content, min_severity=2)
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

        # Write gate: reject low-quality content before encoding
        write_gate_cfg = self.config.write_gate
        if write_gate_cfg.enabled is True:
            from neural_memory.engine.quality_scorer import check_write_gate

            gate_result = check_write_gate(
                content,
                gate_config=write_gate_cfg,
                is_auto_capture=is_auto_capture,
                memory_type=args.get("type"),
                tags=args.get("tags"),
                context=args.get("context") if isinstance(args.get("context"), dict) else None,
            )
            if gate_result.rejected:
                logger.debug(
                    "Write gate rejected: %s (score=%d)",
                    gate_result.rejection_reason,
                    gate_result.score,
                )
                return {
                    "error": "Write gate rejected",
                    "rejection_reason": gate_result.rejection_reason,
                    **gate_result.to_dict(),
                }

        # Determine memory type
        if "type" in args:
            try:
                mem_type = MemoryType(args["type"])
            except ValueError:
                return {"error": f"Invalid memory type: {args['type']}"}
        else:
            mem_type = suggest_memory_type(content)

        # Layer routing: determine save destination (project vs global brain)
        layer_decision = None
        global_storage = None
        try:
            from neural_memory.engine.layer_router import MemoryLayer, route_memory

            layer_decision = route_memory(
                memory_type=mem_type.value,
                is_ephemeral=bool(args.get("ephemeral", False)),
                explicit_layer=args.get("layer"),
            )

            if layer_decision.layer == MemoryLayer.GLOBAL:
                global_storage = await self._get_global_storage()
                if global_storage is not None:
                    storage = global_storage
                    brain, err = await _get_brain_or_error(storage)
                    if err:
                        # Fall back to project brain if global brain has issues
                        logger.debug("Global brain error, falling back to project")
                        storage = await self.get_storage()
                        brain, err = await _get_brain_or_error(storage)
                        if err:
                            return err
                        layer_decision = None  # reset — saved to project
            elif layer_decision.layer == MemoryLayer.SESSION:
                # SESSION layer = ephemeral flag (handled downstream)
                # Don't report layer=session if ephemeral was already set
                if not bool(args.get("ephemeral", False)):
                    layer_decision = None  # auto-routing to session without ephemeral — skip
        except Exception:
            logger.debug("Layer routing failed, using project brain", exc_info=True)

        try:
            return await self._remember_save(
                args,
                content,
                mem_type,
                storage,
                brain,
                layer_decision,
                is_auto_capture,
                redacted_matches,
                remaining_matches,
                sensitive_detected,
                should_encrypt,
                encrypted_content,
                encryption_meta,
            )
        finally:
            if global_storage is not None:
                try:
                    await global_storage.close()
                except Exception:
                    logger.debug("Global storage cleanup failed", exc_info=True)

    async def _remember_save(
        self,
        args: dict[str, Any],
        content: str,
        mem_type: MemoryType,
        storage: NeuralStorage,
        brain: Any,
        layer_decision: Any,
        is_auto_capture: bool,
        redacted_matches: list[Any],
        remaining_matches: list[Any],
        sensitive_detected: bool,
        should_encrypt: bool,
        encrypted_content: str | None,
        encryption_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Inner save logic for _remember — extracted for global storage cleanup."""
        # Phase A: Merge structured context into content
        raw_context = args.get("context")
        if raw_context and isinstance(raw_context, dict):
            from neural_memory.engine.context_merger import merge_context

            content = merge_context(content, raw_context, mem_type.value)

        # Auto-importance scoring when priority not explicitly set
        raw_priority = args.get("priority")
        if raw_priority is not None:
            priority = Priority.from_int(raw_priority)
        else:
            from neural_memory.engine.importance import auto_importance_score

            auto_score = auto_importance_score(content, mem_type.value, args.get("tags", []))

            # A8 T3.4: Surface gap bonus — sparse topic clusters get +1 importance
            try:
                if hasattr(self, "_surface_text") and self._surface_text:
                    from neural_memory.surface.parser import parse as parse_surface

                    surface = parse_surface(self._surface_text)
                    if surface.meta and surface.meta.top_entities:
                        # Check if content topics overlap with well-covered entities
                        content_lower = content.lower()
                        is_covered = any(
                            e.lower() in content_lower for e in surface.meta.top_entities[:5]
                        )
                        if not is_covered and auto_score < 9:
                            auto_score += 1  # Filling a knowledge gap
            except Exception:
                pass

            priority = Priority.from_int(auto_score)

        # SimHash dedup always runs (pure Python, zero cost).
        # Full dedup (embedding + LLM) only when [dedup] enabled = true.
        from neural_memory.engine.dedup import build_dedup_pipeline

        dedup_pipeline = build_dedup_pipeline(self.config.dedup, storage)

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
            # Auto-inject agent identity tag
            agent_id = getattr(self, "_agent_id", "")
            if agent_id:
                tags.add(f"agent:{agent_id}")
            # Domain scope for boundary memories
            raw_domain = args.get("domain")
            if raw_domain and isinstance(raw_domain, str):
                raw_domain = raw_domain.lower().strip()[:50]
                if mem_type == MemoryType.BOUNDARY and raw_domain:
                    tags.add(f"domain:{raw_domain}")
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
                content=encode_content,
                timestamp=event_timestamp,
                tags=tags if tags else None,
                metadata={"type": mem_type.value},
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

            # Mark neurons as ephemeral if requested
            is_ephemeral = bool(args.get("ephemeral", False))
            if is_ephemeral:
                ephemeral_ids = [n.id for n in result.neurons_created]
                if ephemeral_ids:
                    await storage.update_neurons_ephemeral_batch(ephemeral_ids, ephemeral=True)

            expiry_days = args.get("expires_days")
            # Ephemeral memories default to 1-day expiry if not explicitly set
            if is_ephemeral and expiry_days is None:
                expiry_days = 1

            raw_trust = args.get("trust_score")
            trust_score: float | None = None
            if raw_trust is not None:
                try:
                    trust_score = float(raw_trust)
                    if not (0.0 <= trust_score <= 1.0):
                        return {"error": f"trust_score must be 0.0-1.0, got {raw_trust}"}
                except (TypeError, ValueError):
                    return {"error": f"Invalid trust_score: {raw_trust}"}

            raw_tier = str(args.get("tier", "warm")).lower().strip()
            try:
                from neural_memory.core.memory_types import MemoryTier

                MemoryTier(raw_tier)
            except ValueError:
                return {"error": f"Invalid tier: {raw_tier}. Must be hot, warm, or cold."}

            typed_mem = TypedMemory.create(
                fiber_id=result.fiber.id,
                memory_type=mem_type,
                priority=priority,
                source=mcp_source,
                expires_in_days=expiry_days,
                tags=tags if tags else None,
                trust_score=trust_score,
                tier=raw_tier,
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

            # Decision intelligence: detect overlapping prior decisions
            decision_overlaps_out: list[dict[str, Any]] = []
            if mem_type == MemoryType.DECISION:
                try:
                    from neural_memory.engine.decision_intel import (
                        extract_decision_components,
                        find_overlapping_decisions,
                    )

                    decision_context = args.get("context") or {}
                    components = extract_decision_components(
                        content, decision_context if isinstance(decision_context, dict) else None
                    )
                    if components is not None:
                        overlaps = await find_overlapping_decisions(
                            storage,
                            components,
                            tags,
                        )
                        for ov in overlaps:
                            decision_overlaps_out.append(ov.to_dict())
                            # Create EVOLVES_FROM synapse
                            ov_fiber = await storage.get_fiber(ov.fiber_id)
                            if ov_fiber and ov_fiber.anchor_neuron_id:
                                evo_syn = Synapse.create(
                                    source_id=result.fiber.anchor_neuron_id,
                                    target_id=ov_fiber.anchor_neuron_id,
                                    type=SynapseType.EVOLVES_FROM,
                                    weight=ov.overlap_score,
                                    metadata={"relationship": ov.relationship},
                                )
                                await storage.add_synapse(evo_syn)
                except Exception:
                    logger.debug("Decision overlap detection failed (non-critical)", exc_info=True)

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

        # Accumulate importance for reflection trigger
        try:
            from neural_memory.engine.reflection import ReflectionEngine

            if not hasattr(self, "_reflection_engine"):
                self._reflection_engine = ReflectionEngine(threshold=50.0)
            self._reflection_engine.accumulate(float(priority.value))
        except Exception:
            pass  # Non-critical

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

        # Phase B: Quality scoring (soft gate — always stores, returns hints)
        from neural_memory.engine.quality_scorer import score_memory

        raw_tags_list = list(tags) if tags else []
        quality_result = score_memory(
            content,
            memory_type=mem_type.value,
            tags=raw_tags_list,
            context=raw_context if isinstance(raw_context, dict) else None,
        )

        # Compact mode: minimal response for agent efficiency (saves ~200-400 tokens)
        compact = bool(args.get("compact", True))
        if compact:
            compact_response: dict[str, Any] = {
                "success": True,
                "fiber_id": result.fiber.id,
                "memory_type": mem_type.value,
            }
            # Only surface critical warnings in compact mode
            if redacted_matches:
                compact_response["auto_redacted"] = True
            try:
                conflicts_detected = int(result.conflicts_detected)
            except (TypeError, ValueError, AttributeError):
                conflicts_detected = 0
            if conflicts_detected > 0:
                compact_response["conflicts_detected"] = conflicts_detected
            dedup_alias_of = result.fiber.metadata.get("_dedup_alias_of")
            if dedup_alias_of is None and result.neurons_created:
                for neuron in result.neurons_created:
                    dedup_alias_of = neuron.metadata.get("_dedup_alias_of")
                    if dedup_alias_of:
                        break
            if dedup_alias_of:
                compact_response["dedup_hint"] = (
                    "Similar memory exists — consider nmem_edit to update"
                )
            deja_vu = (result.fiber.metadata or {}).get("_deja_vu")
            if deja_vu:
                compact_response["deja_vu_warnings"] = len(deja_vu)
            return compact_response

        response: dict[str, Any] = {
            "success": True,
            "fiber_id": result.fiber.id,
            "memory_type": mem_type.value,
            "tier": typed_mem.tier,
            "priority": priority.value,
            "auto_importance": raw_priority is None,
            "neurons_created": len(result.neurons_created),
            "message": f"Remembered: {content[:50]}{'...' if len(content) > 50 else ''}",
            **quality_result.to_dict(),
        }

        if source_id and isinstance(source_id, str):
            response["source_id"] = source_id

        # A8 T3.3: Surface type classification hints
        type_hint = result.fiber.metadata.get("_type_hint")
        if type_hint:
            response["type_hint"] = type_hint
        type_confidence = result.fiber.metadata.get("_type_confidence")
        if type_confidence is not None:
            response["type_confidence"] = type_confidence

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

        # Quality hint: warn about long content (skip ephemeral — scratch notes are often long)
        if len(content) > 500 and not is_ephemeral:
            response["quality_warning"] = (
                f"Content is {len(content)} chars — long memories reduce recall precision. "
                "Consider splitting into 2-3 focused memories."
            )

        if is_ephemeral:
            response["ephemeral"] = True
            response["message"] += " [ephemeral — auto-expires, never synced]"

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
            # A8 T3.2: Enhanced dedup feedback with similarity score and fiber_id
            dedup_hint: dict[str, Any] = {
                "similar_existing": dedup_alias_of,
                "message": "Similar memory exists — consider nmem_edit to update instead",
            }
            # Include similarity score if propagated from DedupCheckStep
            dedup_sim = result.fiber.metadata.get("_dedup_similarity")
            if dedup_sim is not None:
                dedup_hint["similarity"] = round(float(dedup_sim), 3)
            dedup_tier = result.fiber.metadata.get("_dedup_tier")
            if dedup_tier is not None:
                dedup_hint["dedup_tier"] = int(dedup_tier)
            # Find the fiber_id for the existing anchor neuron
            try:
                existing_fibers = await storage.find_fibers(contains_neuron=dedup_alias_of, limit=1)
                if existing_fibers:
                    dedup_hint["duplicate_of"] = existing_fibers[0].id
            except Exception:
                pass
            response["dedup_hint"] = dedup_hint

        # Decision overlap results
        if decision_overlaps_out:
            response["decision_overlaps"] = decision_overlaps_out

        try:
            conflicts_detected = int(result.conflicts_detected)
        except (TypeError, ValueError, AttributeError):
            conflicts_detected = 0
        if conflicts_detected > 0:
            response["conflicts_detected"] = conflicts_detected
            response["message"] += f" ({conflicts_detected} conflict(s) detected)"

        # Déjà vu warnings: surface scar tissue from encode pipeline
        deja_vu = (result.fiber.metadata or {}).get("_deja_vu")
        if deja_vu:
            response["deja_vu_warnings"] = deja_vu
            response["message"] += (
                f" ⚠ {len(deja_vu)} déjà vu warning(s): similar content was involved"
                " in past error/decision chains"
            )

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
                activations, _trace = await activator.activate(
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
        alert_info = await self._surface_pending_alerts()
        if alert_info:
            response.update(alert_info)

        # Layer routing info
        if layer_decision is not None:
            response["layer"] = layer_decision.to_dict()

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
                "ephemeral",
                "tier",
                "domain",
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
                results.append({"index": idx, "status": "error", "reason": "failed to store"})
                failed += 1

        return {
            "success": saved > 0,
            "saved": saved,
            "failed": failed,
            "total": len(memories),
            "results": results,
        }
