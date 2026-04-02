"""InfinityDB Cognitive + Alerts + Reviews + Sources + Extras mixin.

In-memory implementations for cognitive state, alerts, review schedules,
co-activation, actions, depth priors, compression backups, neuron snapshots,
knowledge gaps, sources, and hot index. All volatile (session-scoped).
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.alert import Alert
    from neural_memory.core.review_schedule import ReviewSchedule


class InfinityDBExtrasMixin:
    """Mixin providing cognitive, alerts, reviews, and misc ops.

    Composing class must call ``_init_extras_stores()`` in __init__.
    """

    if TYPE_CHECKING:
        _current_brain_id: str | None

    def _init_extras_stores(self) -> None:
        self._cognitive_states: dict[str, dict[str, Any]] = {}
        self._alerts: list[Alert] = []
        self._review_schedules: dict[str, ReviewSchedule] = {}
        self._co_activations: list[dict[str, Any]] = []
        self._action_events: list[dict[str, Any]] = []
        self._depth_priors: dict[str, list[Any]] = {}
        self._compression_backups: dict[str, dict[str, Any]] = {}
        self._neuron_snapshots: dict[str, dict[str, Any]] = {}
        self._knowledge_gaps: dict[str, dict[str, Any]] = {}
        self._sources: dict[str, Any] = {}
        self._hot_index: list[dict[str, Any]] = []

    def _get_brain_id(self) -> str:
        bid: str | None = getattr(self, "_current_brain_id", None)
        if not bid:
            msg = "Brain ID not set"
            raise ValueError(msg)
        return str(bid)

    # ========== Cognitive State ==========

    async def upsert_cognitive_state(
        self,
        neuron_id: str,
        *,
        confidence: float = 0.5,
        evidence_for_count: int = 0,
        evidence_against_count: int = 0,
        status: str = "active",
        predicted_at: str | None = None,
        resolved_at: str | None = None,
        schema_version: int = 1,
        parent_schema_id: str | None = None,
        last_evidence_at: str | None = None,
    ) -> None:
        self._cognitive_states[neuron_id] = {
            "neuron_id": neuron_id,
            "confidence": confidence,
            "evidence_for_count": evidence_for_count,
            "evidence_against_count": evidence_against_count,
            "status": status,
            "predicted_at": predicted_at,
            "resolved_at": resolved_at,
            "schema_version": schema_version,
            "parent_schema_id": parent_schema_id,
            "last_evidence_at": last_evidence_at,
            "created_at": utcnow().isoformat(),
        }

    async def get_cognitive_state(self, neuron_id: str) -> dict[str, Any] | None:
        return self._cognitive_states.get(neuron_id)

    async def list_cognitive_states(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for cs in self._cognitive_states.values():
            if status is not None and cs.get("status") != status:
                continue
            results.append(cs)
            if len(results) >= limit:
                break
        return results

    async def update_cognitive_evidence(
        self,
        neuron_id: str,
        *,
        confidence: float,
        evidence_for_count: int,
        evidence_against_count: int,
        status: str,
        resolved_at: str | None = None,
        last_evidence_at: str | None = None,
    ) -> None:
        cs = self._cognitive_states.get(neuron_id)
        if cs is None:
            return
        self._cognitive_states[neuron_id] = {
            **cs,
            "confidence": confidence,
            "evidence_for_count": evidence_for_count,
            "evidence_against_count": evidence_against_count,
            "status": status,
            "resolved_at": resolved_at,
            "last_evidence_at": last_evidence_at,
        }

    async def list_predictions(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for cs in self._cognitive_states.values():
            if cs.get("predicted_at") is None:
                continue
            if status is not None and cs.get("status") != status:
                continue
            results.append(cs)
            if len(results) >= limit:
                break
        return results

    async def get_calibration_stats(self) -> dict[str, int]:
        stats: dict[str, int] = {
            "total": 0,
            "resolved_correct": 0,
            "resolved_incorrect": 0,
            "active": 0,
        }
        for cs in self._cognitive_states.values():
            if cs.get("predicted_at") is None:
                continue
            stats["total"] += 1
            s = cs.get("status", "active")
            if s == "active":
                stats["active"] += 1
            elif s == "resolved_correct":
                stats["resolved_correct"] += 1
            elif s == "resolved_incorrect":
                stats["resolved_incorrect"] += 1
        return stats

    async def get_schema_history(
        self,
        neuron_id: str,
        *,
        max_depth: int = 20,
    ) -> list[dict[str, Any]]:
        chain: list[dict[str, Any]] = []
        current_id: str | None = neuron_id
        seen: set[str] = set()
        while current_id and current_id not in seen and len(chain) < max_depth:
            seen.add(current_id)
            cs = self._cognitive_states.get(current_id)
            if cs is None:
                break
            chain.append(cs)
            current_id = cs.get("parent_schema_id")
        return chain

    # ========== Alerts ==========

    async def record_alert(self, alert: Alert) -> str:
        # Simple duplicate suppression: same type+brain within 1h
        now = utcnow()
        for existing in self._alerts:
            if (
                existing.alert_type == alert.alert_type
                and existing.brain_id == alert.brain_id
                and existing.status.value != "resolved"
                and (now - existing.created_at).total_seconds() < 3600
            ):
                return ""
        aid = alert.id or str(uuid.uuid4())
        new_alert = replace(alert, id=aid) if alert.id != aid else alert
        self._alerts.append(new_alert)
        return aid

    async def get_active_alerts(self, limit: int = 50) -> list[Alert]:
        results: list[Alert] = []
        for a in self._alerts:
            if a.status.value != "resolved":
                results.append(a)
                if len(results) >= limit:
                    break
        return results

    async def count_pending_alerts(self) -> int:
        return sum(1 for a in self._alerts if a.status.value in ("active", "seen"))

    async def mark_alerts_seen(self, alert_ids: list[str]) -> int:
        from neural_memory.core.alert import AlertStatus

        id_set = set(alert_ids)
        count = 0
        new_alerts: list[Alert] = []
        for a in self._alerts:
            if a.id in id_set and a.status == AlertStatus.ACTIVE:
                new_alerts.append(replace(a, status=AlertStatus.SEEN, seen_at=utcnow()))
                count += 1
            else:
                new_alerts.append(a)
        self._alerts = new_alerts
        return count

    async def mark_alert_acknowledged(self, alert_id: str) -> bool:
        from neural_memory.core.alert import AlertStatus

        for i, a in enumerate(self._alerts):
            if a.id == alert_id and a.status.value != "resolved":
                self._alerts[i] = replace(
                    a, status=AlertStatus.ACKNOWLEDGED, acknowledged_at=utcnow()
                )
                return True
        return False

    async def resolve_alerts_by_type(self, alert_types: list[str]) -> int:
        from neural_memory.core.alert import AlertStatus

        type_set = set(alert_types)
        count = 0
        new_alerts: list[Alert] = []
        for a in self._alerts:
            at_val = a.alert_type.value if hasattr(a.alert_type, "value") else str(a.alert_type)
            if at_val in type_set and a.status.value != "resolved":
                new_alerts.append(replace(a, status=AlertStatus.RESOLVED, resolved_at=utcnow()))
                count += 1
            else:
                new_alerts.append(a)
        self._alerts = new_alerts
        return count

    # ========== Review Schedules ==========

    async def add_review_schedule(self, schedule: ReviewSchedule) -> str:
        self._review_schedules[schedule.fiber_id] = schedule
        return schedule.fiber_id

    async def get_review_schedule(self, fiber_id: str) -> ReviewSchedule | None:
        return self._review_schedules.get(fiber_id)

    async def get_due_reviews(self, limit: int = 20) -> list[ReviewSchedule]:
        now = utcnow()
        due = [
            s
            for s in self._review_schedules.values()
            if s.next_review is not None and s.next_review <= now
        ]
        due.sort(key=lambda s: s.next_review or now)
        return due[: min(limit, 100)]

    async def delete_review_schedule(self, fiber_id: str) -> bool:
        return self._review_schedules.pop(fiber_id, None) is not None

    async def get_review_stats(self) -> dict[str, int]:
        now = utcnow()
        stats: dict[str, int] = {"total": 0, "due": 0}
        for i in range(1, 6):
            stats[f"box_{i}"] = 0
        for s in self._review_schedules.values():
            stats["total"] += 1
            if s.next_review is not None and s.next_review <= now:
                stats["due"] += 1
            box_key = f"box_{min(s.box, 5)}"
            stats[box_key] = stats.get(box_key, 0) + 1
        return stats

    # ========== Co-Activation ==========

    async def record_co_activation(
        self,
        neuron_a: str,
        neuron_b: str,
        binding_strength: float,
        source_anchor: str | None = None,
    ) -> str:
        if neuron_a == neuron_b:
            return ""
        # Canonical order
        a, b = (neuron_a, neuron_b) if neuron_a < neuron_b else (neuron_b, neuron_a)
        event_id = str(uuid.uuid4())
        self._co_activations.append(
            {
                "id": event_id,
                "neuron_a": a,
                "neuron_b": b,
                "binding_strength": binding_strength,
                "source_anchor": source_anchor,
                "created_at": utcnow(),
            }
        )
        return event_id

    async def get_co_activation_counts(
        self,
        since: datetime | None = None,
        min_count: int = 1,
    ) -> list[tuple[str, str, int, float]]:
        pairs: dict[tuple[str, str], list[float]] = {}
        for ev in self._co_activations:
            if since is not None and ev["created_at"] < since:
                continue
            key = (ev["neuron_a"], ev["neuron_b"])
            pairs.setdefault(key, []).append(ev["binding_strength"])
        results: list[tuple[str, str, int, float]] = []
        for (a, b), strengths in pairs.items():
            cnt = len(strengths)
            if cnt >= min_count:
                avg = sum(strengths) / cnt
                results.append((a, b, cnt, avg))
        return results

    async def prune_co_activations(self, older_than: datetime) -> int:
        before = len(self._co_activations)
        self._co_activations = [ev for ev in self._co_activations if ev["created_at"] >= older_than]
        return before - len(self._co_activations)

    # ========== Action Events ==========

    async def record_action(
        self,
        action_type: str,
        action_context: str = "",
        tags: tuple[str, ...] | list[str] = (),
        session_id: str | None = None,
        fiber_id: str | None = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        self._action_events.append(
            {
                "id": event_id,
                "action_type": action_type,
                "action_context": action_context,
                "tags": list(tags),
                "session_id": session_id,
                "fiber_id": fiber_id,
                "created_at": utcnow(),
            }
        )
        return event_id

    async def get_action_sequences(
        self,
        session_id: str | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[Any]:
        results: list[dict[str, Any]] = []
        for ev in self._action_events:
            if session_id is not None and ev.get("session_id") != session_id:
                continue
            if since is not None and ev["created_at"] < since:
                continue
            results.append(ev)
        results.sort(key=lambda e: e["created_at"])
        return results[:limit]

    async def prune_action_events(self, older_than: datetime) -> int:
        before = len(self._action_events)
        self._action_events = [ev for ev in self._action_events if ev["created_at"] >= older_than]
        return before - len(self._action_events)

    # ========== Depth Priors ==========

    async def get_depth_priors_batch(self, entity_texts: list[str]) -> dict[str, list[Any]]:
        return {et: list(self._depth_priors[et]) for et in entity_texts if et in self._depth_priors}

    async def upsert_depth_prior(self, prior: Any) -> None:
        et = getattr(prior, "entity_text", str(prior))
        self._depth_priors.setdefault(et, [])
        # Replace existing with same depth, or append
        existing_depths = {getattr(p, "depth", None) for p in self._depth_priors[et]}
        prior_depth = getattr(prior, "depth", None)
        if prior_depth in existing_depths:
            self._depth_priors[et] = [
                p for p in self._depth_priors[et] if getattr(p, "depth", None) != prior_depth
            ]
        self._depth_priors[et].append(prior)

    async def get_stale_priors(self, older_than: datetime) -> list[Any]:
        stale: list[Any] = []
        for priors in self._depth_priors.values():
            for p in priors:
                last_updated = getattr(p, "last_updated", None)
                if last_updated is not None and last_updated < older_than:
                    stale.append(p)
        return stale

    async def delete_depth_priors(self, entity_text: str) -> int:
        removed = self._depth_priors.pop(entity_text, [])
        return len(removed)

    # ========== Compression Backups ==========

    async def save_compression_backup(
        self,
        fiber_id: str,
        original_content: str,
        compression_tier: int,
        original_token_count: int,
        compressed_token_count: int,
    ) -> None:
        self._compression_backups[fiber_id] = {
            "fiber_id": fiber_id,
            "original_content": original_content,
            "compression_tier": compression_tier,
            "original_token_count": original_token_count,
            "compressed_token_count": compressed_token_count,
            "created_at": utcnow().isoformat(),
        }

    async def get_compression_backup(self, fiber_id: str) -> dict[str, Any] | None:
        return self._compression_backups.get(fiber_id)

    async def delete_compression_backup(self, fiber_id: str) -> bool:
        return self._compression_backups.pop(fiber_id, None) is not None

    async def get_compression_stats(self) -> dict[str, Any]:
        by_tier: dict[int, int] = {}
        total_saved = 0
        for backup in self._compression_backups.values():
            tier = backup["compression_tier"]
            by_tier[tier] = by_tier.get(tier, 0) + 1
            total_saved += backup["original_token_count"] - backup["compressed_token_count"]
        return {
            "total_backups": len(self._compression_backups),
            "by_tier": by_tier,
            "total_tokens_saved": total_saved,
        }

    # ========== Neuron Snapshots ==========

    async def save_neuron_snapshot(
        self,
        neuron_id: str,
        brain_id: str,
        original_content: str,
        compressed_at: str,
        tier: int,
    ) -> None:
        self._neuron_snapshots[neuron_id] = {
            "neuron_id": neuron_id,
            "brain_id": brain_id,
            "original_content": original_content,
            "compressed_at": compressed_at,
            "tier": tier,
        }

    async def get_neuron_snapshot(self, neuron_id: str) -> dict[str, Any] | None:
        return self._neuron_snapshots.get(neuron_id)

    async def delete_neuron_snapshot(self, neuron_id: str) -> bool:
        return self._neuron_snapshots.pop(neuron_id, None) is not None

    # ========== Knowledge Gaps ==========

    async def add_knowledge_gap(
        self,
        *,
        topic: str,
        detection_source: str,
        priority: float = 0.5,
        related_neuron_ids: list[str] | None = None,
    ) -> str:
        gap_id = str(uuid.uuid4())
        self._knowledge_gaps[gap_id] = {
            "id": gap_id,
            "topic": topic,
            "detection_source": detection_source,
            "priority": priority,
            "related_neuron_ids": related_neuron_ids or [],
            "resolved": False,
            "resolved_by_neuron_id": None,
            "created_at": utcnow().isoformat(),
        }
        return gap_id

    async def list_knowledge_gaps(
        self,
        *,
        include_resolved: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for gap in self._knowledge_gaps.values():
            if not include_resolved and gap["resolved"]:
                continue
            results.append(gap)
            if len(results) >= limit:
                break
        return results

    async def get_knowledge_gap(self, gap_id: str) -> dict[str, Any] | None:
        return self._knowledge_gaps.get(gap_id)

    async def resolve_knowledge_gap(
        self,
        gap_id: str,
        *,
        resolved_by_neuron_id: str | None = None,
    ) -> bool:
        gap = self._knowledge_gaps.get(gap_id)
        if gap is None or gap["resolved"]:
            return False
        self._knowledge_gaps[gap_id] = {
            **gap,
            "resolved": True,
            "resolved_by_neuron_id": resolved_by_neuron_id,
        }
        return True

    # ========== Sources ==========

    async def add_source(self, source: Any) -> str:
        sid = getattr(source, "id", None) or str(uuid.uuid4())
        self._sources[sid] = source
        return sid

    async def get_source(self, source_id: str) -> Any:
        return self._sources.get(source_id)

    async def list_sources(
        self,
        source_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Any]:
        results: list[Any] = []
        for s in self._sources.values():
            if source_type is not None and getattr(s, "source_type", None) != source_type:
                continue
            if status is not None and getattr(s, "status", None) != status:
                continue
            results.append(s)
            if len(results) >= limit:
                break
        return results

    async def update_source(
        self,
        source_id: str,
        status: str | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        s = self._sources.get(source_id)
        if s is None:
            return False
        if hasattr(s, "__dataclass_fields__"):
            updates: dict[str, Any] = {}
            if status is not None:
                updates["status"] = status
            if version is not None:
                updates["version"] = version
            if metadata is not None:
                updates["metadata"] = metadata
            if updates:
                self._sources[source_id] = replace(s, **updates)
        return True

    async def delete_source(self, source_id: str) -> bool:
        return self._sources.pop(source_id, None) is not None

    async def count_neurons_for_source(self, source_id: str) -> int:
        return 0  # No neuron-source linking in in-memory store

    async def find_source_by_name(self, name: str) -> Any:
        for s in self._sources.values():
            if getattr(s, "name", None) == name:
                return s
        return None

    # ========== Hot Index ==========

    async def refresh_hot_index(self, items: list[dict[str, Any]]) -> int:
        self._hot_index = list(items)
        return len(items)

    async def get_hot_index(self, limit: int = 10) -> list[dict[str, Any]]:
        return self._hot_index[:limit]
