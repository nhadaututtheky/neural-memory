"""Incremental consolidation — dirty sets and resumable strategy runs.

Uses ``change_log`` as an immutable event source. Checkpoints are stored
separately per strategy and never touch ``change_log.synced``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.core.consolidation_checkpoint import ConsolidationCheckpoint
from neural_memory.utils.timeutils import ensure_naive_utc, utcnow

if TYPE_CHECKING:
    from neural_memory.engine.consolidation import (
        ConsolidationEngine,
        ConsolidationReport,
        ConsolidationStrategy,
    )

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHANGES = 5000
DEFAULT_MAX_CANDIDATES = 10000
DEFAULT_TIME_BUDGET_S = 120.0


@dataclass(frozen=True)
class DirtySet:
    """Bounded set of entities changed since a strategy checkpoint."""

    neuron_ids: frozenset[str]
    synapse_ids: frozenset[str]
    fiber_ids: frozenset[str]
    high_watermark: int
    truncated: bool
    change_count: int = 0
    from_sequence: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.neuron_ids or self.synapse_ids or self.fiber_ids)

    @property
    def total_entities(self) -> int:
        return len(self.neuron_ids) + len(self.synapse_ids) + len(self.fiber_ids)


@dataclass
class IncrementalRunReport:
    """Per-run incremental consolidation outcome."""

    started_at: datetime = field(default_factory=utcnow)
    duration_ms: float = 0.0
    mode: str = "incremental"  # incremental | bootstrap_full | zero_work
    strategies_run: list[str] = field(default_factory=list)
    strategies_advanced: list[str] = field(default_factory=list)
    strategies_failed: list[str] = field(default_factory=list)
    strategies_skipped: list[str] = field(default_factory=list)
    dirty: DirtySet | None = None
    truncated: bool = False
    dry_run: bool = False
    consolidation: ConsolidationReport | None = None
    checkpoints: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Incremental Consolidation ({self.mode})",
            f"  Duration: {self.duration_ms:.1f}ms",
            f"  Advanced: {', '.join(self.strategies_advanced) or 'none'}",
            f"  Failed: {', '.join(self.strategies_failed) or 'none'}",
            f"  Truncated: {self.truncated}",
        ]
        if self.dirty is not None:
            lines.append(
                f"  Dirty: n={len(self.dirty.neuron_ids)} "
                f"s={len(self.dirty.synapse_ids)} f={len(self.dirty.fiber_ids)} "
                f"wm={self.dirty.high_watermark} changes={self.dirty.change_count}"
            )
        if self.consolidation is not None:
            lines.append(self.consolidation.summary())
        return "\n".join(lines)


async def get_change_log_high_watermark(storage: Any) -> int:
    """Return max change_log.id for current brain (0 if empty)."""
    if hasattr(storage, "get_change_log_max_sequence"):
        return int(await storage.get_change_log_max_sequence())
    # Fallback: page to end
    if not hasattr(storage, "get_changes_since"):
        return 0
    seq = 0
    while True:
        batch = await storage.get_changes_since(seq, limit=1000)
        if not batch:
            break
        seq = max(c.id for c in batch)
        if len(batch) < 1000:
            break
    return seq


async def build_dirty_set(
    storage: Any,
    strategy: str,
    *,
    max_changes: int = DEFAULT_MAX_CHANGES,
    from_sequence: int | None = None,
) -> DirtySet:
    """Build a bounded dirty set from change_log after the strategy checkpoint.

    Does **not** modify ``change_log.synced``. Concurrent writes after the
    captured high-watermark are left for the next run.
    """
    del strategy  # strategy selects checkpoint externally; dirty set is global log slice
    max_changes = max(1, min(int(max_changes), 50_000))

    if from_sequence is None:
        from_sequence = 0
        if hasattr(storage, "get_consolidation_checkpoint"):
            # Caller should pass checkpoint; keep API flexible
            pass
    from_sequence = max(0, int(from_sequence))

    if not hasattr(storage, "get_changes_since"):
        return DirtySet(
            neuron_ids=frozenset(),
            synapse_ids=frozenset(),
            fiber_ids=frozenset(),
            high_watermark=from_sequence,
            truncated=False,
            change_count=0,
            from_sequence=from_sequence,
        )

    # Snapshot high-watermark first so concurrent inserts are deferred
    high_watermark = await get_change_log_high_watermark(storage)
    if high_watermark <= from_sequence:
        return DirtySet(
            neuron_ids=frozenset(),
            synapse_ids=frozenset(),
            fiber_ids=frozenset(),
            high_watermark=from_sequence,
            truncated=False,
            change_count=0,
            from_sequence=from_sequence,
        )

    neuron_ids: set[str] = set()
    synapse_ids: set[str] = set()
    fiber_ids: set[str] = set()
    change_count = 0
    truncated = False
    cursor = from_sequence
    page = min(1000, max_changes)

    while change_count < max_changes and cursor < high_watermark:
        remaining = max_changes - change_count
        batch = await storage.get_changes_since(cursor, limit=min(page, remaining))
        if not batch:
            break
        for entry in batch:
            if entry.id > high_watermark:
                break
            change_count += 1
            cursor = entry.id
            et = (entry.entity_type or "").lower()
            eid = entry.entity_id
            if not eid:
                # delete payload fallback
                payload = entry.payload or {}
                eid = str(payload.get("id") or payload.get("entity_id") or "")
            if not eid:
                continue
            if et == "neuron":
                neuron_ids.add(eid)
            elif et == "synapse":
                synapse_ids.add(eid)
                # Expand to endpoint neurons when payload available
                payload = entry.payload or {}
                for key in ("source_id", "target_id"):
                    nid = payload.get(key)
                    if isinstance(nid, str) and nid:
                        neuron_ids.add(nid)
            elif et == "fiber":
                fiber_ids.add(eid)
                payload = entry.payload or {}
                for nid in payload.get("neuron_ids") or []:
                    if isinstance(nid, str):
                        neuron_ids.add(nid)
                anchor = payload.get("anchor_neuron_id")
                if isinstance(anchor, str) and anchor:
                    neuron_ids.add(anchor)
        if len(batch) < page:
            break
        if change_count >= max_changes and cursor < high_watermark:
            truncated = True
            break

    # Bound neighbor expansion: fibers containing dirty neurons
    if neuron_ids and hasattr(storage, "get_fibers_for_neurons"):
        try:
            extra_fibers = await storage.get_fibers_for_neurons(list(neuron_ids)[:500])
            for f in extra_fibers or []:
                fid = getattr(f, "id", None)
                if isinstance(fid, str):
                    fiber_ids.add(fid)
        except Exception:
            logger.debug("Dirty-set fiber expansion failed", exc_info=True)
    elif neuron_ids and hasattr(storage, "get_fibers"):
        # Best-effort: sample recent fibers for membership (bounded)
        try:
            recent = await storage.get_fibers(limit=200, order_by="created_at", descending=True)
            for f in recent or []:
                nids = getattr(f, "neuron_ids", None) or set()
                if nids & neuron_ids:
                    fiber_ids.add(f.id)
        except Exception:
            logger.debug("Dirty-set fiber scan failed", exc_info=True)

    effective_wm = cursor if truncated else high_watermark
    return DirtySet(
        neuron_ids=frozenset(neuron_ids),
        synapse_ids=frozenset(synapse_ids),
        fiber_ids=frozenset(fiber_ids),
        high_watermark=effective_wm,
        truncated=truncated,
        change_count=change_count,
        from_sequence=from_sequence,
    )


async def run_incremental(
    engine: ConsolidationEngine,
    strategies: list[ConsolidationStrategy] | None = None,
    *,
    max_changes: int = DEFAULT_MAX_CHANGES,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    dry_run: bool = False,
    bootstrap_full: bool = False,
    reference_time: datetime | None = None,
) -> IncrementalRunReport:
    """Run consolidation incrementally per strategy with independent checkpoints.

    - Advances a strategy checkpoint only after success, and never on
      dry-run / truncation / failure / timeout.
    - First run with no checkpoint: if ``bootstrap_full``, runs full
      consolidation once then stamps all strategies at high-watermark;
      otherwise starts from sequence 0 (full change log history, bounded).
    """
    from neural_memory.engine.consolidation import ConsolidationStrategy

    started = ensure_naive_utc(reference_time) if reference_time else utcnow()
    t0 = time.perf_counter()
    storage = engine._storage
    out = IncrementalRunReport(started_at=started, dry_run=dry_run)

    if strategies is None:
        strategies = [
            ConsolidationStrategy.PRUNE,
            ConsolidationStrategy.MERGE,
            ConsolidationStrategy.MATURE,
        ]
    # Normalize
    normalized: list[ConsolidationStrategy] = []
    for s in strategies:
        if s is ConsolidationStrategy.ALL:
            normalized = [x for x in ConsolidationStrategy if x is not ConsolidationStrategy.ALL]
            break
        normalized.append(s if isinstance(s, ConsolidationStrategy) else ConsolidationStrategy(s))

    # Use Any for optional checkpoint/outbox APIs (not all storage backends).
    store: Any = storage
    brain_id = getattr(store, "brain_id", None) or getattr(store, "_current_brain_id", "") or ""
    has_cp = hasattr(store, "get_consolidation_checkpoint") and hasattr(
        store, "save_consolidation_checkpoint"
    )

    # Bootstrap: no checkpoints exist and policy requests full pass
    if bootstrap_full and has_cp:
        any_cp = False
        for s in normalized:
            cp = await store.get_consolidation_checkpoint(s.value)
            if cp is None:
                continue
            # Row existence means bootstrapped (including last_sequence==0 on empty log)
            if cp is not None:
                any_cp = True
                break
        if not any_cp:
            out.mode = "bootstrap_full"
            report = await engine.run(
                strategies=normalized,
                dry_run=dry_run,
                reference_time=started,
            )
            out.consolidation = report
            out.strategies_run = [s.value for s in normalized]
            wm = await get_change_log_high_watermark(store)
            if not dry_run and not report.fiber_scan_truncated:
                for s in normalized:
                    try:
                        await store.save_consolidation_checkpoint(
                            ConsolidationCheckpoint.create(
                                brain_id=str(brain_id),
                                strategy=s.value,
                                last_sequence=wm,
                            )
                        )
                        out.strategies_advanced.append(s.value)
                        out.checkpoints[s.value] = wm
                    except Exception as exc:
                        logger.error("Failed to stamp bootstrap checkpoint %s: %s", s.value, exc)
                        out.strategies_failed.append(s.value)
            out.duration_ms = (time.perf_counter() - t0) * 1000
            return out

    # Shared dirty set from the lowest checkpoint among strategies
    min_seq = 0
    if has_cp:
        seqs: list[int] = []
        for s in normalized:
            cp = await store.get_consolidation_checkpoint(s.value)
            if cp is None:
                seqs.append(0)
                continue
            try:
                seqs.append(int(getattr(cp, "last_sequence", 0) or 0))
            except (TypeError, ValueError):
                seqs.append(0)
        min_seq = min(seqs) if seqs else 0

    dirty = await build_dirty_set(
        store,
        "shared",
        max_changes=max_changes,
        from_sequence=min_seq,
    )
    out.dirty = dirty
    out.truncated = dirty.truncated

    if dirty.is_empty and dirty.change_count == 0:
        out.mode = "zero_work"
        # Still advance checkpoints to current watermark when log is empty ahead
        wm = dirty.high_watermark
        if has_cp and not dry_run and not dirty.truncated:
            for s in normalized:
                try:
                    cp = await store.get_consolidation_checkpoint(s.value)
                    cur = 0
                    if cp is not None:
                        try:
                            cur = int(getattr(cp, "last_sequence", 0) or 0)
                        except (TypeError, ValueError):
                            cur = 0
                    if wm >= cur:
                        await store.save_consolidation_checkpoint(
                            ConsolidationCheckpoint.create(
                                brain_id=str(brain_id),
                                strategy=s.value,
                                last_sequence=wm,
                            )
                        )
                        out.strategies_advanced.append(s.value)
                        out.checkpoints[s.value] = wm
                except Exception:
                    out.strategies_failed.append(s.value)
        out.duration_ms = (time.perf_counter() - t0) * 1000
        return out

    # Attach dirty scope for strategies that honor it
    engine._incremental_dirty = dirty  # type: ignore[attr-defined]
    engine._incremental_max_candidates = max_candidates  # type: ignore[attr-defined]

    # Per-strategy run with independent checkpoint advancement
    budget_deadline = time.perf_counter() + max(1.0, float(time_budget_s))
    combined_report = None

    for s in normalized:
        if time.perf_counter() > budget_deadline:
            out.strategies_skipped.append(s.value)
            out.extra["timeout"] = True
            break

        out.strategies_run.append(s.value)
        # Per-strategy dirty from its own checkpoint
        strat_from = 0
        if has_cp:
            cp = await store.get_consolidation_checkpoint(s.value)
            if cp is not None:
                try:
                    strat_from = int(getattr(cp, "last_sequence", 0) or 0)
                except (TypeError, ValueError):
                    strat_from = 0
        strat_dirty = await build_dirty_set(
            store,
            s.value,
            max_changes=max_changes,
            from_sequence=strat_from,
        )
        if strat_dirty.truncated:
            out.truncated = True
        engine._incremental_dirty = strat_dirty  # type: ignore[attr-defined]

        try:
            report = await engine.run(
                strategies=[s],
                dry_run=dry_run,
                reference_time=started,
            )
            if combined_report is None:
                combined_report = report
            else:
                # Merge counters lightly
                for attr in (
                    "synapses_pruned",
                    "neurons_pruned",
                    "fibers_merged",
                    "fibers_removed",
                    "fibers_created",
                    "summaries_created",
                    "stages_advanced",
                ):
                    setattr(
                        combined_report,
                        attr,
                        getattr(combined_report, attr) + getattr(report, attr),
                    )
                combined_report.duration_ms += report.duration_ms
                if report.fiber_scan_truncated:
                    combined_report.fiber_scan_truncated = True

            timed_out = bool(report.extra.get("timed_out_strategies"))
            success = not report.fiber_scan_truncated and not dry_run and not timed_out
            if success and has_cp and not strat_dirty.truncated:
                await store.save_consolidation_checkpoint(
                    ConsolidationCheckpoint.create(
                        brain_id=str(brain_id),
                        strategy=s.value,
                        last_sequence=strat_dirty.high_watermark,
                    )
                )
                out.strategies_advanced.append(s.value)
                out.checkpoints[s.value] = strat_dirty.high_watermark
            elif strat_dirty.truncated or report.fiber_scan_truncated:
                out.strategies_skipped.append(s.value)
                out.extra[f"{s.value}_partial"] = True
            elif dry_run:
                out.extra[f"{s.value}_dry_run"] = True
        except Exception as exc:
            logger.error("Incremental strategy %s failed: %s", s.value, exc, exc_info=True)
            out.strategies_failed.append(s.value)
            # Do not advance checkpoint

    out.consolidation = combined_report
    # Clear scope
    engine._incremental_dirty = None  # type: ignore[attr-defined]
    out.duration_ms = (time.perf_counter() - t0) * 1000
    return out
