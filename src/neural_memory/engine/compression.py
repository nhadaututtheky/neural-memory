"""Tiered memory compression engine — age-based, zero-LLM, entity-density scoring.

Compression is applied to fibers based on their age. Five tiers exist:

  Tier 0 (FULL)        < 7d    No compression; content is unchanged.
  Tier 1 (EXTRACTIVE)  7-30d   Keep the top-N sentences by entity density.
  Tier 2 (ENTITY_ONLY) 30-90d  Keep only sentences that contain an entity ref.
  Tier 3 (TEMPLATE)    90-180d Render "{entity} {relation} {entity}" triples.
  Tier 4 (GRAPH_ONLY)  180d+   Delete content entirely; keep graph structure.

Tiers 1-2 are reversible (original content saved in compression_backups).
Tiers 3-4 are irreversible.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.core.fiber import Fiber
    from neural_memory.core.synapse import Synapse
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence splitting helpers
# ---------------------------------------------------------------------------

# Abbreviations that should NOT be treated as sentence boundaries.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "fig",
        "approx",
        "est",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    }
)

# Sentence-boundary pattern: punctuation followed by whitespace + capital letter.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------


class CompressionTier(IntEnum):
    """Age-based compression tier for a fiber."""

    FULL = 0
    EXTRACTIVE = 1
    ENTITY_ONLY = 2
    TEMPLATE = 3
    GRAPH_ONLY = 4


@dataclass(frozen=True)
class CompressionConfig:
    """Configuration for the compression engine."""

    # Age thresholds (days) for each tier boundary.
    tier1_days: float = 7.0
    tier2_days: float = 30.0
    tier3_days: float = 90.0
    tier4_days: float = 180.0

    # Tier 1: how many top sentences to keep.
    tier1_max_sentences: int = 5

    # Tier 1/2: always preserve the first sentence regardless of score.
    preserve_first_sentence: bool = True

    # Minimum entity density score to keep a sentence in tier-2.
    tier2_min_density: float = 0.0


@dataclass(frozen=True)
class CompressionResult:
    """Result of compressing a single fiber."""

    fiber_id: str
    original_tier: int
    new_tier: int
    original_token_count: int
    compressed_token_count: int
    entities_preserved: int
    backup_created: bool
    skipped: bool = False
    skip_reason: str = ""

    @property
    def tokens_saved(self) -> int:
        """Tokens removed by this compression."""
        return max(0, self.original_token_count - self.compressed_token_count)


@dataclass
class CompressionReport:
    """Aggregate report for a full compression run."""

    started_at: datetime = field(default_factory=utcnow)
    duration_ms: float = 0.0
    fibers_compressed: int = 0
    fibers_skipped: int = 0
    tokens_saved: int = 0
    backups_created: int = 0
    dry_run: bool = False
    results: list[CompressionResult] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of the compression run."""
        mode = " (dry run)" if self.dry_run else ""
        lines = [
            f"Compression Report{mode} ({self.started_at.strftime('%Y-%m-%d %H:%M')})",
            f"  Fibers compressed: {self.fibers_compressed}",
            f"  Fibers skipped: {self.fibers_skipped}",
            f"  Tokens saved: {self.tokens_saved}",
            f"  Backups created: {self.backups_created}",
            f"  Duration: {self.duration_ms:.1f}ms",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, handling common abbreviations.

    The function avoids splitting on abbreviation-terminated periods by
    checking the token immediately before the boundary.

    Args:
        text: The text to split.

    Returns:
        A list of non-empty sentences (stripped).
    """
    # Normalise whitespace first.
    text = text.strip()
    if not text:
        return []

    # Candidate split positions from regex.
    candidates = list(_SENTENCE_RE.finditer(text))
    if not candidates:
        return [text] if text else []

    sentences: list[str] = []
    start = 0
    for match in candidates:
        fragment = text[start : match.start() + 1].strip()
        # Check if the word before the period is an abbreviation.
        words = fragment.split()
        if words:
            last_word = words[-1].rstrip(".!?").lower()
            if last_word in _ABBREVIATIONS:
                # Not a real sentence boundary — skip.
                continue
        if fragment:
            sentences.append(fragment)
        start = match.end()

    # Remainder after last boundary.
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)

    return sentences if sentences else [text]


def _token_count(text: str) -> int:
    """Approximate token count as whitespace-split word count."""
    return len(text.split())


def compute_entity_density(sentence: str, neuron_contents: list[str]) -> float:
    """Score a sentence by how many neuron contents it references per word.

    The score is *count_of_matching_neurons / word_count*, clamped to [0, 1].
    Matching is case-insensitive substring search.

    Args:
        sentence: The sentence to score.
        neuron_contents: Content strings of neurons in the same fiber.

    Returns:
        Entity density in [0.0, 1.0].
    """
    if not sentence:
        return 0.0

    words = sentence.split()
    word_count = len(words)
    if word_count == 0:
        return 0.0

    sentence_lower = sentence.lower()
    matches = sum(1 for nc in neuron_contents if nc and nc.lower() in sentence_lower)
    return min(1.0, matches / word_count)


def compress_tier1_extractive(
    content: str,
    neuron_contents: list[str],
    config: CompressionConfig,
) -> tuple[str, int]:
    """Keep the top-N sentences ranked by entity density (Tier 1).

    When *preserve_first_sentence* is set in config, the first sentence is
    always included regardless of its density score.

    Args:
        content: The original fiber content.
        neuron_contents: Content strings of neurons in the fiber.
        config: Compression configuration.

    Returns:
        A tuple of (compressed_text, entities_preserved_count).
    """
    sentences = split_sentences(content)
    if not sentences:
        return content, 0

    scored: list[tuple[float, int, str]] = [
        (compute_entity_density(s, neuron_contents), idx, s) for idx, s in enumerate(sentences)
    ]

    # Always preserve the first sentence if configured.
    first_idx: int | None = 0 if config.preserve_first_sentence else None

    # Sort by descending density, then keep top-N.
    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    top = ranked[: config.tier1_max_sentences]

    # Add first sentence if not already included.
    selected_indices: set[int] = {idx for _, idx, _ in top}
    if first_idx is not None and first_idx not in selected_indices:
        # Replace the lowest-ranked entry with the first sentence.
        if len(top) >= config.tier1_max_sentences and ranked:
            top = top[:-1]
        first_entry = scored[first_idx]
        top.append(first_entry)

    # Reconstruct in original order.
    keep_indices: set[int] = {idx for _, idx, _ in top}
    result_sentences = [s for idx, s in enumerate(sentences) if idx in keep_indices]

    entities_found = sum(1 for _, _, s in top if compute_entity_density(s, neuron_contents) > 0)
    return " ".join(result_sentences), entities_found


def compress_tier2_entity_preserving(
    content: str,
    neuron_contents: list[str],
    relations: list[str],
    config: CompressionConfig,
) -> tuple[str, int]:
    """Keep only sentences that reference at least one entity (Tier 2).

    A sentence qualifies if its entity density exceeds
    *config.tier2_min_density* (default 0.0, meaning at least one match).

    Args:
        content: The original fiber content.
        neuron_contents: Content strings of neurons in the fiber.
        relations: Relation labels from synapses (currently unused but
                   available for future scoring).
        config: Compression configuration.

    Returns:
        A tuple of (compressed_text, entities_preserved_count).
    """
    sentences = split_sentences(content)
    if not sentences:
        return content, 0

    first_idx: int | None = 0 if config.preserve_first_sentence else None
    kept_sentences: list[str] = []
    entities_preserved = 0

    for idx, sentence in enumerate(sentences):
        density = compute_entity_density(sentence, neuron_contents)
        keep = density > config.tier2_min_density or (first_idx is not None and idx == first_idx)
        if keep:
            kept_sentences.append(sentence)
            if density > 0:
                entities_preserved += 1

    if not kept_sentences:
        # Fall back to original if nothing survives.
        return content, 0

    return " ".join(kept_sentences), entities_preserved


def compress_tier3_template(
    neuron_contents: list[str],
    relations: list[str],
) -> tuple[str, int]:
    """Render entity-relation-entity triples as a template string (Tier 3).

    Each triple takes the form ``"{entity_a} {relation} {entity_b}"``.
    If fewer than two entities exist, returns a simple entity list.

    Args:
        neuron_contents: Content strings of neurons in the fiber.
        relations: Relation type labels from synapses.

    Returns:
        A tuple of (template_text, entities_preserved_count).
    """
    entities = [nc for nc in neuron_contents if nc]
    if not entities:
        return "", 0

    if len(entities) < 2:
        return entities[0], 1

    parts: list[str] = []
    relation = relations[0] if relations else "related_to"

    # Pair consecutive entities with the primary relation.
    for i in range(len(entities) - 1):
        rel = relations[i] if i < len(relations) else relation
        parts.append(f"{entities[i]} {rel} {entities[i + 1]}")

    return "; ".join(parts), len(entities)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CompressionEngine:
    """Engine for tiered memory compression.

    Reads fibers from storage, determines their target compression tier
    based on age, and applies the appropriate compression strategy.

    Usage::

        engine = CompressionEngine(storage)
        report = await engine.run()
    """

    def __init__(
        self,
        storage: NeuralStorage,
        config: CompressionConfig | None = None,
    ) -> None:
        self._storage = storage
        self._config = config or CompressionConfig()

    def determine_target_tier(
        self,
        fiber: Fiber,
        reference_time: datetime,
    ) -> CompressionTier:
        """Return the target CompressionTier for *fiber* given *reference_time*.

        The tier is determined solely from the fiber's age (in days since
        *created_at*). The current *compression_tier* is not taken into account
        — callers should compare this value against ``fiber.compression_tier``
        to decide whether compression is needed.

        Args:
            fiber: The fiber to evaluate.
            reference_time: The reference UTC datetime for age calculation.

        Returns:
            The target CompressionTier.
        """
        age_seconds = (reference_time - fiber.created_at).total_seconds()
        age_days = age_seconds / 86400.0

        if age_days < self._config.tier1_days:
            return CompressionTier.FULL
        if age_days < self._config.tier2_days:
            return CompressionTier.EXTRACTIVE
        if age_days < self._config.tier3_days:
            return CompressionTier.ENTITY_ONLY
        if age_days < self._config.tier4_days:
            return CompressionTier.TEMPLATE
        return CompressionTier.GRAPH_ONLY

    async def compress_fiber(
        self,
        fiber: Fiber,
        target_tier: CompressionTier,
        *,
        dry_run: bool = False,
    ) -> CompressionResult:
        """Compress a single fiber to *target_tier*.

        For reversible tiers (1-2) a backup of the original content is saved
        before modification.  For tier 4 (GRAPH_ONLY) all neuron contents are
        cleared in storage.

        Args:
            fiber: The fiber to compress.
            target_tier: The tier to compress to.
            dry_run: If True, calculate but do not apply changes.

        Returns:
            A CompressionResult describing what happened.
        """
        current_tier = CompressionTier(fiber.compression_tier)

        if target_tier <= current_tier:
            return CompressionResult(
                fiber_id=fiber.id,
                original_tier=fiber.compression_tier,
                new_tier=fiber.compression_tier,
                original_token_count=0,
                compressed_token_count=0,
                entities_preserved=0,
                backup_created=False,
                skipped=True,
                skip_reason="already_at_or_beyond_target_tier",
            )

        # Fetch neurons in this fiber.
        neurons = await self._storage.get_neurons_batch(list(fiber.neuron_ids))
        neuron_contents = [n.content for n in neurons.values()]

        # Fetch synapses for entity-level relation labels.
        all_synapses: list[Synapse] = []
        if fiber.synapse_ids:
            for syn_id in fiber.synapse_ids:
                syn = await self._storage.get_synapse(syn_id)
                if syn is not None:
                    all_synapses.append(syn)
        relations = [str(s.type) for s in all_synapses]

        # Build a representative "content" string from neuron contents for
        # tiers that operate on text (1-2).
        original_content = "\n".join(nc for nc in neuron_contents if nc)
        original_token_count = _token_count(original_content)

        backup_created = False
        entities_preserved = 0

        if target_tier == CompressionTier.GRAPH_ONLY:
            compressed_content = ""
            compressed_token_count = 0
            entities_preserved = 0
        elif target_tier == CompressionTier.TEMPLATE:
            compressed_content, entities_preserved = compress_tier3_template(
                neuron_contents, relations
            )
            compressed_token_count = _token_count(compressed_content)
        elif target_tier == CompressionTier.ENTITY_ONLY:
            compressed_content, entities_preserved = compress_tier2_entity_preserving(
                original_content, neuron_contents, relations, self._config
            )
            compressed_token_count = _token_count(compressed_content)
        elif target_tier == CompressionTier.EXTRACTIVE:
            compressed_content, entities_preserved = compress_tier1_extractive(
                original_content, neuron_contents, self._config
            )
            compressed_token_count = _token_count(compressed_content)
        else:
            # FULL — no compression needed (should be filtered before here).
            return CompressionResult(
                fiber_id=fiber.id,
                original_tier=fiber.compression_tier,
                new_tier=fiber.compression_tier,
                original_token_count=original_token_count,
                compressed_token_count=original_token_count,
                entities_preserved=len(neuron_contents),
                backup_created=False,
                skipped=True,
                skip_reason="full_tier_no_compression",
            )

        if dry_run:
            return CompressionResult(
                fiber_id=fiber.id,
                original_tier=fiber.compression_tier,
                new_tier=int(target_tier),
                original_token_count=original_token_count,
                compressed_token_count=compressed_token_count,
                entities_preserved=entities_preserved,
                backup_created=False,
            )

        # Persist backup for reversible tiers (1-2).
        reversible = target_tier in (CompressionTier.EXTRACTIVE, CompressionTier.ENTITY_ONLY)
        if reversible:
            try:
                await self._storage.save_compression_backup(
                    fiber_id=fiber.id,
                    original_content=original_content,
                    compression_tier=int(target_tier),
                    original_token_count=original_token_count,
                    compressed_token_count=compressed_token_count,
                )
                backup_created = True
            except Exception:
                logger.error(
                    "Failed to save compression backup for fiber %s", fiber.id, exc_info=True
                )

        # Apply compression: update each neuron's content if tier < GRAPH_ONLY,
        # or clear all content for GRAPH_ONLY.
        if target_tier == CompressionTier.GRAPH_ONLY:
            for neuron in neurons.values():
                from dataclasses import replace as dc_replace

                cleared = dc_replace(neuron, content="[graph-only]")
                try:
                    await self._storage.update_neuron(cleared)
                except Exception:
                    logger.error(
                        "Failed to clear neuron %s content for fiber %s",
                        neuron.id,
                        fiber.id,
                        exc_info=True,
                    )
        elif target_tier in (
            CompressionTier.TEMPLATE,
            CompressionTier.ENTITY_ONLY,
            CompressionTier.EXTRACTIVE,
        ):
            # For tier 1-3, we store the compressed content in the fiber's
            # anchor neuron and truncate non-anchor neurons.
            anchor_id = fiber.anchor_neuron_id
            anchor_neuron = neurons.get(anchor_id)
            if anchor_neuron is not None:
                from dataclasses import replace as dc_replace

                updated_anchor = dc_replace(anchor_neuron, content=compressed_content)
                try:
                    await self._storage.update_neuron(updated_anchor)
                except Exception:
                    logger.error(
                        "Failed to update anchor neuron %s for fiber %s",
                        anchor_id,
                        fiber.id,
                        exc_info=True,
                    )

        # Update fiber's compression_tier in storage.
        from dataclasses import replace as dc_replace

        updated_fiber = dc_replace(fiber, compression_tier=int(target_tier))
        try:
            await self._storage.update_fiber(updated_fiber)
        except Exception:
            logger.error("Failed to update compression_tier for fiber %s", fiber.id, exc_info=True)

        return CompressionResult(
            fiber_id=fiber.id,
            original_tier=fiber.compression_tier,
            new_tier=int(target_tier),
            original_token_count=original_token_count,
            compressed_token_count=compressed_token_count,
            entities_preserved=entities_preserved,
            backup_created=backup_created,
        )

    async def decompress_fiber(self, fiber_id: str) -> bool:
        """Restore a fiber's original content from its compression backup.

        Only works for tiers 1-2 (reversible). Returns False if no backup
        exists or if the fiber is at tier 3-4 (irreversible).

        Args:
            fiber_id: ID of the fiber to decompress.

        Returns:
            True if decompression succeeded, False otherwise.
        """
        backup: dict[str, Any] | None = await self._storage.get_compression_backup(fiber_id)
        if backup is None:
            logger.warning("No compression backup found for fiber %s", fiber_id)
            return False

        fiber = await self._storage.get_fiber(fiber_id)
        if fiber is None:
            logger.warning("Fiber %s not found for decompression", fiber_id)
            return False

        original_content: str = backup.get("original_content", "")
        if not original_content:
            logger.warning("Backup for fiber %s has no original content", fiber_id)
            return False

        # Restore content to the anchor neuron.
        anchor_id = fiber.anchor_neuron_id
        anchor_neuron = await self._storage.get_neuron(anchor_id)
        if anchor_neuron is None:
            logger.error("Anchor neuron %s missing for fiber %s", anchor_id, fiber_id)
            return False

        from dataclasses import replace as dc_replace

        restored_neuron = dc_replace(anchor_neuron, content=original_content)
        try:
            await self._storage.update_neuron(restored_neuron)
        except Exception:
            logger.error("Failed to restore neuron content for fiber %s", fiber_id, exc_info=True)
            return False

        # Reset fiber compression tier to 0 (FULL).
        updated_fiber = dc_replace(fiber, compression_tier=int(CompressionTier.FULL))
        try:
            await self._storage.update_fiber(updated_fiber)
        except Exception:
            logger.error("Failed to reset compression_tier for fiber %s", fiber_id, exc_info=True)
            return False

        # Remove the backup.
        try:
            await self._storage.delete_compression_backup(fiber_id)
        except Exception:
            logger.warning(
                "Failed to delete compression backup for fiber %s", fiber_id, exc_info=True
            )

        return True

    async def run(
        self,
        *,
        reference_time: datetime | None = None,
        dry_run: bool = False,
    ) -> CompressionReport:
        """Compress all eligible fibers in the current brain.

        A fiber is eligible when its target tier is higher than its current
        ``compression_tier`` value.  Fibers already at the correct tier are
        skipped.

        Args:
            reference_time: UTC reference time for age calculation (default: now).
            dry_run: If True, compute but do not apply changes.

        Returns:
            A CompressionReport with aggregate statistics.
        """
        import time

        reference_time = reference_time or utcnow()
        report = CompressionReport(started_at=reference_time, dry_run=dry_run)
        start = time.perf_counter()

        if not self._storage.current_brain_id:
            logger.warning("CompressionEngine.run() called with no brain context")
            report.duration_ms = (time.perf_counter() - start) * 1000
            return report

        fibers = await self._storage.get_fibers(limit=10000)

        for fiber in fibers:
            # Pinned (KB) fibers stay at tier 0 forever
            if fiber.pinned:
                report.fibers_skipped += 1
                continue

            target_tier = self.determine_target_tier(fiber, reference_time)

            if int(target_tier) <= fiber.compression_tier:
                report.fibers_skipped += 1
                continue

            try:
                result = await self.compress_fiber(fiber, target_tier, dry_run=dry_run)
            except Exception:
                logger.error("Compression failed for fiber %s", fiber.id, exc_info=True)
                report.fibers_skipped += 1
                continue

            report.results.append(result)

            if result.skipped:
                report.fibers_skipped += 1
            else:
                report.fibers_compressed += 1
                report.tokens_saved += result.tokens_saved
                if result.backup_created:
                    report.backups_created += 1

        report.duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Compression run complete: %d compressed, %d skipped, %d tokens saved",
            report.fibers_compressed,
            report.fibers_skipped,
            report.tokens_saved,
        )
        return report
