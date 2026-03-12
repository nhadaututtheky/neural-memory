"""Bayesian adaptive depth selection for retrieval queries.

Uses Beta distribution conjugate priors to learn which depth level
works best for different entity patterns. Zero-LLM, pure Bayesian.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from neural_memory.engine.retrieval_types import DepthLevel
from neural_memory.extraction.parser import Stimulus
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthPrior:
    """A single Bayesian prior for an (entity, depth) pair."""

    entity_text: str
    depth_level: DepthLevel
    alpha: float = 1.0
    beta: float = 1.0
    total_queries: int = 0
    last_updated: datetime = field(default_factory=utcnow)
    created_at: datetime = field(default_factory=utcnow)

    @property
    def expected_success_rate(self) -> float:
        """E[Beta(a,b)] = a / (a + b)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence_width(self) -> float:
        """Variance indicator: lower = more confident."""
        n = self.alpha + self.beta
        return math.sqrt((self.alpha * self.beta) / (n * n * (n + 1)))

    def update_success(self) -> DepthPrior:
        return replace(
            self,
            alpha=self.alpha + 1.0,
            total_queries=self.total_queries + 1,
            last_updated=utcnow(),
        )

    def update_failure(self) -> DepthPrior:
        return replace(
            self, beta=self.beta + 1.0, total_queries=self.total_queries + 1, last_updated=utcnow()
        )

    def decay(self, factor: float = 0.9) -> DepthPrior:
        return replace(
            self,
            alpha=max(1.0, self.alpha * factor),
            beta=max(1.0, self.beta * factor),
            last_updated=utcnow(),
        )


@dataclass(frozen=True)
class DepthDecision:
    """Result of adaptive depth selection with explanation."""

    depth: DepthLevel
    reason: str
    method: str  # "bayesian" | "rule_based" | "exploration"
    entity_priors: dict[str, float] = field(default_factory=dict)
    exploration: bool = False


class AdaptiveDepthSelector:
    """Selects retrieval depth using Bayesian priors on entity patterns."""

    EPSILON = 0.05
    SUCCESS_THRESHOLD = 0.3
    MIN_FIBERS_SUCCESS = 1
    DECAY_INTERVAL_DAYS = 30
    DECAY_FACTOR = 0.9
    MIN_QUERIES_FOR_BAYESIAN = 5

    # Session-aware depth bias thresholds
    SESSION_TOPIC_BOOST_EMA = 0.5  # Topic EMA above this → already primed
    SESSION_PRIMED_BIAS = -1  # Bias toward shallower depth when primed
    SESSION_NEW_TOPIC_BIAS = 1  # Bias toward deeper depth for new topics

    def __init__(self, storage: NeuralStorage, *, epsilon: float | None = None) -> None:
        self._storage = storage
        if epsilon is not None:
            self.EPSILON = epsilon
        self._rng = random.Random()

    async def select_depth(
        self,
        stimulus: Stimulus,
        fallback_depth: DepthLevel,
        session_state: Any | None = None,
    ) -> DepthDecision:
        """Select optimal depth based on Bayesian priors and session context.

        Args:
            stimulus: Parsed query stimulus with entities and keywords.
            fallback_depth: Rule-based depth to use when priors insufficient.
            session_state: Optional SessionState for session-aware biasing.
        """
        entity_texts = [e.text for e in stimulus.entities] if stimulus.entities else []

        if not entity_texts:
            # Even without entities, session context can bias depth
            session_bias = self._compute_session_bias(stimulus, session_state)
            if session_bias != 0:
                biased = self._apply_depth_bias(fallback_depth, session_bias)
                return DepthDecision(
                    depth=biased,
                    reason=f"No entities, session bias {session_bias:+d} → {biased.name}",
                    method="session_biased",
                )
            return DepthDecision(
                depth=fallback_depth,
                reason="No entities detected, using rule-based depth",
                method="rule_based",
            )

        # Batch-fetch priors for all entities
        priors_by_entity = await self._storage.get_depth_priors_batch(entity_texts)

        if not any(priors_by_entity.values()):
            session_bias = self._compute_session_bias(stimulus, session_state)
            if session_bias != 0:
                biased = self._apply_depth_bias(fallback_depth, session_bias)
                return DepthDecision(
                    depth=biased,
                    reason=f"No prior data, session bias {session_bias:+d} → {biased.name}",
                    method="session_biased",
                )
            return DepthDecision(
                depth=fallback_depth,
                reason="No prior data for entities, using rule-based depth",
                method="rule_based",
            )

        # Aggregate expected success rate per depth level
        depth_scores: dict[DepthLevel, list[float]] = {d: [] for d in DepthLevel}
        total_entity_queries = 0

        for priors in priors_by_entity.values():
            for prior in priors:
                depth_scores[prior.depth_level].append(prior.expected_success_rate)
                total_entity_queries += prior.total_queries

        # Need enough data before Bayesian overrides rule-based
        if total_entity_queries < self.MIN_QUERIES_FOR_BAYESIAN:
            session_bias = self._compute_session_bias(stimulus, session_state)
            if session_bias != 0:
                biased = self._apply_depth_bias(fallback_depth, session_bias)
                return DepthDecision(
                    depth=biased,
                    reason=f"Insufficient data ({total_entity_queries}q), session bias {session_bias:+d} → {biased.name}",
                    method="session_biased",
                )
            return DepthDecision(
                depth=fallback_depth,
                reason=f"Insufficient data ({total_entity_queries} queries), using rule-based depth",
                method="rule_based",
            )

        # Compute mean score per depth
        depth_means: dict[DepthLevel, float] = {}
        for depth, scores in depth_scores.items():
            if scores:
                depth_means[depth] = sum(scores) / len(scores)

        if not depth_means:
            return DepthDecision(
                depth=fallback_depth,
                reason="No depth scores computed, using rule-based depth",
                method="rule_based",
            )

        # Sort by mean score descending
        sorted_depths = sorted(depth_means.items(), key=lambda x: x[1], reverse=True)
        best_depth, best_score = sorted_depths[0]

        # Epsilon exploration
        if self._rng.random() < self.EPSILON and len(sorted_depths) > 1:
            other_depths = [d for d, _ in sorted_depths[1:]]
            explore_depth = self._rng.choice(other_depths)
            return DepthDecision(
                depth=explore_depth,
                reason=f"Exploration: trying {explore_depth.name} instead of best {best_depth.name} (score={best_score:.2f})",
                method="exploration",
                entity_priors={e: depth_means.get(best_depth, 0.0) for e in entity_texts},
                exploration=True,
            )

        # Apply session bias to Bayesian result
        session_bias = self._compute_session_bias(stimulus, session_state)

        # Use Bayesian choice only if score is reasonable
        if best_score > 0.5:
            final_depth = (
                self._apply_depth_bias(best_depth, session_bias) if session_bias else best_depth
            )
            method = "bayesian+session" if session_bias else "bayesian"
            return DepthDecision(
                depth=final_depth,
                reason=f"Bayesian: {best_depth.name} (score={best_score:.2f}), session bias {session_bias:+d} → {final_depth.name}",
                method=method,
                entity_priors=dict.fromkeys(entity_texts, best_score),
            )

        # Score too low, still apply session bias to fallback
        if session_bias:
            biased = self._apply_depth_bias(fallback_depth, session_bias)
            return DepthDecision(
                depth=biased,
                reason=f"Bayesian score low ({best_score:.2f}), session bias {session_bias:+d} → {biased.name}",
                method="session_biased",
            )

        return DepthDecision(
            depth=fallback_depth,
            reason=f"Best Bayesian score too low ({best_score:.2f}), using rule-based depth",
            method="rule_based",
        )

    async def record_outcome(
        self,
        stimulus: Stimulus,
        depth_used: DepthLevel,
        confidence: float,
        fibers_matched: int,
        agent_used_result: bool | None = None,
    ) -> None:
        """Update priors based on retrieval outcome.

        Args:
            stimulus: The query stimulus.
            depth_used: Depth level used for retrieval.
            confidence: Result confidence.
            fibers_matched: Number of fibers matched.
            agent_used_result: If True, agent remembered something after recalling
                (strong positive signal). If False, recall was unused (weak negative).
                If None, no signal available (use default heuristic).
        """
        entity_texts = [e.text for e in stimulus.entities] if stimulus.entities else []
        if not entity_texts:
            return

        # Enhanced success signal: combine confidence threshold with agent behavior
        base_success = (
            confidence >= self.SUCCESS_THRESHOLD and fibers_matched >= self.MIN_FIBERS_SUCCESS
        )
        if agent_used_result is True:
            # Agent remembered after recall → strong positive signal
            success = True
        elif agent_used_result is False:
            # Agent didn't use the result → weaker signal (but not necessarily failure)
            success = base_success and confidence >= 0.5  # Raise bar
        else:
            success = base_success

        priors_by_entity = await self._storage.get_depth_priors_batch(entity_texts)

        for entity_text in entity_texts:
            existing = priors_by_entity.get(entity_text, [])
            found: DepthPrior | None = None
            for p in existing:
                if p.depth_level == depth_used:
                    found = p
                    break

            if found is None:
                found = DepthPrior(
                    entity_text=entity_text,
                    depth_level=depth_used,
                )

            updated = found.update_success() if success else found.update_failure()
            await self._storage.upsert_depth_prior(updated)

    def _compute_session_bias(self, stimulus: Stimulus, session_state: Any | None) -> int:
        """Compute depth bias from session context.

        Returns:
            -1 (shallower) if query topic is already primed in session,
            +1 (deeper) if topic is new to session but session is established,
            0 (no bias) if no session or session too young.
        """
        if session_state is None:
            return 0

        topic_ema: dict[str, float] = getattr(session_state, "topic_ema", {})
        query_count: int = getattr(session_state, "query_count", 0)

        # Need at least 3 queries for session context to be meaningful
        if query_count < 3 or not topic_ema:
            return 0

        # Extract query topics (entities + keywords)
        query_topics: list[str] = []
        if stimulus.entities:
            query_topics.extend(e.text.lower().strip() for e in stimulus.entities)
        if stimulus.keywords:
            query_topics.extend(k.lower().strip() for k in stimulus.keywords)

        if not query_topics:
            return 0

        # Check overlap with session topics
        primed_count = sum(
            1 for t in query_topics if topic_ema.get(t, 0.0) >= self.SESSION_TOPIC_BOOST_EMA
        )

        if primed_count > 0:
            # Most query topics are already warm in session → go shallower
            return self.SESSION_PRIMED_BIAS

        # Topics are new but session is established → explore deeper
        return self.SESSION_NEW_TOPIC_BIAS

    @staticmethod
    def _apply_depth_bias(depth: DepthLevel, bias: int) -> DepthLevel:
        """Apply integer bias to a depth level, clamping to valid range."""
        new_val = max(0, min(3, depth.value + bias))
        return DepthLevel(new_val)

    async def decay_stale_priors(self) -> int:
        """Apply decay to priors older than DECAY_INTERVAL_DAYS. Returns count decayed."""
        from datetime import timedelta

        cutoff = utcnow() - timedelta(days=self.DECAY_INTERVAL_DAYS)
        stale = await self._storage.get_stale_priors(cutoff)

        decayed_count = 0
        for prior in stale:
            decayed = prior.decay(self.DECAY_FACTOR)
            # Delete priors that have decayed back to near-uniform with few queries
            if decayed.alpha < 1.1 and decayed.beta < 1.1 and decayed.total_queries < 3:
                await self._storage.delete_depth_priors(prior.entity_text)
            else:
                await self._storage.upsert_depth_prior(decayed)
            decayed_count += 1

        return decayed_count
