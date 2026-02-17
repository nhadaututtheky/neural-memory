"""Tag normalizer — synonym mapping, SimHash near-match, and drift detection.

Normalizes tags at ingestion time to prevent semantic drift. Uses a curated
synonym map for common equivalences and falls back to SimHash fuzzy matching
for near-duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass

from neural_memory.utils.simhash import hamming_distance, simhash

# Canonical tag → known synonyms/aliases
SYNONYM_MAP: dict[str, list[str]] = {
    "frontend": ["ui", "client-side", "client side", "front-end", "front end"],
    "backend": ["server-side", "server side", "back-end", "back end"],
    "database": ["db", "datastore", "data store", "data-store"],
    "auth": ["authentication", "authorization", "authn", "authz"],
    "config": ["configuration", "settings", "conf"],
    "deploy": ["deployment", "deploying", "release"],
    "testing": ["tests", "test", "unit-test", "unit test", "unittest"],
    "docs": ["documentation", "doc", "readme"],
    "perf": ["performance", "optimization", "optimisation", "speed"],
    "infra": ["infrastructure", "devops", "ops"],
    "monitoring": ["observability", "metrics", "alerting"],
    "caching": ["cache", "memoization", "memoize"],
    "security": ["sec", "vulnerability", "vuln"],
    "refactoring": ["refactor", "cleanup", "clean-up", "clean up"],
    "debugging": ["debug", "troubleshooting", "troubleshoot"],
    "container": ["docker", "containerization", "containerisation"],
    "k8s": ["kubernetes", "kube"],
    "ci": ["ci/cd", "cicd", "continuous-integration", "continuous integration"],
    "js": ["javascript", "ecmascript"],
    "ts": ["typescript"],
    "py": ["python"],
    "api": ["rest", "restful", "rest-api", "rest api", "endpoint", "endpoints"],
    "ml": ["machine-learning", "machine learning"],
    "ai": ["artificial-intelligence", "artificial intelligence"],
}

# Build reverse lookup: synonym → canonical
_REVERSE_MAP: dict[str, str] = {}
for canonical, synonyms in SYNONYM_MAP.items():
    for syn in synonyms:
        _REVERSE_MAP[syn.lower()] = canonical
    _REVERSE_MAP[canonical.lower()] = canonical


@dataclass(frozen=True)
class DriftReport:
    """Report of tag drift for a canonical tag."""

    canonical: str
    variants: tuple[str, ...]
    recommendation: str


class TagNormalizer:
    """Normalizes tags via synonym mapping and SimHash near-match detection.

    Args:
        extra_synonyms: Additional synonym mappings to merge
        simhash_threshold: Max Hamming distance for SimHash near-match (default 6)
    """

    def __init__(
        self,
        extra_synonyms: dict[str, list[str]] | None = None,
        simhash_threshold: int = 6,
    ) -> None:
        self._reverse_map = dict(_REVERSE_MAP)
        self._simhash_threshold = simhash_threshold
        self._canonical_hashes: dict[str, int] = {}

        if extra_synonyms:
            for canonical, synonyms in extra_synonyms.items():
                canonical_lower = canonical.lower()
                self._reverse_map[canonical_lower] = canonical
                for syn in synonyms:
                    self._reverse_map[syn.lower()] = canonical

        # Pre-compute SimHash for all canonical tags
        for canonical in set(self._reverse_map.values()):
            self._canonical_hashes[canonical] = simhash(canonical)

    def normalize(self, tag: str) -> str:
        """Normalize a single tag.

        Priority:
        1. Exact synonym match → canonical form
        2. SimHash near-match → closest canonical form
        3. Lowercase fallback

        Args:
            tag: The tag to normalize

        Returns:
            Normalized tag string
        """
        tag_lower = tag.lower().strip()

        # 1. Synonym lookup
        canonical = self._reverse_map.get(tag_lower)
        if canonical is not None:
            return canonical

        # 2. SimHash near-match against canonical tags
        tag_hash = simhash(tag_lower)
        if tag_hash != 0:
            best_dist = self._simhash_threshold + 1
            best_match: str | None = None
            for canonical, canonical_hash in self._canonical_hashes.items():
                if canonical_hash == 0:
                    continue
                dist = hamming_distance(tag_hash, canonical_hash)
                if dist < best_dist:
                    best_dist = dist
                    best_match = canonical
            if best_match is not None and best_dist <= self._simhash_threshold:
                return best_match

        # 3. Lowercase fallback
        return tag_lower

    def normalize_set(self, tags: set[str]) -> set[str]:
        """Normalize a set of tags, deduplicating after normalization.

        Args:
            tags: Set of tags to normalize

        Returns:
            Deduplicated set of normalized tags
        """
        return {self.normalize(tag) for tag in tags}

    def detect_drift(self, all_tags: set[str]) -> list[DriftReport]:
        """Detect tag drift — multiple variants mapping to the same canonical.

        Args:
            all_tags: Complete set of tags to analyze

        Returns:
            List of DriftReport for canonicals with 2+ input variants
        """
        canonical_to_variants: dict[str, list[str]] = {}

        for tag in all_tags:
            normalized = self.normalize(tag)
            canonical_to_variants.setdefault(normalized, []).append(tag)

        reports: list[DriftReport] = []
        for canonical, variants in canonical_to_variants.items():
            unique_variants = sorted(set(variants))
            if len(unique_variants) >= 2:
                reports.append(
                    DriftReport(
                        canonical=canonical,
                        variants=tuple(unique_variants),
                        recommendation=f"Normalize all to '{canonical}'",
                    )
                )

        return reports
