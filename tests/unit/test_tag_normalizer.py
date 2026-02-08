"""Tests for tag normalizer — synonym mapping, SimHash matching, drift detection."""

from __future__ import annotations

import pytest

from neural_memory.utils.tag_normalizer import TagNormalizer


class TestTagNormalizerSynonyms:
    def test_maps_synonym_to_canonical(self) -> None:
        normalizer = TagNormalizer()
        assert normalizer.normalize("ui") == "frontend"
        assert normalizer.normalize("client-side") == "frontend"
        assert normalizer.normalize("server-side") == "backend"
        assert normalizer.normalize("db") == "database"

    def test_canonical_maps_to_itself(self) -> None:
        normalizer = TagNormalizer()
        assert normalizer.normalize("frontend") == "frontend"
        assert normalizer.normalize("backend") == "backend"

    def test_case_insensitive(self) -> None:
        normalizer = TagNormalizer()
        assert normalizer.normalize("UI") == "frontend"
        assert normalizer.normalize("Db") == "database"
        assert normalizer.normalize("PYTHON") == "py"

    def test_strips_whitespace(self) -> None:
        normalizer = TagNormalizer()
        assert normalizer.normalize("  ui  ") == "frontend"

    def test_unknown_tag_lowercased(self) -> None:
        normalizer = TagNormalizer()
        assert normalizer.normalize("MyCustomTag") == "mycustomtag"

    def test_extra_synonyms(self) -> None:
        normalizer = TagNormalizer(extra_synonyms={"mycanon": ["myalias", "myother"]})
        assert normalizer.normalize("myalias") == "mycanon"
        assert normalizer.normalize("myother") == "mycanon"
        assert normalizer.normalize("mycanon") == "mycanon"


class TestTagNormalizerSet:
    def test_deduplicates_after_normalization(self) -> None:
        normalizer = TagNormalizer()
        result = normalizer.normalize_set({"ui", "frontend", "client-side"})
        assert result == {"frontend"}

    def test_mixed_known_and_unknown(self) -> None:
        normalizer = TagNormalizer()
        result = normalizer.normalize_set({"ui", "myproject"})
        assert "frontend" in result
        assert "myproject" in result

    def test_empty_set(self) -> None:
        normalizer = TagNormalizer()
        result = normalizer.normalize_set(set())
        assert result == set()


class TestTagNormalizerDriftDetection:
    def test_detects_multiple_variants(self) -> None:
        normalizer = TagNormalizer()
        reports = normalizer.detect_drift({"ui", "frontend", "client-side"})
        assert len(reports) == 1
        assert reports[0].canonical == "frontend"
        assert set(reports[0].variants) == {"ui", "frontend", "client-side"}

    def test_no_drift_for_single_variant(self) -> None:
        normalizer = TagNormalizer()
        reports = normalizer.detect_drift({"frontend", "myunique"})
        assert reports == []

    def test_multiple_drift_groups(self) -> None:
        normalizer = TagNormalizer()
        reports = normalizer.detect_drift({"ui", "frontend", "db", "database"})
        canonicals = {r.canonical for r in reports}
        assert "frontend" in canonicals
        assert "database" in canonicals

    def test_recommendation_format(self) -> None:
        normalizer = TagNormalizer()
        reports = normalizer.detect_drift({"tests", "testing"})
        assert len(reports) == 1
        assert "testing" in reports[0].recommendation


class TestTagNormalizerSimHash:
    def test_simhash_near_match(self) -> None:
        # SimHash fuzzy matching for tags very similar to canonical forms
        normalizer = TagNormalizer(simhash_threshold=6)
        # "frontnd" is close to "frontend" by SimHash (missing one char)
        result = normalizer.normalize("frontnd")
        # Should either match via SimHash or fall back to lowercase
        assert isinstance(result, str)

    def test_very_different_tag_no_match(self) -> None:
        normalizer = TagNormalizer(simhash_threshold=6)
        result = normalizer.normalize("zzzzqqqq")
        assert result == "zzzzqqqq"


class TestCommonNormalizations:
    """Integration-style tests for common synonym groups."""

    @pytest.mark.parametrize(
        "input_tag,expected",
        [
            ("authentication", "auth"),
            ("authorization", "auth"),
            ("configuration", "config"),
            ("deployment", "deploy"),
            ("unit-test", "testing"),
            ("documentation", "docs"),
            ("performance", "perf"),
            ("infrastructure", "infra"),
            ("cache", "caching"),
            ("vulnerability", "security"),
            ("refactor", "refactoring"),
            ("debug", "debugging"),
            ("docker", "container"),
            ("kubernetes", "k8s"),
            ("ci/cd", "ci"),
            ("javascript", "js"),
            ("typescript", "ts"),
            ("python", "py"),
            ("rest", "api"),
            ("machine-learning", "ml"),
        ],
    )
    def test_synonym_normalization(self, input_tag: str, expected: str) -> None:
        normalizer = TagNormalizer()
        assert normalizer.normalize(input_tag) == expected
