"""Tests for conversational stop words in keyword extraction.

Verifies that contractions without apostrophes, filler words, profanity,
and other conversational noise are filtered during keyword extraction.
"""

from __future__ import annotations

from neural_memory.extraction.keywords import extract_weighted_keywords


class TestConversationalStopWords:
    """Conversational English noise should not appear in extracted keywords."""

    def test_contractions_without_apostrophes_filtered(self) -> None:
        """dont, ive, im, thats should not appear as keywords."""
        kws = extract_weighted_keywords("i dont think ive seen thats the case with im not sure")
        kw_texts = {kw.text for kw in kws}
        for word in ("dont", "ive", "im", "thats"):
            assert word not in kw_texts, f"Contraction '{word}' should be filtered"

    def test_profanity_filtered(self) -> None:
        """Profanity should not appear as keywords."""
        kws = extract_weighted_keywords("the fucking thing just fucking crashed again")
        kw_texts = {kw.text for kw in kws}
        assert "fucking" not in kw_texts
        assert "fuck" not in kw_texts

    def test_filler_words_filtered(self) -> None:
        """Common filler words should not appear as keywords."""
        kws = extract_weighted_keywords("thats literally basically the same thing honestly")
        kw_texts = {kw.text for kw in kws}
        for word in ("literally", "basically", "honestly", "thing"):
            assert word not in kw_texts, f"Filler '{word}' should be filtered"

    def test_no_garbage_bigrams(self) -> None:
        """Garbage bigrams from casual text should not appear."""
        kws = extract_weighted_keywords("i dont think ive had something like gpu accel etc")
        kw_texts = {kw.text for kw in kws}
        assert "dont think" not in kw_texts
        assert "think ive" not in kw_texts
        assert "something like" not in kw_texts

    def test_meaningful_words_preserved(self) -> None:
        """Substantive words should still be extracted."""
        kws = extract_weighted_keywords("decided to use Redis for caching instead of Memcached")
        kw_texts = {kw.text for kw in kws}
        # Redis and Memcached should survive
        redis_found = any("redis" in kw for kw in kw_texts)
        assert redis_found, f"Expected 'redis' in keywords, got {kw_texts}"

    def test_like_filtered_as_filler(self) -> None:
        """'like' used as filler should be filtered, not when part of a comparison."""
        kws = extract_weighted_keywords("its like windows but i dont think ive seen that")
        kw_texts = {kw.text for kw in kws}
        # 'like' alone or in 'windows like' bigram should not appear
        assert "like" not in kw_texts

    def test_conversational_abbreviations_filtered(self) -> None:
        """Common abbreviations (idk, tbh, lol) should be filtered."""
        kws = extract_weighted_keywords("idk tbh the lol cache was kinda slow")
        kw_texts = {kw.text for kw in kws}
        for word in ("idk", "tbh", "lol", "kinda"):
            assert word not in kw_texts, f"Abbreviation '{word}' should be filtered"

    def test_capitalized_contractions_filtered(self) -> None:
        """Contractions should be filtered regardless of casing."""
        kws = extract_weighted_keywords("IM ASKING DONT YOU GET IT")
        kw_texts = {kw.text for kw in kws}
        assert "im" not in kw_texts
        assert "dont" not in kw_texts
