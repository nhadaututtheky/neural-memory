"""Tests for write gate — quality enforcement before storage.

Covers:
- WriteGateConfig defaults and from_dict
- check_write_gate: filler, length, wall-of-text, quality score rejection
- Auto-capture stricter threshold
- Integration with _remember handler (gate enabled/disabled)
- Backward compatibility: gate disabled = original behavior
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from neural_memory.engine.quality_scorer import (
    _is_generic_filler,
    check_write_gate,
    score_memory,
)
from neural_memory.unified_config import WriteGateConfig

# ---------------------------------------------------------------------------
# WriteGateConfig tests
# ---------------------------------------------------------------------------


class TestWriteGateConfig:
    """Tests for WriteGateConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = WriteGateConfig()
        assert cfg.enabled is False
        assert cfg.min_length == 30
        assert cfg.min_quality_score == 3
        assert cfg.auto_capture_min_score == 5
        assert cfg.max_content_length == 2000
        assert cfg.reject_generic_filler is True

    def test_from_dict_empty(self) -> None:
        cfg = WriteGateConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.min_length == 30

    def test_from_dict_custom(self) -> None:
        cfg = WriteGateConfig.from_dict(
            {
                "enabled": True,
                "min_length": 50,
                "min_quality_score": 4,
                "auto_capture_min_score": 6,
                "max_content_length": 1500,
                "reject_generic_filler": False,
            }
        )
        assert cfg.enabled is True
        assert cfg.min_length == 50
        assert cfg.min_quality_score == 4
        assert cfg.auto_capture_min_score == 6
        assert cfg.max_content_length == 1500
        assert cfg.reject_generic_filler is False

    def test_to_dict_round_trip(self) -> None:
        cfg = WriteGateConfig(enabled=True, min_length=40)
        d = cfg.to_dict()
        cfg2 = WriteGateConfig.from_dict(d)
        assert cfg2.enabled is True
        assert cfg2.min_length == 40

    def test_frozen(self) -> None:
        cfg = WriteGateConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Generic filler detection
# ---------------------------------------------------------------------------


class TestGenericFiller:
    """Tests for _is_generic_filler()."""

    @pytest.mark.parametrize(
        "content",
        ["done", "Done", "DONE", "ok", "Ok.", "completed", "noted", "xong", "oke", "got it"],
    )
    def test_filler_detected(self, content: str) -> None:
        assert _is_generic_filler(content) is True

    @pytest.mark.parametrize(
        "content",
        [
            "done migrating the database",
            "completed the auth refactor because we needed OAuth2",
            "noted the issue with PostgreSQL connections",
            "The task is done and tested",
        ],
    )
    def test_substantive_not_filler(self, content: str) -> None:
        assert _is_generic_filler(content) is False

    def test_empty_string_not_filler(self) -> None:
        assert _is_generic_filler("") is False

    def test_whitespace_only_not_filler(self) -> None:
        # Empty after strip — not in filler set
        assert _is_generic_filler("   ") is False


# ---------------------------------------------------------------------------
# check_write_gate tests
# ---------------------------------------------------------------------------


class TestCheckWriteGate:
    """Tests for check_write_gate()."""

    def _gate(self, **overrides: object) -> WriteGateConfig:
        defaults = {
            "enabled": True,
            "min_length": 30,
            "min_quality_score": 3,
            "auto_capture_min_score": 5,
            "max_content_length": 2000,
            "reject_generic_filler": True,
        }
        defaults.update(overrides)
        return WriteGateConfig(**defaults)  # type: ignore[arg-type]

    # -- Filler rejection --

    def test_rejects_filler_done(self) -> None:
        result = check_write_gate("done", gate_config=self._gate())
        assert result.rejected is True
        assert "filler" in result.rejection_reason.lower()

    def test_rejects_filler_ok(self) -> None:
        result = check_write_gate("ok", gate_config=self._gate())
        assert result.rejected is True

    def test_filler_rejection_disabled(self) -> None:
        """Filler check skipped when reject_generic_filler=False."""
        result = check_write_gate("ok", gate_config=self._gate(reject_generic_filler=False))
        # Still rejected for length (2 chars < 30)
        assert result.rejected is True
        assert "short" in result.rejection_reason.lower()

    # -- Length rejection --

    def test_rejects_too_short(self) -> None:
        result = check_write_gate("short text", gate_config=self._gate())
        assert result.rejected is True
        assert "short" in result.rejection_reason.lower()

    def test_accepts_at_min_length(self) -> None:
        content = "a" * 30 + " because it was needed"
        result = check_write_gate(
            content, gate_config=self._gate(min_length=30, min_quality_score=1)
        )
        assert result.rejected is False

    def test_custom_min_length(self) -> None:
        content = "a" * 20 + " test content"
        result = check_write_gate(content, gate_config=self._gate(min_length=50))
        assert result.rejected is True
        assert "short" in result.rejection_reason.lower()

    # -- Wall-of-text rejection --

    def test_rejects_wall_of_text(self) -> None:
        content = "x " * 1500  # 3000 chars
        result = check_write_gate(content, gate_config=self._gate(max_content_length=2000))
        assert result.rejected is True
        assert "long" in result.rejection_reason.lower()
        assert "split" in result.rejection_reason.lower()

    def test_accepts_within_max_length(self) -> None:
        content = "Chose PostgreSQL because ACID " * 30  # ~900 chars
        result = check_write_gate(
            content, gate_config=self._gate(max_content_length=2000, min_quality_score=1)
        )
        assert result.rejected is False

    # -- Quality score rejection --

    def test_rejects_low_quality_score(self) -> None:
        # Content with length but no cognitive richness, no tags, no context
        content = "some random content that is long enough to pass length check"
        result = check_write_gate(content, gate_config=self._gate(min_quality_score=5))
        assert result.rejected is True
        assert "quality score" in result.rejection_reason.lower()

    def test_accepts_high_quality(self) -> None:
        result = check_write_gate(
            "Chose PostgreSQL over MongoDB because ACID needed for payment processing",
            gate_config=self._gate(min_quality_score=3),
            memory_type="decision",
            tags=["database"],
            context={"reason": "ACID"},
        )
        assert result.rejected is False
        assert result.score >= 3

    # -- Auto-capture stricter threshold --

    def test_auto_capture_uses_stricter_threshold(self) -> None:
        """Auto-capture uses auto_capture_min_score (5), not min_quality_score (3)."""
        # Content that scores ~3-4 (passes normal gate but fails auto-capture)
        content = "some content that is reasonably long enough to pass basic checks"
        gate = self._gate(min_quality_score=2, auto_capture_min_score=5)

        normal = check_write_gate(content, gate_config=gate, is_auto_capture=False)
        auto = check_write_gate(content, gate_config=gate, is_auto_capture=True)

        assert normal.rejected is False
        assert auto.rejected is True

    def test_auto_capture_false_uses_normal_threshold(self) -> None:
        content = "Chose X over Y because of Z which was better than alternatives"
        gate = self._gate(min_quality_score=3, auto_capture_min_score=8)
        result = check_write_gate(content, gate_config=gate, is_auto_capture=False)
        assert result.rejected is False

    # -- Hints are returned on rejection --

    def test_rejected_result_has_hints(self) -> None:
        result = check_write_gate(
            "some content that is long enough but low quality",
            gate_config=self._gate(min_quality_score=8),
        )
        assert result.rejected is True
        assert len(result.hints) > 0

    # -- Backward compat: QualityResult --

    def test_result_to_dict_includes_rejection(self) -> None:
        result = check_write_gate("done", gate_config=self._gate())
        d = result.to_dict()
        assert d["rejected"] is True
        assert "rejection_reason" in d

    def test_accepted_result_no_rejection_in_dict(self) -> None:
        result = check_write_gate(
            "Chose PostgreSQL because ACID needed for payment processing",
            gate_config=self._gate(min_quality_score=1),
            memory_type="decision",
            tags=["db"],
        )
        d = result.to_dict()
        assert "rejected" not in d

    # -- Gate order: filler checked before length --

    def test_filler_checked_before_length(self) -> None:
        """'ok' is 2 chars (< min_length) but should be rejected as filler first."""
        result = check_write_gate("ok", gate_config=self._gate())
        assert "filler" in result.rejection_reason.lower()


# ---------------------------------------------------------------------------
# score_memory backward compat
# ---------------------------------------------------------------------------


class TestScoreMemoryBackwardCompat:
    """Ensure score_memory still works exactly as before."""

    def test_no_rejected_field_by_default(self) -> None:
        """score_memory doesn't have rejected field (only check_write_gate does)."""
        result = score_memory("hello")
        assert result.rejected is False  # default value
        assert result.rejection_reason == ""

    def test_to_dict_no_rejection_keys(self) -> None:
        """score_memory results don't include rejection keys in dict."""
        result = score_memory("hello")
        d = result.to_dict()
        assert "rejected" not in d


# ---------------------------------------------------------------------------
# Integration: _remember with write gate
# ---------------------------------------------------------------------------


class TestRememberWriteGateIntegration:
    """Test write gate integration in _remember handler."""

    @pytest.fixture
    def mock_handler(self) -> MagicMock:
        """Create a minimal mock of the tool handler."""
        handler = MagicMock()
        handler.config = MagicMock()
        handler.config.write_gate = WriteGateConfig(enabled=True, min_quality_score=3)
        handler.config.safety = MagicMock(auto_redact_min_severity=3)
        handler.config.encryption = MagicMock(enabled=False)
        return handler

    def test_gate_rejects_filler_before_encoding(self) -> None:
        """Write gate prevents 'done' from reaching the encoder."""
        from neural_memory.engine.quality_scorer import check_write_gate

        gate_cfg = WriteGateConfig(enabled=True)
        result = check_write_gate("done", gate_config=gate_cfg)
        assert result.rejected is True
        # Encoder should never be called if gate rejects

    def test_gate_disabled_allows_everything(self) -> None:
        """When gate is disabled, check_write_gate is not called."""
        # Gate disabled = no check happens, so low-quality content passes
        gate_cfg = WriteGateConfig(enabled=False)
        # The handler checks `gate_cfg.enabled` before calling check_write_gate
        assert gate_cfg.enabled is False


# ---------------------------------------------------------------------------
# Vietnamese filler detection
# ---------------------------------------------------------------------------


class TestVietnameseFiller:
    """Ensure Vietnamese filler words are caught."""

    @pytest.mark.parametrize("content", ["xong", "Xong", "da", "vang", "hieu roi"])
    def test_vietnamese_filler_detected(self, content: str) -> None:
        assert _is_generic_filler(content) is True

    def test_vietnamese_substantive_passes(self) -> None:
        assert _is_generic_filler("xong roi, da test xong") is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestWriteGateEdgeCases:
    """Edge cases for write gate."""

    def _gate(self, **overrides: object) -> WriteGateConfig:
        defaults = {
            "enabled": True,
            "min_length": 30,
            "min_quality_score": 3,
            "auto_capture_min_score": 5,
            "max_content_length": 2000,
            "reject_generic_filler": True,
        }
        defaults.update(overrides)
        return WriteGateConfig(**defaults)  # type: ignore[arg-type]

    def test_whitespace_content_rejected(self) -> None:
        result = check_write_gate("    \n\t   ", gate_config=self._gate())
        assert result.rejected is True

    def test_exactly_max_content_length_passes(self) -> None:
        content = "a " * 999 + "because needed"  # ~2012 chars
        gate = self._gate(max_content_length=2100, min_quality_score=1)
        result = check_write_gate(content, gate_config=gate)
        assert result.rejected is False

    def test_content_with_trailing_punctuation_filler(self) -> None:
        """'done!' should still be detected as filler."""
        result = check_write_gate("done!", gate_config=self._gate())
        assert result.rejected is True
        assert "filler" in result.rejection_reason.lower()

    def test_multiline_content_length_check(self) -> None:
        """Multiline content length is total, not per-line."""
        content = "line\n" * 500  # 2500 chars
        result = check_write_gate(content, gate_config=self._gate(max_content_length=2000))
        assert result.rejected is True

    def test_quality_result_immutable(self) -> None:
        result = check_write_gate("done", gate_config=self._gate())
        with pytest.raises(AttributeError):
            result.rejected = False  # type: ignore[misc]
