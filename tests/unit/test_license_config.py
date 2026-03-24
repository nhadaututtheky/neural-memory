"""Tests for LicenseConfig — tier validation, is_pro(), load/save roundtrip."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from neural_memory.unified_config import (
    LicenseConfig,
    UnifiedConfig,
    _sanitize_iso_datetime,
)

# ── LicenseConfig dataclass ─────────────────────────────────────


class TestLicenseConfig:
    def test_defaults(self) -> None:
        cfg = LicenseConfig()
        assert cfg.tier == "free"
        assert cfg.activated_at == ""
        assert cfg.expires_at == ""

    def test_from_dict_valid_tiers(self) -> None:
        for tier in ("free", "pro", "team"):
            cfg = LicenseConfig.from_dict({"tier": tier})
            assert cfg.tier == tier

    def test_from_dict_invalid_tier_falls_back_to_free(self) -> None:
        cfg = LicenseConfig.from_dict({"tier": "enterprise"})
        assert cfg.tier == "free"

    def test_from_dict_case_insensitive(self) -> None:
        cfg = LicenseConfig.from_dict({"tier": "PRO"})
        assert cfg.tier == "pro"

    def test_from_dict_empty(self) -> None:
        cfg = LicenseConfig.from_dict({})
        assert cfg.tier == "free"
        assert cfg.activated_at == ""

    def test_to_dict_roundtrip(self) -> None:
        original = LicenseConfig(tier="pro", activated_at="2026-03-24", expires_at="2027-03-24")
        restored = LicenseConfig.from_dict(original.to_dict())
        assert restored.tier == original.tier
        assert restored.activated_at == original.activated_at
        assert restored.expires_at == original.expires_at

    def test_frozen(self) -> None:
        cfg = LicenseConfig()
        with pytest.raises(AttributeError):
            cfg.tier = "pro"  # type: ignore[misc]


# ── UnifiedConfig.is_pro() ──────────────────────────────────────


class TestIsPro:
    def test_free_is_not_pro(self) -> None:
        cfg = UnifiedConfig(license=LicenseConfig(tier="free"))
        assert cfg.is_pro() is False

    def test_pro_is_pro(self) -> None:
        cfg = UnifiedConfig(license=LicenseConfig(tier="pro"))
        assert cfg.is_pro() is True

    def test_team_is_pro(self) -> None:
        cfg = UnifiedConfig(license=LicenseConfig(tier="team"))
        assert cfg.is_pro() is True

    def test_default_is_not_pro(self) -> None:
        cfg = UnifiedConfig()
        assert cfg.is_pro() is False


# ── TOML save/load roundtrip ────────────────────────────────────


class TestLicenseTomlRoundtrip:
    def test_save_and_load_preserves_license(self, tmp_path: Path) -> None:
        data_dir = tmp_path / ".neuralmemory"
        data_dir.mkdir()

        original = UnifiedConfig(
            data_dir=data_dir,
            current_brain="default",
            license=LicenseConfig(
                tier="pro",
                activated_at="2026-03-24T10:00:00",
                expires_at="2027-03-24T10:00:00",
            ),
        )
        original.save()

        # Verify TOML contains [license] section
        toml_content = (data_dir / "config.toml").read_text()
        assert "[license]" in toml_content
        assert 'tier = "pro"' in toml_content

        loaded = UnifiedConfig.load(data_dir / "config.toml")
        assert loaded.license.tier == "pro"
        assert loaded.license.activated_at == "2026-03-24T10:00:00"
        assert loaded.license.expires_at == "2027-03-24T10:00:00"
        assert loaded.is_pro() is True

    def test_load_without_license_section_defaults_to_free(self, tmp_path: Path) -> None:
        data_dir = tmp_path / ".neuralmemory"
        data_dir.mkdir()

        # Save a config, then strip the [license] section
        cfg = UnifiedConfig(data_dir=data_dir, current_brain="default")
        cfg.save()

        toml_path = data_dir / "config.toml"
        lines = toml_path.read_text().splitlines()
        filtered = []
        skip = False
        for line in lines:
            if line.strip() == "[license]":
                skip = True
                continue
            if skip and line.strip().startswith("["):
                skip = False
            if not skip:
                filtered.append(line)
        toml_path.write_text("\n".join(filtered) + "\n")

        loaded = UnifiedConfig.load(toml_path)
        assert loaded.license.tier == "free"
        assert loaded.is_pro() is False


# ── is_pro() expiry checks ──────────────────────────────────────


class TestIsProExpiry:
    def test_no_expiry_is_perpetual(self) -> None:
        cfg = UnifiedConfig(license=LicenseConfig(tier="pro", expires_at=""))
        assert cfg.is_pro() is True

    def test_future_expiry_is_pro(self) -> None:
        future = (datetime.now(UTC) + timedelta(days=30)).isoformat()
        cfg = UnifiedConfig(license=LicenseConfig(tier="pro", expires_at=future))
        assert cfg.is_pro() is True

    def test_past_expiry_is_not_pro(self) -> None:
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        cfg = UnifiedConfig(license=LicenseConfig(tier="pro", expires_at=past))
        assert cfg.is_pro() is False

    def test_malformed_expiry_treated_as_perpetual(self) -> None:
        cfg = UnifiedConfig(license=LicenseConfig(tier="pro", expires_at="not-a-date"))
        # Malformed expiry sanitized to "" by from_dict, but direct construction bypasses
        assert cfg.is_pro() is True

    def test_naive_datetime_expiry(self) -> None:
        future = (datetime.now(UTC) + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
        cfg = UnifiedConfig(license=LicenseConfig(tier="pro", expires_at=future))
        assert cfg.is_pro() is True


# ── _sanitize_iso_datetime ───────────────────────────────────────


class TestSanitizeIsoDatetime:
    def test_valid_iso(self) -> None:
        assert _sanitize_iso_datetime("2026-03-24T10:00:00") == "2026-03-24T10:00:00"

    def test_valid_iso_with_timezone(self) -> None:
        assert _sanitize_iso_datetime("2026-03-24T10:00:00+07:00") == "2026-03-24T10:00:00+07:00"

    def test_valid_iso_with_z(self) -> None:
        assert _sanitize_iso_datetime("2026-03-24T10:00:00Z") == "2026-03-24T10:00:00Z"

    def test_empty_string(self) -> None:
        assert _sanitize_iso_datetime("") == ""

    def test_rejects_quotes(self) -> None:
        assert _sanitize_iso_datetime('2026-01-01"\\n[sync]') == ""

    def test_rejects_newlines(self) -> None:
        assert _sanitize_iso_datetime("2026-01-01\ninjected") == ""

    def test_rejects_spaces(self) -> None:
        assert _sanitize_iso_datetime("2026-01-01 10:00:00") == ""

    def test_rejects_alpha(self) -> None:
        assert _sanitize_iso_datetime("not-a-date") == ""

    def test_truncates_long_input(self) -> None:
        long_input = "2026-03-24T10:00:00" + "0" * 100
        result = _sanitize_iso_datetime(long_input)
        assert len(result) <= 64

    def test_non_string_returns_empty(self) -> None:
        assert _sanitize_iso_datetime(12345) == ""  # type: ignore[arg-type]


# ── H3: from_dict sanitizes datetime fields ──────────────────────


class TestFromDictSanitization:
    def test_injection_in_activated_at_stripped(self) -> None:
        cfg = LicenseConfig.from_dict(
            {
                "tier": "pro",
                "activated_at": '2026-01-01"\n[sync]\napi_key = "injected',
            }
        )
        assert cfg.activated_at == ""

    def test_injection_in_expires_at_stripped(self) -> None:
        cfg = LicenseConfig.from_dict(
            {
                "tier": "pro",
                "expires_at": '2099-01-01"; DROP TABLE neurons; --',
            }
        )
        assert cfg.expires_at == ""
