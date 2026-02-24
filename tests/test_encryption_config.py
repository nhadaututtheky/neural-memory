"""Tests for EncryptionConfig in unified_config."""

from __future__ import annotations


class TestEncryptionConfig:
    def test_default_values(self) -> None:
        from neural_memory.unified_config import EncryptionConfig

        cfg = EncryptionConfig()
        assert cfg.enabled is True
        assert cfg.auto_encrypt_sensitive is True
        assert cfg.keys_dir == ""

    def test_from_dict(self) -> None:
        from neural_memory.unified_config import EncryptionConfig

        cfg = EncryptionConfig.from_dict(
            {
                "enabled": False,
                "auto_encrypt_sensitive": False,
                "keys_dir": "custom/path",
            }
        )
        assert cfg.enabled is False
        assert cfg.auto_encrypt_sensitive is False
        assert cfg.keys_dir == "custom/path"

    def test_from_dict_defaults(self) -> None:
        from neural_memory.unified_config import EncryptionConfig

        cfg = EncryptionConfig.from_dict({})
        assert cfg.enabled is True
        assert cfg.auto_encrypt_sensitive is True

    def test_to_dict(self) -> None:
        from neural_memory.unified_config import EncryptionConfig

        cfg = EncryptionConfig()
        d = cfg.to_dict()
        assert "enabled" in d
        assert "auto_encrypt_sensitive" in d
        assert "keys_dir" in d

    def test_unified_config_has_encryption(self) -> None:
        from neural_memory.unified_config import EncryptionConfig, UnifiedConfig

        config = UnifiedConfig()
        assert hasattr(config, "encryption")
        assert isinstance(config.encryption, EncryptionConfig)

    def test_frozen(self) -> None:
        import pytest

        from neural_memory.unified_config import EncryptionConfig

        cfg = EncryptionConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = False  # type: ignore[misc]
