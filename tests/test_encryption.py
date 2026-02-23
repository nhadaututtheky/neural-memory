"""Tests for safety.encryption module."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest


class TestMemoryEncryptor:
    """Unit tests for MemoryEncryptor."""

    def test_encrypt_decrypt_roundtrip(self, tmp_path: Path) -> None:
        from neural_memory.safety.encryption import MemoryEncryptor

        encryptor = MemoryEncryptor(keys_dir=tmp_path)
        plaintext = "api_key = sk-proj-abc123def456"
        brain_id = "default"

        result = encryptor.encrypt(plaintext, brain_id)
        assert result.ciphertext != plaintext
        assert result.key_id == brain_id
        assert result.algorithm == "fernet"

        decrypted = encryptor.decrypt(result.ciphertext, brain_id)
        assert decrypted == plaintext

    def test_key_created_on_first_encrypt(self, tmp_path: Path) -> None:
        from neural_memory.safety.encryption import MemoryEncryptor

        encryptor = MemoryEncryptor(keys_dir=tmp_path)
        encryptor.encrypt("secret data", "test-brain")

        key_file = tmp_path / "test-brain.key"
        assert key_file.exists()

    def test_key_reused_across_calls(self, tmp_path: Path) -> None:
        from neural_memory.safety.encryption import MemoryEncryptor

        encryptor = MemoryEncryptor(keys_dir=tmp_path)
        r1 = encryptor.encrypt("first", "mybrain")
        r2 = encryptor.encrypt("second", "mybrain")

        assert r1.ciphertext != r2.ciphertext
        assert encryptor.decrypt(r1.ciphertext, "mybrain") == "first"
        assert encryptor.decrypt(r2.ciphertext, "mybrain") == "second"

    def test_different_brains_different_keys(self, tmp_path: Path) -> None:
        from neural_memory.safety.encryption import MemoryEncryptor

        encryptor = MemoryEncryptor(keys_dir=tmp_path)
        r1 = encryptor.encrypt("same content", "brain-a")
        r2 = encryptor.encrypt("same content", "brain-b")

        assert r1.ciphertext != r2.ciphertext
        assert r1.key_id == "brain-a"
        assert r2.key_id == "brain-b"

    def test_decrypt_missing_key_returns_fallback(self, tmp_path: Path) -> None:
        from neural_memory.safety.encryption import MemoryEncryptor

        encryptor = MemoryEncryptor(keys_dir=tmp_path)
        result = encryptor.decrypt("bogus-ciphertext", "nonexistent-brain")
        assert result == "[Encrypted - key unavailable]"

    def test_decrypt_wrong_key_returns_fallback(self, tmp_path: Path) -> None:
        from neural_memory.safety.encryption import MemoryEncryptor

        enc1 = MemoryEncryptor(keys_dir=tmp_path)
        ciphertext = enc1.encrypt("secret", "brain-a").ciphertext

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        enc2 = MemoryEncryptor(keys_dir=other_dir)
        enc2.encrypt("dummy", "brain-a")

        result = enc2.decrypt(ciphertext, "brain-a")
        assert result == "[Encrypted - key unavailable]"

    def test_encrypt_empty_string(self, tmp_path: Path) -> None:
        from neural_memory.safety.encryption import MemoryEncryptor

        encryptor = MemoryEncryptor(keys_dir=tmp_path)
        result = encryptor.encrypt("", "default")
        assert encryptor.decrypt(result.ciphertext, "default") == ""

    def test_encrypt_unicode_content(self, tmp_path: Path) -> None:
        from neural_memory.safety.encryption import MemoryEncryptor

        encryptor = MemoryEncryptor(keys_dir=tmp_path)
        content = "mật khẩu: p@$$w0rd_安全_🔐"
        result = encryptor.encrypt(content, "default")
        assert encryptor.decrypt(result.ciphertext, "default") == content

    def test_key_file_permissions_restricted(self, tmp_path: Path) -> None:
        from neural_memory.safety.encryption import MemoryEncryptor

        encryptor = MemoryEncryptor(keys_dir=tmp_path)
        encryptor.encrypt("data", "secure-brain")

        key_file = tmp_path / "secure-brain.key"
        if os.name != "nt":
            mode = key_file.stat().st_mode
            assert not (mode & stat.S_IROTH), "Key file should not be world-readable"
            assert not (mode & stat.S_IWOTH), "Key file should not be world-writable"

    def test_invalid_brain_id_rejected(self, tmp_path: Path) -> None:
        from neural_memory.safety.encryption import MemoryEncryptor

        encryptor = MemoryEncryptor(keys_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid brain ID"):
            encryptor.encrypt("data", "../../../etc/passwd")

    def test_keys_dir_created_if_missing(self, tmp_path: Path) -> None:
        from neural_memory.safety.encryption import MemoryEncryptor

        keys_dir = tmp_path / "nested" / "keys"
        encryptor = MemoryEncryptor(keys_dir=keys_dir)
        encryptor.encrypt("data", "default")
        assert keys_dir.exists()


class TestEncryptionResult:
    def test_frozen(self) -> None:
        from neural_memory.safety.encryption import EncryptionResult

        result = EncryptionResult(ciphertext="abc", key_id="brain")
        with pytest.raises(AttributeError):
            result.ciphertext = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        from neural_memory.safety.encryption import EncryptionResult

        result = EncryptionResult(ciphertext="abc", key_id="brain")
        assert result.algorithm == "fernet"
