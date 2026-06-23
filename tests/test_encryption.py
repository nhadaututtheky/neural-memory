"""Tests for safety.encryption module."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

try:
    import cryptography

    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False


@pytest.mark.skipif(not _HAS_CRYPTOGRAPHY, reason="cryptography not installed")
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

    @pytest.mark.skipif(os.name == "nt", reason="O_EXCL race path is POSIX-only")
    def test_first_write_toctou_race_handled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression (#68): if a concurrent process wins the O_EXCL race,
        encrypt() must read back the existing key instead of raising
        FileExistsError."""
        from cryptography.fernet import Fernet

        from neural_memory.safety import encryption as enc_mod
        from neural_memory.safety.encryption import MemoryEncryptor

        encryptor = MemoryEncryptor(keys_dir=tmp_path)
        brain_id = "race-brain"
        key_path = tmp_path / f"{brain_id}.key"

        # Simulate the racing process: a valid key already on disk, and our
        # O_EXCL open losing the race (FileExistsError) on the first attempt.
        winner_key = Fernet.generate_key()
        real_open = os.open

        def fake_open(path: str, flags: int, *args: object) -> int:
            if str(path) == str(key_path) and (flags & os.O_EXCL):
                # The winner wrote its key just before our open.
                key_path.write_bytes(winner_key)
                raise FileExistsError(17, "File exists", str(path))
            return real_open(path, flags, *args)  # type: ignore[arg-type]

        monkeypatch.setattr(enc_mod.os, "open", fake_open)

        result = encryptor.encrypt("secret payload", brain_id)
        # Must succeed and be decryptable with the winner's key (fresh encryptor).
        fresh = MemoryEncryptor(keys_dir=tmp_path)
        assert fresh.decrypt(result.ciphertext, brain_id) == "secret payload"


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
