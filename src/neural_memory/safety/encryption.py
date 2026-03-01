"""Memory encryption for sensitive neuron content.

Provides Fernet symmetric encryption with per-brain keys.
Keys are stored on disk at ~/.neuralmemory/keys/{brain_id}.key.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

_BRAIN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


@dataclass(frozen=True)
class EncryptionResult:
    """Result of encrypting content."""

    ciphertext: str
    key_id: str
    algorithm: str = "fernet"


class MemoryEncryptor:
    """Per-brain Fernet encryption for neuron content."""

    _FALLBACK = "[Encrypted - key unavailable]"

    def __init__(self, keys_dir: Path) -> None:
        self._keys_dir = keys_dir
        self._ciphers: dict[str, Fernet] = {}

    def _validate_brain_id(self, brain_id: str) -> None:
        if not _BRAIN_ID_PATTERN.match(brain_id):
            raise ValueError(f"Invalid brain ID for encryption: {brain_id!r}")

    def _key_path(self, brain_id: str) -> Path:
        return self._keys_dir / f"{brain_id}.key"

    @staticmethod
    def _restrict_windows_acl(path: Path) -> None:
        """Restrict file access on Windows using icacls (owner-only read/write)."""
        import subprocess

        username = os.environ.get("USERNAME", "")
        if not username:
            return
        # Remove inherited permissions, grant only current user full control
        subprocess.run(  # noqa: S603
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{username}:(R,W)"],  # noqa: S607
            capture_output=True,
            check=False,
        )

    def _get_or_create_cipher(self, brain_id: str) -> Fernet:
        from cryptography.fernet import Fernet

        if brain_id in self._ciphers:
            return self._ciphers[brain_id]

        self._keys_dir.mkdir(parents=True, exist_ok=True)
        key_path = self._key_path(brain_id)

        if key_path.exists():
            key = key_path.read_bytes().strip()
            logger.debug("Loaded encryption key for brain %s", brain_id)
        else:
            key = Fernet.generate_key()
            key_path.write_bytes(key)
            try:
                if os.name == "nt":
                    self._restrict_windows_acl(key_path)
                else:
                    key_path.chmod(0o600)
            except OSError:
                logger.warning("Could not restrict key file permissions: %s", key_path)
            logger.info(
                "Generated new encryption key for brain %s at %s",
                brain_id,
                key_path,
            )

        cipher = Fernet(key)
        self._ciphers[brain_id] = cipher
        return cipher

    def _load_cipher(self, brain_id: str) -> Fernet | None:
        from cryptography.fernet import Fernet

        if brain_id in self._ciphers:
            return self._ciphers[brain_id]

        key_path = self._key_path(brain_id)
        if not key_path.exists():
            return None

        try:
            key = key_path.read_bytes().strip()
            cipher = Fernet(key)
            self._ciphers[brain_id] = cipher
            return cipher
        except Exception:
            logger.warning("Failed to load encryption key for brain %s", brain_id)
            return None

    def encrypt(self, content: str, brain_id: str) -> EncryptionResult:
        """Encrypt content for the given brain.

        Args:
            content: Plaintext content to encrypt.
            brain_id: Brain identifier (used as key namespace).

        Returns:
            EncryptionResult with ciphertext and metadata.

        Raises:
            ValueError: If brain_id contains invalid characters.
        """
        self._validate_brain_id(brain_id)
        cipher = self._get_or_create_cipher(brain_id)
        token = cipher.encrypt(content.encode("utf-8"))
        return EncryptionResult(
            ciphertext=token.decode("ascii"),
            key_id=brain_id,
        )

    def decrypt(self, ciphertext: str, brain_id: str) -> str:
        """Decrypt ciphertext for the given brain.

        Returns the fallback string if the key is missing or decryption fails.

        Args:
            ciphertext: Previously encrypted token string.
            brain_id: Brain identifier matching the encryption key.

        Returns:
            Decrypted plaintext, or ``"[Encrypted - key unavailable]"`` on failure.
        """
        try:
            self._validate_brain_id(brain_id)
        except ValueError:
            return self._FALLBACK

        cipher = self._load_cipher(brain_id)
        if cipher is None:
            return self._FALLBACK

        try:
            plaintext: str = cipher.decrypt(ciphertext.encode("ascii")).decode("utf-8")
            return plaintext
        except Exception:
            logger.warning("Decryption failed for brain %s", brain_id)
            return self._FALLBACK
