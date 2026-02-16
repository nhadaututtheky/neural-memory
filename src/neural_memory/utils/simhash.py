"""SimHash: locality-sensitive hashing for near-duplicate text detection.

Uses 64-bit fingerprints built from character n-gram shingles.
No external dependencies — stdlib only.

The key idea: similar texts produce fingerprints with small Hamming distance,
while dissimilar texts produce fingerprints with large Hamming distance.
"""

from __future__ import annotations

import hashlib
import struct

# Number of bits in the fingerprint
_BITS = 64

# Mask for 64-bit unsigned integer
_MASK = (1 << _BITS) - 1

# Default shingle size (3-character n-grams)
_SHINGLE_SIZE = 3

# Default Hamming distance threshold for near-duplicate detection.
# 10 bits out of 64 ≈ ~85% similarity threshold.
DEFAULT_THRESHOLD = 10


def _shingles(text: str, size: int = _SHINGLE_SIZE) -> list[str]:
    """Extract character n-gram shingles from text.

    Args:
        text: Input text (will be lowercased and whitespace-normalized).
        size: Size of each shingle.

    Returns:
        List of shingle strings.
    """
    normalized = " ".join(text.lower().split())
    if len(normalized) < size:
        return [normalized] if normalized else []
    return [normalized[i : i + size] for i in range(len(normalized) - size + 1)]


def simhash(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint for the given text.

    Args:
        text: The text to fingerprint.

    Returns:
        A 64-bit integer fingerprint.
    """
    if not text or not text.strip():
        return 0

    # Accumulate weighted bit vectors
    v = [0] * _BITS

    for shingle in _shingles(text):
        # Use MD5 for deterministic hashing (not security-sensitive here).
        # surrogatepass handles lone surrogates that can appear on Windows.
        digest = hashlib.md5(shingle.encode("utf-8", errors="surrogatepass")).digest()
        h = struct.unpack("<Q", digest[:8])[0]
        for i in range(_BITS):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    # Build fingerprint from sign of each dimension
    fingerprint = 0
    for i in range(_BITS):
        if v[i] > 0:
            fingerprint |= 1 << i

    # Convert to signed 64-bit to fit SQLite INTEGER
    if fingerprint >= (1 << 63):
        fingerprint -= 1 << 64

    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two fingerprints.

    Args:
        a: First fingerprint.
        b: Second fingerprint.

    Returns:
        Number of differing bits (0-64).
    """
    # Mask to 64 bits to handle signed integers correctly
    return bin((a ^ b) & _MASK).count("1")


def is_near_duplicate(a: int, b: int, threshold: int = DEFAULT_THRESHOLD) -> bool:
    """Check if two fingerprints are near-duplicates.

    Args:
        a: First fingerprint.
        b: Second fingerprint.
        threshold: Maximum Hamming distance to be considered near-duplicate.

    Returns:
        True if texts are near-duplicates.
    """
    return hamming_distance(a, b) <= threshold
