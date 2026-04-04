"""Vector quantization and compression for InfinityDB.

Provides lossless and lossy compression for float32 vectors:
- float32 (Active): full precision, 1536 bytes/384D
- float16 (Warm): half precision, 768 bytes/384D
- int8 (Cool): 8-bit quantized, 384 bytes/384D
- binary (Frozen): 1-bit hash, 48 bytes/384D
- None (Crystal): metadata-only, 0 bytes vector

Each tier trades accuracy for space. Higher tiers are used for
less-frequently accessed neurons.
"""

from __future__ import annotations

import logging
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CompressionTier(IntEnum):
    """Compression tiers ordered by quality (highest first)."""

    ACTIVE = 0  # float32 — full precision
    WARM = 1  # float16 — half precision
    COOL = 2  # int8 — 8-bit quantized
    FROZEN = 3  # binary — 1-bit per dimension
    CRYSTAL = 4  # no vector — metadata only


# Bytes per dimension for each tier
BYTES_PER_DIM: dict[CompressionTier, float] = {
    CompressionTier.ACTIVE: 4.0,  # float32
    CompressionTier.WARM: 2.0,  # float16
    CompressionTier.COOL: 1.0,  # int8
    CompressionTier.FROZEN: 0.125,  # 1 bit
    CompressionTier.CRYSTAL: 0.0,  # nothing
}


class VectorCompressor:
    """Compress and decompress vectors between quality tiers."""

    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def compress(
        self,
        vector: NDArray[np.float32],
        target_tier: CompressionTier,
    ) -> bytes:
        """Compress a float32 vector to a target tier.

        Args:
            vector: source vector (float32, shape=(dimensions,))
            target_tier: desired compression level

        Returns:
            Compressed bytes representation.
        """
        if vector.shape != (self._dimensions,):
            msg = f"Expected shape ({self._dimensions},), got {vector.shape}"
            raise ValueError(msg)

        if target_tier == CompressionTier.ACTIVE:
            return bytes(vector.astype(np.float32).tobytes())

        if target_tier == CompressionTier.WARM:
            return bytes(vector.astype(np.float16).tobytes())

        if target_tier == CompressionTier.COOL:
            return self._quantize_int8(vector)

        if target_tier == CompressionTier.FROZEN:
            return self._quantize_binary(vector)

        if target_tier == CompressionTier.CRYSTAL:
            return b""  # No vector data

        msg = f"Unknown tier: {target_tier}"
        raise ValueError(msg)

    def decompress(
        self,
        data: bytes,
        source_tier: CompressionTier,
    ) -> NDArray[np.float32]:
        """Decompress bytes back to float32 vector.

        Args:
            data: compressed bytes
            source_tier: compression level of the data

        Returns:
            Reconstructed float32 vector. Lossy for tiers > ACTIVE.
        """
        if source_tier == CompressionTier.ACTIVE:
            return np.frombuffer(data, dtype=np.float32).copy()

        if source_tier == CompressionTier.WARM:
            return np.frombuffer(data, dtype=np.float16).astype(np.float32)

        if source_tier == CompressionTier.COOL:
            return self._dequantize_int8(data)

        if source_tier == CompressionTier.FROZEN:
            return self._dequantize_binary(data)

        if source_tier == CompressionTier.CRYSTAL:
            return np.zeros(self._dimensions, dtype=np.float32)

        msg = f"Unknown tier: {source_tier}"
        raise ValueError(msg)

    def _quantize_int8(self, vector: NDArray[np.float32]) -> bytes:
        """Quantize float32 to int8 with min/max scaling.

        Layout: [min_val:f32][max_val:f32][int8_data:N]
        Total: 8 + dimensions bytes.
        """
        vmin = float(vector.min())
        vmax = float(vector.max())
        val_range = vmax - vmin

        if val_range < 1e-10:
            # Constant vector — store as all zeros
            header = bytes(np.array([vmin, vmax], dtype=np.float32).tobytes())
            return header + bytes(np.zeros(self._dimensions, dtype=np.int8).tobytes())

        # Scale to [-127, 127]
        scaled = ((vector - vmin) / val_range * 254 - 127).clip(-127, 127)
        quantized = scaled.astype(np.int8)

        header = bytes(np.array([vmin, vmax], dtype=np.float32).tobytes())
        return header + bytes(quantized.tobytes())

    def _dequantize_int8(self, data: bytes) -> NDArray[np.float32]:
        """Restore float32 from int8 quantized data."""
        header = np.frombuffer(data[:8], dtype=np.float32)
        vmin, vmax = float(header[0]), float(header[1])
        val_range = vmax - vmin

        quantized = np.frombuffer(data[8:], dtype=np.int8).astype(np.float32)

        if val_range < 1e-10:
            return np.full(self._dimensions, vmin, dtype=np.float32)

        return ((quantized + 127) / 254 * val_range + vmin).astype(np.float32)

    def _quantize_binary(self, vector: NDArray[np.float32]) -> bytes:
        """Quantize to 1-bit per dimension (sign hash).

        Each dimension becomes 1 if > 0, else 0. Packed into bytes.
        Total: ceil(dimensions / 8) bytes.
        """
        bits = (vector > 0).astype(np.uint8)
        # Pad to multiple of 8
        pad_len = (8 - (self._dimensions % 8)) % 8
        if pad_len:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
        return bytes(np.packbits(bits).tobytes())

    def _dequantize_binary(self, data: bytes) -> NDArray[np.float32]:
        """Restore approximate float32 from binary hash.

        Maps 1 -> +1.0, 0 -> -1.0. Very lossy but preserves direction.
        """
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        # Trim to actual dimensions
        bits = bits[: self._dimensions]
        return bits.astype(np.float32) * 2 - 1

    def estimate_size(self, tier: CompressionTier, count: int) -> int:
        """Estimate total bytes for N vectors at a given tier."""
        bytes_per = BYTES_PER_DIM[tier] * self._dimensions
        if tier == CompressionTier.COOL:
            bytes_per += 8  # int8 header
        return int(bytes_per * count)

    def compression_ratio(
        self, source_tier: CompressionTier, target_tier: CompressionTier
    ) -> float:
        """Get compression ratio between two tiers."""
        source_bpd = BYTES_PER_DIM[source_tier]
        target_bpd = BYTES_PER_DIM[target_tier]
        if target_bpd == 0:
            return float("inf")
        return source_bpd / target_bpd
