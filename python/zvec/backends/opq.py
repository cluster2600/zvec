"""Optimized Product Quantization (OPQ) implementation."""

from __future__ import annotations

import logging

import numpy as np

from zvec.backends.quantization import PQEncoder

logger = logging.getLogger(__name__)


class OPQEncoder:
    """Optimized Product Quantization encoder.

    OPQ rotates vectors before applying PQ to improve compression quality.
    The rotation aligns the data with the quantization axes.

    Example:
        >>> encoder = OPQEncoder(m=8, nbits=8, k=256)
        >>> encoder.train(vectors)
        >>> codes = encoder.encode(vectors)
        >>> rotated = encoder.rotate(vectors)
    """

    def __init__(self, m: int = 8, nbits: int = 8, k: int = 256):
        """Initialize OPQ encoder.

        Args:
            m: Number of sub-vectors (subquantizers).
            nbits: Number of bits per sub-vector.
            k: Number of centroids per sub-vector.
        """
        self.m = m
        self.nbits = nbits
        self.k = k
        self.pq = PQEncoder(m=m, nbits=nbits, k=k)
        self.rotation_matrix: np.ndarray | None = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Check if encoder is trained."""
        return self._is_trained

    def train(self, vectors: np.ndarray, n_iter: int = 20) -> None:
        """Train the OPQ encoder on vectors.

        This iteratively optimizes:
        1. The rotation matrix R
        2. The PQ codebooks

        Args:
            vectors: Training vectors (N x dim).
            n_iter: Number of optimization iterations.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        _n_vectors, dim = vectors.shape

        if dim % self.m != 0:
            raise ValueError(f"Dimension {dim} must be divisible by m={self.m}")

        # Initialize rotation matrix as identity
        self.rotation_matrix = np.eye(dim, dtype=np.float32)

        # Iterative optimization
        for iteration in range(n_iter):
            # Step 1: Rotate vectors
            rotated = vectors @ self.rotation_matrix.T

            # Step 2: Train PQ on rotated vectors
            self.pq.train(rotated)

            # Step 3: Learn optimal rotation
            # Simple SVD-based rotation learning
            self._learn_rotation(vectors)

            if iteration % 5 == 0:
                logger.info("OPQ iteration %d/%d", iteration, n_iter)

        self._is_trained = True
        logger.info("OPQ training complete")

    def _learn_rotation(self, vectors: np.ndarray) -> None:
        """Learn optimal rotation matrix.

        Uses a simplified SVD approach to find rotation that
        minimizes quantization error.

        Args:
            vectors: Original vectors (N x dim).
        """
        # Encode with current rotation
        rotated = vectors @ self.rotation_matrix.T
        codes = self.pq.encode(rotated)

        # Decode to get approximate vectors
        decoded = self.pq.decode(codes)

        # Compute error
        error = rotated - decoded

        # Learn rotation from error (simplified)
        # In full OPQ, this uses more sophisticated optimization
        U, _ = np.linalg.qr(error.T)
        self.rotation_matrix = U[: vectors.shape[1], : vectors.shape[1]].T

    def rotate(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate vectors using the learned rotation matrix.

        Args:
            vectors: Vectors to rotate (N x dim).

        Returns:
            Rotated vectors.
        """
        if self.rotation_matrix is None:
            raise RuntimeError("Encoder not trained. Call train() first.")

        return vectors @ self.rotation_matrix.T

    def inverse_rotate(self, vectors: np.ndarray) -> np.ndarray:
        """Inverse rotate vectors.

        Args:
            vectors: Rotated vectors (N x dim).

        Returns:
            Original vectors.
        """
        if self.rotation_matrix is None:
            raise RuntimeError("Encoder not trained. Call train() first.")

        return vectors @ self.rotation_matrix

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors using OPQ.

        Args:
            vectors: Vectors to encode (N x dim).

        Returns:
            PQ codes (N x m).
        """
        if not self._is_trained:
            raise RuntimeError("Encoder not trained. Call train() first.")

        rotated = self.rotate(vectors)
        return self.pq.encode(rotated)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode PQ codes back to original vectors.

        Args:
            codes: PQ codes (N x m).

        Returns:
            Reconstructed vectors (N x dim).
        """
        if not self._is_trained:
            raise RuntimeError("Encoder not trained. Call train() first.")

        decoded_rotated = self.pq.decode(codes)
        return self.inverse_rotate(decoded_rotated)


class ScalarQuantizer:
    """Scalar quantizer for simple value quantization.

    Supports 8-bit and 16-bit quantization.
    """

    def __init__(self, bits: int = 8):
        """Initialize scalar quantizer.

        Args:
            bits: Number of bits (8 or 16).
        """
        if bits not in (8, 16):
            raise ValueError("bits must be 8 or 16")

        self.bits = bits
        self.scale: float | None = None
        self.zero_point: float | None = None

    def train(self, vectors: np.ndarray) -> None:
        """Compute quantization parameters.

        Args:
            vectors: Training vectors.
        """
        vectors = np.asarray(vectors, dtype=np.float32)

        # Compute min/max for symmetric quantization
        vmin = vectors.min()
        vmax = vectors.max()

        # Symmetric quantization around zero
        abs_max = max(abs(vmin), abs(vmax))
        self.scale = abs_max / (2 ** (self.bits - 1))
        self.zero_point = 0.0

        logger.info(
            "Scalar quantizer trained: bits=%d, scale=%.6f", self.bits, self.scale
        )

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Quantize vectors to integers.

        Args:
            vectors: Vectors to quantize.

        Returns:
            Quantized integers.
        """
        if self.scale is None:
            raise RuntimeError("Quantizer not trained. Call train() first.")

        scaled = vectors / self.scale
        return np.round(scaled).astype(np.int8 if self.bits == 8 else np.int16)

    def decode(self, quantized: np.ndarray) -> np.ndarray:
        """Dequantize vectors.

        Args:
            quantized: Quantized integers.

        Returns:
            Dequantized vectors.
        """
        if self.scale is None:
            raise RuntimeError("Quantizer not trained. Call train() first.")

        return quantized.astype(np.float32) * self.scale


def create_quantizer(
    quantizer_type: str = "pq", **kwargs
) -> PQEncoder | OPQEncoder | ScalarQuantizer:
    """Create a quantizer by type.

    Args:
        quantizer_type: Type of quantizer ("pq", "opq", "scalar").
        **kwargs: Arguments passed to quantizer constructor.

    Returns:
        Quantizer instance.
    """
    if quantizer_type == "pq":
        return PQEncoder(**kwargs)
    if quantizer_type == "opq":
        return OPQEncoder(**kwargs)
    if quantizer_type == "scalar":
        return ScalarQuantizer(**kwargs)
    raise ValueError(f"Unknown quantizer type: {quantizer_type}")
