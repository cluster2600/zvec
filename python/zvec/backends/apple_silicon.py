"""Apple Silicon optimization using Accelerate framework and MPS."""

from __future__ import annotations

import logging
import platform
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Check for Apple Silicon
IS_APPLE_SILICON = platform.machine() == "arm64" and platform.system() == "Darwin"

# Try to import Accelerate
ACCELERATE_AVAILABLE = False
try:
    from accelerate import init_backend  # noqa: F401

    ACCELERATE_AVAILABLE = True
except ImportError:
    pass

# Try to import PyTorch MPS
MPS_AVAILABLE = False
if IS_APPLE_SILICON:
    try:
        import torch

        MPS_AVAILABLE = torch.backends.mps.is_available()
        if MPS_AVAILABLE:
            logger.info("Apple MPS (Metal Performance Shaders) available")
    except ImportError:
        pass


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return IS_APPLE_SILICON


def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available."""
    return MPS_AVAILABLE


def is_accelerate_available() -> bool:
    """Check if Accelerate framework is available."""
    return ACCELERATE_AVAILABLE


class AppleSiliconBackend:
    """Apple Silicon optimized backend for vector operations.

    Uses the following priority:
    1. PyTorch MPS (GPU)
    2. Accelerate (BLAS)
    3. NumPy (fallback)
    """

    def __init__(self, backend: str = "auto"):
        """Initialize Apple Silicon backend.

        Args:
            backend: Backend to use ("auto", "mps", "accelerate", "numpy").
        """
        self._backend = backend
        self._selected = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect the best available backend."""
        if self._backend == "auto":
            if MPS_AVAILABLE:
                return "mps"
            elif ACCELERATE_AVAILABLE:
                return "accelerate"
            else:
                return "numpy"
        return self._backend

    @property
    def backend(self) -> str:
        """Get selected backend."""
        return self._selected

    def matrix_multiply(
        self, a: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """Matrix multiplication.

        Args:
            a: First matrix (M x K).
            b: Second matrix (K x N).

        Returns:
            Result matrix (M x N).
        """
        if self._selected == "mps":
            return self._mps_matmul(a, b)
        elif self._selected == "accelerate":
            return self._accelerate_matmul(a, b)
        else:
            return a @ b

    def _mps_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication using PyTorch MPS."""
        import torch

        a_torch = torch.from_numpy(a).to("mps")
        b_torch = torch.from_numpy(b).to("mps")
        result = torch.mm(a_torch, b_torch)
        return result.cpu().numpy()

    def _accelerate_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication using Accelerate."""
        # Accelerate is already used by NumPy on Apple Silicon
        return a @ b

    def l2_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute L2 distance between row vectors.

        Args:
            a: First set of vectors (N x D).
            b: Second set of vectors (M x D).

        Returns:
            Distance matrix (N x M).
        """
        if self._selected == "mps":
            return self._mps_l2_distance(a, b)
        else:
            # NumPy implementation (already optimized with Accelerate)
            return self._numpy_l2_distance(a, b)

    def _mps_l2_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """L2 distance using PyTorch MPS."""
        import torch

        a_torch = torch.from_numpy(a).to("mps")
        b_torch = torch.from_numpy(b).to("mps")

        # Compute squared distances: ||a||^2 - 2*a.b + ||b||^2
        a_sq = torch.sum(a_torch ** 2, dim=1)
        b_sq = torch.sum(b_torch ** 2, dim=1)
        ab = torch.mm(a_torch, b_torch.T)

        distances = a_sq.unsqueeze(1) - 2 * ab + b_sq.unsqueeze(0)
        distances = torch.clamp(distances, min=0)  # Numerical stability
        return torch.sqrt(distances).cpu().numpy()

    def _numpy_l2_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """L2 distance using NumPy."""
        a_sq = np.sum(a ** 2, axis=1, keepdims=True)
        b_sq = np.sum(b ** 2, axis=1)
        ab = a @ b.T
        distances = a_sq + b_sq - 2 * ab
        distances = np.clip(distances, 0, None)  # Numerical stability
        return np.sqrt(distances)

    def search_knn(
        self, queries: np.ndarray, database: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search k-nearest neighbors.

        Args:
            queries: Query vectors (Q x D).
            database: Database vectors (N x D).
            k: Number of neighbors.

        Returns:
            Tuple of (distances, indices).
        """
        distances = self.l2_distance(queries, database)
        indices = np.argsort(distances, axis=1)[:, :k]
        distances = np.take_along_axis(distances, indices, axis=1)
        return distances, indices

    def batch_search_knn(
        self,
        queries: np.ndarray,
        database: np.ndarray,
        k: int = 10,
        batch_size: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch search for memory efficiency.

        Args:
            queries: Query vectors (Q x D).
            database: Database vectors (N x D).
            k: Number of neighbors.
            batch_size: Batch size for queries.

        Returns:
            Tuple of (distances, indices).
        """
        n_queries = queries.shape[0]
        all_distances = []

        for i in range(0, n_queries, batch_size):
            batch = queries[i : i + batch_size]
            distances = self.l2_distance(batch, database)
            all_distances.append(distances)

        all_distances = np.vstack(all_distances)
        indices = np.argsort(all_distances, axis=1)[:, :k]
        distances = np.take_along_axis(all_distances, indices, axis=1)
        return distances, indices


def get_apple_silicon_backend(backend: str = "auto") -> AppleSiliconBackend:
    """Get Apple Silicon optimized backend.

    Args:
        backend: Backend to use ("auto", "mps", "accelerate", "numpy").

    Returns:
        AppleSiliconBackend instance.
    """
    return AppleSiliconBackend(backend=backend)


def get_available_backends() -> dict[str, bool]:
    """Get available backends on this system.

    Returns:
        Dictionary of available backends.
    """
    return {
        "apple_silicon": IS_APPLE_SILICON,
        "mps": MPS_AVAILABLE,
        "accelerate": ACCELERATE_AVAILABLE,
    }
