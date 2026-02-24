"""
Accelerated operations module for zvec using FAISS and NumPy.

This module provides high-performance vector operations using:
- FAISS (Facebook AI Similarity Search) - fastest for large datasets
- NumPy with Accelerate (Apple's BLAS) - optimal for small/medium datasets

Usage:
    from zvec.accelerate import AcceleratedBackend, get_optimal_backend

    # Auto-detect best backend (FAISS > NumPy/Accelerate)
    backend = get_optimal_backend()
"""

from __future__ import annotations

import platform
from typing import Literal, Optional

import numpy as np

__all__ = [
    "FAISS_AVAILABLE",
    "AcceleratedBackend",
    "get_accelerate_info",
    "get_optimal_backend",
    "search_faiss",
    "search_numpy",
]

# Check what's available
FAISS_AVAILABLE = False
BACKEND_TYPE = "numpy"

# Try to import FAISS
try:
    import faiss

    FAISS_AVAILABLE = True
    BACKEND_TYPE = "faiss"
except ImportError:
    FAISS_AVAILABLE = False


def get_optimal_backend() -> str:
    """Get the optimal backend for the current platform."""
    return BACKEND_TYPE


def get_accelerate_info() -> dict:
    """Get information about available acceleration backends."""
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "backends": {
            "faiss": FAISS_AVAILABLE,
        },
        "selected": BACKEND_TYPE,
    }


class AcceleratedBackend:
    """
    Accelerated backend using FAISS for large-scale vector search.

    FAISS provides the fastest approximate nearest neighbor search,
    optimized for both CPU and GPU (NVIDIA).
    """

    def __init__(self, backend: Optional[str] = None):
        """
        Initialize accelerated backend.

        Args:
            backend: "faiss" or "numpy" (auto-detect if None)
        """
        self.backend = backend or get_optimal_backend()

        if self.backend not in ["faiss", "numpy"]:
            raise ValueError(f"Unknown backend: {self.backend}")

    @staticmethod
    def is_faiss_available() -> bool:
        """Check if FAISS is available."""
        return FAISS_AVAILABLE

    def create_index(
        self,
        dim: int,
        metric: Literal["L2", "IP"] = "L2",
        nlist: int = 100,
    ):
        """Create an index for vector search."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")

        if metric == "L2":
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        else:  # IP = inner product
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )

        return index

    def search(
        self,
        index,
        queries: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the index."""
        return index.search(queries.astype("float32"), k)

    def __repr__(self) -> str:
        return f"AcceleratedBackend(backend={self.backend}, faiss={FAISS_AVAILABLE})"


# Convenience functions
def search_faiss(
    queries: np.ndarray,
    database: np.ndarray,
    k: int = 10,
    nlist: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast vector search using FAISS.

    Args:
        queries: Query vectors (N x D)
        database: Database vectors (M x D)
        k: Number of nearest neighbors
        nlist: Number of clusters for IVF index

    Returns:
        Tuple of (distances, indices)
    """
    if not FAISS_AVAILABLE:
        raise RuntimeError("FAISS not available")

    dim = database.shape[1]

    # Create index (use IVF for large datasets)
    if len(database) > 10000 and nlist > 0:
        # Use IVF index for better performance on large datasets
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(nlist, len(database) // 10))
        index.train(database.astype("float32"))
    else:
        # Use flat index for small datasets
        index = faiss.IndexFlatL2(dim)

    index.add(database.astype("float32"))

    # Search
    return index.search(queries.astype("float32"), k)


def search_numpy(
    queries: np.ndarray,
    database: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vector search using NumPy with Accelerate (Apple's BLAS).

    This is very fast for small to medium datasets.

    Args:
        queries: Query vectors (N x D)
        database: Database vectors (M x D)
        k: Number of nearest neighbors

    Returns:
        Tuple of (distances, indices)
    """
    # Compute all pairwise L2 distances using matrix operations
    # ||q - d||^2 = ||q||^2 + ||d||^2 - 2*q.d
    q_norm = np.sum(queries**2, axis=1, keepdims=True)
    d_norm = np.sum(database**2, axis=1)
    distances = q_norm + d_norm - 2 * (queries @ database.T)

    # Get top-k
    indices = np.argpartition(distances, k - 1, axis=1)[:, :k]

    # Sort by distance
    row_idx = np.arange(len(queries))[:, None]
    sorted_dist = distances[row_idx, indices]
    sorted_idx = np.argsort(sorted_dist, axis=1)

    return np.take_along_axis(distances, indices, axis=1)[
        row_idx, sorted_idx
    ], np.take_along_axis(indices, sorted_idx, axis=1)


# Auto-initialize
_default_backend = AcceleratedBackend() if FAISS_AVAILABLE else None
