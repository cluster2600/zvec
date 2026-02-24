"""zvec.backends - Hardware detection and backend selection."""

from __future__ import annotations

from zvec.backends.detect import (
    FAISS_AVAILABLE,
    FAISS_CPU_AVAILABLE,
    FAISS_GPU_AVAILABLE,
    get_available_backends,
    get_backend_info,
    get_optimal_backend,
    is_gpu_available,
)

__all__ = [
    "FAISS_AVAILABLE",
    "FAISS_CPU_AVAILABLE",
    "FAISS_GPU_AVAILABLE",
    "get_available_backends",
    "get_backend_info",
    "get_optimal_backend",
    "is_gpu_available",
]
