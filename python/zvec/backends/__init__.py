"""zvec.backends - Hardware detection and backend selection."""

from __future__ import annotations

from zvec.backends.detect import (
    CPP_CUVS_AVAILABLE,
    CUVS_AVAILABLE,
    FAISS_AVAILABLE,
    FAISS_CPU_AVAILABLE,
    FAISS_GPU_AVAILABLE,
    get_available_backends,
    get_backend_info,
    get_optimal_backend,
    is_gpu_available,
)
from zvec.backends.gpu import (
    GPUIndex,
    create_index,
    create_index_with_fallback,
)
from zvec.backends.unified import (
    UnifiedGpuIndex,
    select_backend,
)

__all__ = [
    "CPP_CUVS_AVAILABLE",
    "CUVS_AVAILABLE",
    "FAISS_AVAILABLE",
    "FAISS_CPU_AVAILABLE",
    "FAISS_GPU_AVAILABLE",
    "GPUIndex",
    "UnifiedGpuIndex",
    "create_index",
    "create_index_with_fallback",
    "get_available_backends",
    "get_backend_info",
    "get_optimal_backend",
    "is_gpu_available",
    "select_backend",
]
