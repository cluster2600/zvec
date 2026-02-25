"""Hardware detection and backend selection for zvec."""

from __future__ import annotations

import logging
import platform
import sys

logger = logging.getLogger(__name__)

# Try to import FAISS
FAISS_AVAILABLE = False
FAISS_GPU_AVAILABLE = False
FAISS_CPU_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
    FAISS_CPU_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore[assignment]

# Check for GPU support
if FAISS_AVAILABLE:
    try:
        # Try to create a GPU resources to check if CUDA is available
        resources = faiss.StandardGpuResources()
        FAISS_GPU_AVAILABLE = True
    except Exception:
        FAISS_GPU_AVAILABLE = False

# Try to detect NVIDIA GPU
NVIDIA_GPU_DETECTED = False

if FAISS_GPU_AVAILABLE:
    try:
        # Additional check using nvidia-smi if available
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            NVIDIA_GPU_DETECTED = True
            logger.info("NVIDIA GPU detected: %s", result.stdout.strip())
    except FileNotFoundError:
        # nvidia-smi not found, but FAISS GPU is available
        NVIDIA_GPU_DETECTED = True
    except Exception:
        pass

# Try to detect Apple Silicon
APPLE_SILICON = platform.machine() == "arm64" and platform.system() == "Darwin"

# Try to detect AMD GPU
AMD_GPU_DETECTED = False

# Check for MPS (Apple Silicon GPU)
MPS_AVAILABLE = False
if APPLE_SILICON:
    try:
        import torch

        MPS_AVAILABLE = torch.backends.mps.is_available()
        if MPS_AVAILABLE:
            logger.info("Apple MPS (Metal Performance Shaders) available")
    except ImportError:
        pass

# Try to detect cuVS (NVIDIA RAPIDS)
CUVS_AVAILABLE = False
try:
    import cuvs  # noqa: F401

    CUVS_AVAILABLE = True
    logger.info("cuVS (NVIDIA RAPIDS) available")
except ImportError:
    pass

# Try to detect C++ cuVS bindings (via _zvec pybind11)
CPP_CUVS_AVAILABLE = False
try:
    import _zvec

    CPP_CUVS_AVAILABLE = hasattr(_zvec, "create_cagra_float")
    if CPP_CUVS_AVAILABLE:
        logger.info("C++ cuVS bindings available (preferred path)")
except ImportError:
    pass


def get_available_backends() -> dict[str, bool]:
    """Return a dictionary of available backends.

    Returns:
        Dictionary with backend availability information.
    """
    return {
        "cpp_cuvs": CPP_CUVS_AVAILABLE,
        "cuvs": CUVS_AVAILABLE,
        "faiss": FAISS_AVAILABLE,
        "faiss_gpu": FAISS_GPU_AVAILABLE,
        "faiss_cpu": FAISS_CPU_AVAILABLE,
        "nvidia_gpu": NVIDIA_GPU_DETECTED,
        "amd_gpu": AMD_GPU_DETECTED,
        "apple_silicon": APPLE_SILICON,
        "mps": MPS_AVAILABLE,
    }


def get_optimal_backend() -> str:
    """Determine the optimal backend for the current system.

    Priority: C++ cuVS > Python cuVS > FAISS GPU > MPS > FAISS CPU > NumPy.

    Returns:
        Name of the optimal backend.
    """
    if CPP_CUVS_AVAILABLE:
        logger.info("Using C++ cuVS backend (native, preferred)")
        return "cpp_cuvs"

    if CUVS_AVAILABLE:
        logger.info("Using Python cuVS backend")
        return "cuvs"

    if FAISS_GPU_AVAILABLE and NVIDIA_GPU_DETECTED:
        logger.info("Using FAISS GPU backend")
        return "faiss_gpu"

    if MPS_AVAILABLE:
        logger.info("Using FAISS CPU with MPS fallback (Apple Silicon)")
        return "faiss_cpu"

    if FAISS_CPU_AVAILABLE:
        logger.info("Using FAISS CPU backend")
        return "faiss_cpu"

    logger.info("Using NumPy backend (fallback)")
    return "numpy"


def is_gpu_available() -> bool:
    """Check if a GPU is available for vector operations.

    Returns:
        True if GPU acceleration is available.
    """
    return CPP_CUVS_AVAILABLE or CUVS_AVAILABLE or FAISS_GPU_AVAILABLE or MPS_AVAILABLE


def get_backend_info() -> dict:
    """Get detailed information about the current backend.

    Returns:
        Dictionary with backend details.
    """
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": sys.version,
        "backends": get_available_backends(),
        "selected": get_optimal_backend(),
    }
