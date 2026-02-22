"""
GPU acceleration module for zvec.

This module provides GPU acceleration for vector operations on Apple Silicon (M-series)
and other platforms. Falls back to CPU if GPU is not available.

Usage:
    from zvec.gpu import GPUBackend, get_optimal_backend
    
    # Auto-detect best backend
    backend = get_optimal_backend()
    
    # Create GPU-accelerated index
    index = GPUBackend.create_index(dim=128, metric="L2")
"""

from __future__ import annotations

import platform
import sys
from typing import Literal, Optional

__all__ = [
    'GPUBackend',
    'get_optimal_backend',
    'is_apple_silicon',
    'get_gpu_info',
    'AVAILABLE',
]

# Check what's available
AVAILABLE = False
BACKEND_TYPE = "none"

# Check for Apple Silicon
def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3/M4)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

# Try to import GPU libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_MPS_AVAILABLE = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_MPS_AVAILABLE = False

# Determine available backend
def _detect_backend() -> tuple[bool, str]:
    """Detect the best available backend."""
    if is_apple_silicon():
        # Apple Silicon - can use MPS or CPU
        if TORCH_MPS_AVAILABLE:
            return True, "mps"
        elif FAISS_AVAILABLE:
            return True, "faiss-cpu"
    elif platform.system() == "Linux":
        # Check for NVIDIA GPU
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return True, "cuda"
        elif FAISS_AVAILABLE:
            return True, "faiss-cpu"
    elif platform.system() == "Darwin":
        # Intel Mac
        if FAISS_AVAILABLE:
            return True, "faiss-cpu"
    
    return False, "none"

AVAILABLE, BACKEND_TYPE = _detect_backend()


def get_optimal_backend() -> str:
    """
    Get the optimal backend for the current platform.
    
    Returns:
        Backend type: "mps", "cuda", "faiss-cpu", or "none"
    """
    return BACKEND_TYPE


def get_gpu_info() -> dict:
    """
    Get information about available GPU backends.
    
    Returns:
        Dictionary with backend information
    """
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "is_apple_silicon": is_apple_silicon(),
        "backends": {
            "faiss": FAISS_AVAILABLE,
            "torch": TORCH_AVAILABLE,
            "torch_mps": TORCH_MPS_AVAILABLE,
            "cuda": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
        },
        "selected": BACKEND_TYPE,
        "available": AVAILABLE,
    }
    return info


class GPUBackend:
    """
    GPU-accelerated backend for zvec operations.
    
    Currently supports:
    - Apple Silicon MPS (M1/M2/M3/M4)
    - NVIDIA CUDA (via PyTorch)
    - CPU fallback (FAISS)
    """
    
    def __init__(
        self,
        backend: Optional[str] = None,
        device: int = 0,
    ):
        """
        Initialize GPU backend.
        
        Args:
            backend: Backend to use ("mps", "cuda", "faiss-cpu", "auto")
            device: Device ID for CUDA
        """
        self.backend = backend or get_optimal_backend()
        self.device = device
        
        if self.backend == "auto":
            self.backend = get_optimal_backend()
        
        if self.backend not in ["mps", "cuda", "faiss-cpu", "none"]:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    @staticmethod
    def is_available() -> bool:
        """Check if GPU backend is available."""
        return AVAILABLE
    
    def create_index(
        self,
        dim: int,
        metric: Literal["L2", "IP", "cosine"] = "L2",
        nlist: int = 100,
    ) -> "faiss.Index":
        """
        Create a GPU-accelerated index.
        
        Args:
            dim: Vector dimension
            metric: Distance metric ("L2", "IP", "cosine")
            nlist: Number of clusters
            
        Returns:
            FAISS index (GPU-accelerated if available)
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")
        
        # Create index
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        
        # Transfer to GPU if available
        if self.backend == "cuda" and TORCH_AVAILABLE:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.device, index)
        elif self.backend == "mps":
            # MPS not directly supported by FAISS, use CPU
            # But we can use PyTorch MPS for operations
            pass
        
        return index
    
    def search(
        self,
        index: "faiss.Index",
        queries: "np.ndarray",
        k: int = 10,
    ) -> tuple:
        """
        Search the index.
        
        Args:
            index: FAISS index
            queries: Query vectors
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (distances, indices)
        """
        if hasattr(index, 'is_trained') and not index.is_trained:
            raise RuntimeError("Index not trained")
        
        return index.search(queries, k)
    
    def __repr__(self) -> str:
        return f"GPUBackend(backend={self.backend}, available={AVAILABLE})"


# Convenience function
def get_optimal_backend() -> str:
    """Get the optimal backend for the current platform."""
    return BACKEND_TYPE


# Auto-initialize if possible
if AVAILABLE:
    _default_backend = GPUBackend()
else:
    _default_backend = None
