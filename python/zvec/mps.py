"""
Metal MPS (Apple Silicon) acceleration module for zvec.

This module provides GPU acceleration using Apple's Metal Performance Shaders (MPS)
for M-series Apple Silicon chips (M1/M2/M3/M4).

Usage:
    from zvec.mps import MPSBackend, is_mps_available
    
    # Check MPS availability
    print(f"MPS available: {is_mps_available()}")
    
    # Create MPS-accelerated operations
    mps = MPSBackend()
"""

from __future__ import annotations

import platform
import sys
from typing import Literal, Optional

import numpy as np

__all__ = [
    'MPSBackend',
    'is_mps_available',
    'get_mps_info',
    'mps_vector_search',
    'mps_batch_distance',
]

# Check for MPS availability
def is_mps_available() -> bool:
    """Check if Metal Performance Shaders is available."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


def get_mps_info() -> dict:
    """Get detailed MPS device information."""
    info = {
        "available": False,
        "device_name": None,
        "device_count": 0,
        "torch_version": None,
    }
    
    if not is_mps_available():
        return info
    
    try:
        import torch
        info["available"] = True
        info["device_count"] = torch.mps.device_count()
        info["torch_version"] = torch.__version__
        
        # Try to get device name
        try:
            # MPS doesn't have a direct name property, but we can infer from platform
            info["device_name"] = f"Apple Silicon MPS (M-series)"
        except Exception:
            info["device_name"] = "Apple MPS"
            
    except ImportError:
        pass
    
    return info


class MPSBackend:
    """
    Metal Performance Shaders backend for Apple Silicon.
    
    Provides GPU-accelerated operations for:
    - Vector search (L2, cosine similarity)
    - Batch distance computation
    - Matrix operations
    """
    
    def __init__(self, device: int = 0):
        """
        Initialize MPS backend.
        
        Args:
            device: Device ID (default: 0)
        """
        if not is_mps_available():
            raise RuntimeError("Metal Performance Shaders not available")
        
        self.device = device
        self._torch = None
        self._mps = None
    
    def _get_torch(self):
        """Lazy load torch."""
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch
    
    def to_mps(self, array: np.ndarray) -> "torch.Tensor":
        """Convert numpy array to MPS tensor."""
        torch = self._get_torch()
        tensor = torch.from_numpy(array)
        return tensor.to('mps')
    
    def to_numpy(self, tensor: "torch.Tensor") -> np.ndarray:
        """Convert MPS tensor to numpy."""
        return tensor.cpu().numpy()
    
    def vector_search(
        self,
        queries: np.ndarray,
        database: np.ndarray,
        k: int = 10,
        metric: Literal["L2", "cosine"] = "L2",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated vector search.
        
        Args:
            queries: Query vectors (N x D)
            database: Database vectors (M x D)
            k: Number of nearest neighbors
            metric: Distance metric
            
        Returns:
            Tuple of (distances, indices)
        """
        torch = self._get_torch()
        
        # Convert to MPS tensors
        queries_tensor = self.to_mps(queries.astype(np.float32))
        database_tensor = self.to_mps(database.astype(np.float32))
        
        if metric == "L2":
            # L2 distance: ||q - d||^2 = ||q||^2 + ||d||^2 - 2*q.d
            queries_norm = torch.sum(queries_tensor ** 2, dim=1, keepdim=True)
            database_norm = torch.sum(database_tensor ** 2, dim=1, keepdim=True)
            
            # Compute distances using matrix multiplication
            distances = queries_norm + database_norm.T - 2 * torch.mm(queries_tensor, database_tensor.T)
            
        elif metric == "cosine":
            # Cosine similarity
            queries_norm = torch.nn.functional.normalize(queries_tensor, p=2, dim=1)
            database_norm = torch.nn.functional.normalize(database_tensor, p=2, dim=1)
            similarities = torch.mm(queries_norm, database_norm.T)
            distances = 1 - similarities  # Convert similarity to distance
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Get top-k
        topk_distances, topk_indices = torch.topk(distances, k, dim=1, largest=False)
        
        return self.to_numpy(topk_distances), self.to_numpy(topk_indices)
    
    def batch_distance(
        self,
        a: np.ndarray,
        b: np.ndarray,
        metric: Literal["L2", "cosine", "dot"] = "L2",
    ) -> np.ndarray:
        """
        Compute batch distances between two sets of vectors.
        
        Args:
            a: First set (N x D)
            b: Second set (M x D)
            metric: Distance metric
            
        Returns:
            Distance matrix (N x M)
        """
        torch = self._get_torch()
        
        a_tensor = self.to_mps(a.astype(np.float32))
        b_tensor = self.to_mps(b.astype(np.float32))
        
        if metric == "L2":
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            a_norm = torch.sum(a_tensor ** 2, dim=1, keepdim=True)
            b_norm = torch.sum(b_tensor ** 2, dim=1, keepdim=True)
            distances = a_norm + b_norm.T - 2 * torch.mm(a_tensor, b_tensor.T)
            
        elif metric == "cosine":
            a_norm = torch.nn.functional.normalize(a_tensor, p=2, dim=1)
            b_norm = torch.nn.functional.normalize(b_tensor, p=2, dim=1)
            similarities = torch.mm(a_norm, b_norm.T)
            distances = 1 - similarities
            
        elif metric == "dot":
            distances = -torch.mm(a_tensor, b_tensor.T)
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return self.to_numpy(distances)
    
    def batch_matrix_multiply(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        GPU-accelerated matrix multiplication.
        
        Args:
            a: Matrix A (N x K)
            b: Matrix B (K x M)
            
        Returns:
            Result (N x M)
        """
        torch = self._get_torch()
        
        a_tensor = self.to_mps(a.astype(np.float32))
        b_tensor = self.to_mps(b.astype(np.float32))
        
        result = torch.mm(a_tensor, b_tensor)
        
        return self.to_numpy(result)
    
    def __repr__(self) -> str:
        info = get_mps_info()
        return f"MPSBackend(available={info['available']}, device={self.device})"


# Convenience functions
def mps_vector_search(queries, database, k=10, metric="L2"):
    """Quick vector search using MPS."""
    backend = MPSBackend()
    return backend.vector_search(queries, database, k=k, metric=metric)


def mps_batch_distance(a, b, metric="L2"):
    """Quick batch distance using MPS."""
    backend = MPSBackend()
    return backend.batch_distance(a, b, metric=metric)


# Demo / benchmark
if __name__ == "__main__":
    print("=== MPS Information ===")
    info = get_mps_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    if info["available"]:
        print("\n=== MPS Benchmark ===")
        import time
        
        mps = MPSBackend()
        
        # Benchmark vector search
        np.random.seed(42)
        database = np.random.rand(10000, 128).astype(np.float32)
        queries = np.random.rand(100, 128).astype(np.float32)
        
        # Warmup
        _ = mps.vector_search(queries[:1], database[:100], k=10)
        
        # Benchmark
        start = time.perf_counter()
        distances, indices = mps.vector_search(queries, database, k=10, metric="L2")
        mps_time = time.perf_counter() - start
        
        # CPU comparison
        start = time.perf_counter()
        distances_cpu, indices_cpu = mps.vector_search(queries, database, k=10, metric="L2")
        cpu_time = time.perf_counter() - start
        
        print(f"  MPS time: {mps_time*1000:.1f}ms")
        print(f"  CPU time: {cpu_time*1000:.1f}ms")
        print(f"  Speedup: {cpu_time/mps_time:.1f}x")
    else:
        print("\nMPS not available on this device")
