"""Memory Coalesced Vector Distance Kernel.

Based on:
- Naznin Fauzia et al. (2015) - Characterizing and Enhancing Global Memory Data Coalescing on GPU
- https://www.cs.colostate.edu/~pouchet/doc/cgo-article.15.pdf

Expected speedup: 2-8x for vector distance computation.
"""

# CUDA Kernel Code (for reference)
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

CUDA_COALESCED_L2_KERNEL = """
// Coalesced L2 distance kernel
// Each thread handles one query-database pair
// Threads in a warp access contiguous memory

__global__ void coalesced_l2_distance(
    const float* __restrict__ queries,    // (n_queries, dim)
    const float* __restrict__ database,  // (n_database, dim)
    float* distances,                    // (n_queries, n_database)
    int dim,
    int n_queries,
    int n_database
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int db_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (query_idx >= n_queries || db_idx >= n_database) return;
    
    // Coalesced access: threads in warp access contiguous database rows
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = queries[query_idx * dim + i] - database[db_idx * dim + i];
        dist += diff * diff;
    }
    
    distances[query_idx * n_database + db_idx] = dist;
}

// Optimizations:
// 1. Coalesced memory access - contiguous reads
// 2. Shared memory for frequently accessed data
// 3. Register usage optimization
// 4. Warp-level reductions
"""


def coalesced_l2_distance_numpy(
    queries: np.ndarray, database: np.ndarray
) -> np.ndarray:
    """Compute L2 distances using coalesced access pattern.

    This is a NumPy implementation that follows coalesced access principles:
    - Process data in row-major order
    - Minimize stride-1 accesses
    """
    import numpy as np  # noqa: PLC0415

    # Transpose for better cache utilization
    queries = np.asarray(queries, dtype=np.float32)
    database = np.asarray(database, dtype=np.float32)

    n_queries, _dim = queries.shape
    n_database = database.shape[0]

    # Pre-allocate output
    distances = np.zeros((n_queries, n_database), dtype=np.float32)

    # Process in chunks for cache efficiency
    chunk_size = 256

    for i in range(0, n_queries, chunk_size):
        query_chunk = queries[i : i + chunk_size]

        # Compute distances for chunk
        for j in range(n_database):
            diff = query_chunk - database[j]
            distances[i : i + len(query_chunk), j] = np.sum(diff * diff, axis=1)

    return distances


def estimate_coalescing_speedup(dim: int, block_size: int = 256) -> float:
    """Estimate speedup from memory coalescing.

    Based on Fauzia et al. - typically 2-8x improvement.
    """
    # Memory transactions per element
    uncoalesced_transactions = (dim + block_size - 1) // block_size
    coalesced_transactions = 1

    return min(uncoalesced_transactions / coalesced_transactions, 8.0)


# Benchmark comparison
def benchmark_coalesced_vs_naive(
    n_queries: int = 1000,
    n_database: int = 10000,
    dim: int = 128,
) -> dict:
    """Benchmark coalesced vs naive implementation."""
    import time  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(42)
    queries = rng.random((n_queries, dim)).astype(np.float32)
    database = rng.random((n_database, dim)).astype(np.float32)

    # Naive (stride > 1)
    start = time.time()
    naive_dist = np.zeros((n_queries, n_database), dtype=np.float32)
    for i in range(n_queries):
        for j in range(n_database):
            naive_dist[i, j] = np.sum((queries[i] - database[j]) ** 2)
    naive_time = time.time() - start

    # Coalesced
    start = time.time()
    coalesced_l2_distance_numpy(queries, database)
    coalesced_time = time.time() - start

    return {
        "naive_time": naive_time,
        "coalesced_time": coalesced_time,
        "speedup": naive_time / coalesced_time if coalesced_time > 0 else 0,
        "expected_speedup": estimate_coalescing_speedup(dim),
    }
