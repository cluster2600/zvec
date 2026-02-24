"""Benchmark script for comparing CPU vs GPU performance."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_random_vectors(
    n_vectors: int, dim: int, seed: int = 42
) -> np.ndarray:
    """Generate random vectors for benchmarking.

    Args:
        n_vectors: Number of vectors to generate.
        dim: Dimensionality of vectors.
        seed: Random seed.

    Returns:
        Random vectors as numpy array.
    """
    np.random.seed(seed)
    return np.random.random((n_vectors, dim)).astype(np.float32)


def benchmark_numpy(
    database: np.ndarray, queries: np.ndarray, k: int = 10
) -> dict[str, Any]:
    """Benchmark using NumPy (brute force).

    Args:
        database: Database vectors.
        queries: Query vectors.
        k: Number of neighbors.

    Returns:
        Dictionary with timing results.
    """
    # Compute pairwise distances
    start = time.perf_counter()
    distances = np.linalg.norm(
        database[np.newaxis, :, :] - queries[:, np.newaxis, :], axis=2
    )
    # Get k nearest
    np.argsort(distances, axis=1)[:, :k]
    end = time.perf_counter()

    return {
        "backend": "numpy",
        "time": end - start,
        "queries_per_second": len(queries) / (end - start),
    }


def benchmark_faiss_cpu(
    database: np.ndarray, queries: np.ndarray, k: int = 10
) -> dict[str, Any]:
    """Benchmark using FAISS CPU.

    Args:
        database: Database vectors.
        queries: Query vectors.
        k: Number of neighbors.

    Returns:
        Dictionary with timing results.
    """
    try:
        import faiss

        # Create index
        dim = database.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(database)

        # Search
        start = time.perf_counter()
        _distances, _indices = index.search(queries, k)
        end = time.perf_counter()

        return {
            "backend": "faiss-cpu",
            "time": end - start,
            "queries_per_second": len(queries) / (end - start),
        }
    except ImportError:
        logger.warning("FAISS CPU not available")
        return None


def benchmark_faiss_gpu(
    database: np.ndarray, queries: np.ndarray, k: int = 10
) -> dict[str, Any]:
    """Benchmark using FAISS GPU.

    Args:
        database: Database vectors.
        queries: Query vectors.
        k: Number of neighbors.

    Returns:
        Dictionary with timing results.
    """
    try:
        import faiss

        # Create GPU index
        dim = database.shape[1]
        index = faiss.IndexFlatL2(dim)
        gpu_resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
        index.add(database)

        # Search
        start = time.perf_counter()
        _distances, _indices = index.search(queries, k)
        end = time.perf_counter()

        del gpu_resources

        return {
            "backend": "faiss-gpu",
            "time": end - start,
            "queries_per_second": len(queries) / (end - start),
        }
    except Exception as e:
        logger.warning(f"FAISS GPU not available: {e}")
        return None


def run_benchmarks(
    n_vectors: int,
    dim: int = 128,
    n_queries: int = 100,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Run all benchmarks.

    Args:
        n_vectors: Number of vectors in database.
        dim: Vector dimensionality.
        n_queries: Number of query vectors.
        k: Number of neighbors to search.

    Returns:
        List of benchmark results.
    """
    logger.info(
        f"Generating data: {n_vectors:,} vectors, dim={dim}, {n_queries} queries"
    )

    database = generate_random_vectors(n_vectors, dim)
    queries = generate_random_vectors(n_queries, dim, seed=123)

    results = []

    # NumPy
    logger.info("Running NumPy benchmark...")
    result = benchmark_numpy(database, queries, k)
    results.append(result)
    logger.info(f"  NumPy: {result['time']:.4f}s")

    # FAISS CPU
    result = benchmark_faiss_cpu(database, queries, k)
    if result:
        results.append(result)
        logger.info(f"  FAISS CPU: {result['time']:.4f}s")

    # FAISS GPU
    result = benchmark_faiss_gpu(database, queries, k)
    if result:
        results.append(result)
        logger.info(f"  FAISS GPU: {result['time']:.4f}s")

    return results


def print_results(results: list[dict[str, Any]]) -> None:
    """Print benchmark results in a table.

    Args:
        results: List of benchmark results.
    """

    baseline = None
    for r in results:
        if baseline is None:
            baseline = r["time"]
        else:
            f"{baseline / r['time']:.1f}x"




def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark vector search performance"
    )
    parser.add_argument(
        "--vectors",
        type=int,
        default=100000,
        help="Number of vectors in database (default: 100000)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Vector dimensionality (default: 128)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of query vectors (default: 100)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors (default: 10)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="10000,100000,1000000",
        help="Comma-separated list of sizes to benchmark",
    )

    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else [args.vectors]

    for n_vectors in sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with {n_vectors:,} vectors")
        logger.info(f"{'='*60}")

        results = run_benchmarks(
            n_vectors=n_vectors,
            dim=args.dim,
            n_queries=args.queries,
            k=args.k,
        )
        print_results(results)


if __name__ == "__main__":
    main()
