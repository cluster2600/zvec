"""Optimized search functions for vector databases."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def asymmetric_distance_computation(
    queries: np.ndarray,
    codes: np.ndarray,
    distance_table: np.ndarray,
) -> np.ndarray:
    """Compute distances using Asymmetric Distance Computation (ADC).

    This is faster than symmetric distance computation because we only
    decode the database codes, not the queries.

    Args:
        queries: Query vectors (Q x dim).
        codes: PQ codes for database (N x m).
        distance_table: Precomputed distance table (Q x m x k).

    Returns:
        Distances (Q x N).
    """
    n_queries = queries.shape[0]
    n_codes = codes.shape[0]

    distances = np.zeros((n_queries, n_codes), dtype=np.float32)

    for i in range(codes.shape[1]):  # m sub-vectors
        distances += distance_table[:, i, codes[:, i]].T

    return distances


def compute_distance_table_fast(
    queries: np.ndarray,
    codebooks: np.ndarray,
) -> np.ndarray:
    """Compute distance table efficiently using matrix operations.

    Args:
        queries: Query vectors (Q x dim).
        codebooks: PQ codebooks (m x k x sub_dim).

    Returns:
        Distance table (Q x m x k).
    """
    n_queries, dim = queries.shape
    m = codebooks.shape[0]
    sub_dim = codebooks.shape[2]

    # Reshape queries
    queries_reshaped = queries.reshape(n_queries, m, sub_dim)

    # Compute distances for each sub-vector
    distance_table = np.zeros(
        (n_queries, m, codebooks.shape[1]), dtype=np.float32
    )

    for i in range(m):
        # Broadcasting: (Q, 1, sub_dim) - (1, k, sub_dim) -> (Q, k, sub_dim)
        diff = queries_reshaped[:, i:i+1, :] - codebooks[i:i+1, :, :]
        distance_table[:, i, :] = np.sum(diff ** 2, axis=2)

    return distance_table


def batch_search(
    queries: np.ndarray,
    database: np.ndarray,
    codes: np.ndarray,
    codebooks: np.ndarray,
    k: int = 10,
    batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform batched search for memory efficiency.

    Args:
        queries: Query vectors (Q x dim).
        database: Database vectors (N x dim).
        codes: PQ codes (N x m).
        codebooks: PQ codebooks (m x k x sub_dim).
        k: Number of nearest neighbors.
        batch_size: Number of queries to process at once.

    Returns:
        Tuple of (distances, indices).
    """
    n_queries = queries.shape[0]
    n_database = database.shape[0]

    all_distances = np.full((n_queries, n_database), np.inf, dtype=np.float32)

    # Process in batches
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch_queries = queries[start:end]

        # Compute distance table
        distance_table = compute_distance_table_fast(batch_queries, codebooks)

        # Compute all distances
        batch_distances = asymmetric_distance_computation(
            batch_queries, codes, distance_table
        )
        all_distances[start:end] = batch_distances

        logger.info(f"Processed {end}/{n_queries} queries")

    # Get top k for each query
    indices = np.argsort(all_distances, axis=1)[:, :k]
    distances = np.take_along_axis(all_distances, indices, axis=1)[:, :k]

    return distances, indices


def search_with_reranking(
    queries: np.ndarray,
    database: np.ndarray,
    codes: np.ndarray,
    codebooks: np.ndarray,
    k: int = 10,
    rerank_top: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Search with PQ and rerank top candidates using exact distances.

    Args:
        queries: Query vectors (Q x dim).
        database: Database vectors (N x dim).
        codes: PQ codes (N x m).
        codebooks: PQ codebooks (m x k x sub_dim).
        k: Number of nearest neighbors to return.
        rerank_top: Number of candidates to rerank exactly.

    Returns:
        Tuple of (distances, indices).
    """
    n_queries = queries.shape[0]
    n_database = database.shape[0]

    # Initial PQ search
    distance_table = compute_distance_table_fast(queries, codebooks)
    pq_distances = asymmetric_distance_computation(queries, codes, distance_table)

    # Get top candidates
    top_indices = np.argsort(pq_distances, axis=1)[:, :rerank_top]

    # Rerank with exact distances
    final_distances = np.zeros((n_queries, k), dtype=np.float32)
    final_indices = np.zeros((n_queries, k), dtype=np.int64)

    for i in range(n_queries):
        # Get candidates
        candidates = top_indices[i]
        candidate_vectors = database[candidates]

        # Compute exact L2 distances
        diff = candidate_vectors - queries[i]
        exact_distances = np.sum(diff ** 2, axis=1)

        # Sort by exact distance
        sorted_order = np.argsort(exact_distances)
        final_indices[i] = candidates[sorted_order[:k]]
        final_distances[i] = exact_distances[sorted_order[:k]]

    return final_distances, final_indices
