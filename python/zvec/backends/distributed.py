"""Distributed vector database implementation."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ShardManager:
    """Manages vector sharding for distributed deployment.

    Supports different sharding strategies:
    - Hash-based (consistent hashing)
    - Range-based
    - Random
    """

    def __init__(
        self,
        n_shards: int = 4,
        strategy: str = "hash",
        replication_factor: int = 1,
    ):
        """Initialize shard manager.

        Args:
            n_shards: Number of shards.
            strategy: Sharding strategy ("hash", "range", "random").
            replication_factor: Number of replicas per vector.
        """
        self.n_shards = n_shards
        self.strategy = strategy
        self.replication_factor = replication_factor
        self._shards: dict[int, list[np.ndarray]] = {}

    def _hash_key(self, key: str) -> int:
        """Compute hash for a key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.n_shards

    def get_shard(self, vector_id: str | int) -> int:
        """Get shard index for a vector.

        Args:
            vector_id: Unique vector identifier.

        Returns:
            Shard index.
        """
        key = str(vector_id)

        if self.strategy == "hash":
            return self._hash_key(key)
        if self.strategy == "random":
            return hash(key) % self.n_shards
        # Range-based
        return int(vector_id) % self.n_shards

    def get_shard_for_query(self, query: np.ndarray) -> list[int]:  # noqa: ARG002
        """Get shards to query for a search.

        For full search, returns all shards.
        For approximate search, can return subset.

        Args:
            query: Query vector.

        Returns:
            List of shard indices to query.
        """
        return list(range(self.n_shards))

    def add_vector(
        self, vector: np.ndarray, vector_id: str | int
    ) -> None:
        """Add a vector to the appropriate shard.

        Args:
            vector: Vector to add.
            vector_id: Unique vector identifier.
        """
        shard = self.get_shard(vector_id)
        if shard not in self._shards:
            self._shards[shard] = []
        self._shards[shard].append(vector)

    def get_shard_vectors(self, shard: int) -> list[np.ndarray]:
        """Get all vectors in a shard.

        Args:
            shard: Shard index.

        Returns:
            List of vectors in the shard.
        """
        return self._shards.get(shard, [])


class DistributedIndex:
    """Distributed vector index across multiple shards.

    Provides:
    - Sharding
    - Scatter-gather query processing
    - Result merging
    """

    def __init__(
        self,
        n_shards: int = 4,
        sharding_strategy: str = "hash",
        replication_factor: int = 1,
    ):
        """Initialize distributed index.

        Args:
            n_shards: Number of shards.
            sharding_strategy: Strategy for distributing vectors.
            replication_factor: Number of replicas.
        """
        self.shard_manager = ShardManager(
            n_shards=n_shards,
            strategy=sharding_strategy,
            replication_factor=replication_factor,
        )
        self.n_shards = n_shards
        self._local_indexes: dict[int, Any] = {}

    def add(
        self,
        vectors: np.ndarray,
        vector_ids: list[str | int] | None = None,
    ) -> None:
        """Add vectors to the index.

        Args:
            vectors: Vectors to add (N x dim).
            vector_ids: Optional unique IDs for vectors.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        n_vectors = vectors.shape[0]

        if vector_ids is None:
            vector_ids = list(range(n_vectors))

        # Distribute vectors to shards
        for _i, (vector, vid) in enumerate(zip(vectors, vector_ids, strict=False)):
            shard = self.shard_manager.get_shard(vid)
            if shard not in self._local_indexes:
                self._local_indexes[shard] = []
            self._local_indexes[shard].append((vid, vector))

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        shards_to_search: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors across shards.

        Uses scatter-gather pattern:
        1. Scatter: Send query to all relevant shards
        2. Gather: Collect and merge results

        Args:
            query: Query vector (1 x dim).
            k: Number of neighbors to return.
            shards_to_search: Optional list of shards to search.

        Returns:
            Tuple of (distances, indices).
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if shards_to_search is None:
            shards_to_search = self.shard_manager.get_shard_for_query(query)

        all_results: list[tuple[float, int, int]] = []  # (distance, shard, index)

        # Search each shard
        for shard in shards_to_search:
            if shard not in self._local_indexes:
                continue

            vectors = self._local_indexes[shard]
            if not vectors:
                continue

            # Compute distances in shard
            db = np.array([v for _, v in vectors])
            distances = np.linalg.norm(db - query[0], axis=1)

            # Get top k from this shard
            top_k_idx = np.argsort(distances)[:k]
            for idx in top_k_idx:
                vid, _ = vectors[idx]
                all_results.append((distances[idx], shard, vid))

        # Merge and get global top k
        all_results.sort(key=lambda x: x[0])
        top_results = all_results[:k]

        distances = np.array([d for d, _, _ in top_results], dtype=np.float32)
        indices = np.array([v for _, _, v in top_results], dtype=np.int64)

        return distances, indices


class QueryRouter:
    """Routes queries to appropriate shards.

    Supports:
    - Full search (all shards)
    - Selective search (subset of shards)
    - Routing based on query characteristics
    """

    def __init__(self, shard_manager: ShardManager):
        """Initialize query router.

        Args:
            shard_manager: ShardManager instance.
        """
        self.shard_manager = shard_manager

    def route_query(
        self,
        query: np.ndarray,  # noqa: ARG002
        strategy: str = "all",
    ) -> list[int]:
        """Route query to appropriate shards.

        Args:
            query: Query vector.
            strategy: Routing strategy ("all", "random", "local_first").

        Returns:
            List of shard indices to search.
        """
        if strategy == "all":
            return list(range(self.shard_manager.n_shards))
        if strategy == "random":
            import random  # noqa: PLC0415
            n = max(1, self.shard_manager.n_shards // 2)
            return random.sample(range(self.shard_manager.n_shards), n)
        return list(range(self.shard_manager.n_shards))


class ResultMerger:
    """Merges results from multiple shards.

    Supports different merge strategies:
    - Score-based (simple concatenation and sort)
    - Distributed scoring
    """

    @staticmethod
    def merge_knn(
        shard_results: list[tuple[np.ndarray, np.ndarray]],
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merge k-NN results from multiple shards.

        Args:
            shard_results: List of (distances, indices) tuples from shards.
            k: Number of results to return.

        Returns:
            Merged (distances, indices).
        """
        all_distances = []
        all_indices = []

        for distances, indices in shard_results:
            all_distances.append(distances)
            all_indices.append(indices)

        if not all_distances:
            return np.array([]), np.array([])

        # Concatenate all results
        all_distances = np.concatenate(all_distances)
        all_indices = np.concatenate(all_indices)

        # Get top k
        top_k_idx = np.argsort(all_distances)[:k]

        return all_distances[top_k_idx], all_indices[top_k_idx]


def create_distributed_index(
    n_shards: int = 4,
    sharding_strategy: str = "hash",
) -> DistributedIndex:
    """Create a distributed index.

    Args:
        n_shards: Number of shards.
        sharding_strategy: Sharding strategy.

    Returns:
        DistributedIndex instance.
    """
    return DistributedIndex(
        n_shards=n_shards,
        sharding_strategy=sharding_strategy,
    )
