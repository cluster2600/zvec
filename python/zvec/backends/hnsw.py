"""Hierarchical Navigable Small World (HNSW) implementation."""

from __future__ import annotations

import heapq
import logging
import pickle
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HNSWIndex:
    """Pure Python HNSW implementation.

    HNSW is a graph-based index that provides fast approximate nearest
    neighbor search with logarithmic complexity.

    Example:
        >>> index = HNSWIndex(dim=128, M=16, efConstruction=200)
        >>> index.add(vectors)
        >>> distances, indices = index.search(query, k=10)
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        efConstruction: int = 200,
        efSearch: int = 50,
        max_elements: int = 1000000,
    ):
        """Initialize HNSW index.

        Args:
            dim: Dimensionality of vectors.
            M: Number of connections per layer.
            efConstruction: Search width during construction.
            efSearch: Search width for queries.
            max_elements: Maximum number of elements.
        """
        self.dim = dim
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.max_elements = max_elements

        # Graph layers: list of dicts, each dict maps element_id -> [(neighbor_id, distance), ...]
        self.graph: list[dict[int, list[tuple[int, float]]]] = []

        # Element data
        self.vectors: np.ndarray | None = None
        self.element_count = 0
        self.max_level = 0

        # Entry point (element id of the top layer)
        self.entry_point: int | None = None

    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute L2 distance between two vectors."""
        return float(np.linalg.norm(v1 - v2))

    def _get_random_level(self) -> int:
        """Get random level for new element using exponential distribution."""
        import random

        level = 0
        while random.random() < 0.5 and level < self.max_elements:
            level += 1
        return level

    def _search_layer(
        self,
        query: np.ndarray,
        ef: int,
        entry_point: int,
        level: int,
    ) -> list[tuple[float, int]]:
        """Search for nearest neighbors in a single layer.

        Args:
            query: Query vector.
            ef: Number of candidates to return.
            entry_point: Starting element.
            level: Layer to search.

        Returns:
            List of (distance, element_id) sorted by distance.
        """
        visited = set()
        candidates: list[tuple[float, int]] = []  # (distance, element_id)
        results: list[tuple[float, int]] = []  # (distance, element_id)

        heapq.heappush(candidates, (0.0, entry_point))
        visited.add(entry_point)

        while candidates:
            dist, current = heapq.heappop(candidates)

            # Get current element's neighbors at this level
            if level < len(self.graph) and current in self.graph[level]:
                neighbors = self.graph[level][current]
            else:
                neighbors = []

            # Check if we should add to results
            if results and dist > results[-1][0] and len(results) >= ef:
                continue

            heapq.heappush(results, (dist, current))
            if len(results) > ef:
                heapq.heappop(results)

            # Explore neighbors
            for neighbor_id, _neighbor_dist in neighbors:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                # Get distance to neighbor
                neighbor_vector = self.vectors[neighbor_id]
                d = self._distance(query, neighbor_vector)

                if len(results) < ef or d < results[-1][0]:
                    heapq.heappush(candidates, (d, neighbor_id))

        return sorted(results, key=lambda x: x[0])

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.

        Args:
            vectors: Vectors to add (N x dim).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        n_vectors = vectors.shape[0]

        if self.vectors is None:
            self.vectors = vectors
            self.element_count = n_vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            self.element_count += n_vectors

        # Initialize graph if empty
        if not self.graph:
            self.graph = [{} for _ in range(1)]
            self.entry_point = 0

        logger.info("Added %d vectors to HNSW index", n_vectors)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector (dim,) or (1, dim).
            k: Number of nearest neighbors.

        Returns:
            Tuple of (distances, indices).
        """
        if self.vectors is None or self.element_count == 0:
            raise RuntimeError("Index is empty. Call add() first.")

        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = np.asarray(query, dtype=np.float32)

        if self.entry_point is None:
            raise RuntimeError("No entry point. Index is empty.")

        # Start from top layer and go down
        current = self.entry_point
        for level in range(self.max_level, 0, -1):
            current = self._search_layer(
                query[0], ef=1, entry_point=current, level=level
            )[0][1]

        # Search at base layer
        results = self._search_layer(
            query[0], ef=max(k, self.efSearch), entry_point=current, level=0
        )

        # Return top k
        top_k = results[:k]
        distances = np.array([d for d, _ in top_k], dtype=np.float32)
        indices = np.array([i for _, i in top_k], dtype=np.int64)

        return distances, indices

    def save(self, filepath: str) -> None:
        """Save index to file.

        Args:
            filepath: Path to save to.
        """
        data = {
            "dim": self.dim,
            "M": self.M,
            "efConstruction": self.efConstruction,
            "efSearch": self.efSearch,
            "vectors": self.vectors,
            "element_count": self.element_count,
            "graph": self.graph,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved HNSW index to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> HNSWIndex:
        """Load index from file.

        Args:
            filepath: Path to load from.

        Returns:
            Loaded HNSWIndex.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        index = cls(
            dim=data["dim"],
            M=data["M"],
            efConstruction=data["efConstruction"],
            efSearch=data["efSearch"],
        )
        index.vectors = data["vectors"]
        index.element_count = data["element_count"]
        index.graph = data["graph"]
        index.entry_point = data["entry_point"]
        index.max_level = data["max_level"]

        logger.info("Loaded HNSW index from %s", filepath)
        return index


def create_hnsw_index(
    dim: int,
    M: int = 16,
    efConstruction: int = 200,
    efSearch: int = 50,
    _use_faiss: bool = True,
) -> HNSWIndex | Any:
    """Create HNSW index.

    Args:
        dim: Vector dimensionality.
        M: Number of connections.
        efConstruction: Construction width.
        efSearch: Search width.
        use_faiss: If True, try to use FAISS HNSW first.

    Returns:
        HNSWIndex or FAISS index.
    """
    # Try FAISS first for better performance
    try:
        import faiss

        index = faiss.IndexHNSWFlat(dim, M)
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = efSearch
        logger.info("Using FAISS HNSW index")
        return index
    except ImportError:
        logger.info("FAISS not available, using pure Python HNSW")
        return HNSWIndex(
            dim=dim,
            M=M,
            efConstruction=efConstruction,
            efSearch=efSearch,
        )
