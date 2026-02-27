"""Product Quantization (PQ) implementation for vector compression."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PQEncoder:
    """Product Quantization encoder.

    Splits vectors into sub-vectors and quantizes each independently
    using k-means clustering.
    """

    def __init__(self, m: int = 8, nbits: int = 8, k: int = 256):
        """Initialize PQ encoder.

        Args:
            m: Number of sub-vectors (subquantizers).
            nbits: Number of bits per sub-vector (code size = 2^nbits).
            k: Number of centroids per sub-vector.
        """
        self.m = m
        self.nbits = nbits
        self.k = min(k, 256)  # Cap at 256
        self.code_size = 1 << nbits  # 2^nbits
        self.codebooks: np.ndarray | None = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Check if encoder is trained."""
        return self._is_trained

    def train(self, vectors: np.ndarray) -> None:
        """Train the PQ encoder on vectors using k-means.

        Args:
            vectors: Training vectors (N x dim).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        n_vectors, dim = vectors.shape

        if dim % self.m != 0:
            raise ValueError(f"Dimension {dim} must be divisible by m={self.m}")

        # Adjust k if needed
        actual_k = min(self.k, max(1, n_vectors // 4))

        sub_dim = dim // self.m

        # Split vectors into sub-vectors
        sub_vectors = vectors.reshape(n_vectors, self.m, sub_dim)

        # Train k-means for each sub-vector
        self.codebooks = np.zeros((self.m, actual_k, sub_dim), dtype=np.float32)

        rng = np.random.default_rng(42)

        for i in range(self.m):
            sub = sub_vectors[:, i, :]
            # Initialize centroids randomly
            indices = rng.choice(n_vectors, actual_k, replace=False)
            centroids = sub[indices].copy()

            # K-means iterations
            for _ in range(10):
                # Assign to nearest centroid
                distances = np.linalg.norm(
                    sub[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
                )
                labels = np.argmin(distances, axis=1)

                # Update centroids
                new_centroids = np.zeros_like(centroids)
                counts = np.zeros(actual_k)
                for j in range(n_vectors):
                    c = labels[j]
                    new_centroids[c] += sub[j]
                    counts[c] += 1

                # Avoid division by zero
                counts = np.maximum(counts, 1)
                centroids = new_centroids / counts[:, np.newaxis]

            self.codebooks[i] = centroids
            self.k = actual_k  # Update to actual k used

        self._is_trained = True
        logger.info(
            "PQ trained: m=%d, nbits=%d, k=%d",
            self.m,
            self.nbits,
            actual_k,
        )

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors to PQ codes.

        Args:
            vectors: Vectors to encode (N x dim).

        Returns:
            PQ codes (N x m), each value is centroid index (0 to k-1).
        """
        if not self._is_trained:
            raise RuntimeError("Encoder not trained. Call train() first.")

        vectors = np.asarray(vectors, dtype=np.float32)
        n_vectors, dim = vectors.shape
        sub_dim = dim // self.m

        sub_vectors = vectors.reshape(n_vectors, self.m, sub_dim)
        codes = np.zeros((n_vectors, self.m), dtype=np.uint8)

        for i in range(self.m):
            sub = sub_vectors[:, i, :]
            # Find nearest centroid
            distances = np.linalg.norm(
                sub[:, np.newaxis, :] - self.codebooks[i][np.newaxis, :, :], axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode PQ codes back to vectors.

        Args:
            codes: PQ codes (N x m).

        Returns:
            Reconstructed vectors (N x dim).
        """
        if not self._is_trained:
            raise RuntimeError("Encoder not trained. Call train() first.")

        codes = np.asarray(codes, dtype=np.uint8)
        n_codes = codes.shape[0]
        dim = self.m * (self.codebooks.shape[2])

        # Look up centroids
        reconstructed = np.zeros((n_codes, self.m, dim // self.m), dtype=np.float32)
        for i in range(self.m):
            reconstructed[:, i, :] = self.codebooks[i][codes[:, i]]

        return reconstructed.reshape(n_codes, dim)


class PQIndex:
    """PQ index for fast approximate nearest neighbor search."""

    def __init__(self, m: int = 8, nbits: int = 8, k: int = 256):
        """Initialize PQ index.

        Args:
            m: Number of sub-vectors.
            nbits: Number of bits per sub-vector.
            k: Number of centroids per sub-vector.
        """
        self.encoder = PQEncoder(m=m, nbits=nbits, k=k)
        self.database: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.

        Args:
            vectors: Vectors to add (N x dim).
        """
        self.database = vectors
        self.encoder.train(vectors)
        self.codes = self.encoder.encode(vectors)

    def search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Args:
            queries: Query vectors (Q x dim).
            k: Number of nearest neighbors.

        Returns:
            Tuple of (distances, indices).
        """
        if self.database is None:
            raise RuntimeError("No vectors in index. Call add() first.")

        # Compute distances using decoded vectors
        n_queries = queries.shape[0]
        n_database = self.database.shape[0]

        # Simple brute force using decoded vectors
        self.encoder.decode(self.codes)

        all_distances = np.zeros((n_queries, n_database), dtype=np.float32)
        for i in range(n_queries):
            all_distances[i] = np.linalg.norm(self.database - queries[i], axis=1)

        # Get k nearest
        indices = np.argsort(all_distances, axis=1)[:, :k]
        distances = np.take_along_axis(all_distances, indices, axis=1)[:, :k]

        return distances, indices
