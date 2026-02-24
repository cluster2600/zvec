"""GPU-accelerated index implementations using FAISS."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from zvec.backends.detect import (
    FAISS_AVAILABLE,
    FAISS_GPU_AVAILABLE,
)

if TYPE_CHECKING:
    import faiss

logger = logging.getLogger(__name__)

# Lazy import FAISS
faiss: Any = None
if FAISS_AVAILABLE:
    import faiss as _faiss

    faiss = _faiss


class GPUIndex:
    """GPU-accelerated index wrapper for FAISS.

    This class provides a unified interface for creating and using
    GPU-accelerated indexes for vector similarity search.

    Example:
        >>> index = GPUIndex(dim=128, index_type="IVF", nlist=100)
        >>> index.add(vectors)
        >>> distances, indices = index.search(query_vectors, k=10)
    """

    def __init__(
        self,
        dim: int,
        index_type: Literal["flat", "IVF", "IVF-PQ", "HNSW"] = "flat",
        metric: Literal["L2", "IP"] = "L2",
        nlist: int = 100,
        nprobe: int = 10,
        m: int = 8,
        nbits: int = 8,
        M: int = 32,
        efConstruction: int = 200,
        efSearch: int = 50,
        use_gpu: bool | None = None,
    ):
        """Initialize a GPU index.

        Args:
            dim: Dimensionality of the vectors.
            index_type: Type of index to create ("flat", "IVF", "IVF-PQ", "HNSW").
            metric: Distance metric ("L2" for Euclidean, "IP" for inner product).
            nlist: Number of clusters for IVF indexes.
            nprobe: Number of clusters to search for IVF indexes.
            m: Number of subquantizers for PQ.
            nbits: Number of bits per subquantizer.
            M: Number of connections for HNSW.
            efConstruction: Search width during construction for HNSW.
            efSearch: Search width for HNSW queries.
            use_gpu: Force GPU usage (None for auto-detect).
        """
        self.dim = dim
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.nbits = nbits
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch

        # Determine backend
        if use_gpu is None:
            self.use_gpu = FAISS_GPU_AVAILABLE
        else:
            self.use_gpu = use_gpu and FAISS_GPU_AVAILABLE

        self._index: Any = None
        self._gpu_resources: Any = None

        if not FAISS_AVAILABLE:
            raise RuntimeError(
                "FAISS is not available. Install with: pip install faiss-cpu "
                "or pip install faiss-gpu"
            )

        self._create_index()

    def _create_index(self) -> None:
        """Create the FAISS index."""
        # Create quantizer
        if self.metric == "L2":
            quantizer = faiss.IndexFlatL2(self.dim)
        else:
            quantizer = faiss.IndexFlatIP(self.dim)

        # Create index based on type
        if self.index_type == "flat":
            if self.metric == "L2":
                self._index = faiss.IndexFlatL2(self.dim)
            else:
                self._index = faiss.IndexFlatIP(self.dim)

        elif self.index_type == "IVF":
            self._index = faiss.IndexIVFFlat(
                quantizer, self.dim, self.nlist, faiss.METRIC_L2
            )

        elif self.index_type == "IVF-PQ":
            self._index = faiss.IndexIVFPQ(
                quantizer,
                self.dim,
                self.nlist,
                self.m,
                self.nbits,
            )

        elif self.index_type == "HNSW":
            if not hasattr(faiss, "IndexHNSW"):
                logger.warning("HNSW not available in this FAISS build")
                self._index = faiss.IndexFlatL2(self.dim)
            else:
                self._index = faiss.IndexHNSWFlat(self.dim, self.M)
                self._index.hnsw.efConstruction = self.efConstruction
                self._index.hnsw.efSearch = self.efSearch

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Move to GPU if requested
        if self.use_gpu:
            try:
                self._gpu_resources = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(
                    self._gpu_resources, 0, self._index
                )
                logger.info("Moved %s index to GPU", self.index_type)
            except Exception as e:
                logger.warning("Failed to move index to GPU: %s", e)
                logger.info("Falling back to CPU index")
                self.use_gpu = False

    def train(self, vectors: np.ndarray) -> None:
        """Train the index on the given vectors.

        Args:
            vectors: Training vectors (N x dim).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} != index dimension {self.dim}"
            )
        self._index.train(vectors)

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.

        Args:
            vectors: Vectors to add (N x dim).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        self._index.add(vectors)

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Args:
            query: Query vectors (N x dim).
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (distances, indices).
        """
        query = np.asarray(query, dtype=np.float32)
        return self._index.search(query, k)

    def set_nprobe(self, nprobe: int) -> None:
        """Set the number of clusters to search.

        Args:
            nprobe: Number of clusters to search.
        """
        self.nprobe = nprobe
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = nprobe

    def set_ef(self, ef: int) -> None:
        """Set the search width for HNSW.

        Args:
            ef: Search width.
        """
        self.efSearch = ef
        if hasattr(self._index, "hnsw"):
            self._index.hnsw.efSearch = ef

    @property
    def ntotal(self) -> int:
        """Return the number of vectors in the index."""
        return self._index.ntotal

    def fallback_to_cpu(self) -> None:
        """Fallback to CPU index if GPU fails.

        This method moves the index from GPU to CPU and updates
        the internal state to use CPU for all operations.
        """
        if not self.use_gpu:
            logger.info("Already using CPU backend")
            return

        try:
            # Move index from GPU to CPU
            self._index = faiss.index_gpu_to_cpu(self._index)
            self.use_gpu = False

            # Cleanup GPU resources
            if self._gpu_resources is not None:
                with contextlib.suppress(Exception):
                    del self._gpu_resources
                self._gpu_resources = None

            logger.info("Successfully fallback to CPU index")
        except Exception as e:
            logger.error("Failed to fallback to CPU: %s", e)
            raise

    def __del__(self):
        """Cleanup GPU resources."""
        if self._gpu_resources is not None:
            with contextlib.suppress(Exception):
                del self._gpu_resources


def create_index(
    dim: int,
    index_type: str = "flat",
    metric: str = "L2",
    nlist: int = 100,
    use_gpu: bool | None = None,
) -> GPUIndex:
    """Create a GPU-accelerated index.

    Args:
        dim: Dimensionality of the vectors.
        index_type: Type of index ("flat", "IVF", "IVF-PQ", "HNSW").
        metric: Distance metric ("L2" or "IP").
        nlist: Number of clusters for IVF indexes.
        use_gpu: Force GPU usage (None for auto-detect).

    Returns:
        GPUIndex instance.
    """
    return GPUIndex(
        dim=dim,
        index_type=index_type,
        metric=metric,
        nlist=nlist,
        use_gpu=use_gpu,
    )


def create_index_with_fallback(
    dim: int,
    index_type: str = "flat",
    metric: str = "L2",
    nlist: int = 100,
    use_gpu: bool | None = None,
    fallback_on_error: bool = True,
) -> GPUIndex:
    """Create an index with automatic fallback to CPU on GPU errors.

    This function creates an index and automatically falls back to CPU
    if GPU operations fail.

    Args:
        dim: Dimensionality of the vectors.
        index_type: Type of index ("flat", "IVF", "IVF-PQ", "HNSW").
        metric: Distance metric ("L2" or "IP").
        nlist: Number of clusters for IVF indexes.
        use_gpu: Force GPU usage (None for auto-detect).
        fallback_on_error: If True, automatically fallback to CPU on errors.

    Returns:
        GPUIndex instance.

    Example:
        >>> index = create_index_with_fallback(128, use_gpu=True)
        >>> index.add(vectors)  # Falls back to CPU automatically if GPU fails
    """
    index = GPUIndex(
        dim=dim,
        index_type=index_type,
        metric=metric,
        nlist=nlist,
        use_gpu=use_gpu,
    )

    if not fallback_on_error:
        return index

    # Wrap search and add methods to fallback on error
    original_search = index.search
    original_add = index.add

    def search_with_fallback(query: np.ndarray, k: int = 10):
        try:
            return original_search(query, k)
        except Exception as e:
            logger.warning("GPU search failed, fallback to CPU: %s", e)
            index.fallback_to_cpu()
            return original_search(query, k)

    def add_with_fallback(vectors: np.ndarray):
        try:
            return original_add(vectors)
        except Exception as e:
            logger.warning("GPU add failed, fallback to CPU: %s", e)
            index.fallback_to_cpu()
            return original_add(vectors)

    index.search = search_with_fallback
    index.add = add_with_fallback

    return index
