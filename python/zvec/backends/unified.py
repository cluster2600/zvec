# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified GPU backend adapter for zvec.

Provides a common interface across all GPU backends so that ``GpuIndex`` can
switch backends transparently.

Backend priority (C++ first, then Python):

1. C++ native cuVS  (via ``_zvec`` pybind11 — IVFPQIndex, CAGRAIndex, HNSWIndex)
2. Python cuVS CAGRA / IVF-PQ  (``cuvs.neighbors``)
3. FAISS GPU
4. Apple MPS
5. FAISS CPU (fallback)

The C++ path is preferred because it avoids Python-side data copies and
integrates directly with zvec's ``IndexProvider`` / ``GpuBufferLoader``
infrastructure.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Configurable backend priority via environment variable.
# Comma-separated list of backend names.  When set, overrides the default
# priority chain for auto-selection.
# Example: ZVEC_GPU_BACKEND_PRIORITY=faiss_gpu,cuvs_cagra,cuvs_ivf_pq,faiss_cpu
_ENV_PRIORITY_KEY = "ZVEC_GPU_BACKEND_PRIORITY"


class UnifiedGpuIndex(ABC):
    """Abstract base class for all GPU/accelerated index backends.

    Every adapter normalizes its wrapped backend to this interface:
    ``train`` + ``add`` + ``search``.
    """

    @abstractmethod
    def train(self, vectors: np.ndarray) -> None:
        """Train/build the index from base vectors.

        Args:
            vectors: Training vectors with shape ``(n, dim)``, dtype float32.
        """

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to a previously trained index.

        For backends that build the full index in ``train`` (CAGRA, HNSW),
        this may be a no-op.

        Args:
            vectors: Vectors to add with shape ``(n, dim)``, dtype float32.
        """

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for *k* nearest neighbors.

        Args:
            queries: Query vectors with shape ``(n_queries, dim)``, dtype float32.
            k: Number of neighbors to return.

        Returns:
            ``(distances, indices)`` each with shape ``(n_queries, k)``.
        """

    @abstractmethod
    def size(self) -> int:
        """Return the number of vectors currently in the index."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable name of the backend."""


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class FaissGpuAdapter(UnifiedGpuIndex):
    """Wraps :class:`zvec.backends.gpu.GPUIndex` (FAISS GPU/CPU)."""

    def __init__(self, dim: int, index_type: str = "flat", **kwargs: Any) -> None:
        from zvec.backends.gpu import GPUIndex  # noqa: PLC0415

        self._index = GPUIndex(dim=dim, index_type=index_type, use_gpu=True, **kwargs)

    def train(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if (
            hasattr(self._index._index, "is_trained")
            and not self._index._index.is_trained
        ):
            self._index.train(vectors)
        self._index.add(vectors)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._index.add(vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        return self._index.search(queries, k)

    def size(self) -> int:
        return self._index.ntotal

    @property
    def backend_name(self) -> str:
        suffix = "GPU" if self._index.use_gpu else "CPU"
        return f"FAISS {suffix} ({self._index.index_type})"


class FaissCpuAdapter(UnifiedGpuIndex):
    """Wraps :class:`zvec.backends.gpu.GPUIndex` forced to CPU."""

    def __init__(self, dim: int, index_type: str = "flat", **kwargs: Any) -> None:
        from zvec.backends.gpu import GPUIndex  # noqa: PLC0415

        self._index = GPUIndex(dim=dim, index_type=index_type, use_gpu=False, **kwargs)

    def train(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if (
            hasattr(self._index._index, "is_trained")
            and not self._index._index.is_trained
        ):
            self._index.train(vectors)
        self._index.add(vectors)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._index.add(vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        return self._index.search(queries, k)

    def size(self) -> int:
        return self._index.ntotal

    @property
    def backend_name(self) -> str:
        return f"FAISS CPU ({self._index.index_type})"


class CuvsCAGRAAdapter(UnifiedGpuIndex):
    """Wraps :class:`zvec.backends.cuvs_cagra.cuVSCAGRAIndex`."""

    def __init__(self, **kwargs: Any) -> None:
        from zvec.backends.cuvs_cagra import cuVSCAGRAIndex  # noqa: PLC0415

        self._index = cuVSCAGRAIndex(**kwargs)
        self._size = 0

    def train(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._index.train(vectors)
        self._size = vectors.shape[0]

    def add(self, vectors: np.ndarray) -> None:  # noqa: ARG002
        # CAGRA builds the full graph in train(); add is a no-op.
        logger.debug("CAGRA: add() is a no-op (graph built during train)")

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        return self._index.search(queries, k)

    def size(self) -> int:
        return self._size

    @property
    def backend_name(self) -> str:
        return "cuVS CAGRA"


class CuvsIvfPqAdapter(UnifiedGpuIndex):
    """Wraps :class:`zvec.backends.cuvs_ivf_pq.cuVSIVFPQIndex`."""

    def __init__(self, **kwargs: Any) -> None:
        from zvec.backends.cuvs_ivf_pq import cuVSIVFPQIndex  # noqa: PLC0415

        self._index = cuVSIVFPQIndex(**kwargs)
        self._size = 0

    def train(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._index.train(vectors)
        self._size = vectors.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._index.add(vectors)
        self._size += vectors.shape[0]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        return self._index.search(queries, k)

    def size(self) -> int:
        return self._size

    @property
    def backend_name(self) -> str:
        return "cuVS IVF-PQ"


class CppCuvsAdapter(UnifiedGpuIndex):
    """Wraps the C++ cuVS bindings exposed via ``_zvec`` pybind11.

    This adapter is the **preferred path** when available because it avoids
    Python-side data copies and leverages zvec's native ``GpuBufferLoader``
    to stream vectors directly from ``IndexProvider`` to the GPU.

    The C++ layer is defined in ``src/ailego/gpu/cuvs/zvec_cuvs.h`` and
    exposes ``IVFPQIndex<float>``, ``CAGRAIndex<float>``, ``HNSWIndex<float>``
    via factory functions.
    """

    def __init__(self, algo: str = "cagra", **kwargs: Any) -> None:
        self._algo = algo.lower()
        self._size = 0
        self._dim = 0

        try:
            import _zvec  # noqa: PLC0415

            if self._algo == "cagra":
                self._index = _zvec.create_cagra_float(**kwargs)
            elif self._algo == "ivf_pq":
                self._index = _zvec.create_ivf_pq_float(**kwargs)
            elif self._algo == "hnsw":
                self._index = _zvec.create_hnsw_float(**kwargs)
            else:
                raise ValueError(f"Unknown C++ cuVS algorithm: {algo}")
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"C++ cuVS bindings not available for '{algo}'. "
                "Ensure _zvec is built with CUDA/cuVS support."
            ) from exc

    def train(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, dim = vectors.shape
        self._dim = dim

        if self._algo == "ivf_pq":
            self._index.train(vectors, n, dim)
            self._index.add(vectors, n)
        else:
            # CAGRA and HNSW build in one shot
            self._index.build(vectors, n, dim)
        self._size = n

    def add(self, vectors: np.ndarray) -> None:
        if self._algo == "ivf_pq":
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            self._index.add(vectors, vectors.shape[0])
            self._size += vectors.shape[0]
        else:
            logger.debug("C++ %s: add() is a no-op (built during train)", self._algo)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        result = self._index.search(queries, queries.shape[0], k)
        # C++ SearchResult has .distances and .indices vectors
        n_queries = queries.shape[0]
        distances = np.array(result.distances, dtype=np.float32).reshape(n_queries, k)
        indices = np.array(result.indices, dtype=np.int64).reshape(n_queries, k)
        return distances, indices

    def size(self) -> int:
        return self._size

    @property
    def backend_name(self) -> str:
        return f"C++ cuVS {self._algo.upper()}"


class AppleMpsAdapter(UnifiedGpuIndex):
    """Wraps :class:`zvec.backends.apple_silicon.AppleSiliconBackend`."""

    def __init__(self) -> None:
        from zvec.backends.apple_silicon import AppleSiliconBackend  # noqa: PLC0415

        self._backend = AppleSiliconBackend(backend="auto")
        self._database: np.ndarray | None = None

    def train(self, vectors: np.ndarray) -> None:
        # MPS is brute-force; just store the database.
        self._database = np.asarray(vectors, dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._database is None:
            self._database = vectors
        else:
            self._database = np.vstack([self._database, vectors])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._database is None:
            raise RuntimeError("Index not built. Call train() first.")
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        return self._backend.search_knn(queries, self._database, k)

    def size(self) -> int:
        return 0 if self._database is None else self._database.shape[0]

    @property
    def backend_name(self) -> str:
        return f"Apple MPS ({self._backend.backend})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _try_create_backend(
    name: str,
    dim: int,
    n_vectors: int,
    **kwargs: Any,
) -> UnifiedGpuIndex | None:
    """Try to create a single backend by name.  Returns *None* on failure."""
    name = name.lower().replace("-", "_")

    # Map name → constructor thunk (deferred so imports only run on match)
    _CONSTRUCTORS: dict[str, Any] = {
        "cpp_cuvs_cagra": lambda: CppCuvsAdapter(algo="cagra", **kwargs),
        "cpp_cuvs_ivf_pq": lambda: CppCuvsAdapter(algo="ivf_pq", **kwargs),
        "cpp_cuvs_hnsw": lambda: CppCuvsAdapter(algo="hnsw", **kwargs),
        "cpp_cuvs": lambda: CppCuvsAdapter(
            algo="ivf_pq" if n_vectors > 1_000_000 else "cagra", **kwargs
        ),
        "cuvs_cagra": lambda: CuvsCAGRAAdapter(**kwargs),
        "cuvs_ivf_pq": lambda: CuvsIvfPqAdapter(**kwargs),
        "faiss_gpu": lambda: FaissGpuAdapter(dim=dim, **kwargs),
        "apple_mps": lambda: AppleMpsAdapter(),
        "faiss_cpu": lambda: FaissCpuAdapter(dim=dim, **kwargs),
    }

    factory = _CONSTRUCTORS.get(name)
    if factory is None:
        return None
    try:
        return factory()
    except Exception as exc:
        logger.warning("Backend '%s' requested but init failed: %s", name, exc)
        return None


def _resolve_preference(preference: str) -> str:
    """Normalise a device / backend preference string."""
    pref = preference.lower().replace("-", "_")
    if pref in ("gpu", "cuda", "cuda:0"):
        return "auto_gpu"
    if pref == "cpu":
        return "faiss_cpu"
    return pref


def _probe_availability() -> tuple[bool, bool]:
    """Return ``(cpp_cuvs_available, py_cuvs_available)``."""
    cpp_cuvs = False
    try:
        import _zvec  # noqa: PLC0415

        cpp_cuvs = hasattr(_zvec, "create_cagra_float")
    except ImportError:
        pass

    py_cuvs = False
    try:
        import cuvs  # noqa: PLC0415, F401

        py_cuvs = True
    except ImportError:
        pass

    return cpp_cuvs, py_cuvs


def select_backend(
    dim: int,
    n_vectors: int = 0,
    preference: str = "auto",
    **kwargs: Any,
) -> UnifiedGpuIndex:
    """Create the best available :class:`UnifiedGpuIndex`.

    Selection priority (when *preference* is ``"auto"``):

    1. **C++ cuVS**   (native pybind11 — zero-copy, fastest path)
    2. Python cuVS CAGRA  (NVIDIA GPU, best for <10M vectors)
    3. Python cuVS IVF-PQ (NVIDIA GPU, large-scale)
    4. FAISS GPU   (NVIDIA GPU, general purpose)
    5. Apple MPS   (Apple Silicon)
    6. FAISS CPU   (fallback)

    The priority can be overridden via the ``ZVEC_GPU_BACKEND_PRIORITY``
    environment variable — a comma-separated list of backend names, tried
    in order.  Example::

        ZVEC_GPU_BACKEND_PRIORITY=faiss_gpu,cuvs_cagra,faiss_cpu

    Args:
        dim: Vector dimensionality.
        n_vectors: Approximate number of vectors (hint for backend selection).
        preference: Force a specific backend, ``"auto"``, or a device string
            like ``"gpu"`` / ``"cpu"`` / ``"cuda:0"``.
        **kwargs: Passed through to the chosen adapter constructor.

    Returns:
        A ready-to-use :class:`UnifiedGpuIndex` instance.

    Raises:
        RuntimeError: If no backend is available.
    """
    from zvec.backends.detect import (  # noqa: PLC0415
        APPLE_SILICON,
        FAISS_AVAILABLE,
        FAISS_GPU_AVAILABLE,
        MPS_AVAILABLE,
    )

    cpp_cuvs_available, py_cuvs_available = _probe_availability()
    _pref = _resolve_preference(preference)

    # ------- explicit preference -------
    if _pref not in ("auto", "auto_gpu"):
        result = _try_create_backend(_pref, dim, n_vectors, **kwargs)
        if result is not None:
            return result
        logger.warning(
            "Explicit backend '%s' failed, falling through to auto", preference
        )

    # ------- env-var priority override -------
    result = _try_env_priority(dim, n_vectors, **kwargs)
    if result is not None:
        return result

    # ------- auto selection -------
    return _auto_select(
        dim,
        n_vectors,
        _pref == "auto_gpu",
        cpp_cuvs_available,
        py_cuvs_available,
        FAISS_GPU_AVAILABLE,
        APPLE_SILICON and MPS_AVAILABLE,
        FAISS_AVAILABLE,
        **kwargs,
    )


def _try_env_priority(
    dim: int,
    n_vectors: int,
    **kwargs: Any,
) -> UnifiedGpuIndex | None:
    """Try backends listed in ``ZVEC_GPU_BACKEND_PRIORITY``."""
    env_priority = os.environ.get(_ENV_PRIORITY_KEY, "").strip()
    if not env_priority:
        return None
    backends = [b.strip() for b in env_priority.split(",") if b.strip()]
    logger.info(
        "Using custom backend priority from %s: %s",
        _ENV_PRIORITY_KEY,
        backends,
    )
    for name in backends:
        result = _try_create_backend(name, dim, n_vectors, **kwargs)
        if result is not None:
            logger.info("Selected backend '%s' from env priority", name)
            return result
    logger.warning("No backend from %s succeeded, trying defaults", _ENV_PRIORITY_KEY)
    return None


def _auto_select(
    dim: int,
    n_vectors: int,
    gpu_only: bool,
    cpp_cuvs: bool,
    py_cuvs: bool,
    faiss_gpu: bool,
    apple_mps: bool,
    faiss_cpu: bool,
    **kwargs: Any,
) -> UnifiedGpuIndex:
    """Run the default backend priority chain."""
    # 1. C++ native cuVS — zero-copy, fastest
    if cpp_cuvs:
        algo = "ivf_pq" if n_vectors > 1_000_000 else "cagra"
        logger.info("Auto-selected C++ cuVS %s (n=%d)", algo.upper(), n_vectors)
        try:
            return CppCuvsAdapter(algo=algo, **kwargs)
        except RuntimeError:
            logger.warning("C++ cuVS %s init failed, trying Python fallback", algo)

    # 2. Python cuVS
    if py_cuvs:
        if n_vectors > 1_000_000:
            logger.info("Auto-selected Python cuVS IVF-PQ (n=%d)", n_vectors)
            return CuvsIvfPqAdapter(**kwargs)
        logger.info("Auto-selected Python cuVS CAGRA (n=%d)", n_vectors)
        return CuvsCAGRAAdapter(**kwargs)

    # 3. FAISS GPU
    if faiss_gpu:
        logger.info("Auto-selected FAISS GPU")
        return FaissGpuAdapter(dim=dim, **kwargs)

    # 4. Apple MPS
    if apple_mps:
        logger.info("Auto-selected Apple MPS")
        return AppleMpsAdapter()

    if gpu_only:
        raise RuntimeError(
            "device='gpu' requested but no GPU backend is available. "
            "Install one of: cuvs, faiss-gpu, or torch (for Apple MPS)."
        )

    # 5. FAISS CPU (fallback)
    if faiss_cpu:
        logger.info("Auto-selected FAISS CPU (fallback)")
        return FaissCpuAdapter(dim=dim, **kwargs)

    raise RuntimeError(
        "No vector search backend available. "
        "Install one of: faiss-cpu, faiss-gpu, cuvs, or torch (for Apple MPS)."
    )
