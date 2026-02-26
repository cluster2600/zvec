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

"""GPU-accelerated index integrated with the zvec Collection API.

This module provides :class:`GpuIndex`, a bridge between the GPU backends
(cuVS, FAISS GPU, Apple MPS) and zvec's ``Collection`` / ``Doc`` data model.

Architecture
------------
The preferred execution path goes through the **C++ layer** whenever possible:

    Collection  ──►  IndexProvider::Iterator
                          │
                          ▼
                     GpuBufferLoader          (C++, zero-copy)
                          │
                          ▼
                     zvec::cuvs::CAGRAIndex   (C++ cuVS)
                          │
                          ▼
                     SearchResult { distances, indices }

When the C++ pybind11 bindings are not compiled with GPU support, the module
transparently falls back to the Python cuVS / FAISS GPU / MPS backends.

Usage
-----
::

    import zvec
    import numpy as np

    col = zvec.open("my_collection")

    # Create a GPU index bound to the "embedding" vector field
    gpu = col.index("embedding", device="gpu")        # PyTorch-style device
    gpu = col.index("embedding", device="cuda:0")     # explicit CUDA device
    gpu = col.index("embedding", backend="cuvs_cagra") # explicit backend

    # Build from vectors + doc IDs
    gpu.build(vectors, ids)

    # Or build directly from the collection (streams in batches)
    gpu.build_from_collection(batch_size=10000)

    # Search — returns (doc_id, distance) pairs
    results = gpu.search(query_vector, k=10)

    # Full query — returns Doc objects just like collection.query()
    docs = gpu.query(query_vector, topk=10, output_fields=["title"])
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from zvec.backends.unified import UnifiedGpuIndex, select_backend

if TYPE_CHECKING:
    from zvec.model.collection import Collection
    from zvec.model.doc import Doc

logger = logging.getLogger(__name__)

__all__ = ["GpuIndex"]

# Default threshold: below this number of vectors, the CPU path is used
# when device="auto" to avoid GPU transfer overhead.  Can be overridden
# via the ZVEC_GPU_AUTO_THRESHOLD environment variable.
_DEFAULT_GPU_THRESHOLD = 50_000


class GpuIndex:
    """GPU-accelerated index bound to a :class:`Collection` vector field.

    Bridges the gap between zvec's standalone GPU backends and the
    Collection query workflow.  After calling :meth:`build` or
    :meth:`build_from_collection`, the index can be queried with
    :meth:`search` (raw results) or :meth:`query` (returns full ``Doc``
    objects, same format as ``Collection.query``).

    Args:
        collection: The zvec Collection this index is associated with.
        field_name: Name of the vector field to index.
        backend: Backend preference — ``"auto"`` (default) lets the factory
            pick the fastest available backend (C++ cuVS first).
            See :func:`~zvec.backends.unified.select_backend` for options.
        device: Device string (PyTorch-style). When set, overrides *backend*.
            ``"gpu"`` — any GPU, ``"cuda:0"`` — specific CUDA device,
            ``"cpu"`` — force CPU.  Default ``None`` (use *backend*).
        gpu_threshold: Number of vectors below which the auto-selector
            prefers CPU over GPU.  Only effective when *backend* and
            *device* are both ``"auto"`` / ``None``.
            Default 50 000.  Set to 0 to always use GPU.
        **params: Extra parameters forwarded to the backend adapter.
    """

    def __init__(
        self,
        collection: Collection,
        field_name: str,
        backend: str = "auto",
        *,
        device: str | None = None,
        gpu_threshold: int | None = None,
        **params: Any,
    ) -> None:
        self._collection = collection
        self._field_name = field_name
        self._params = params

        # Resolve device / backend preference
        if device is not None:
            self._backend_pref = device  # device takes precedence
        else:
            self._backend_pref = backend

        # GPU/CPU threshold for hybrid auto-selection
        import os  # noqa: PLC0415

        if gpu_threshold is not None:
            self._gpu_threshold = gpu_threshold
        else:
            self._gpu_threshold = int(
                os.environ.get("ZVEC_GPU_AUTO_THRESHOLD", str(_DEFAULT_GPU_THRESHOLD))
            )

        self._backend: UnifiedGpuIndex | None = None
        self._ids: np.ndarray | None = None  # doc-ID array parallel to index
        self._dim: int = 0
        self._built = False

        # Resolve dimension from schema
        vschema = collection.schema.vector(field_name)
        if vschema is None:
            raise ValueError(
                f"Field '{field_name}' is not a vector field in collection schema"
            )
        self._dim = vschema.dim

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        vectors: np.ndarray,
        ids: Union[list[str], np.ndarray],
    ) -> GpuIndex:
        """Build the GPU index from explicit vectors and document IDs.

        Args:
            vectors: Base vectors with shape ``(n, dim)``, dtype float32.
            ids: Parallel array of document IDs (same length as *vectors*).

        Returns:
            *self* for chaining.

        Raises:
            ValueError: If shapes are inconsistent.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        n, dim = vectors.shape

        if dim != self._dim:
            raise ValueError(
                f"Vector dimension {dim} does not match field "
                f"'{self._field_name}' dimension {self._dim}"
            )

        ids_arr = np.asarray(ids)
        if ids_arr.shape[0] != n:
            raise ValueError(
                f"Number of IDs ({ids_arr.shape[0]}) != number of vectors ({n})"
            )

        # Hybrid auto-selection: use CPU for small collections
        pref = self._backend_pref
        if pref == "auto" and n < self._gpu_threshold:
            logger.info(
                "n=%d < gpu_threshold=%d, using CPU for better latency",
                n,
                self._gpu_threshold,
            )
            pref = "cpu"

        # Create backend (lazy — so we know dim and n_vectors)
        t0 = time.perf_counter()
        self._backend = select_backend(
            dim=dim,
            n_vectors=n,
            preference=pref,
            **self._params,
        )

        # Train + populate
        self._backend.train(vectors)
        self._ids = ids_arr
        self._built = True

        elapsed = time.perf_counter() - t0
        logger.info(
            "GpuIndex built: %d vectors, dim=%d, backend=%s (%.1f ms)",
            n,
            dim,
            self._backend.backend_name,
            elapsed * 1000,
        )
        return self

    def build_from_collection(
        self,
        *,
        batch_size: int = 10_000,
        doc_ids: list[str] | None = None,
    ) -> GpuIndex:
        """Build the index by streaming vectors from the collection.

        This is a convenience method that internally fetches vectors in
        batches, avoiding the need to manually extract and pass arrays.
        The collection must already contain documents with the vector
        field populated.

        Args:
            batch_size: Number of documents to fetch per batch.  Larger
                values use more memory but are faster.
            doc_ids: Explicit list of document IDs to index.  When
                ``None`` (default), all documents in the collection are
                indexed.

        Returns:
            *self* for chaining.
        """
        t0 = time.perf_counter()

        all_vectors: list[np.ndarray] = []
        all_ids: list[str] = []

        if doc_ids is not None:
            # Fetch specific documents in batches
            for start in range(0, len(doc_ids), batch_size):
                batch_ids = doc_ids[start : start + batch_size]
                fetched = self._collection.fetch(batch_ids)
                for doc_id, doc in fetched.items():
                    if doc.vectors and self._field_name in doc.vectors:
                        vec = doc.vectors[self._field_name]
                        all_vectors.append(np.asarray(vec, dtype=np.float32))
                        all_ids.append(doc_id)
        elif hasattr(self._collection, "fetch_all"):
            fetched = self._collection.fetch_all()
            for doc_id, doc in fetched.items():
                if doc.vectors and self._field_name in doc.vectors:
                    vec = doc.vectors[self._field_name]
                    all_vectors.append(np.asarray(vec, dtype=np.float32))
                    all_ids.append(doc_id)
        else:
            raise ValueError(
                "build_from_collection() without doc_ids requires either "
                "a Collection with fetch_all() or explicit doc_ids. "
                "Pass doc_ids=[...] to specify which documents to index."
            )

        if not all_vectors:
            raise ValueError(
                f"No vectors found for field '{self._field_name}' in collection"
            )

        vectors = np.stack(all_vectors)
        elapsed_fetch = time.perf_counter() - t0
        logger.info(
            "Fetched %d vectors in %.1f ms (batch_size=%d)",
            len(all_ids),
            elapsed_fetch * 1000,
            batch_size,
        )

        return self.build(vectors, all_ids)

    # ------------------------------------------------------------------
    # Search (raw)
    # ------------------------------------------------------------------

    def search(
        self,
        query: Union[np.ndarray, list[float]],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search for the *k* nearest neighbors.

        Args:
            query: Query vector(s).  A 1-D array is treated as a single query.
            k: Number of neighbors.

        Returns:
            List of ``(doc_id, distance)`` tuples sorted by distance
            (ascending for L2, descending for IP).
        """
        self._ensure_built()

        query_arr = np.asarray(query, dtype=np.float32)
        if query_arr.ndim == 1:
            query_arr = query_arr.reshape(1, -1)

        distances, indices = self._backend.search(query_arr, k)

        # Map flat indices → doc IDs
        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            idx_int = int(idx)
            if 0 <= idx_int < len(self._ids):
                results.append((str(self._ids[idx_int]), float(dist)))
        return results

    # ------------------------------------------------------------------
    # Query (Collection-compatible)
    # ------------------------------------------------------------------

    def query(
        self,
        query_vector: Union[np.ndarray, list[float]],
        *,
        topk: int = 10,
        include_vector: bool = False,
        output_fields: Optional[list[str]] = None,
    ) -> list[Doc]:
        """Query the GPU index and return full ``Doc`` objects.

        This mirrors the signature of ``Collection.query()`` but uses GPU
        search under the hood.  After the GPU returns candidate IDs,
        ``Collection.fetch()`` retrieves the full document fields.

        Args:
            query_vector: The query embedding.
            topk: Number of nearest neighbors.
            include_vector: Whether to include the vector data in results.
            output_fields: Scalar fields to include.  ``None`` means all.

        Returns:
            ``list[Doc]`` sorted by relevance (best first).
        """
        from zvec.model.doc import Doc  # noqa: PLC0415

        self._ensure_built()

        # 1. GPU search
        hits = self.search(query_vector, k=topk)
        if not hits:
            return []

        doc_ids = [doc_id for doc_id, _ in hits]
        score_map = dict(hits)

        # 2. Fetch full documents from collection
        fetched = self._collection.fetch(doc_ids)

        # 3. Assemble Doc list with scores
        docs: list[Doc] = []
        for doc_id in doc_ids:
            doc = fetched.get(doc_id)
            if doc is None:
                # Doc was deleted between index build and query
                continue

            # Attach the distance as score
            score = score_map.get(doc_id)

            # Filter output fields if requested
            fields = doc.fields
            if output_fields is not None and fields:
                fields = {k: v for k, v in fields.items() if k in output_fields}

            vectors = doc.vectors if include_vector else None

            docs.append(
                Doc(
                    id=doc_id,
                    score=score,
                    vectors=vectors,
                    fields=fields,
                )
            )
        return docs

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def info(self) -> dict[str, Any]:
        """Return metadata about the GPU index."""
        return {
            "field_name": self._field_name,
            "dim": self._dim,
            "built": self._built,
            "n_vectors": self._backend.size() if self._backend else 0,
            "backend": self._backend.backend_name if self._backend else None,
            "gpu_threshold": self._gpu_threshold,
        }

    @property
    def is_built(self) -> bool:
        """Whether :meth:`build` has been called."""
        return self._built

    def __repr__(self) -> str:
        backend = self._backend.backend_name if self._backend else "not built"
        n = self._backend.size() if self._backend else 0
        return (
            f"GpuIndex(field='{self._field_name}', dim={self._dim}, "
            f"n={n}, backend='{backend}')"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_built(self) -> None:
        if not self._built or self._backend is None or self._ids is None:
            raise RuntimeError(
                "GpuIndex not built. Call .build(vectors, ids) or "
                ".build_from_collection() first."
            )
