"""Tests for GPU-accelerated indexing (GpuIndex + unified backends)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# UnifiedGpuIndex & select_backend
# ---------------------------------------------------------------------------


class TestSelectBackend:
    """Tests for backend selection factory."""

    def test_select_faiss_cpu_explicit(self):
        """Explicitly request FAISS CPU."""
        from zvec.backends.unified import select_backend

        backend = select_backend(dim=128, preference="faiss_cpu")
        assert "FAISS CPU" in backend.backend_name

    def test_select_auto_fallback(self):
        """Auto-selection should return *something* (at least FAISS CPU)."""
        from zvec.backends.unified import select_backend

        backend = select_backend(dim=128, n_vectors=1000)
        assert backend is not None
        assert backend.backend_name  # non-empty string

    def test_select_unknown_preference(self):
        """Unknown preference falls through to auto-selection."""
        from zvec.backends.unified import select_backend

        # "bogus" is not a recognised preference → auto
        backend = select_backend(dim=64, preference="bogus")
        assert backend is not None


class TestFaissCpuAdapter:
    """Tests for the FAISS CPU adapter."""

    def test_train_add_search(self):
        """End-to-end train → search on FAISS CPU."""
        from zvec.backends.unified import FaissCpuAdapter

        np.random.seed(42)
        dim = 64
        adapter = FaissCpuAdapter(dim=dim, index_type="flat")

        vectors = np.random.random((200, dim)).astype(np.float32)
        adapter.train(vectors)

        assert adapter.size() == 200

        queries = np.random.random((5, dim)).astype(np.float32)
        distances, indices = adapter.search(queries, k=10)

        assert distances.shape == (5, 10)
        assert indices.shape == (5, 10)
        # L2 distances should be non-negative
        assert np.all(distances >= 0)

    def test_search_single_query(self):
        """1-D query vector is auto-reshaped."""
        from zvec.backends.unified import FaissCpuAdapter

        dim = 32
        adapter = FaissCpuAdapter(dim=dim, index_type="flat")
        adapter.train(np.random.random((50, dim)).astype(np.float32))

        query = np.random.random(dim).astype(np.float32)
        distances, indices = adapter.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)


class TestAppleMpsAdapter:
    """Tests for the Apple MPS adapter (always uses numpy fallback)."""

    def test_numpy_fallback(self):
        """MPS adapter with numpy backend works on all platforms."""
        from zvec.backends.unified import AppleMpsAdapter

        adapter = AppleMpsAdapter()
        dim = 32
        vectors = np.random.random((100, dim)).astype(np.float32)
        adapter.train(vectors)
        assert adapter.size() == 100

        queries = np.random.random((3, dim)).astype(np.float32)
        distances, indices = adapter.search(queries, k=5)

        assert distances.shape == (3, 5)
        assert indices.shape == (3, 5)

    def test_add_extends_database(self):
        """add() should extend the stored database."""
        from zvec.backends.unified import AppleMpsAdapter

        adapter = AppleMpsAdapter()
        dim = 16
        adapter.train(np.random.random((50, dim)).astype(np.float32))
        adapter.add(np.random.random((30, dim)).astype(np.float32))

        assert adapter.size() == 80


# ---------------------------------------------------------------------------
# GpuIndex (with mocked Collection)
# ---------------------------------------------------------------------------


def _make_mock_collection(dim: int = 64):
    """Create a mock Collection with a vector schema."""
    from zvec.model.doc import Doc

    col = MagicMock()

    # schema.vector(field_name) returns a VectorSchema-like object
    vschema = MagicMock()
    vschema.dim = dim
    col.schema.vector.return_value = vschema

    # fetch returns Doc objects
    def _fake_fetch(ids):
        return {
            doc_id: Doc(
                id=doc_id,
                fields={"title": f"Document {doc_id}"},
            )
            for doc_id in ids
        }

    col.fetch.side_effect = _fake_fetch
    return col


class TestGpuIndex:
    """Tests for the GpuIndex bridge class."""

    def test_build_and_search(self):
        """Build GPU index and run raw search."""
        from zvec.gpu_index import GpuIndex

        dim = 64
        col = _make_mock_collection(dim=dim)

        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")

        np.random.seed(42)
        vectors = np.random.random((200, dim)).astype(np.float32)
        ids = [f"doc_{i}" for i in range(200)]

        gpu.build(vectors, ids)
        assert gpu.is_built
        assert gpu.info["n_vectors"] == 200
        assert "FAISS CPU" in gpu.info["backend"]

        query = np.random.random(dim).astype(np.float32)
        results = gpu.search(query, k=5)

        assert len(results) == 5
        for doc_id, distance in results:
            assert doc_id.startswith("doc_")
            assert isinstance(distance, float)
            assert distance >= 0

    def test_query_returns_docs(self):
        """query() should return Doc objects with scores."""
        from zvec.gpu_index import GpuIndex

        dim = 32
        col = _make_mock_collection(dim=dim)

        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")

        np.random.seed(42)
        vectors = np.random.random((100, dim)).astype(np.float32)
        ids = [f"doc_{i}" for i in range(100)]
        gpu.build(vectors, ids)

        query = np.random.random(dim).astype(np.float32)
        docs = gpu.query(query, topk=5)

        assert len(docs) == 5
        for doc in docs:
            assert doc.id.startswith("doc_")
            assert doc.score is not None
            assert doc.fields.get("title") is not None

    def test_query_output_fields(self):
        """query() should filter output fields."""
        from zvec.gpu_index import GpuIndex

        dim = 32
        col = _make_mock_collection(dim=dim)

        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")
        vectors = np.random.random((50, dim)).astype(np.float32)
        ids = [f"d{i}" for i in range(50)]
        gpu.build(vectors, ids)

        query = np.random.random(dim).astype(np.float32)
        docs = gpu.query(query, topk=3, output_fields=["title"])

        for doc in docs:
            assert "title" in doc.fields

    def test_dimension_mismatch_raises(self):
        """build() with wrong dimension should raise."""
        from zvec.gpu_index import GpuIndex

        col = _make_mock_collection(dim=64)
        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")

        wrong_dim = np.random.random((10, 32)).astype(np.float32)
        with pytest.raises(ValueError, match="dimension"):
            gpu.build(wrong_dim, [f"d{i}" for i in range(10)])

    def test_ids_length_mismatch_raises(self):
        """build() with mismatched ID count should raise."""
        from zvec.gpu_index import GpuIndex

        col = _make_mock_collection(dim=64)
        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")

        vectors = np.random.random((10, 64)).astype(np.float32)
        with pytest.raises(ValueError, match="IDs"):
            gpu.build(vectors, ["only_one_id"])

    def test_search_before_build_raises(self):
        """search() before build() should raise RuntimeError."""
        from zvec.gpu_index import GpuIndex

        col = _make_mock_collection(dim=64)
        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")

        with pytest.raises(RuntimeError, match="not built"):
            gpu.search(np.zeros(64), k=5)

    def test_invalid_field_raises(self):
        """Non-vector field should raise ValueError."""
        from zvec.gpu_index import GpuIndex

        col = MagicMock()
        col.schema.vector.return_value = None

        with pytest.raises(ValueError, match="not a vector field"):
            GpuIndex(col, "nonexistent_field")

    def test_repr(self):
        """__repr__ should be informative."""
        from zvec.gpu_index import GpuIndex

        col = _make_mock_collection(dim=64)
        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")
        r = repr(gpu)
        assert "GpuIndex" in r
        assert "embedding" in r

    def test_info_before_build(self):
        """info property should work before build."""
        from zvec.gpu_index import GpuIndex

        col = _make_mock_collection(dim=128)
        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")

        info = gpu.info
        assert info["built"] is False
        assert info["dim"] == 128
        assert info["n_vectors"] == 0


# ---------------------------------------------------------------------------
# Collection.gpu_index integration
# ---------------------------------------------------------------------------


class TestCollectionGpuIndex:
    """Test the Collection.gpu_index() convenience method."""

    def test_gpu_index_method_exists(self):
        """Collection should have gpu_index method."""
        from zvec.model.collection import Collection

        assert hasattr(Collection, "gpu_index")


# ---------------------------------------------------------------------------
# Detection updates
# ---------------------------------------------------------------------------


class TestDetectCuVS:
    """Tests for updated backend detection."""

    def test_cuVS_in_backends(self):
        """get_available_backends should include cuVS keys."""
        from zvec.backends import detect

        backends = detect.get_available_backends()
        assert "cuvs" in backends
        assert "cpp_cuvs" in backends
        assert isinstance(backends["cuvs"], bool)
        assert isinstance(backends["cpp_cuvs"], bool)

    def test_optimal_backend_includes_cuvs(self):
        """get_optimal_backend return value is in the valid set."""
        from zvec.backends import detect

        backend = detect.get_optimal_backend()
        valid = {"cpp_cuvs", "cuvs", "faiss_gpu", "faiss_cpu", "numpy"}
        assert backend in valid

    def test_is_gpu_available(self):
        """is_gpu_available should return bool."""
        from zvec.backends import detect

        assert isinstance(detect.is_gpu_available(), bool)
