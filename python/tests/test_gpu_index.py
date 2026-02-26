"""Tests for GPU-accelerated indexing (GpuIndex + unified backends)."""

from __future__ import annotations

import os
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

    def test_select_device_cpu(self):
        """device='cpu' maps to FAISS CPU."""
        from zvec.backends.unified import select_backend

        backend = select_backend(dim=64, preference="cpu")
        assert "FAISS CPU" in backend.backend_name

    def test_select_device_gpu_without_gpu(self):
        """device='gpu' without GPU hardware raises RuntimeError."""
        from zvec.backends.unified import select_backend

        # Patch all GPU detection to False
        with (
            patch("zvec.backends.detect.FAISS_GPU_AVAILABLE", False),
            patch("zvec.backends.detect.MPS_AVAILABLE", False),
            patch("zvec.backends.detect.APPLE_SILICON", False),
        ):
            # Also patch cuVS imports to fail
            with patch.dict("sys.modules", {"_zvec": None, "cuvs": None}):
                with pytest.raises(RuntimeError, match="no GPU backend"):
                    select_backend(dim=64, preference="gpu")

    def test_env_var_priority(self):
        """ZVEC_GPU_BACKEND_PRIORITY env var overrides auto-selection."""
        from zvec.backends.unified import select_backend, _ENV_PRIORITY_KEY

        # Force a specific priority via env var
        with patch.dict(os.environ, {_ENV_PRIORITY_KEY: "faiss_cpu"}):
            backend = select_backend(dim=64, n_vectors=100)
            assert "FAISS CPU" in backend.backend_name

    def test_env_var_priority_multiple(self):
        """Multiple backends in env var, first available wins."""
        from zvec.backends.unified import select_backend, _ENV_PRIORITY_KEY

        # bogus_backend will fail, faiss_cpu should succeed
        with patch.dict(os.environ, {_ENV_PRIORITY_KEY: "bogus_backend,faiss_cpu"}):
            backend = select_backend(dim=64, n_vectors=100)
            assert "FAISS CPU" in backend.backend_name

    def test_env_var_priority_all_fail_fallback(self):
        """If all env var backends fail, fall through to default auto-selection."""
        from zvec.backends.unified import select_backend, _ENV_PRIORITY_KEY

        with patch.dict(os.environ, {_ENV_PRIORITY_KEY: "bogus_one,bogus_two"}):
            backend = select_backend(dim=64, n_vectors=100)
            # Should still get a working backend via default chain
            assert backend is not None

    def test_try_create_backend_normalizes_name(self):
        """_try_create_backend normalizes dashes to underscores."""
        from zvec.backends.unified import _try_create_backend

        backend = _try_create_backend("faiss-cpu", dim=64, n_vectors=100)
        assert backend is not None
        assert "FAISS CPU" in backend.backend_name

    def test_try_create_backend_unknown(self):
        """_try_create_backend returns None for unknown backends."""
        from zvec.backends.unified import _try_create_backend

        result = _try_create_backend("nonexistent", dim=64, n_vectors=100)
        assert result is None


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


def _make_mock_collection(dim: int = 64, has_fetch_all: bool = False):
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
                vectors={"embedding": list(np.random.random(dim).astype(float))},
            )
            for doc_id in ids
        }

    col.fetch.side_effect = _fake_fetch

    # Optionally add fetch_all for build_from_collection tests
    if has_fetch_all:
        np.random.seed(123)
        all_docs = {}
        for i in range(50):
            doc_id = f"doc_{i}"
            all_docs[doc_id] = Doc(
                id=doc_id,
                fields={"title": f"Document {doc_id}"},
                vectors={"embedding": list(np.random.random(dim).astype(float))},
            )
        col.fetch_all.return_value = all_docs

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
# GpuIndex device= parameter
# ---------------------------------------------------------------------------


class TestGpuIndexDevice:
    """Tests for the device= parameter (PyTorch-style API)."""

    def test_device_cpu(self):
        """device='cpu' should use CPU backend."""
        from zvec.gpu_index import GpuIndex

        dim = 32
        col = _make_mock_collection(dim=dim)

        gpu = GpuIndex(col, "embedding", device="cpu")

        vectors = np.random.random((50, dim)).astype(np.float32)
        ids = [f"d{i}" for i in range(50)]
        gpu.build(vectors, ids)

        assert gpu.is_built
        assert "CPU" in gpu.info["backend"]

    def test_device_overrides_backend(self):
        """device= should take precedence over backend=."""
        from zvec.gpu_index import GpuIndex

        dim = 32
        col = _make_mock_collection(dim=dim)

        # device="cpu" should win over backend="faiss_gpu"
        gpu = GpuIndex(col, "embedding", backend="faiss_gpu", device="cpu")

        vectors = np.random.random((50, dim)).astype(np.float32)
        ids = [f"d{i}" for i in range(50)]
        gpu.build(vectors, ids)

        assert "CPU" in gpu.info["backend"]

    def test_device_none_uses_backend(self):
        """device=None should defer to backend parameter."""
        from zvec.gpu_index import GpuIndex

        dim = 32
        col = _make_mock_collection(dim=dim)

        gpu = GpuIndex(col, "embedding", backend="faiss_cpu", device=None)

        vectors = np.random.random((50, dim)).astype(np.float32)
        ids = [f"d{i}" for i in range(50)]
        gpu.build(vectors, ids)

        assert "FAISS CPU" in gpu.info["backend"]


# ---------------------------------------------------------------------------
# GpuIndex hybrid CPU/GPU auto-selector (gpu_threshold)
# ---------------------------------------------------------------------------


class TestGpuThreshold:
    """Tests for hybrid CPU/GPU auto-selection via gpu_threshold."""

    def test_small_collection_uses_cpu(self):
        """Collections below threshold should use CPU in auto mode."""
        from zvec.gpu_index import GpuIndex

        dim = 32
        col = _make_mock_collection(dim=dim)

        # Set threshold higher than our vector count
        gpu = GpuIndex(col, "embedding", backend="auto", gpu_threshold=1000)

        vectors = np.random.random((100, dim)).astype(np.float32)
        ids = [f"d{i}" for i in range(100)]
        gpu.build(vectors, ids)

        # Should fall through to CPU since 100 < 1000
        assert "CPU" in gpu.info["backend"]

    def test_large_collection_allows_gpu(self):
        """Collections above threshold should not be forced to CPU."""
        from zvec.gpu_index import GpuIndex

        dim = 32
        col = _make_mock_collection(dim=dim)

        # Set threshold lower than our vector count
        gpu = GpuIndex(col, "embedding", backend="auto", gpu_threshold=10)

        vectors = np.random.random((100, dim)).astype(np.float32)
        ids = [f"d{i}" for i in range(100)]
        gpu.build(vectors, ids)

        # Should use whatever auto-selects (likely CPU on test machine,
        # but won't be forced to CPU by threshold logic)
        assert gpu.is_built

    def test_threshold_zero_always_auto(self):
        """gpu_threshold=0 should never force CPU."""
        from zvec.gpu_index import GpuIndex

        dim = 32
        col = _make_mock_collection(dim=dim)

        gpu = GpuIndex(col, "embedding", backend="auto", gpu_threshold=0)

        vectors = np.random.random((10, dim)).astype(np.float32)
        ids = [f"d{i}" for i in range(10)]
        gpu.build(vectors, ids)

        # With threshold=0, even small collections go through auto
        assert gpu.is_built

    def test_env_threshold_override(self):
        """ZVEC_GPU_AUTO_THRESHOLD env var overrides default."""
        from zvec.gpu_index import GpuIndex

        dim = 32
        col = _make_mock_collection(dim=dim)

        with patch.dict(os.environ, {"ZVEC_GPU_AUTO_THRESHOLD": "500"}):
            gpu = GpuIndex(col, "embedding", backend="auto")
            assert gpu._gpu_threshold == 500

    def test_explicit_threshold_overrides_env(self):
        """Explicit gpu_threshold parameter takes precedence over env var."""
        from zvec.gpu_index import GpuIndex

        dim = 32
        col = _make_mock_collection(dim=dim)

        with patch.dict(os.environ, {"ZVEC_GPU_AUTO_THRESHOLD": "500"}):
            gpu = GpuIndex(col, "embedding", backend="auto", gpu_threshold=100)
            assert gpu._gpu_threshold == 100

    def test_threshold_only_applies_to_auto(self):
        """Explicit backend should not be affected by threshold."""
        from zvec.gpu_index import GpuIndex

        dim = 32
        col = _make_mock_collection(dim=dim)

        gpu = GpuIndex(col, "embedding", backend="faiss_cpu", gpu_threshold=1_000_000)

        vectors = np.random.random((10, dim)).astype(np.float32)
        ids = [f"d{i}" for i in range(10)]
        gpu.build(vectors, ids)

        assert "FAISS CPU" in gpu.info["backend"]


# ---------------------------------------------------------------------------
# GpuIndex.build_from_collection
# ---------------------------------------------------------------------------


class TestBuildFromCollection:
    """Tests for the build_from_collection() method."""

    def test_build_from_fetch_all(self):
        """build_from_collection() with fetch_all support."""
        from zvec.gpu_index import GpuIndex

        dim = 64
        col = _make_mock_collection(dim=dim, has_fetch_all=True)

        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")
        gpu.build_from_collection()

        assert gpu.is_built
        assert gpu.info["n_vectors"] == 50  # _make_mock_collection creates 50 docs

    def test_build_from_explicit_doc_ids(self):
        """build_from_collection(doc_ids=...) fetches specific docs."""
        from zvec.gpu_index import GpuIndex

        dim = 64
        col = _make_mock_collection(dim=dim)

        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")

        doc_ids = [f"doc_{i}" for i in range(20)]
        gpu.build_from_collection(doc_ids=doc_ids)

        assert gpu.is_built
        assert gpu.info["n_vectors"] == 20

    def test_build_from_collection_with_batch_size(self):
        """build_from_collection with small batch_size fetches in batches."""
        from zvec.gpu_index import GpuIndex

        dim = 64
        col = _make_mock_collection(dim=dim)

        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")

        doc_ids = [f"doc_{i}" for i in range(25)]
        gpu.build_from_collection(doc_ids=doc_ids, batch_size=10)

        assert gpu.is_built
        assert gpu.info["n_vectors"] == 25
        # fetch should have been called ceil(25/10) = 3 times
        assert col.fetch.call_count == 3

    def test_build_from_collection_no_fetch_all_no_ids_raises(self):
        """build_from_collection() without fetch_all or doc_ids should raise."""
        from zvec.gpu_index import GpuIndex

        dim = 64
        col = _make_mock_collection(dim=dim, has_fetch_all=False)
        # Remove fetch_all attribute to simulate missing API
        del col.fetch_all

        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")

        with pytest.raises(ValueError, match="fetch_all"):
            gpu.build_from_collection()

    def test_build_from_collection_empty_raises(self):
        """build_from_collection() with no vectors found should raise."""
        from zvec.gpu_index import GpuIndex

        dim = 64
        col = _make_mock_collection(dim=dim)
        # Make fetch return docs without vectors
        from zvec.model.doc import Doc

        col.fetch.side_effect = lambda ids: {
            doc_id: Doc(id=doc_id, fields={"title": "empty"}, vectors={})
            for doc_id in ids
        }

        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")

        with pytest.raises(ValueError, match="No vectors found"):
            gpu.build_from_collection(doc_ids=["doc_0", "doc_1"])

    def test_build_from_collection_chains(self):
        """build_from_collection() returns self for chaining."""
        from zvec.gpu_index import GpuIndex

        dim = 64
        col = _make_mock_collection(dim=dim, has_fetch_all=True)

        gpu = GpuIndex(col, "embedding", backend="faiss_cpu")
        result = gpu.build_from_collection()

        assert result is gpu


# ---------------------------------------------------------------------------
# Collection.index() and Collection.gpu_index()
# ---------------------------------------------------------------------------


class TestCollectionIndex:
    """Test the Collection.index() and Collection.gpu_index() methods."""

    def test_index_method_exists(self):
        """Collection should have index method."""
        from zvec.model.collection import Collection

        assert hasattr(Collection, "index")

    def test_gpu_index_method_exists(self):
        """Collection should still have gpu_index method (backward compat)."""
        from zvec.model.collection import Collection

        assert hasattr(Collection, "gpu_index")

    def test_gpu_index_deprecation_warning(self):
        """Collection.gpu_index() should emit DeprecationWarning."""
        import warnings
        from zvec.model.collection import Collection

        col = _make_mock_collection(dim=64)

        # Call the unbound method directly on the mock, simulating
        # col.gpu_index("embedding", backend="faiss_cpu")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Collection.gpu_index(col, "embedding", "faiss_cpu")

        deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecations) >= 1
        assert "deprecated" in str(deprecations[0].message).lower()


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
