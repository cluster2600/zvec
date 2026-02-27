"""Tests for backends module."""

import numpy as np
import pytest

from zvec.backends import detect


class TestHardwareDetection:
    """Tests for hardware detection."""

    def test_get_available_backends(self):
        """Test getting available backends."""
        backends = detect.get_available_backends()
        assert isinstance(backends, dict)
        assert "faiss" in backends
        assert "faiss_cpu" in backends

    def test_get_optimal_backend(self):
        """Test optimal backend detection."""
        backend = detect.get_optimal_backend()
        assert backend in ["faiss_gpu", "faiss_cpu", "numpy"]

    def test_is_gpu_available(self):
        """Test GPU detection."""
        # Should return boolean
        result = detect.is_gpu_available()
        assert isinstance(result, bool)

    def test_get_backend_info(self):
        """Test getting full backend info."""
        info = detect.get_backend_info()
        assert "system" in info
        assert "backends" in info
        assert "selected" in info


class TestGPUIndex:
    """Tests for GPU index."""

    def test_create_index(self):
        """Test creating GPU index."""
        from zvec.backends.gpu import create_index  # noqa: PLC0415

        index = create_index(dim=128, index_type="flat")
        assert index is not None

    def test_add_vectors(self):
        """Test adding vectors to index."""
        from zvec.backends.gpu import GPUIndex  # noqa: PLC0415

        index = GPUIndex(dim=128, index_type="flat")
        vectors = np.random.random((100, 128)).astype(np.float32)  # noqa: NPY002
        index.add(vectors)
        assert index.ntotal == 100

    def test_search(self):
        """Test searching index."""
        from zvec.backends.gpu import GPUIndex  # noqa: PLC0415

        index = GPUIndex(dim=128, index_type="flat")
        vectors = np.random.random((100, 128)).astype(np.float32)  # noqa: NPY002
        index.add(vectors)

        query = np.random.random((5, 128)).astype(np.float32)  # noqa: NPY002
        distances, indices = index.search(query, k=10)

        assert distances.shape == (5, 10)
        assert indices.shape == (5, 10)

    def test_fallback_to_cpu(self):
        """Test CPU fallback."""
        from zvec.backends.gpu import GPUIndex  # noqa: PLC0415

        index = GPUIndex(dim=128, index_type="flat", use_gpu=False)
        assert not index.use_gpu


class TestQuantization:
    """Tests for PQ quantization."""

    def test_pq_encoder_init(self):
        """Test PQ encoder initialization."""
        from zvec.backends.quantization import PQEncoder  # noqa: PLC0415

        encoder = PQEncoder(m=8, nbits=8, k=256)
        assert encoder.m == 8
        assert encoder.nbits == 8
        assert encoder.k == 256

    def test_pq_train(self):
        """Test PQ training."""
        from zvec.backends.quantization import PQEncoder  # noqa: PLC0415

        np.random.seed(42)  # noqa: NPY002
        vectors = np.random.random((1000, 128)).astype(np.float32)  # noqa: NPY002

        encoder = PQEncoder(m=8, nbits=8, k=256)
        encoder.train(vectors)

        assert encoder.is_trained

    def test_pq_encode_decode(self):
        """Test PQ encode/decode."""
        from zvec.backends.quantization import PQEncoder  # noqa: PLC0415

        np.random.seed(42)  # noqa: NPY002
        vectors = np.random.random((100, 128)).astype(np.float32)  # noqa: NPY002

        encoder = PQEncoder(m=8, nbits=8, k=256)
        encoder.train(vectors)

        codes = encoder.encode(vectors)
        assert codes.shape == (100, 8)

        decoded = encoder.decode(codes)
        assert decoded.shape == vectors.shape

    def test_pq_index(self):
        """Test PQ index."""
        from zvec.backends.quantization import PQIndex  # noqa: PLC0415

        np.random.seed(42)  # noqa: NPY002
        vectors = np.random.random((100, 128)).astype(np.float32)  # noqa: NPY002

        index = PQIndex(m=8, nbits=8, k=256)
        index.add(vectors)

        query = np.random.random((5, 128)).astype(np.float32)  # noqa: NPY002
        distances, indices = index.search(query, k=10)

        assert distances.shape == (5, 10)
        assert indices.shape == (5, 10)


class TestOPQ:
    """Tests for OPQ."""

    def test_opq_encoder_init(self):
        """Test OPQ encoder initialization."""
        from zvec.backends.opq import OPQEncoder  # noqa: PLC0415

        encoder = OPQEncoder(m=8, nbits=8, k=256)
        assert encoder.m == 8

    def test_scalar_quantizer(self):
        """Test scalar quantizer."""
        from zvec.backends.opq import ScalarQuantizer  # noqa: PLC0415

        np.random.seed(42)  # noqa: NPY002
        vectors = np.random.random((100, 128)).astype(np.float32)  # noqa: NPY002

        quantizer = ScalarQuantizer(bits=8)
        quantizer.train(vectors)

        encoded = quantizer.encode(vectors)
        assert encoded.dtype == np.int8

        decoded = quantizer.decode(encoded)
        assert decoded.shape == vectors.shape


class TestSearchOptimization:
    """Tests for search optimization."""

    def test_adc(self):
        """Test asymmetric distance computation."""
        from zvec.backends.search import asymmetric_distance_computation  # noqa: PLC0415

        np.random.seed(42)  # noqa: NPY002
        queries = np.random.random((10, 128)).astype(np.float32)  # noqa: NPY002
        codes = np.random.randint(0, 256, (100, 8), dtype=np.uint8)  # noqa: NPY002
        distance_table = np.random.random((10, 8, 256)).astype(np.float32)  # noqa: NPY002

        distances = asymmetric_distance_computation(queries, codes, distance_table)
        assert distances.shape == (10, 100)


class TestHNSW:
    """Tests for HNSW."""

    def test_hnsw_creation(self):
        """Test HNSW index creation."""
        from zvec.backends.hnsw import HNSWIndex  # noqa: PLC0415

        index = HNSWIndex(dim=128, M=16)
        assert index.dim == 128

    def test_hnsw_add(self):
        """Test adding vectors to HNSW."""
        from zvec.backends.hnsw import HNSWIndex  # noqa: PLC0415

        index = HNSWIndex(dim=128, M=8)
        vectors = np.random.random((50, 128)).astype(np.float32)  # noqa: NPY002
        index.add(vectors)

        # Basic check - just verify no error
        assert index.element_count == 50


class TestAppleSilicon:
    """Tests for Apple Silicon backend."""

    def test_apple_silicon_detection(self):
        """Test Apple Silicon detection."""
        from zvec.backends import apple_silicon  # noqa: PLC0415

        # Just verify functions exist and are callable
        assert callable(apple_silicon.is_apple_silicon)
        assert callable(apple_silicon.is_mps_available)

    def test_apple_backend_init(self):
        """Test Apple Silicon backend initialization."""
        from zvec.backends.apple_silicon import AppleSiliconBackend  # noqa: PLC0415

        backend = AppleSiliconBackend(backend="numpy")
        assert backend.backend == "numpy"

    def test_l2_distance(self):
        """Test L2 distance computation."""
        from zvec.backends.apple_silicon import AppleSiliconBackend  # noqa: PLC0415

        backend = AppleSiliconBackend(backend="numpy")

        a = np.random.random((10, 128)).astype(np.float32)  # noqa: NPY002
        b = np.random.random((20, 128)).astype(np.float32)  # noqa: NPY002

        distances = backend.l2_distance(a, b)
        assert distances.shape == (10, 20)


class TestDistributed:
    """Tests for distributed index."""

    def test_shard_manager(self):
        """Test shard manager."""
        from zvec.backends.distributed import ShardManager  # noqa: PLC0415

        manager = ShardManager(n_shards=4, strategy="hash")
        assert manager.n_shards == 4

        shard = manager.get_shard("vector_1")
        assert 0 <= shard < 4

    def test_distributed_index(self):
        """Test distributed index."""
        from zvec.backends.distributed import DistributedIndex  # noqa: PLC0415

        index = DistributedIndex(n_shards=4)
        vectors = np.random.random((100, 128)).astype(np.float32)  # noqa: NPY002
        vector_ids = [f"v_{i}" for i in range(100)]

        index.add(vectors, vector_ids)
        assert 4 in index._local_indexes

    def test_result_merger(self):
        """Test result merging."""
        from zvec.backends.distributed import ResultMerger  # noqa: PLC0415

        results = [
            (np.array([1.0, 2.0]), np.array([0, 1])),
            (np.array([1.5, 2.5]), np.array([2, 3])),
        ]

        distances, indices = ResultMerger.merge_knn(results, k=2)
        assert len(distances) == 2
