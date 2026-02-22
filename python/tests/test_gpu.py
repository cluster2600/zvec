"""
Tests for zvec GPU module.
"""

import platform
import numpy as np
import pytest

from zvec.gpu import (
    GPUBackend,
    get_optimal_backend,
    get_gpu_info,
    is_apple_silicon,
    AVAILABLE,
)


class TestGPUDetection:
    """Tests for GPU detection."""
    
    def test_platform_detection(self):
        """Test platform detection."""
        info = get_gpu_info()
        
        assert info['platform'] == platform.system()
        assert info['machine'] == platform.machine()
    
    def test_apple_silicon(self):
        """Test Apple Silicon detection."""
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            assert is_apple_silicon() is True
        else:
            assert is_apple_silicon() is False
    
    def test_backend_selection(self):
        """Test automatic backend selection."""
        backend = get_optimal_backend()
        
        # Should return a valid backend string
        assert backend in ["mps", "cuda", "faiss-cpu", "none"]
    
    def test_gpu_info(self):
        """Test GPU info dictionary."""
        info = get_gpu_info()
        
        assert 'platform' in info
        assert 'machine' in info
        assert 'backends' in info
        assert 'selected' in info
        assert 'available' in info


class TestGPUBackend:
    """Tests for GPUBackend class."""
    
    @pytest.fixture
    def backend(self):
        """Create GPU backend instance."""
        return GPUBackend()
    
    def test_backend_creation(self, backend):
        """Test backend creation."""
        assert backend is not None
        assert backend.backend in ["mps", "cuda", "faiss-cpu", "none"]
    
    def test_is_available(self):
        """Test availability check."""
        # FAISS is available on this machine
        result = GPUBackend.is_available()
        assert isinstance(result, bool)
    
    def test_create_index(self):
        """Test index creation."""
        import faiss
        
        backend = GPUBackend()
        
        # Create a small index
        index = backend.create_index(dim=128, metric="L2", nlist=10)
        
        assert index is not None
        assert isinstance(index, faiss.Index)
    
    def test_search(self):
        """Test search functionality."""
        backend = GPUBackend()
        
        # Create and train index
        dim = 128
        nlist = 10
        index = backend.create_index(dim=dim, metric="L2", nlist=nlist)
        
        # Generate random training data
        np.random.seed(42)
        training_data = np.random.rand(1000, dim).astype('float32')
        index.train(training_data)
        
        # Add some vectors
        vectors = np.random.rand(100, dim).astype('float32')
        index.add(vectors)
        
        # Search
        query = np.random.rand(1, dim).astype('float32')
        distances, indices = backend.search(index, query, k=10)
        
        assert distances.shape == (1, 10)
        assert indices.shape == (1, 10)
    
    def test_metric_options(self):
        """Test different metric options."""
        for metric in ["L2", "IP"]:
            backend = GPUBackend()
            index = backend.create_index(dim=64, metric=metric, nlist=4)
            assert index is not None
    
    def test_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError):
            GPUBackend(backend="invalid_backend")


class TestGPUPerformance:
    """Performance tests for GPU vs CPU."""
    
    def test_index_performance(self):
        """Test index creation performance."""
        import time
        
        backend = GPUBackend()
        
        # Create index
        start = time.perf_counter()
        index = backend.create_index(dim=512, metric="L2", nlist=100)
        create_time = time.perf_counter() - start
        
        # Train index
        np.random.seed(42)
        train_data = np.random.rand(10000, 512).astype('float32')
        
        start = time.perf_counter()
        index.train(train_data)
        train_time = time.perf_counter() - start
        
        # Add data
        start = time.perf_counter()
        index.add(train_data[:5000])
        add_time = time.perf_counter() - start
        
        # Should be relatively fast
        assert create_time < 1.0  # Index creation < 1 second
        assert train_time < 5.0    # Training < 5 seconds
        
        print(f"\nPerformance: create={create_time:.3f}s, train={train_time:.3f}s, add={add_time:.3f}s")
    
    def test_search_performance(self):
        """Test search performance."""
        import time
        
        backend = GPUBackend()
        
        # Create and populate index
        dim = 256
        nlist = 50
        index = backend.create_index(dim=dim, metric="L2", nlist=nlist)
        
        np.random.seed(42)
        data = np.random.rand(10000, dim).astype('float32')
        index.train(data)
        index.add(data)
        
        # Search
        queries = np.random.rand(100, dim).astype('float32')
        
        start = time.perf_counter()
        distances, indices = backend.search(index, queries, k=10)
        search_time = time.perf_counter() - start
        
        # Should be fast
        assert search_time < 1.0  # 100 queries < 1 second
        
        print(f"\nSearch performance: {search_time*1000:.2f}ms for 100 queries")


class TestIntegration:
    """Integration tests."""
    
    def test_gpu_module_importable(self):
        """Test that GPU module is importable."""
        # Just verify module is importable
        import zvec.gpu
        assert hasattr(zvec.gpu, 'GPUBackend')
        assert hasattr(zvec.gpu, 'get_optimal_backend')
