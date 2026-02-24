# zvec Backends Module

GPU optimization modules for zvec vector database.

## Modules

### Hardware Detection (`detect.py`)
Automatic detection of available hardware and backends.

```python
from zvec.backends import get_available_backends, get_optimal_backend, is_gpu_available

# Get all available backends
backends = get_available_backends()
# {'faiss': True, 'faiss_gpu': False, 'faiss_cpu': True, ...}

# Get optimal backend
backend = get_optimal_backend()  # 'faiss_gpu', 'faiss_cpu', or 'numpy'

# Check if GPU available
if is_gpu_available():
    print("GPU acceleration available!")
```

### GPU Index (`gpu.py`)
FAISS-backed GPU-accelerated index.

```python
from zvec.backends.gpu import GPUIndex, create_index, create_index_with_fallback

# Create GPU index
index = create_index(dim=128, index_type="IVF", nlist=100, use_gpu=True)

# Add vectors
vectors = np.random.random((10000, 128)).astype(np.float32)
index.add(vectors)

# Search
query = np.random.random((5, 128)).astype(np.float32)
distances, indices = index.search(query, k=10)

# With automatic CPU fallback
index = create_index_with_fallback(dim=128, use_gpu=True)
```

### Product Quantization (`quantization.py`)
Vector compression using PQ.

```python
from zvec.backends.quantization import PQEncoder, PQIndex

# Create encoder
encoder = PQEncoder(m=8, nbits=8, k=256)

# Train on your vectors
vectors = np.random.random((10000, 128)).astype(np.float32)
encoder.train(vectors)

# Encode vectors (compression)
codes = encoder.encode(vectors)
# codes.shape = (10000, 8) - 4-8x compression!

# Decode
decoded = encoder.decode(codes)

# Or use PQIndex for search
index = PQIndex(m=8, nbits=8, k=256)
index.add(vectors)
distances, indices = index.search(query, k=10)
```

### OPQ & Scalar Quantization (`opq.py`)
Optimized Product Quantization and simple scalar quantization.

```python
from zvec.backends.opq import OPQEncoder, ScalarQuantizer

# OPQ - rotates vectors for better compression
opq = OPQEncoder(m=8, nbits=8, k=256)
opq.train(vectors)
codes = opq.encode(vectors)

# Scalar Quantization - simple 8-bit or 16-bit
sq = ScalarQuantizer(bits=8)
sq.train(vectors)
encoded = sq.encode(vectors)  # int8
decoded = sq.decode(encoded)
```

### Search Optimization (`search.py`)
Fast search functions.

```python
from zvec.backends.search import (
    asymmetric_distance_computation,
    batch_search,
    search_with_reranking,
)

# ADC - Asymmetric Distance Computation
distance_table = compute_distance_table_fast(queries, codebooks)
distances = asymmetric_distance_computation(queries, codes, distance_table)

# Batch search for memory efficiency
distances, indices = batch_search(queries, database, codes, codebooks, k=10, batch_size=1000)

# Search with reranking
distances, indices = search_with_reranking(queries, database, codes, codebooks, k=10)
```

### HNSW (`hnsw.py`)
Hierarchical Navigable Small World graph index.

```python
from zvec.backends.hnsw import HNSWIndex, create_hnsw_index

# Pure Python implementation
index = HNSWIndex(dim=128, M=16, efConstruction=200, efSearch=50)
index.add(vectors)
distances, indices = index.search(query, k=10)

# Save/load
index.save("hnsw_index.pkl")
loaded = HNSWIndex.load("hnsw_index.pkl")

# Or use FAISS HNSW (faster)
index = create_hnsw_index(dim=128, use_faiss=True)
```

### Apple Silicon (`apple_silicon.py`)
Optimized for M1/M2/M3/M4 Macs.

```python
from zvec.backends.apple_silicon import (
    get_apple_silicon_backend,
    is_apple_silicon,
    is_mps_available,
)

# Check hardware
print(f"Apple Silicon: {is_apple_silicon()}")
print(f"MPS Available: {is_mps_available()}")

# Get optimized backend
backend = get_apple_silicon_backend()  # auto-detects best backend

# Vector operations
distances = backend.l2_distance(queries, database)
distances, indices = backend.search_knn(queries, database, k=10)
```

### Distributed (`distributed.py`)
Distributed vector index with sharding.

```python
from zvec.backends.distributed import (
    DistributedIndex,
    ShardManager,
    QueryRouter,
    ResultMerger,
)

# Create distributed index
index = DistributedIndex(n_shards=4, sharding_strategy="hash")

# Add vectors with IDs
vectors = np.random.random((10000, 128)).astype(np.float32)
vector_ids = [f"v_{i}" for i in range(10000)]
index.add(vectors, vector_ids)

# Search (scatter-gather)
distances, indices = index.search(query, k=10)

# Shard management
shard_manager = ShardManager(n_shards=8, strategy="hash")
shard = shard_manager.get_shard("vector_id")

# Query routing
router = QueryRouter(shard_manager)
shards = router.route_query(query, strategy="all")
```

## Installation

```bash
# Core dependencies
pip install numpy

# For CPU acceleration
pip install faiss-cpu

# For GPU acceleration (NVIDIA)
pip install faiss-gpu

# For Apple Silicon
pip install torch  # MPS support included
```

## Benchmarks

See `BENCHMARK_README.md` for detailed benchmarks.

## Testing

```bash
pytest python/tests/test_backends.py -v
```
