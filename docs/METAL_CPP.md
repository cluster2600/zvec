# Metal C++ Backend

GPU-accelerated vector operations for Apple Silicon using Metal shaders.

## Architecture

```
VectorStorage (RocksDB)  -->  load_all()  -->  Metal GPU Buffers
                                                    |
                                              Metal Kernels
                                           (L2, IP, Cosine, TopK)
                                                    |
                                              Results Buffer
```

## Metal Kernels

### Distance Kernels

| Kernel | Description |
|--------|-------------|
| `metal_l2_distance` | Basic L2 distance (1 thread per pair) |
| `metal_l2_distance_simd` | float4 vectorized L2 |
| `metal_l2_distance_fp16` | Half-precision L2 |
| `metal_l2_distance_batch` | One query vs all database |
| `metal_l2_distance_simdgroup` | Simdgroup cooperative L2 (32 threads per pair) |
| `metal_inner_product` | Basic inner product |
| `metal_inner_product_simdgroup` | Simdgroup cooperative inner product |
| `metal_cosine_similarity_simdgroup` | Simdgroup cosine similarity |

### Utility Kernels

| Kernel | Description |
|--------|-------------|
| `metal_matmul_batch` | Basic matrix multiplication (C = A * B^T) |
| `metal_matmul_tiled` | Tiled matmul with shared memory |
| `metal_normalize_simdgroup` | In-place L2 normalization |
| `metal_topk_simdgroup` | Per-query top-k selection |

## Simdgroup Optimization

The `*_simdgroup` kernels use Metal's cooperative SIMD intrinsics (`simd_sum`, `simd_min`, `simd_shuffle`, `simd_ballot`) to perform reductions across 32 SIMD lanes without shared memory barriers. Each simdgroup of 32 threads collaborates on a single (query, database) distance computation, splitting the dimension across lanes and reducing with hardware-accelerated cross-lane operations.

Dispatch model:
- Threadgroup size: 32 (one simdgroup)
- Grid: `(n_database, n_queries)` threadgroups

## C++ Quantization

### Product Quantizer (`product_quantizer.h`)

Splits D-dimensional vectors into M sub-vectors and quantizes each with k-means.

```cpp
#include <ailego/algorithm/product_quantizer.h>

zvec::ailego::ProductQuantizer pq(/*m=*/8, /*k=*/256);
pq.train(data, n_vectors, dim);

std::vector<uint8_t> codes(n * 8);
pq.encode(data, n, codes.data());
```

### Optimized PQ (`opq.h`)

Learns an orthogonal rotation matrix R via SVD-based Procrustes before PQ, minimizing quantization distortion.

```cpp
#include <ailego/algorithm/opq.h>

zvec::ailego::OptimizedProductQuantizer opq(/*m=*/8, /*k=*/256, /*n_iter=*/20);
opq.train(data, n_vectors, dim);

std::vector<uint8_t> codes(n * 8);
opq.encode(data, n, codes.data());
```

## RocksDB Vector Storage (`vector_storage.h`)

Persistent vector storage with column families for raw vectors, PQ codes, and metadata. Provides `load_all()` to stream vectors into contiguous GPU-ready buffers.

```cpp
#include <zvec/db/index/vector/vector_storage.h>

zvec::VectorStorage store;
store.create("/path/to/db", 128);  // 128-dim vectors
store.put_vectors_batch(ids, vectors, n);

std::vector<uint64_t> all_ids;
std::vector<float> all_vecs;
store.load_all(all_ids, all_vecs);
// all_vecs is now a contiguous (N x 128) buffer ready for Metal
```

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Metal shaders are compiled automatically on macOS via CMake.

## Future Work

- CUDA backend for NVIDIA GPUs (cuVS integration)
- ANE (Apple Neural Engine) backend via Core ML
- Distributed vector search across multiple nodes
