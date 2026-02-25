# Metal C++ Backend

GPU-accelerated vector operations for Apple Silicon using Metal shaders.

## Architecture

```
IndexProvider (Flat/HNSW/IVF)
    │
    ├── Iterator  ──→  GpuBufferLoader::load()
    │                        │
    │                  GpuBuffer (contiguous float32)
    │                        │
    │                  ┌─────┴──────┐
    │                  │            │
    │            Metal device   cudaMemcpy
    │              buffer       (CUDA/cuVS)
    │                  │
    │            Metal Kernels
    │         (L2, IP, Cosine, TopK)
    │                  │
    │            Results Buffer
    │
    └── get_vector(key)  ──→  single vector lookup
```

## GPU Buffer Loading

The `GpuBufferLoader` bridges zvec's segment-based storage with GPU compute
pipelines. It streams vectors through `IndexProvider::Iterator` into a
contiguous float32 buffer ready for GPU transfer.

```cpp
#include <ailego/gpu/gpu_buffer_loader.h>

// Load all vectors from any index type
auto provider = index->create_provider();
auto buffer = zvec::GpuBufferLoader::load(provider);

// buffer.vectors is contiguous (N x dim) float32
// buffer.keys[i] corresponds to buffer.vector_at(i)

// Metal: create device buffer
id<MTLBuffer> mtl_buf = [device newBufferWithBytes:buffer.vectors.data()
                                            length:buffer.byte_size()
                                           options:MTLResourceStorageModeShared];

// CUDA: copy to device
cudaMemcpy(d_vectors, buffer.vectors.data(),
           buffer.byte_size(), cudaMemcpyHostToDevice);
```

### Chunked Loading

For datasets larger than GPU memory:

```cpp
auto iter = provider->create_iterator();
size_t chunk_size = 100000;  // vectors per chunk

while (iter->is_valid()) {
    auto chunk = zvec::GpuBufferLoader::load_chunk(
        iter.get(), provider->dimension(),
        provider->data_type(), chunk_size);

    // Process chunk on GPU...
}
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

The `*_simdgroup` kernels use Metal's cooperative SIMD intrinsics (`simd_sum`, `simd_min`, `simd_shuffle`) to perform reductions across 32 SIMD lanes without shared memory barriers. Each simdgroup of 32 threads collaborates on a single (query, database) distance computation, splitting the dimension across lanes and reducing with hardware-accelerated cross-lane operations.

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
