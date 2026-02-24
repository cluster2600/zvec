/**
 * Memory Coalesced Vector Distance CUDA Kernels
 * 
 * Based on:
 * - Naznin Fauzia et al. (2015) - Characterizing and Enhancing Global Memory Data Coalescing on GPU
 * - Expected speedup: 2-8x
 * 
 * Key optimizations:
 * 1. Coalesced memory access - threads in warp access contiguous memory
 * 2. Shared memory for frequently accessed data
 * 3. Register optimization
 * 4. Warp-level reductions
 */

#ifndef ZVEC_GPU_COALESCE_CUH_
#define ZVEC_GPU_COALESCE_CUH_

#include <cuda_runtime.h>
#include <stdint.h>

namespace zvec {
namespace gpu {

// Utility macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Block sizes
constexpr uint32_t COALESCE_BLOCK_SIZE = 256;
constexpr uint32_t WARP_SIZE = 32;

/**
 * Coalesced L2 Distance Kernel
 * 
 * Each thread handles one query-database pair
 * Warp accesses contiguous database rows for coalesced reads
 * 
 * Memory access pattern:
 * - Thread t reads database[t % WARP_SIZE][dim * (t / WARP_SIZE) + i]
 * - This ensures consecutive threads read consecutive memory
 */
__global__ void coalesced_l2_distance_kernel(
    const float* __restrict__ queries,     // (n_queries, dim)
    const float* __restrict__ database,   // (n_database, dim)
    float* __restrict__ distances,        // (n_queries, n_database)
    uint32_t dim,
    uint32_t n_queries,
    uint32_t n_database
);

/**
 * Optimized L2 with shared memory tiling
 * 
 * Uses shared memory to cache database rows for reuse
 */
__global__ void tiled_l2_distance_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ distances,
    uint32_t dim,
    uint32_t n_queries,
    uint32_t n_database
);

/**
 * Warp-level reduction for distance accumulation
 * 
 * Uses shuffle instructions for efficient reduction
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Block-level reduction
 */
__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE];
    int tid = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (tid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[tid] : 0;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

/**
 * Batch L2 distance with maximum coalescing
 * 
 * Processes multiple queries in parallel with optimal memory access
 */
__global__ void batch_coalesced_l2_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ distances,
    uint32_t dim,
    uint32_t n_queries,
    uint32_t n_database
);

/**
 * Inner product (cosine similarity) kernel
 */
__global__ void coalesced_inner_product_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ distances,
    uint32_t dim,
    uint32_t n_queries,
    uint32_t n_database
);

/**
 * Half-precision (FP16) L2 distance
 * 
 * Uses FP16 for reduced memory bandwidth
 */
__global__ void coalesced_l2_fp16_kernel(
    const half* __restrict__ queries,
    const half* __restrict__ database,
    float* __restrict__ distances,
    uint32_t dim,
    uint32_t n_queries,
    uint32_t n_database
);

/**
 * Utility functions
 */
struct CoalesceConfig {
    uint32_t block_size;
    uint32_t grid_size;
    uint32_t shared_mem_bytes;
    
    CoalesceConfig(uint32_t n_queries, uint32_t n_database, uint32_t dim) {
        block_size = COALESCE_BLOCK_SIZE;
        grid_size = (n_queries * n_database + block_size - 1) / block_size;
        shared_mem_bytes = 0;
    }
};

void launch_coalesced_l2(
    const float* queries,
    const float* database,
    float* distances,
    uint32_t dim,
    uint32_t n_queries,
    uint32_t n_database,
    cudaStream_t stream = 0
);

} // namespace gpu
} // namespace zvec

#endif // ZVEC_GPU_COALESCE_CUH_
