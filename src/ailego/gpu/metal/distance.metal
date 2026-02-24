/**
 * Metal Performance Shaders (MPS) Vector Distance Kernels for Apple Silicon
 * 
 * Based on:
 * - Apple ML Research: Deploying Transformers on ANE (2022)
 * - Ben Brown (2023): Neural Search on Modern Consumer Devices
 * 
 * Optimizations:
 * - FP16 compute
 * - SIMD/NEON vectorization
 * - Unified memory access
 */

#ifndef ZVEC_GPU_METAL_DISTANCE_METAL_H_
#define ZVEC_GPU_METAL_DISTANCE_METAL_H_

#include <metal_stdlib>
using namespace metal;

// Constants
constant uint WARP_SIZE = 32;

// =============================================================================
// L2 Distance Kernels
// =============================================================================

/**
 * Basic L2 distance kernel
 * Each thread computes distance between one query and one database vector
 */
kernel void metal_l2_distance(
    device const float* queries [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_queries [[buffer(4)]],
    constant uint& n_database [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint q_idx = gid.y;
    uint d_idx = gid.x;
    
    if (q_idx >= n_queries || d_idx >= n_database) return;
    
    float dist = 0.0f;
    
    for (uint i = 0; i < dim; i++) {
        float diff = queries[q_idx * dim + i] - database[d_idx * dim + i];
        dist += diff * diff;
    }
    
    distances[q_idx * n_database + d_idx] = dist;
}

/**
 * Optimized L2 using SIMD/NEON vector types
 */
kernel void metal_l2_distance_simd(
    device const float* queries [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_queries [[buffer(4)]],
    constant uint& n_database [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint q_idx = gid.y;
    uint d_idx = gid.x;
    
    if (q_idx >= n_queries || d_idx >= n_database) return;
    
    // Use SIMD for faster computation
    simd_float4 sum = 0.0f;
    
    uint vectorized_dim = (dim / 4) * 4;
    
    for (uint i = 0; i < vectorized_dim; i += 4) {
        simd_float4 q = simd_make_float4(
            queries[q_idx * dim + i],
            queries[q_idx * dim + i + 1],
            queries[q_idx * dim + i + 2],
            queries[q_idx * dim + i + 3]
        );
        simd_float4 d = simd_make_float4(
            database[d_idx * dim + i],
            database[d_idx * dim + i + 1],
            database[d_idx * dim + i + 2],
            database[d_idx * dim + i + 3]
        );
        simd_float4 diff = q - d;
        sum += diff * diff;
    }
    
    // Horizontal sum of simd vector
    float dist = sum.x + sum.y + sum.z + sum.w;
    
    // Handle remaining elements
    for (uint i = vectorized_dim; i < dim; i++) {
        float diff = queries[q_idx * dim + i] - database[d_idx * dim + i];
        dist += diff * diff;
    }
    
    distances[q_idx * n_database + d_idx] = dist;
}

// =============================================================================
// FP16 (Half) Kernels for Better Performance
// =============================================================================

/**
 * FP16 L2 distance kernel
 * Uses half precision for faster computation on Apple Silicon
 */
kernel void metal_l2_distance_fp16(
    device const half* queries [[buffer(0)]],
    device const half* database [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_queries [[buffer(4)]],
    constant uint& n_database [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint q_idx = gid.y;
    uint d_idx = gid.x;
    
    if (q_idx >= n_queries || d_idx >= n_database) return;
    
    simd_float4 sum = 0.0f;
    
    uint vectorized_dim = (dim / 4) * 4;
    
    // Convert and compute in FP32 for accumulation
    for (uint i = 0; i < vectorized_dim; i += 4) {
        simd_float4 q = simd_make_float4(
            float(queries[q_idx * dim + i]),
            float(queries[q_idx * dim + i + 1]),
            float(queries[q_idx * dim + i + 2]),
            float(queries[q_idx * dim + i + 3])
        );
        simd_float4 d = simd_make_float4(
            float(database[d_idx * dim + i]),
            float(database[d_idx * dim + i + 1]),
            float(database[d_idx * dim + i + 2]),
            float(database[d_idx * dim + i + 3])
        );
        simd_float4 diff = q - d;
        sum += diff * diff;
    }
    
    float dist = sum.x + sum.y + sum.z + sum.w;
    
    for (uint i = vectorized_dim; i < dim; i++) {
        float diff = float(queries[q_idx * dim + i]) - float(database[d_idx * dim + i]);
        dist += diff * diff;
    }
    
    distances[q_idx * n_database + d_idx] = dist;
}

// =============================================================================
// Batch Kernel - Multiple Queries at Once
// =============================================================================

/**
 * Batch L2 distance - processes one query against all database vectors
 */
kernel void metal_l2_distance_batch(
    device const float* queries [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_database [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint q_idx = gid;
    
    const float* query = queries + q_idx * dim;
    float* dist_row = distances + q_idx * n_database;
    
    for (uint d_idx = 0; d_idx < n_database; d_idx++) {
        float dist = 0.0f;
        
        for (uint i = 0; i < dim; i++) {
            float diff = query[i] - database[d_idx * dim + i];
            dist += diff * diff;
        }
        
        dist_row[d_idx] = dist;
    }
}

// =============================================================================
// Inner Product / Cosine Similarity
// =============================================================================

/**
 * Inner product (cosine similarity) kernel
 */
kernel void metal_inner_product(
    device const float* queries [[buffer(0)]],
    device const float* database [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant uint& n_queries [[buffer(4)]],
    constant uint& n_database [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint q_idx = gid.y;
    uint d_idx = gid.x;
    
    if (q_idx >= n_queries || d_idx >= n_database) return;
    
    float dot = 0.0f;
    
    for (uint i = 0; i < dim; i++) {
        dot += queries[q_idx * dim + i] * database[d_idx * dim + i];
    }
    
    similarities[q_idx * n_database + d_idx] = dot;
}

// =============================================================================
// Matrix Multiplication (for batch operations)
// =============================================================================

/**
 * Matrix multiplication kernel for vector batch processing
 * C = A * B where A is (M x K) queries, B is (K x N) database transposed
 */
kernel void metal_matmul_batch(
    device const float* A [[buffer(0)]],  // Queries: (n_queries x dim)
    device const float* B [[buffer(1)]],  // Database: (n_database x dim) 
    device float* C [[buffer(2)]],        // Output: (n_queries x n_database)
    constant uint& M [[buffer(3)]],       // n_queries
    constant uint& K [[buffer(4)]],       // dim
    constant uint& N [[buffer(5)]],       // n_database
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    
    // Dot product of row from A and column from B
    // B is stored as (n_database x dim), we want column col
    for (uint i = 0; i < K; i++) {
        sum += A[row * K + i] * B[col * K + i];
    }
    
    C[row * N + col] = sum;
}

#endif // ZVEC_GPU_METAL_DISTANCE_METAL_H_
