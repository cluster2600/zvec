//
//  zvec_metal.metal
//  Metal compute shaders for vector operations
//
//  Created by cluster2600 on 2026-02-22.
//

#include <metal_stdlib>
using namespace metal;

// Compute L2 distance squared between query and database vector
// Each thread computes one distance
kernel void l2_distance_kernel(constant float *queries [[buffer(0)]],
                               constant float *database [[buffer(1)]],
                               device float *distances [[buffer(2)]],
                               constant uint64_t &num_queries [[buffer(3)]],
                               constant uint64_t &num_db [[buffer(4)]],
                               constant uint64_t &dim [[buffer(5)]],
                               uint2 gid [[thread_position_in_grid]]) {
  uint64_t q_idx = gid.x;
  uint64_t d_idx = gid.y;

  if (q_idx >= num_queries || d_idx >= num_db) return;

  float sum = 0.0f;
  for (uint64_t i = 0; i < dim; i++) {
    float diff = queries[q_idx * dim + i] - database[d_idx * dim + i];
    sum += diff * diff;
  }

  distances[q_idx * num_db + d_idx] = sum;
}

// Optimized L2 distance: compute all distances for one query against all
// database
kernel void l2_distance_query_kernel(constant float *queries [[buffer(0)]],
                                     constant float *database [[buffer(1)]],
                                     device float *distances [[buffer(2)]],
                                     constant uint64_t &num_db [[buffer(3)]],
                                     constant uint64_t &dim [[buffer(4)]],
                                     uint tid [[thread_position_in_grid]]) {
  if (tid >= num_db) return;

  // Compute query 0 (expand for batch later)
  float query_norm = 0.0f;
  for (uint64_t i = 0; i < dim; i++) {
    float v = queries[i];
    query_norm += v * v;
  }

  float db_norm = 0.0f;
  for (uint64_t i = 0; i < dim; i++) {
    float v = database[tid * dim + i];
    db_norm += v * v;
  }

  float dot = 0.0f;
  for (uint64_t i = 0; i < dim; i++) {
    dot += queries[i] * database[tid * dim + i];
  }

  // ||q - d||^2 = ||q||^2 + ||d||^2 - 2*q.d
  distances[tid] = query_norm + db_norm - 2.0f * dot;
}

// Inner product (dot product)
kernel void inner_product_kernel(constant float *queries [[buffer(0)]],
                                 constant float *database [[buffer(1)]],
                                 device float *results [[buffer(2)]],
                                 constant uint64_t &num_queries [[buffer(3)]],
                                 constant uint64_t &num_db [[buffer(4)]],
                                 constant uint64_t &dim [[buffer(5)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  uint64_t q_idx = gid.x;
  uint64_t d_idx = gid.y;

  if (q_idx >= num_queries || d_idx >= num_db) return;

  float sum = 0.0f;
  for (uint64_t i = 0; i < dim; i++) {
    sum += queries[q_idx * dim + i] * database[d_idx * dim + i];
  }

  results[q_idx * num_db + d_idx] = sum;
}

// L2 normalize vectors
kernel void normalize_kernel(device float *vectors [[buffer(0)]],
                             constant uint64_t &num_vectors [[buffer(1)]],
                             constant uint64_t &dim [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
  if (tid >= num_vectors) return;

  float norm = 0.0f;
  for (uint64_t i = 0; i < dim; i++) {
    float v = vectors[tid * dim + i];
    norm += v * v;
  }
  norm = sqrt(norm);

  if (norm > 1e-8f) {
    for (uint64_t i = 0; i < dim; i++) {
      vectors[tid * dim + i] /= norm;
    }
  }
}

// Matrix multiplication (float32)
kernel void matmul_kernel(constant float *A [[buffer(0)]],
                          constant float *B [[buffer(1)]],
                          device float *C [[buffer(2)]],
                          constant uint64_t &M [[buffer(3)]],
                          constant uint64_t &N [[buffer(4)]],
                          constant uint64_t &K [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
  uint64_t row = gid.x;
  uint64_t col = gid.y;

  if (row >= M || col >= N) return;

  float sum = 0.0f;
  for (uint64_t k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
  }

  C[row * N + col] = sum;
}

// Top-K reduction (simple version)
// Returns indices of k smallest values
kernel void topk_indices_kernel(constant float *distances [[buffer(0)]],
                                device uint64_t *indices [[buffer(1)]],
                                device float *topk_distances [[buffer(2)]],
                                constant uint64_t &num_distances [[buffer(3)]],
                                constant uint64_t &k [[buffer(4)]],
                                uint tid [[thread_position_in_grid]]) {
  if (tid >= num_distances) return;

  // Simple sequential top-k for each query (would need parallel for batch)
  // This is a placeholder - real implementation would use wavefront reduction
}

// Add two vectors
kernel void add_vectors_kernel(device float *result [[buffer(0)]],
                               constant float *a [[buffer(1)]],
                               constant float *b [[buffer(2)]],
                               constant uint64_t &size [[buffer(3)]],
                               uint tid [[thread_position_in_grid]]) {
  if (tid >= size) return;
  result[tid] = a[tid] + b[tid];
}

// Scale vector
kernel void scale_vector_kernel(device float *result [[buffer(0)]],
                                constant float *input [[buffer(1)]],
                                constant float &scale [[buffer(2)]],
                                constant uint64_t &size [[buffer(3)]],
                                uint tid [[thread_position_in_grid]]) {
  if (tid >= size) return;
  result[tid] = input[tid] * scale;
}
