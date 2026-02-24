//
//  zvec_metal.h
//  Metal-accelerated operations for zvec
//
//  Created by cluster2600 on 2026-02-22.
//

#ifndef ZVEC_METAL_H
#define ZVEC_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for Metal device
typedef struct ZvecMetalDevice ZvecMetalDevice;

// Initialize Metal device (returns NULL if not available)
ZvecMetalDevice *zvec_metal_create(void);

// Destroy Metal device
void zvec_metal_destroy(ZvecMetalDevice *device);

// Check if Metal is available
int zvec_metal_available(void);

// Get device name
const char *zvec_metal_device_name(ZvecMetalDevice *device);

// Get device memory in bytes
uint64_t zvec_metal_device_memory(ZvecMetalDevice *device);

// L2 distance squared (float32)
int zvec_metal_l2_distance(ZvecMetalDevice *device, const float *queries,
                           const float *database, float *distances,
                           uint64_t num_queries, uint64_t num_db, uint64_t dim);

// Batch L2 distance matrix
int zvec_metal_l2_distance_matrix(ZvecMetalDevice *device, const float *a,
                                  const float *b, float *result,
                                  uint64_t a_rows, uint64_t b_rows,
                                  uint64_t dim);

// Inner product (for cosine similarity)
int zvec_metal_inner_product(ZvecMetalDevice *device, const float *queries,
                             const float *database, float *results,
                             uint64_t num_queries, uint64_t num_db,
                             uint64_t dim);

// Normalize vectors (L2)
int zvec_metal_normalize(ZvecMetalDevice *device, float *vectors,
                         uint64_t num_vectors, uint64_t dim);

#ifdef __cplusplus
}
#endif

#endif  // ZVEC_METAL_H
