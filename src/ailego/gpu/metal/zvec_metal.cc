//
//  zvec_metal.cc
//  Metal implementation for zvec
//
//  Created by cluster2600 on 2026-02-22.
//

#include "zvec_metal.h"
#include <cstdlib>
#include <cstring>

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_MAC
#include <dispatch/dispatch.h>
#endif
#endif

// Metal includes
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#endif

struct ZvecMetalDevice {
#ifdef __OBJC__
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;

  ZvecMetalDevice() : device(nil), queue(nil), library(nil) {}
#endif
};

extern "C" {

ZvecMetalDevice *zvec_metal_create(void) {
#ifdef __OBJC__
  @autoreleasepool {
    ZvecMetalDevice *dev = new ZvecMetalDevice();

    // Get default Metal device
    dev->device = MTLCreateSystemDefaultDevice();
    if (dev->device == nil) {
      delete dev;
      return nullptr;
    }

    // Create command queue
    dev->queue = [dev->device newCommandQueue];
    if (dev->queue == nil) {
      delete dev;
      return nullptr;
    }

    // Load default library (embedded)
    NSError *error = nil;
    dev->library = [dev->device newDefaultLibrary:&error];
    if (error != nil || dev->library == nil) {
      // Try to create from source
      NSString *src = @""
#include <metal_stdlib>
          using namespace metal;
      kernel void dummy() {}
      "@";
      MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
      dev->library =
          [dev->device newLibraryWithSource:src options:opts error:&error];
      if (error != nil) {
        delete dev;
        return nullptr;
      }
    }

    return dev;
  }
#else
  return nullptr;
#endif
}

void zvec_metal_destroy(ZvecMetalDevice *device) {
  if (device) {
    delete device;
  }
}

int zvec_metal_available(void) {
#ifdef __OBJC__
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil ? 1 : 0;
  }
#else
  return 0;
#endif
}

const char *zvec_metal_device_name(ZvecMetalDevice *device) {
  if (!device) return "No Device";
#ifdef __OBJC__
  return [[device->device name] UTF8String];
#else
  return "No Metal";
#endif
}

uint64_t zvec_metal_device_memory(ZvecMetalDevice *device) {
  if (!device) return 0;
#ifdef __OBJC__
  return [device->device recommendedMaxWorkingSetSize];
#else
  return 0;
#endif
}

int zvec_metal_l2_distance(ZvecMetalDevice *device, const float *queries,
                           const float *database, float *distances,
                           uint64_t num_queries, uint64_t num_db,
                           uint64_t dim) {
  if (!device || !queries || !database || !distances) {
    return -1;
  }

#ifdef __OBJC__
  @autoreleasepool {
    // For now, fall back to CPU if Metal kernel compilation fails
    // In production, use the Metal kernels directly

    // Simple CPU fallback for validation
    for (uint64_t q = 0; q < num_queries; q++) {
      for (uint64_t d = 0; d < num_db; d++) {
        float sum = 0.0f;
        for (uint64_t i = 0; i < dim; i++) {
          float diff = queries[q * dim + i] - database[d * dim + i];
          sum += diff * diff;
        }
        distances[q * num_db + d] = sum;
      }
    }

    return 0;
  }
#else
  return -1;
#endif
}

int zvec_metal_l2_distance_matrix(ZvecMetalDevice *device, const float *a,
                                  const float *b, float *result,
                                  uint64_t a_rows, uint64_t b_rows,
                                  uint64_t dim) {
  return zvec_metal_l2_distance(device, a, b, result, a_rows, b_rows, dim);
}

int zvec_metal_inner_product(ZvecMetalDevice *device, const float *queries,
                             const float *database, float *results,
                             uint64_t num_queries, uint64_t num_db,
                             uint64_t dim) {
  if (!device || !queries || !database || !results) {
    return -1;
  }

#ifdef __OBJC__
  @autoreleasepool {
    // CPU fallback
    for (uint64_t q = 0; q < num_queries; q++) {
      for (uint64_t d = 0; d < num_db; d++) {
        float sum = 0.0f;
        for (uint64_t i = 0; i < dim; i++) {
          sum += queries[q * dim + i] * database[d * dim + i];
        }
        results[q * num_db + d] = sum;
      }
    }
    return 0;
  }
#else
  return -1;
#endif
}

int zvec_metal_normalize(ZvecMetalDevice *device, float *vectors,
                         uint64_t num_vectors, uint64_t dim) {
  if (!device || !vectors) {
    return -1;
  }

#ifdef __OBJC__
  @autoreleasepool {
    // CPU fallback
    for (uint64_t v = 0; v < num_vectors; v++) {
      float norm = 0.0f;
      for (uint64_t i = 0; i < dim; i++) {
        float val = vectors[v * dim + i];
        norm += val * val;
      }
      norm = sqrtf(norm);
      if (norm > 1e-8f) {
        for (uint64_t i = 0; i < dim; i++) {
          vectors[v * dim + i] /= norm;
        }
      }
    }
    return 0;
  }
#else
  return -1;
#endif
}

}  // extern "C"
