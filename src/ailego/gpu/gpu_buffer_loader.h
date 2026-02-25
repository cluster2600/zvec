// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_provider.h>

namespace zvec {

/**
 * @brief Result of loading vectors into a contiguous GPU-ready buffer.
 *
 * Contains parallel arrays: keys[i] corresponds to vectors[i * dim .. (i+1) *
 * dim). The vectors buffer is contiguous row-major float32, suitable for direct
 * transfer to Metal (device buffer) or CUDA (cudaMemcpy).
 */
struct GpuBuffer {
  std::vector<uint64_t> keys;   ///< Document keys, one per vector
  std::vector<float> vectors;   ///< Contiguous (n × dim) float32 buffer
  size_t n_vectors = 0;         ///< Number of vectors loaded
  size_t dim = 0;               ///< Dimensionality of each vector

  /// @brief Get a pointer to the i-th vector
  const float *vector_at(size_t i) const { return vectors.data() + i * dim; }

  /// @brief Total bytes in the vector buffer
  size_t byte_size() const { return vectors.size() * sizeof(float); }
};

/**
 * @brief Loads vectors from an IndexProvider into a contiguous GPU-ready buffer.
 *
 * This bridges zvec's segment-based storage with GPU compute pipelines
 * (Metal, CUDA/cuVS). It streams vectors through the IndexProvider::Iterator
 * into a single contiguous float32 buffer that can be directly mapped or
 * copied to GPU memory.
 *
 * Architecture:
 *   IndexProvider (Flat/HNSW/IVF) → Iterator → GpuBufferLoader → GpuBuffer
 *                                                                    │
 *                                                          Metal device buffer
 *                                                          or cudaMemcpy
 *
 * Usage:
 * @code
 *   auto provider = index->create_provider();
 *   auto buffer = GpuBufferLoader::load(provider);
 *
 *   // Metal: create device buffer from contiguous data
 *   id<MTLBuffer> mtl_buf = [device newBufferWithBytes:buffer.vectors.data()
 *                                              length:buffer.byte_size()
 *                                             options:MTLResourceStorageModeShared];
 *
 *   // CUDA: copy to device
 *   cudaMemcpy(d_vectors, buffer.vectors.data(), buffer.byte_size(),
 *              cudaMemcpyHostToDevice);
 *
 *   // cuVS: build index directly
 *   cagra::build(params, buffer.vectors.data(), buffer.n_vectors, buffer.dim);
 * @endcode
 */
class GpuBufferLoader {
 public:
  /**
   * @brief Load all vectors from a provider into a contiguous GPU buffer.
   *
   * Iterates through the provider's vectors and packs them into a single
   * contiguous float32 array. Handles FP32, FP16, INT8 source types with
   * automatic conversion to float32.
   *
   * @param provider  The index provider to stream vectors from.
   * @return GpuBuffer with contiguous vectors and associated keys.
   *
   * @note For large datasets, consider load_range() to load in chunks
   *       that fit in GPU memory.
   */
  static GpuBuffer load(const core::IndexProvider::Pointer &provider) {
    GpuBuffer buf;
    buf.dim = provider->dimension();
    buf.n_vectors = provider->count();

    // Pre-allocate for the known count
    buf.keys.reserve(buf.n_vectors);
    buf.vectors.reserve(buf.n_vectors * buf.dim);

    auto data_type = provider->data_type();
    auto elem_size = provider->element_size();
    auto iter = provider->create_iterator();

    while (iter->is_valid()) {
      buf.keys.push_back(iter->key());
      append_as_float32(buf.vectors, iter->data(), buf.dim, data_type);
      iter->next();
    }

    // Update actual count (may differ if iterator had fewer than count())
    buf.n_vectors = buf.keys.size();
    return buf;
  }

  /**
   * @brief Load a range of vectors (for chunked GPU transfer).
   *
   * Useful when the full dataset doesn't fit in GPU memory. The caller
   * manages the iterator lifetime across multiple calls.
   *
   * @param iter       Iterator (caller manages; position is advanced).
   * @param dim        Vector dimensionality.
   * @param data_type  Source data type for conversion.
   * @param max_count  Maximum number of vectors to load in this chunk.
   * @return GpuBuffer with up to max_count vectors.
   */
  static GpuBuffer load_chunk(core::IndexHolder::Iterator *iter, size_t dim,
                               core::IndexMeta::DataType data_type,
                               size_t max_count) {
    GpuBuffer buf;
    buf.dim = dim;

    buf.keys.reserve(max_count);
    buf.vectors.reserve(max_count * dim);

    size_t loaded = 0;
    while (iter->is_valid() && loaded < max_count) {
      buf.keys.push_back(iter->key());
      append_as_float32(buf.vectors, iter->data(), dim, data_type);
      iter->next();
      ++loaded;
    }

    buf.n_vectors = buf.keys.size();
    return buf;
  }

 private:
  /**
   * @brief Append a single vector to the float32 buffer, converting if needed.
   */
  static void append_as_float32(std::vector<float> &dst, const void *src,
                                 size_t dim,
                                 core::IndexMeta::DataType data_type) {
    size_t offset = dst.size();
    dst.resize(offset + dim);

    switch (data_type) {
      case core::IndexMeta::DT_FP32: {
        std::memcpy(dst.data() + offset, src, dim * sizeof(float));
        break;
      }
      case core::IndexMeta::DT_FP16: {
        // Convert half → float. Metal and CUDA both use IEEE 754 half.
        const uint16_t *half_ptr = static_cast<const uint16_t *>(src);
        for (size_t i = 0; i < dim; ++i) {
          dst[offset + i] = half_to_float(half_ptr[i]);
        }
        break;
      }
      case core::IndexMeta::DT_INT8: {
        const int8_t *int8_ptr = static_cast<const int8_t *>(src);
        for (size_t i = 0; i < dim; ++i) {
          dst[offset + i] = static_cast<float>(int8_ptr[i]);
        }
        break;
      }
      default: {
        // Fallback: assume float-sized elements, memcpy
        std::memcpy(dst.data() + offset, src, dim * sizeof(float));
        break;
      }
    }
  }

  /**
   * @brief Convert IEEE 754 half-precision to single-precision.
   *
   * Handles normals, denormals, inf, and NaN.
   */
  static float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x03FFu;

    if (exponent == 0) {
      if (mantissa == 0) {
        // Zero
        uint32_t bits = sign;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
      }
      // Denormalized: convert to normalized float
      while (!(mantissa & 0x0400u)) {
        mantissa <<= 1;
        exponent--;
      }
      exponent++;
      mantissa &= ~0x0400u;
      exponent += (127 - 15);
      uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      return f;
    } else if (exponent == 31) {
      // Inf or NaN
      uint32_t bits = sign | 0x7F800000u | (mantissa << 13);
      float f;
      std::memcpy(&f, &bits, sizeof(f));
      return f;
    }

    // Normalized
    exponent += (127 - 15);
    uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  }
};

}  // namespace zvec
