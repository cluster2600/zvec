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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace ailego {

/*! Product Quantization (PQ) Algorithm
 *
 * Splits vectors into m sub-vectors and quantizes each independently
 * using k-means clustering. Encodes each sub-vector as an index into
 * its sub-codebook.
 *
 * Reference:
 *   H. Jegou, M. Douze, C. Schmid. "Product Quantization for Nearest
 *   Neighbor Search." IEEE TPAMI, 2011.
 */
class ProductQuantizer {
 public:
  //! Constructor
  //! @param m Number of sub-quantizers (sub-vectors).
  //! @param k Number of centroids per sub-quantizer.
  //! @param n_iter Number of k-means iterations.
  ProductQuantizer(size_t m, size_t k, size_t n_iter = 20)
      : m_(m), k_(k), n_iter_(n_iter) {}

  //! Retrieve the number of sub-quantizers
  size_t m() const {
    return m_;
  }

  //! Retrieve the number of centroids per sub-quantizer
  size_t k() const {
    return k_;
  }

  //! Retrieve the sub-vector dimension
  size_t sub_dim() const {
    return sub_dim_;
  }

  //! Retrieve the full vector dimension
  size_t dim() const {
    return m_ * sub_dim_;
  }

  //! Check if the quantizer is trained
  bool is_trained() const {
    return is_trained_;
  }

  //! Retrieve codebook data (m * k * sub_dim)
  const std::vector<float> &codebooks() const {
    return codebooks_;
  }

  //! Retrieve a pointer to centroids for sub-quantizer i, centroid j
  const float *centroid(size_t i, size_t j) const {
    return codebooks_.data() + (i * k_ + j) * sub_dim_;
  }

  //! Train the PQ codebooks on a set of vectors.
  //! @param data Training vectors (n x dim), row-major.
  //! @param n Number of training vectors.
  //! @param dim Vector dimension (must be divisible by m).
  void train(const float *data, size_t n, size_t dim) {
    ailego_check_with(dim % m_ == 0, "Dimension must be divisible by m");
    sub_dim_ = dim / m_;
    size_t actual_k = std::min(k_, std::max<size_t>(1, n / 4));

    codebooks_.resize(m_ * actual_k * sub_dim_);

    std::mt19937 rng(42);

    for (size_t s = 0; s < m_; ++s) {
      // Extract sub-vectors for this partition
      std::vector<float> sub(n * sub_dim_);
      for (size_t i = 0; i < n; ++i) {
        std::memcpy(sub.data() + i * sub_dim_, data + i * dim + s * sub_dim_,
                    sub_dim_ * sizeof(float));
      }

      // Initialize centroids randomly
      std::vector<size_t> indices(n);
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), rng);

      float *centroids = codebooks_.data() + s * actual_k * sub_dim_;
      for (size_t j = 0; j < actual_k; ++j) {
        std::memcpy(centroids + j * sub_dim_,
                    sub.data() + indices[j] * sub_dim_,
                    sub_dim_ * sizeof(float));
      }

      // K-means iterations
      std::vector<size_t> labels(n);
      std::vector<float> new_centroids(actual_k * sub_dim_);
      std::vector<size_t> counts(actual_k);

      for (size_t iter = 0; iter < n_iter_; ++iter) {
        // Assignment step
        for (size_t i = 0; i < n; ++i) {
          float best_dist = std::numeric_limits<float>::max();
          size_t best_j = 0;
          for (size_t j = 0; j < actual_k; ++j) {
            float dist = 0.0f;
            for (size_t d = 0; d < sub_dim_; ++d) {
              float diff = sub[i * sub_dim_ + d] - centroids[j * sub_dim_ + d];
              dist += diff * diff;
            }
            if (dist < best_dist) {
              best_dist = dist;
              best_j = j;
            }
          }
          labels[i] = best_j;
        }

        // Update step
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < n; ++i) {
          size_t c = labels[i];
          counts[c]++;
          for (size_t d = 0; d < sub_dim_; ++d) {
            new_centroids[c * sub_dim_ + d] += sub[i * sub_dim_ + d];
          }
        }
        for (size_t j = 0; j < actual_k; ++j) {
          if (counts[j] > 0) {
            float inv = 1.0f / static_cast<float>(counts[j]);
            for (size_t d = 0; d < sub_dim_; ++d) {
              centroids[j * sub_dim_ + d] =
                  new_centroids[j * sub_dim_ + d] * inv;
            }
          }
        }
      }
    }

    k_ = actual_k;
    is_trained_ = true;
  }

  //! Encode vectors to PQ codes.
  //! @param data Input vectors (n x dim), row-major.
  //! @param n Number of vectors.
  //! @param codes Output codes (n x m), row-major uint8_t.
  void encode(const float *data, size_t n, uint8_t *codes) const {
    ailego_check_with(is_trained_, "Quantizer not trained");
    size_t dim = m_ * sub_dim_;

    for (size_t i = 0; i < n; ++i) {
      for (size_t s = 0; s < m_; ++s) {
        const float *sub = data + i * dim + s * sub_dim_;
        const float *centroids = codebooks_.data() + s * k_ * sub_dim_;
        float best_dist = std::numeric_limits<float>::max();
        uint8_t best_j = 0;
        for (size_t j = 0; j < k_; ++j) {
          float dist = 0.0f;
          for (size_t d = 0; d < sub_dim_; ++d) {
            float diff = sub[d] - centroids[j * sub_dim_ + d];
            dist += diff * diff;
          }
          if (dist < best_dist) {
            best_dist = dist;
            best_j = static_cast<uint8_t>(j);
          }
        }
        codes[i * m_ + s] = best_j;
      }
    }
  }

  //! Decode PQ codes back to approximate vectors.
  //! @param codes Input codes (n x m), row-major uint8_t.
  //! @param n Number of vectors.
  //! @param out Output vectors (n x dim), row-major.
  void decode(const uint8_t *codes, size_t n, float *out) const {
    ailego_check_with(is_trained_, "Quantizer not trained");
    size_t dim = m_ * sub_dim_;

    for (size_t i = 0; i < n; ++i) {
      for (size_t s = 0; s < m_; ++s) {
        uint8_t c = codes[i * m_ + s];
        const float *centroid_ptr = codebooks_.data() + (s * k_ + c) * sub_dim_;
        std::memcpy(out + i * dim + s * sub_dim_, centroid_ptr,
                    sub_dim_ * sizeof(float));
      }
    }
  }

  //! Compute quantization distortion (mean squared error).
  //! @param data Original vectors (n x dim).
  //! @param n Number of vectors.
  //! @return Average squared reconstruction error.
  float distortion(const float *data, size_t n) const {
    size_t dim = m_ * sub_dim_;
    std::vector<uint8_t> codes(n * m_);
    encode(data, n, codes.data());
    std::vector<float> decoded(n * dim);
    decode(codes.data(), n, decoded.data());

    double total = 0.0;
    for (size_t i = 0; i < n * dim; ++i) {
      double diff = data[i] - decoded[i];
      total += diff * diff;
    }
    return static_cast<float>(total / n);
  }

 private:
  ProductQuantizer(const ProductQuantizer &) = delete;
  ProductQuantizer &operator=(const ProductQuantizer &) = delete;

  size_t m_;
  size_t k_;
  size_t n_iter_;
  size_t sub_dim_{0};
  bool is_trained_{false};
  std::vector<float> codebooks_;
};

}  // namespace ailego
}  // namespace zvec
