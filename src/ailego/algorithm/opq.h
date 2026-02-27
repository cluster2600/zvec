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
#include <vector>
#include "product_quantizer.h"

namespace zvec {
namespace ailego {

/*! Optimized Product Quantization (OPQ)
 *
 * Learns an orthogonal rotation matrix R that minimizes quantization error
 * when applied before PQ encoding. Uses the Orthogonal Procrustes method
 * (SVD-based) to iteratively refine the rotation.
 *
 * Reference:
 *   T. Ge, K. He, Q. Ke, J. Sun. "Optimized Product Quantization."
 *   IEEE TPAMI, 2014.
 */
class OptimizedProductQuantizer {
 public:
  //! Constructor
  //! @param m Number of sub-quantizers.
  //! @param k Number of centroids per sub-quantizer.
  //! @param n_iter Number of outer OPQ iterations.
  //! @param pq_iter Number of inner k-means iterations per PQ training.
  OptimizedProductQuantizer(size_t m, size_t k, size_t n_iter = 20,
                            size_t pq_iter = 10)
      : m_(m), k_(k), n_iter_(n_iter), pq_iter_(pq_iter) {}

  //! Retrieve the number of sub-quantizers
  size_t m() const {
    return m_;
  }

  //! Retrieve the number of centroids per sub-quantizer
  size_t k() const {
    return k_;
  }

  //! Retrieve the vector dimension
  size_t dim() const {
    return dim_;
  }

  //! Check if the quantizer is trained
  bool is_trained() const {
    return is_trained_;
  }

  //! Retrieve the learned rotation matrix (dim x dim, row-major)
  const std::vector<float> &rotation_matrix() const {
    return rotation_;
  }

  //! Retrieve the underlying PQ quantizer
  const ProductQuantizer &pq() const {
    return *pq_;
  }

  //! Train the OPQ on a set of vectors.
  //! Alternates between learning the rotation R and retraining PQ codebooks.
  //! @param data Training vectors (n x dim), row-major.
  //! @param n Number of training vectors.
  //! @param dim Vector dimension (must be divisible by m).
  void train(const float *data, size_t n, size_t dim) {
    ailego_check_with(dim % m_ == 0, "Dimension must be divisible by m");
    dim_ = dim;

    // Initialize rotation as identity
    rotation_.assign(dim * dim, 0.0f);
    for (size_t i = 0; i < dim; ++i) {
      rotation_[i * dim + i] = 1.0f;
    }

    std::vector<float> rotated(n * dim);

    for (size_t iter = 0; iter < n_iter_; ++iter) {
      // Step 1: Rotate vectors: rotated = data * R^T
      MatMul(data, rotation_.data(), rotated.data(), n, dim, dim, false, true);

      // Step 2: Train PQ on rotated vectors
      pq_.reset(new ProductQuantizer(m_, k_, pq_iter_));
      pq_->train(rotated.data(), n, dim);

      // Step 3: Decode to get reconstruction
      std::vector<uint8_t> codes(n * m_);
      pq_->encode(rotated.data(), n, codes.data());
      std::vector<float> decoded(n * dim);
      pq_->decode(codes.data(), n, decoded.data());

      // Step 4: Solve Orthogonal Procrustes for optimal rotation
      //   minimize ||X * R^T - Y_hat|| where X = data, Y_hat = decoded
      //   M = X^T * Y_hat, SVD(M) = U * S * V^T, then R = V * U^T
      LearnRotation(data, decoded.data(), n, dim);
    }

    is_trained_ = true;
  }

  //! Rotate vectors using the learned rotation.
  //! @param data Input vectors (n x dim).
  //! @param n Number of vectors.
  //! @param out Output rotated vectors (n x dim).
  void rotate(const float *data, size_t n, float *out) const {
    ailego_check_with(is_trained_, "OPQ not trained");
    MatMul(data, rotation_.data(), out, n, dim_, dim_, false, true);
  }

  //! Inverse-rotate vectors (multiply by R, since R is orthogonal).
  //! @param data Input rotated vectors (n x dim).
  //! @param n Number of vectors.
  //! @param out Output original-space vectors (n x dim).
  void inverse_rotate(const float *data, size_t n, float *out) const {
    ailego_check_with(is_trained_, "OPQ not trained");
    MatMul(data, rotation_.data(), out, n, dim_, dim_, false, false);
  }

  //! Encode vectors using OPQ (rotate then PQ encode).
  //! @param data Input vectors (n x dim).
  //! @param n Number of vectors.
  //! @param codes Output PQ codes (n x m).
  void encode(const float *data, size_t n, uint8_t *codes) const {
    ailego_check_with(is_trained_, "OPQ not trained");
    std::vector<float> rotated(n * dim_);
    rotate(data, n, rotated.data());
    pq_->encode(rotated.data(), n, codes);
  }

  //! Decode PQ codes back to vectors (PQ decode then inverse rotate).
  //! @param codes Input PQ codes (n x m).
  //! @param n Number of vectors.
  //! @param out Output reconstructed vectors (n x dim).
  void decode(const uint8_t *codes, size_t n, float *out) const {
    ailego_check_with(is_trained_, "OPQ not trained");
    std::vector<float> decoded(n * dim_);
    pq_->decode(codes, n, decoded.data());
    inverse_rotate(decoded.data(), n, out);
  }

  //! Compute quantization distortion (mean squared error).
  float distortion(const float *data, size_t n) const {
    std::vector<uint8_t> codes(n * m_);
    encode(data, n, codes.data());
    std::vector<float> decoded(n * dim_);
    decode(codes.data(), n, decoded.data());

    double total = 0.0;
    for (size_t i = 0; i < n * dim_; ++i) {
      double diff = data[i] - decoded[i];
      total += diff * diff;
    }
    return static_cast<float>(total / n);
  }

 private:
  OptimizedProductQuantizer(const OptimizedProductQuantizer &) = delete;
  OptimizedProductQuantizer &operator=(const OptimizedProductQuantizer &) =
      delete;

  //! Matrix multiplication: C = A * B (or A * B^T)
  //! A is (M x K), B is (K x N) or (N x K) if transpose_b.
  static void MatMul(const float *A, const float *B, float *C, size_t M,
                     size_t K, size_t N, bool transpose_a, bool transpose_b) {
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
          float a = transpose_a ? A[p * M + i] : A[i * K + p];
          float b = transpose_b ? B[j * K + p] : B[p * N + j];
          sum += a * b;
        }
        C[i * N + j] = sum;
      }
    }
  }

  //! Solve Orthogonal Procrustes problem via SVD.
  //! Given original data X and decoded Y_hat, find R such that
  //! ||X * R^T - Y_hat||_F is minimized.
  //! Solution: M = X^T * Y_hat, SVD(M) = U * S * V^T, R = V * U^T.
  void LearnRotation(const float *X, const float *Y, size_t n, size_t dim) {
    // Compute M = X^T * Y (dim x dim)
    std::vector<float> M(dim * dim, 0.0f);
    MatMul(X, Y, M.data(), dim, n, dim, true, false);

    // SVD via Jacobi one-sided method
    std::vector<float> U(dim * dim);
    std::vector<float> S(dim);
    std::vector<float> Vt(dim * dim);
    JacobiSVD(M.data(), U.data(), S.data(), Vt.data(), dim);

    // R = V * U^T  (V = Vt^T, so R = Vt^T * U^T = (U * Vt)^T)
    // Compute U * Vt first, then transpose
    std::vector<float> UVt(dim * dim);
    MatMul(U.data(), Vt.data(), UVt.data(), dim, dim, dim, false, false);

    // R = (U * Vt)^T
    for (size_t i = 0; i < dim; ++i) {
      for (size_t j = 0; j < dim; ++j) {
        rotation_[i * dim + j] = UVt[j * dim + i];
      }
    }
  }

  //! Jacobi one-sided SVD for a square matrix.
  //! Computes A = U * diag(S) * V^T where A, U, V^T are dim x dim.
  //! Uses cyclic Jacobi rotations for convergence.
  static void JacobiSVD(const float *A, float *U, float *S, float *Vt,
                        size_t dim) {
    size_t n = dim;

    // Copy A into working matrix (will become U * S)
    std::vector<float> W(n * n);
    std::memcpy(W.data(), A, n * n * sizeof(float));

    // V starts as identity (we build V, then Vt = V^T)
    std::vector<float> V(n * n, 0.0f);
    for (size_t i = 0; i < n; ++i) V[i * n + i] = 1.0f;

    // Jacobi iterations
    const size_t max_sweeps = 100;
    const float eps = 1e-10f;

    for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
      float off_norm = 0.0f;

      for (size_t p = 0; p < n; ++p) {
        for (size_t q = p + 1; q < n; ++q) {
          // Compute 2x2 Gram matrix entries for columns p and q
          float app = 0.0f, aqq = 0.0f, apq = 0.0f;
          for (size_t i = 0; i < n; ++i) {
            app += W[i * n + p] * W[i * n + p];
            aqq += W[i * n + q] * W[i * n + q];
            apq += W[i * n + p] * W[i * n + q];
          }

          off_norm += apq * apq;

          if (std::abs(apq) < eps * std::sqrt(app * aqq)) continue;

          // Compute Jacobi rotation angle
          float tau = (aqq - app) / (2.0f * apq);
          float t;
          if (tau >= 0.0f) {
            t = 1.0f / (tau + std::sqrt(1.0f + tau * tau));
          } else {
            t = -1.0f / (-tau + std::sqrt(1.0f + tau * tau));
          }
          float c = 1.0f / std::sqrt(1.0f + t * t);
          float s = t * c;

          // Apply rotation to W columns p and q
          for (size_t i = 0; i < n; ++i) {
            float wp = W[i * n + p];
            float wq = W[i * n + q];
            W[i * n + p] = c * wp - s * wq;
            W[i * n + q] = s * wp + c * wq;
          }

          // Apply rotation to V columns p and q
          for (size_t i = 0; i < n; ++i) {
            float vp = V[i * n + p];
            float vq = V[i * n + q];
            V[i * n + p] = c * vp - s * vq;
            V[i * n + q] = s * vp + c * vq;
          }
        }
      }

      if (off_norm < eps * eps) break;
    }

    // Extract singular values and normalize columns of W to get U
    for (size_t j = 0; j < n; ++j) {
      float norm = 0.0f;
      for (size_t i = 0; i < n; ++i) {
        norm += W[i * n + j] * W[i * n + j];
      }
      S[j] = std::sqrt(norm);

      float inv_norm = (S[j] > 0.0f) ? (1.0f / S[j]) : 0.0f;
      for (size_t i = 0; i < n; ++i) {
        U[i * n + j] = W[i * n + j] * inv_norm;
      }
    }

    // Vt = V^T
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        Vt[i * n + j] = V[j * n + i];
      }
    }
  }

  size_t m_;
  size_t k_;
  size_t n_iter_;
  size_t pq_iter_;
  size_t dim_{0};
  bool is_trained_{false};
  std::vector<float> rotation_;
  std::unique_ptr<ProductQuantizer> pq_;
};

}  // namespace ailego
}  // namespace zvec
