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

#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "quantizer/rotator/rotator.h"

using zvec::core::Rotator;

namespace {

// Independent reference for out = in * matrix (row-major, dim x dim):
//   out[j] = sum_i in[i] * matrix[i * dim + j]
// Written in the straightforward j-outer / i-inner order so it does not share
// the loop structure of MatrixRotator::rotate(); it pins the value the rotate
// kernel must produce regardless of how its loops are ordered.
void reference_rotate(const float *in, const float *matrix, size_t dim,
                      float *out) {
  for (size_t j = 0; j < dim; ++j) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      sum += in[i] * matrix[i * dim + j];
    }
    out[j] = sum;
  }
}

}  // namespace

// MatrixRotator::rotate() is a cache-friendly (loop-interchanged) matrix-vector
// product. This guards the invariant that the interchange preserves: the output
// must match the plain row-major matvec across a range of dimensions, including
// non-power-of-two and odd sizes that stress the tail of the inner loop.
TEST(MatrixRotatorTest, RotateMatchesReferenceMatvec) {
  std::mt19937 gen(0x5eed);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  for (size_t dim : {1u, 2u, 3u, 7u, 16u, 31u, 64u, 128u, 257u}) {
    std::vector<float> matrix(dim * dim);
    for (auto &m : matrix) m = dist(gen);

    std::unique_ptr<Rotator> rotator = Rotator::load_matrix(matrix.data(), dim);
    ASSERT_NE(rotator, nullptr) << "dim=" << dim;
    ASSERT_EQ(rotator->dimension(), dim);

    std::vector<float> in(dim);
    for (auto &x : in) x = dist(gen);

    std::vector<float> out(dim), ref(dim);
    rotator->rotate(in.data(), out.data());
    reference_rotate(in.data(), matrix.data(), dim, ref.data());

    // Difference is only FMA/vectorization rounding; scale tolerance with dim
    // since the sums grow with the number of accumulated terms.
    const float tol = 1e-4f * static_cast<float>(dim);
    for (size_t j = 0; j < dim; ++j) {
      EXPECT_NEAR(out[j], ref[j], tol) << "dim=" << dim << " j=" << j;
    }
  }
}
