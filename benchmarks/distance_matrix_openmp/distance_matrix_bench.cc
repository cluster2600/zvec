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
//
// Benchmark: thread-parallel squared-Euclidean distance matrix.
//
// zvec's distance-matrix path (src/ailego/math/euclidean_distance_matrix*.cc)
// computes, for M database vectors and N query vectors of dimension D:
//
//     out[i][j] = sum_k (m[i][k] - q[j][k])^2
//
// The per-pair reduction over D is already SIMD-vectorized (SSE/AVX/AVX512).
// The OUTER loop over database rows i is data-parallel: each row writes a
// disjoint slice of `out` and the reduction order within a pair is untouched,
// so distributing rows across threads is legal and the result is bit-identical
// to the serial computation.
//
// This benchmark measures that outer-loop parallelization (OpenMP) against the
// serial baseline, using the same scalar squared-Euclidean kernel as zvec
// (ailego::SquaredEuclideanDistanceScalar). The schedule (parallelize the row
// loop, never the reduction) was derived and proven legal with the polyhedral
// legality engine in https://github.com/cluster2600/cluster_compilot, which
// rejects parallelizing the reduction dimension and accepts the row loop.
//
// Build (self-contained, no libzvec required):
//     clang++ -O3 -std=c++17 -fopenmp distance_matrix_bench.cc -o bench
//     ./bench
//
// Self-validating: aborts (non-zero exit) if the parallel result diverges from
// the serial result, so it doubles as a correctness test.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// Same kernel as zvec ailego::SquaredEuclideanDistanceScalar: float accumulator,
// sum of squared per-dimension differences.
inline float SquaredEuclideanDistanceScalar(const float *m, const float *q,
                                            std::size_t dim) {
  float sum = 0.0f;
  for (std::size_t k = 0; k < dim; ++k) {
    float d = m[k] - q[k];
    sum += d * d;
  }
  return sum;
}

// out[i*N + j] = dist(db[i], query[j]); row loop runs serially.
void DistanceMatrixSerial(const float *db, const float *query, float *out,
                          std::size_t M, std::size_t N, std::size_t dim) {
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      out[i * N + j] =
          SquaredEuclideanDistanceScalar(db + i * dim, query + j * dim, dim);
    }
  }
}

// Identical, but the data-parallel row loop is distributed across threads.
void DistanceMatrixParallel(const float *db, const float *query, float *out,
                            std::size_t M, std::size_t N, std::size_t dim) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(M); ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      out[i * N + j] =
          SquaredEuclideanDistanceScalar(db + i * dim, query + j * dim, dim);
    }
  }
}

double TimeSeconds(void (*fn)(const float *, const float *, float *,
                              std::size_t, std::size_t, std::size_t),
                   const float *db, const float *query, float *out,
                   std::size_t M, std::size_t N, std::size_t dim) {
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {  // ponytail: best-of-3, warm cache, kills noise
    auto t0 = std::chrono::steady_clock::now();
    fn(db, query, out, M, N, dim);
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    if (dt < best) best = dt;
  }
  return best;
}

}  // namespace

int main(int argc, char **argv) {
  std::size_t M = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 1024;  // db rows
  std::size_t N = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : 1024;  // queries
  std::size_t dim = (argc > 3) ? std::strtoul(argv[3], nullptr, 10) : 128;

  std::vector<float> db(M * dim), query(N * dim);
  std::vector<float> out_serial(M * N), out_parallel(M * N);

  // Deterministic, reproducible inputs (no RNG seed dependence).
  for (std::size_t i = 0; i < db.size(); ++i)
    db[i] = static_cast<float>((i * 7 + 13) % 97) / 97.0f;
  for (std::size_t i = 0; i < query.size(); ++i)
    query[i] = static_cast<float>((i * 11 + 5) % 89) / 89.0f;

  double t_serial = TimeSeconds(DistanceMatrixSerial, db.data(), query.data(),
                                out_serial.data(), M, N, dim);
  double t_parallel = TimeSeconds(DistanceMatrixParallel, db.data(),
                                  query.data(), out_parallel.data(), M, N, dim);

  // Correctness gate: parallel must equal serial (bit-identical here; allow a
  // tiny epsilon for safety). Exits non-zero on divergence -> usable as a test.
  double max_abs_diff = 0.0;
  for (std::size_t i = 0; i < M * N; ++i) {
    double d = std::fabs(static_cast<double>(out_parallel[i]) - out_serial[i]);
    if (d > max_abs_diff) max_abs_diff = d;
  }

  int threads = 1;
#ifdef _OPENMP
  threads = omp_get_max_threads();
#endif

  std::printf("M=%zu N=%zu dim=%zu threads=%d\n", M, N, dim, threads);
  std::printf("serial    : %.4f s\n", t_serial);
  std::printf("parallel  : %.4f s\n", t_parallel);
  std::printf("speedup   : %.2fx\n", t_serial / t_parallel);
  std::printf("max_abs_diff (parallel vs serial): %.3e\n", max_abs_diff);

  if (max_abs_diff > 1e-3) {
    std::fprintf(stderr, "FAIL: parallel result diverges from serial\n");
    return 1;
  }
  std::printf("OK: parallel matches serial\n");
  return 0;
}
