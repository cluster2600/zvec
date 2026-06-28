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
// Benchmark: per-query parallel flat scan using zvec's OWN primitives.
//
// This measures the way zvec actually scales a one-query-vs-many-database scan:
//   * the per-pair distance is zvec's SIMD-dispatched kernel
//     (ailego::Distance::SquaredEuclidean -> SSE/AVX/AVX512/NEON at runtime),
//   * parallelism is applied PER QUERY across zvec's ailego::ThreadPool, never
//     inside a single query's reduction.
// That is exactly zvec's architecture: the single-query scan stays serial, and
// throughput comes from distributing queries over the thread pool. No OpenMP,
// no hand-written intrinsics.
//
// The parallel result is validated bit-for-bit against the serial reference
// (same kernel, same accumulation order, disjoint output rows), so the program
// doubles as a correctness test: it exits non-zero on any mismatch.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>
#include <ailego/internal/cpu_features.h>
#include <ailego/math/distance.h>
#include <ailego/parallel/thread_pool.h>  // pulls in pattern/closure.h

using zvec::ailego::Closure;
using zvec::ailego::Distance;
using zvec::ailego::ThreadPool;

namespace {

// Deterministic synthetic data (no RNG) so runs are reproducible. The kernel is
// a dense, branch-free reduction, so timing depends on shape/access pattern,
// not on the values.
std::vector<float> make_matrix(size_t rows, size_t dim, uint32_t a, uint32_t b,
                               uint32_t mod) {
  std::vector<float> m(rows * dim);
  for (size_t i = 0; i < m.size(); ++i) {
    m[i] = static_cast<float>((i * a + b) % mod) / static_cast<float>(mod);
  }
  return m;
}

// Scan the queries whose index == t (mod stride) against all database vectors,
// writing out[j * num_db + i]. One query is scanned serially with the
// SIMD-dispatched kernel; queries are split across threads by the caller.
void scan_query_shard(const float *db, const float *queries, size_t num_db,
                      size_t num_query, size_t dim, size_t t, size_t stride,
                      float *out) {
  for (size_t j = t; j < num_query; j += stride) {
    const float *q = queries + j * dim;
    float *o = out + j * num_db;
    for (size_t i = 0; i < num_db; ++i) {
      o[i] = Distance::SquaredEuclidean(db + i * dim, q, dim);
    }
  }
}

double now_sec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Best-of-`reps` wall time for the per-query parallel scan over `nthreads`.
double timed_parallel_scan(const float *db, const float *queries, size_t num_db,
                           size_t num_query, size_t dim, size_t nthreads,
                           float *out, int reps) {
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    double t0 = now_sec();
    if (nthreads <= 1) {
      scan_query_shard(db, queries, num_db, num_query, dim, 0, 1, out);
    } else {
      ThreadPool pool(static_cast<uint32_t>(nthreads), false);
      auto group = pool.make_group();
      for (size_t t = 0; t < nthreads; ++t) {
        group->submit(Closure::New(scan_query_shard, db, queries, num_db,
                                   num_query, dim, t, nthreads, out));
      }
      group->wait_finish();
    }
    double dt = now_sec() - t0;
    if (dt < best) best = dt;
  }
  return best;
}

}  // namespace

int main(int argc, char **argv) {
  size_t num_db = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 4096;
  size_t num_query = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : 256;
  size_t dim = (argc > 3) ? std::strtoul(argv[3], nullptr, 10) : 128;

  const int reps = 5;
  std::vector<float> db = make_matrix(num_db, dim, 7, 13, 97);
  std::vector<float> queries = make_matrix(num_query, dim, 11, 5, 89);

  std::printf("zvec per-query parallel flat scan\n");
  std::printf("  database=%zu  queries=%zu  dim=%zu\n", num_db, num_query, dim);
  std::printf("  distance kernel SIMD path: %s\n",
              zvec::ailego::internal::CpuFeatures::Intrinsics());
  std::printf("  hardware_concurrency: %u\n",
              std::thread::hardware_concurrency());

  // Serial reference (no thread pool).
  std::vector<float> ref(num_db * num_query);
  double serial = timed_parallel_scan(db.data(), queries.data(), num_db,
                                      num_query, dim, 1, ref.data(), reps);
  std::printf("\n  serial   : %.4f s\n", serial);

  // Per-query parallel scan at increasing thread counts; validate each.
  std::vector<size_t> thread_counts = {2, 4, 8};
  std::vector<float> out(num_db * num_query);
  std::printf("\n  threads   speedup   max_abs_diff\n");
  std::printf("  %7d  %7.2fx  %12.3e\n", 1, 1.0, 0.0);

  int failures = 0;
  for (size_t nt : thread_counts) {
    if (nt > num_query) break;
    std::fill(out.begin(), out.end(), -1.0f);
    double t = timed_parallel_scan(db.data(), queries.data(), num_db, num_query,
                                   dim, nt, out.data(), reps);

    float max_diff = 0.0f;
    for (size_t k = 0; k < out.size(); ++k) {
      float d = std::abs(out[k] - ref[k]);
      if (d > max_diff) max_diff = d;
    }
    std::printf("  %7zu  %7.2fx  %12.3e\n", nt, serial / t, max_diff);
    if (max_diff != 0.0f) ++failures;  // same kernel + order => must be exact
  }

  if (failures) {
    std::printf("\nFAIL: parallel scan diverged from serial reference\n");
    return 1;
  }
  std::printf("\nOK: per-query parallel scan is bit-identical to serial\n");
  return 0;
}
