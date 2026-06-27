# Distance-matrix OpenMP benchmark

Measures **thread-parallelizing the outer (database-row) loop** of zvec's
squared-Euclidean distance matrix against the serial baseline.

For `M` database vectors and `N` query vectors of dimension `D`, the
distance-matrix path
(`src/ailego/math/euclidean_distance_matrix*.cc`) computes:

```
out[i][j] = sum_k (m[i][k] - q[j][k])^2
```

zvec already SIMD-vectorizes the **inner** per-pair reduction over `D`
(SSE/AVX/AVX512). This benchmark targets the **orthogonal** axis: the outer loop
over database rows `i`. Each row writes a disjoint slice of `out`, and the
reduction order within a pair is unchanged, so distributing rows across threads
is legal and the result is **bit-identical** to the serial computation.

## Why this is safe (and the reduction loop is not)

The schedule — *parallelize the row loop, never the reduction dimension* — was
derived and proven legal with the polyhedral legality engine in
[**cluster_compilot**](https://github.com/cluster2600/cluster_compilot), a
from-scratch implementation of *Agentic Auto-Scheduling* (arXiv:2511.00592).
Modeling this kernel there:

- `parallel(i)` (database rows) → **legal** — no dependence is carried on `i`.
- `parallel(k)` (the `D` reduction) → **rejected** — `k` carries the
  accumulation dependence.

The benchmark encodes exactly that: OpenMP on the `i` loop only, and a runtime
correctness gate (it exits non-zero if the parallel result diverges from the
serial one, so it doubles as a test).

## Build & run

Self-contained — needs only a C++17 compiler and OpenMP, **not** a libzvec build:

```bash
# one-liner
clang++ -O3 -std=c++17 -fopenmp distance_matrix_bench.cc -o bench && ./bench

# or via CMake (adds a ctest correctness test)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build --output-on-failure
./build/distance_matrix_bench [M] [N] [dim]     # default 1024 1024 128
```

## Results

`M=N=1024, dim=128`, Apple Silicon (10 cores), Apache `clang++ -O3`:

| Threads | Speedup |
|--------:|--------:|
| 1 | 1.00× |
| 2 | 1.98× |
| 4 | 3.88× |
| 8 | 5.16× |

`max_abs_diff` (parallel vs serial) = **0.000e+00** at every thread count —
the parallel result is bit-identical.

Scaling is near-linear until it saturates around the physical core count
(this kernel becomes memory-bandwidth-bound once enough threads stream the
input vectors). The win is independent of, and composes with, the existing SIMD
per-pair kernels.

## Notes

- The benchmark reproduces zvec's scalar kernel
  (`ailego::SquaredEuclideanDistanceScalar`) to stay self-contained. Wiring the
  same outer-loop OpenMP schedule into the SIMD-dispatched matrix kernels and
  linking against a built `libzvec` is a natural follow-up.
- Inputs are deterministic (no RNG seed), so results are reproducible.
