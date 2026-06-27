# Benchmarks are opt-in: configure with -DBUILD_BENCHMARKS=ON.
# Each benchmark links zvec_ailego and uses zvec's own primitives
# (ailego::ThreadPool + the SIMD-dispatched distance kernels).

cc_binary(
    NAME distance_scan_bench
    SRCS distance_scan_bench.cc
    LIBS zvec_ailego
    INCS ${PROJECT_ROOT_DIR}/src/include/zvec
  )

# The benchmark self-validates (parallel vs serial, bit-identical) and exits
# non-zero on mismatch, so register it as a correctness test too.
add_test(NAME distance_scan_bench COMMAND distance_scan_bench 512 64 64)
