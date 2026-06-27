# Standalone benchmark — intentionally does NOT depend on libzvec, so it builds
# with nothing but a C++17 compiler and OpenMP:
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
#   ./build/distance_matrix_bench
#
# Or directly:
#   clang++ -O3 -std=c++17 -fopenmp distance_matrix_bench.cc -o bench && ./bench
cmake_minimum_required(VERSION 3.13)
project(distance_matrix_openmp_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(distance_matrix_bench distance_matrix_bench.cc)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(distance_matrix_bench PRIVATE OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found; benchmark builds serial-only (speedup ~1x).")
endif()

enable_testing()
# The benchmark self-validates: it exits non-zero if the parallel result
# diverges from the serial result, so it is also a correctness test.
add_test(NAME distance_matrix_parallel_matches_serial
         COMMAND distance_matrix_bench 256 256 64)
