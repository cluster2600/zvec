# CMakeLists.txt for zvec GPU modules
# 
# Features:
# - CUDA support (coalesced kernels)
# - Metal support (Apple Silicon)
# - SIMD CPU support (AVX2, NEON)
# - cuVS integration (optional)

cmake_minimum_required(VERSION 3.18)
project(zvec_gpu LANGUAGES CXX CUDA)

# Options
option(ZVEC_ENABLE_CUDA "Enable CUDA support" ON)
option(ZVEC_ENABLE_METAL "Enable Metal support (Apple Silicon)" ON)
option(ZVEC_ENABLE_CUVS "Enable cuVS integration" OFF)
option(ZVEC_BUILD_TESTS "Build tests" ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
if(ZVEC_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    # CUDA architectures
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    
    # CUDA flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

# Metal (only on macOS)
if(ZVEC_ENABLE_METAL)
    if(APPLE)
        enable_language(OBJCXX)
        set(METAL_LIBRARY_PATH "/usr/local/lib/libMetal.framework")
    else()
        set(ZVEC_ENABLE_METAL OFF)
        message(STATUS "Metal only available on macOS, disabling")
    endif()
endif()

# cuVS (optional)
if(ZVEC_ENABLE_CUVS)
    find_path(CUVS_INCLUDE_DIR "cuvs" PATHS /usr/local /usr)
    if(CUVS_INCLUDE_DIR)
        message(STATUS "cuVS found at ${CUVS_INCLUDE_DIR}")
    else()
        set(ZVEC_ENABLE_CUVS OFF)
        message(WARNING "cuVS not found, disabling")
    endif()
endif()

# Source files
set(GPU_SOURCES
    src/ailego/gpu/cuda/coalesce.cu
)

set(GPU_HEADERS
    src/ailego/gpu/cuda/coalesce.cuh
    src/ailego/gpu/cuvs/zvec_cuvs.h
)

set(CPU_SOURCES
    src/ailego/cpu/simd_distance.cc
)

set(CPU_HEADERS
    src/ailego/cpu/simd_distance.h
)

# Build GPU library
if(ZVEC_ENABLE_CUDA)
    add_library(zvec_gpu_cuda STATIC ${GPU_SOURCES} ${GPU_HEADERS})
    target_include_directories(zvec_gpu_cuda PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(zvec_gpu_cuda CUDA::cudart)
    set_target_properties(zvec_gpu_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Build Metal library
if(ZVEC_ENABLE_METAL)
    set(METAL_SOURCES
        src/ailego/gpu/metal/distance.metal
    )
    
    # Compile Metal shaders
    find_program(METAL_LIBRARYCompiler metallib)
    if(METAL_LIBRARYCompiler)
        add_custom_target(zvec_metal_shaders ALL
            COMMAND ${METAL_LIBRARYCompiler} 
                ${METAL_SOURCES}
                -o ${CMAKE_BINARY_DIR}/libzvec_metal.air
            COMMENT "Compiling Metal shaders"
        )
    endif()
    
    add_library(zvec_metal STATIC ${METAL_SOURCES})
    set_target_properties(zvec_metal PROPERTIES
        LINKER_LANGUAGE OBJCXX
    )
endif()

# Build CPU SIMD library
add_library(zvec_cpu_simd STATIC ${CPU_SOURCES} ${CPU_HEADERS})
target_include_directories(zvec_cpu_simd PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_compile_options(zvec_cpu_simd PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-march=native -mfma>
    $<$<CXX_COMPILER_ID:Clang>:-march=native -mfma>
    $<$<CXX_COMPILER_ID:AppleClang>:-mcpu=apple-m1>
)

# Build main library
add_library(zvec_gpu INTERFACE)

if(ZVEC_ENABLE_CUDA)
    target_link_libraries(zvec_gpu INTERFACE zvec_gpu_cuda)
endif()

if(ZVEC_ENABLE_METAL)
    target_link_libraries(zvec_gpu INTERFACE zvec_metal)
endif()

target_link_libraries(zvec_gpu INTERFACE zvec_cpu_simd)

# cuVS integration
if(ZVEC_ENABLE_CUVS)
    target_include_directories(zvec_gpu_cuda INTERFACE ${CUVS_INCLUDE_DIR})
    target_compile_definitions(zvec_gpu PUBLIC ZVEC_ENABLE_CUVS)
endif()

# Tests
if(ZVET_BUILD_TESTS)
    enable_testing()
    
    add_executable(test_gpu test_gpu.cc)
    target_link_libraries(test_gpu zvec_gpu)
    
    add_test(NAME gpu_test COMMAND test_gpu)
endif()

# Installation
install(TARGETS zvec_gpu zvec_cpu_simd
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY src/
    DESTINATION include/zvec
    FILES_MATCHING PATTERN "*.h"
)
