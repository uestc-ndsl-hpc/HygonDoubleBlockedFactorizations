cmake_minimum_required(VERSION 3.10)
project(TRSM_Benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROCM_PATH /public/software/compiler/rocm/dtk-23.10)

include_directories(
    ${ROCM_PATH}/include
    ${PROJECT_SOURCE_DIR}/include
)

link_directories(
    ${ROCM_PATH}/lib
)

set(TRSM_SOURCES
    src/trsm/iterative_trsm.cpp
    src/trsm/recursive_trsm.cpp
    src/trsm/test_trsm.cpp
)

add_executable(benchmark_trsm ${TRSM_SOURCES})

target_link_libraries(benchmark_trsm
    rocblas
    hipblas
    amdhip64
    hsa-runtime64
)

target_compile_options(benchmark_trsm PRIVATE -O3 -Wall)