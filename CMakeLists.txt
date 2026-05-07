cmake_minimum_required(VERSION 3.10)

project(TRSM_Benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROCM_PATH /public/software/compiler/rocm/dtk-23.10)

# =========================================================
# Include Directories
# =========================================================

include_directories(
    ${ROCM_PATH}/include
    ${PROJECT_SOURCE_DIR}/include
)

# =========================================================
# Link Directories
# =========================================================

link_directories(
    ${ROCM_PATH}/lib
)

# =========================================================
# Common Compile Settings
# =========================================================

add_compile_definitions(
    __HIP_PLATFORM_AMD__
)

# =========================================================
# TRSM Sources
# =========================================================

set(TRSM_SOURCES
    src/trsm/iterative_trsm.cpp
    src/trsm/recursive_trsm.cpp
    src/trsm/test_trsm.cpp
)

# =========================================================
# SYRK Sources
# =========================================================

set(SYRK_SOURCES
    src/syrk/batched_syrk.cpp
    src/syrk/recursive_syrk.cpp
    src/syrk/test_syrk.cpp
)

# =========================================================
# TRSM Executable
# =========================================================

add_executable(benchmark_trsm
    ${TRSM_SOURCES}
)

target_compile_options(benchmark_trsm PRIVATE
    -O3
    -Wall
)

target_link_libraries(benchmark_trsm
    rocblas
    hipblas
    amdhip64
    hsa-runtime64
)

# =========================================================
# SYRK Executable
# =========================================================

add_executable(benchmark_syrk
    ${SYRK_SOURCES}
)

target_compile_options(benchmark_syrk PRIVATE
    -O3
    -Wall
)

target_link_libraries(benchmark_syrk
    rocblas
    hipblas
    amdhip64
    hsa-runtime64
)