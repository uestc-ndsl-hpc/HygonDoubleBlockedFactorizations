cmake_minimum_required(VERSION 3.16)

set(ROCM_PATH /opt/dtk-26.04 CACHE PATH "DTK/ROCm installation path")

if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(HIPCC_EXECUTABLE
        NAMES hipcc
        HINTS
            ${ROCM_PATH}/bin
            /opt/rocm/bin
    )

    if(HIPCC_EXECUTABLE)
        set(CMAKE_CXX_COMPILER ${HIPCC_EXECUTABLE} CACHE FILEPATH "CXX compiler" FORCE)
    endif()
endif()

project(TRSM_Benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${ROCM_PATH}/include
    ${PROJECT_SOURCE_DIR}/include
)

link_directories(
    ${ROCM_PATH}/lib
    ${ROCM_PATH}/llvm/lib
)

add_compile_definitions(
    __HIP_PLATFORM_AMD__
)

set(KERNEL_SOURCES
    src/kernels/add_damping.hip
    src/kernels/fp16_convert.hip
    src/kernels/matrix_scale.hip
)

set_source_files_properties(${KERNEL_SOURCES} PROPERTIES LANGUAGE CXX)

set(COMMON_UTIL_SOURCES
    src/utils/memory/stream_pool.cpp
    src/utils/benchmark/benchmark_printer.cpp
    src/utils/matrix/matrix_generator.cpp
)

set(GEMM_SOURCES
    src/gemm/mixed_gemm.cpp
)

set(GEMM_BENCHMARK_SOURCES
    src/gemm/gemm_benchmark.cpp
    ${COMMON_UTIL_SOURCES}
    ${KERNEL_SOURCES}
)

set(TRSM_SOURCES
    src/trsm/iterative_trsm.cpp
    src/trsm/recursive_trsm.cpp
    src/trsm/test_trsm.cpp
    ${COMMON_UTIL_SOURCES}
    ${GEMM_SOURCES}
    ${KERNEL_SOURCES}
)

set(SYRK_SOURCES
    src/syrk/batched_syrk.cpp
    src/syrk/recursive_syrk.cpp
    src/syrk/test_syrk.cpp
    ${COMMON_UTIL_SOURCES}
    ${GEMM_SOURCES}
    ${KERNEL_SOURCES}
)

set(POTRF_SOURCES
    src/potrf/recursive_potrf.cpp
    src/potrf/test_potrf.cpp
    src/trsm/recursive_trsm.cpp
    src/syrk/batched_syrk.cpp
    ${GEMM_SOURCES}
    ${COMMON_UTIL_SOURCES}
    ${KERNEL_SOURCES}
)

add_executable(benchmark_gemm
    ${GEMM_BENCHMARK_SOURCES}
)

target_compile_options(benchmark_gemm PRIVATE -O3 -Wall)

target_link_libraries(benchmark_gemm
    rocblas
    hipblas
    amdhip64
    hsa-runtime64
)

add_executable(benchmark_trsm
    ${TRSM_SOURCES}
)

target_compile_options(benchmark_trsm PRIVATE -O3 -Wall)

target_link_libraries(benchmark_trsm
    rocblas
    hipblas
    amdhip64
    hsa-runtime64
)

add_executable(benchmark_syrk
    ${SYRK_SOURCES}
)

target_compile_options(benchmark_syrk PRIVATE -O3 -Wall)

target_link_libraries(benchmark_syrk
    rocblas
    hipblas
    amdhip64
    hsa-runtime64
)

add_executable(benchmark_potrf
    ${POTRF_SOURCES}
)

target_compile_options(benchmark_potrf PRIVATE -O3 -Wall)

target_link_libraries(benchmark_potrf
    rocblas
    rocsolver
    hipblas
    amdhip64
    hsa-runtime64
)
