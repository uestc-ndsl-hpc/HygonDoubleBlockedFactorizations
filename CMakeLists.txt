cmake_minimum_required(VERSION 3.16)

set(ROCM_PATH /opt/dtk CACHE PATH "DTK/ROCm installation path")

if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(HIPCC_EXECUTABLE
        NAMES hipcc
        HINTS
            ${ROCM_PATH}/bin
            /opt/dtk/bin
            /opt/dtk-26.04/bin
            /opt/rocm/bin
            /public/software/compiler/rocm/dtk-24.04/bin
            /public/software/compiler/rocm/dtk-23.10/bin
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

function(apply_benchmark_options target_name)
    target_compile_options(${target_name} PRIVATE -O3 -Wall)
endfunction()

function(link_rocm_runtime target_name)
    target_link_libraries(${target_name}
        rocblas
        hipblas
        amdhip64
        hsa-runtime64
    )
endfunction()

add_executable(benchmark_gemm
    ${GEMM_BENCHMARK_SOURCES}
)

apply_benchmark_options(benchmark_gemm)
link_rocm_runtime(benchmark_gemm)

add_executable(benchmark_trsm
    ${TRSM_SOURCES}
)

apply_benchmark_options(benchmark_trsm)
link_rocm_runtime(benchmark_trsm)

add_executable(benchmark_syrk
    ${SYRK_SOURCES}
)

apply_benchmark_options(benchmark_syrk)
link_rocm_runtime(benchmark_syrk)

add_executable(benchmark_potrf
    ${POTRF_SOURCES}
)

apply_benchmark_options(benchmark_potrf)
link_rocm_runtime(benchmark_potrf)

target_link_libraries(benchmark_potrf
    rocsolver
)
