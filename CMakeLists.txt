cmake_minimum_required(VERSION 3.16)
project(HygonCholeskyDecomposition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROCM_PATH /opt/dtk-26.04 CACHE PATH "DTK/ROCm installation path")

add_compile_definitions(__HIP_PLATFORM_AMD__)

include_directories(
    ${ROCM_PATH}/include
    ${PROJECT_SOURCE_DIR}/include
)

link_directories(
    ${ROCM_PATH}/lib
    ${ROCM_PATH}/llvm/lib
)

function(hygon_apply_benchmark_options target_name)
    target_compile_options(${target_name} PRIVATE -O3 -Wall)
endfunction()

function(hygon_link_rocm target_name)
    target_link_libraries(${target_name}
        rocblas
        hipblas
        amdhip64
        hsa-runtime64
    )
endfunction()

add_subdirectory(src/kernels)
add_subdirectory(src/utils)
add_subdirectory(src/gemm)
add_subdirectory(src/trsm)
add_subdirectory(src/syrk)
add_subdirectory(src/potrf)
