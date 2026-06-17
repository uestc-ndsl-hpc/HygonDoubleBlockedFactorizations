cmake_minimum_required(VERSION 3.16)

set(ROCM_PATH /opt/dtk-26.04 CACHE PATH "DTK/ROCm installation path")
set(HYGON_AMDGPU_TARGET_ID "gfx928:mai-insts+" CACHE STRING
    "AMDGPU target id for K100_AI/DCU builds")
option(HYGON_ENABLE_MFMA_KERNEL "Build the experimental K100_AI MFMA HSS kernel" ON)

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

project(HygonCholeskyDecomposition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(__HIP_PLATFORM_AMD__)
add_compile_definitions(HYGON_AMDGPU_TARGET_ID="${HYGON_AMDGPU_TARGET_ID}")
if(HYGON_ENABLE_MFMA_KERNEL)
    add_compile_definitions(HYGON_ENABLE_MFMA_KERNEL=1)
else()
    add_compile_definitions(HYGON_ENABLE_MFMA_KERNEL=0)
endif()

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:--offload-arch=${HYGON_AMDGPU_TARGET_ID}>")

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
