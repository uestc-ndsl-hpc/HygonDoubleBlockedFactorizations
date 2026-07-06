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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

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
