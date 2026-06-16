set(KERNEL_SOURCES
    add_damping.hip
    fp16_convert.hip
    hss_gemm.hip
    matrix_scale.hip
)

set_source_files_properties(${KERNEL_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(hygon_kernels STATIC
    ${KERNEL_SOURCES}
)

target_compile_options(hygon_kernels PRIVATE -O3 -Wall)

option(HYGON_USE_ROCWMMA "Enable rocWMMA HSS GEMM fast path when headers are available" ON)

if(HYGON_USE_ROCWMMA)
    find_path(ROCWMMA_INCLUDE_DIR
        NAMES rocwmma/rocwmma.hpp
        HINTS
            ${ROCM_PATH}/include
            ${ROCM_PATH}/rocwmma/include
            /opt/rocm/include
            /opt/rocm/rocwmma/include
    )

    if(ROCWMMA_INCLUDE_DIR)
        target_include_directories(hygon_kernels PRIVATE ${ROCWMMA_INCLUDE_DIR})
        target_compile_definitions(hygon_kernels PRIVATE HYGON_ENABLE_ROCWMMA=1)
        message(STATUS "HSS GEMM rocWMMA fast path enabled: ${ROCWMMA_INCLUDE_DIR}")
    else()
        message(STATUS "HSS GEMM rocWMMA fast path disabled: rocwmma/rocwmma.hpp not found")
    endif()
else()
    message(STATUS "HSS GEMM rocWMMA fast path disabled by HYGON_USE_ROCWMMA=OFF")
endif()

target_link_libraries(hygon_kernels
    amdhip64
)
