set(KERNEL_SOURCES
    add_damping.hip
    fp16_convert.hip
    matrix_scale.hip
)

set_source_files_properties(${KERNEL_SOURCES} PROPERTIES LANGUAGE CXX)

add_library(hygon_kernels STATIC
    ${KERNEL_SOURCES}
)

target_compile_options(hygon_kernels PRIVATE -O3 -Wall)
target_link_libraries(hygon_kernels
    amdhip64
)
