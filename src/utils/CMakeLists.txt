add_library(hygon_utils STATIC
    benchmark/benchmark_printer.cpp
    matrix/matrix_generator.cpp
    memory/stream_pool.cpp
)

target_compile_options(hygon_utils PRIVATE -O3 -Wall)
target_link_libraries(hygon_utils
    PUBLIC
        rocblas
        amdhip64
    PRIVATE
        hygon_kernels
)
