add_library(hygon_utils STATIC
    benchmark/benchmark_printer.cpp
    matrix/matrix_generator.cpp
    memory/stream_pool.cpp
)

apply_benchmark_options(hygon_utils)
target_link_libraries(hygon_utils
    PUBLIC
        rocblas
        amdhip64
    PRIVATE
        hygon_kernels
)
