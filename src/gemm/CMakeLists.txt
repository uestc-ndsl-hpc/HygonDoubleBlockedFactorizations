add_library(hygon_gemm STATIC
    mixed_gemm.cpp
)

hygon_apply_benchmark_options(hygon_gemm)
target_link_libraries(hygon_gemm
    PUBLIC
        rocblas
        amdhip64
    PRIVATE
        hygon_kernels
)

add_executable(benchmark_gemm
    gemm_benchmark.cpp
)

hygon_apply_benchmark_options(benchmark_gemm)
hygon_link_rocm(benchmark_gemm)
target_link_libraries(benchmark_gemm
    hygon_utils
)
