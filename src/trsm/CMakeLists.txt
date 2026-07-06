add_library(hygon_trsm STATIC
    iterative_trsm.cpp
    recursive_trsm.cpp
)

apply_benchmark_options(hygon_trsm)
target_link_libraries(hygon_trsm
    PUBLIC
        rocblas
        amdhip64
    PRIVATE
        hygon_gemm
        hygon_kernels
        hygon_utils
)

add_executable(benchmark_trsm
    test_trsm.cpp
    benchmark/trsm_benchmark_common.cpp
    benchmark/trsm_iterative_cases.cpp
    benchmark/trsm_mixed_cases.cpp
    benchmark/trsm_recursive_cases.cpp
    benchmark/trsm_rocblas_cases.cpp
)

apply_benchmark_options(benchmark_trsm)
link_rocm_runtime(benchmark_trsm)
target_link_libraries(benchmark_trsm
    hygon_trsm
    hygon_utils
    hygon_gemm
    hygon_kernels
)
