add_library(hygon_syrk STATIC
    batched_syrk.cpp
    recursive_syrk.cpp
)

apply_benchmark_options(hygon_syrk)
target_link_libraries(hygon_syrk
    PUBLIC
        rocblas
        amdhip64
    PRIVATE
        hygon_gemm
        hygon_utils
)

add_executable(benchmark_syrk
    test_syrk.cpp
    benchmark/syrk_batched_cases.cpp
    benchmark/syrk_benchmark_common.cpp
    benchmark/syrk_mixed_cases.cpp
    benchmark/syrk_recursive_cases.cpp
    benchmark/syrk_rocblas_cases.cpp
)

apply_benchmark_options(benchmark_syrk)
link_rocm_runtime(benchmark_syrk)
target_link_libraries(benchmark_syrk
    hygon_syrk
    hygon_utils
    hygon_gemm
    hygon_kernels
)
