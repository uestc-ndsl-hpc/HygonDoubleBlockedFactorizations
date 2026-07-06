add_library(hygon_potrf STATIC
    recursive_potrf.cpp
)

apply_benchmark_options(hygon_potrf)
target_link_libraries(hygon_potrf
    PUBLIC
        rocblas
        rocsolver
        amdhip64
    PRIVATE
        hygon_trsm
        hygon_syrk
        hygon_kernels
        hygon_utils
)

add_executable(benchmark_potrf
    test_potrf.cpp
    benchmark/potrf_benchmark_common.cpp
    benchmark/potrf_recursive_cases.cpp
    benchmark/potrf_rocsolver_cases.cpp
)

apply_benchmark_options(benchmark_potrf)
link_rocm_runtime(benchmark_potrf)
target_link_libraries(benchmark_potrf
    rocsolver
    hygon_potrf
    hygon_trsm
    hygon_syrk
    hygon_utils
    hygon_gemm
    hygon_kernels
)
