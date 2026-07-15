add_library(hygon_getrf STATIC
    blocked_getrf.cpp
    blocked_getrf_npvt.cpp
    double_blocked_getrf.cpp
    double_blocked_getrf_npvt.cpp
    getrf_pivot_kernels.hip
)

set_source_files_properties(getrf_pivot_kernels.hip PROPERTIES LANGUAGE CXX)

apply_benchmark_options(hygon_getrf)
target_link_libraries(hygon_getrf
    PUBLIC
        rocblas
        rocsolver
        amdhip64
    PRIVATE
        hygon_utils
)

add_executable(benchmark_getrf
    test_getrf.cpp
    benchmark/getrf_benchmark_common.cpp
    benchmark/getrf_blocked_cases.cpp
    benchmark/getrf_double_blocked_cases.cpp
    benchmark/getrf_rocsolver_case.cpp
)

apply_benchmark_options(benchmark_getrf)
link_rocm_runtime(benchmark_getrf)
target_link_libraries(benchmark_getrf
    rocsolver
    hygon_getrf
    hygon_utils
)
