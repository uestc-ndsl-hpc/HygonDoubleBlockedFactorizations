#pragma once

#include <functional>
#include <vector>

#include <rocblas/rocblas.h>

#include "getrf/double_blocked_getrf.h"
#include "utils/benchmark/benchmark_result.h"
#include "utils/benchmark/benchmark_runner.h"
#include "utils/benchmark/error_mode.h"

namespace getrf_benchmark {

struct GetrfBenchmarkOptions {
    int n = 4096;
    int precision_type = 0;
    ErrorMode error_mode = ErrorMode::None;
    bool pivoting = false;
    bool test_rocsolver = true;
    bool test_blocked = true;
    bool test_double_basic = true;
    bool test_batched_pivot = true;
    bool test_mgemm = true;
    bool full_backward_error = false;
    int full_backward_error_max_n = 8192;
    bool cpu_fp64_residual = false;
    int cpu_residual_tile = 512;
    unsigned int seed = 1234567890u;
    getrf_outer_trsm_mode outer_trsm_mode = getrf_outer_trsm_iterative;
    int trsm_block = 2048;
    int pivot_column_tile = 2048;
    int inner_gemm_m_tile = 8192;
    std::vector<int> nbs;
    std::vector<int> double_nbs;
    std::vector<int> micro_blocks;
};

template <typename T>
struct GetrfBenchmarkContext {
    rocblas_handle handle;
    T* d_A;
    int* d_ipiv;
    int* d_info;
    int n;
    int lda;
    bool pivoting;
    bool test_double_basic;
    bool test_batched_pivot;
    bool test_mgemm;
    getrf_outer_trsm_mode outer_trsm_mode;
    int trsm_block;
    int pivot_column_tile;
    int inner_gemm_m_tile;
    getrf_batched_pivot_workspace* batched_pivot_workspace;
    const std::vector<int>& nbs;
    const std::vector<int>& double_nbs;
    const std::vector<int>& micro_blocks;
    BenchmarkRunner& runner;
    std::vector<BenchmarkResult>& results;
    double& baseline_time;
    std::function<void()> reset;
    std::function<void(double&, double&)> verify;
    std::function<double(double)> calc_gflops;
};

void run_getrf_benchmarks(const GetrfBenchmarkOptions& options);

template <typename T>
void run_rocsolver_case(GetrfBenchmarkContext<T>& context);

template <typename T>
void run_blocked_cases(GetrfBenchmarkContext<T>& context);

template <typename T>
void run_double_blocked_cases(GetrfBenchmarkContext<T>& context);

} // namespace getrf_benchmark
