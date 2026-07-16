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
    int trsm_leaf = 512;
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
    int trsm_leaf;
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
