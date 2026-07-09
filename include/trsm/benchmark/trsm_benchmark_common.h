#pragma once

#include <functional>
#include <vector>

#include <rocblas/rocblas.h>

#include "utils/benchmark/benchmark_result.h"
#include "utils/benchmark/benchmark_runner.h"
#include "utils/benchmark/error_mode.h"

namespace trsm_benchmark {

struct TrsmBenchmarkOptions {
    int n = 4096;
    int m = 4096;
    ErrorMode error_mode = ErrorMode::None;
    bool test_recursive = true;
    bool test_rocblas = true;
    bool test_iterative = true;
    bool test_half = true;
    bool test_double_blocked = true;
    std::vector<int> nbs;
    std::vector<int> micro_blocks;
};

struct TrsmBenchmarkContext {
    rocblas_handle handle;
    float* d_A;
    float* d_B;
    int n;
    int m;
    int lda;
    int ldb;
    float alpha;
    const std::vector<int>& nbs;
    const std::vector<int>& micro_blocks;
    BenchmarkRunner& runner;
    std::vector<BenchmarkResult>& results;
    double& baseline_time;
    std::function<void()> reset;
    std::function<void(double&, double&)> verify;
    std::function<double(double)> calc_gflops;
};

void run_trsm_benchmarks(const TrsmBenchmarkOptions& options);
void run_rocblas_case(TrsmBenchmarkContext& context);
void run_recursive_cases(TrsmBenchmarkContext& context);
void run_double_blocked_cases(TrsmBenchmarkContext& context);
void run_mixed_cases(TrsmBenchmarkContext& context);
void run_iterative_cases(TrsmBenchmarkContext& context);

} // namespace trsm_benchmark
