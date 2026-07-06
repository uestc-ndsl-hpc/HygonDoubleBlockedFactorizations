#pragma once

#include <functional>
#include <vector>

#include <rocblas/rocblas.h>

#include "utils/benchmark/benchmark_result.h"
#include "utils/benchmark/benchmark_runner.h"
#include "utils/benchmark/error_mode.h"
#include "utils/memory/stream_pool.h"

namespace syrk_benchmark {

struct SyrkBenchmarkOptions {
    int n = 4096;
    int k = 4096;
    ErrorMode error_mode = ErrorMode::None;
    bool test_recursive = true;
    bool test_rocblas = true;
    bool test_batched = true;
    bool test_paper = true;
    bool test_mixed = true;
    std::vector<int> nbs;
};

struct SyrkBenchmarkContext {
    rocblas_handle handle;
    StreamPool* pool;
    float* d_A;
    float* d_C;
    int n;
    int k;
    int lda;
    int ldc;
    float alpha;
    float beta;
    const std::vector<int>& nbs;
    BenchmarkRunner& runner;
    std::vector<BenchmarkResult>& results;
    double& baseline_time;
    std::function<void()> reset;
    std::function<void(double&, double&)> verify;
    std::function<double(double)> calc_gflops;
};

void run_syrk_benchmarks(const SyrkBenchmarkOptions& options);
void run_rocblas_case(SyrkBenchmarkContext& context);
void run_recursive_cases(SyrkBenchmarkContext& context);
void run_batched_cases(SyrkBenchmarkContext& context);
void run_paper_cases(SyrkBenchmarkContext& context);
void run_mixed_cases(SyrkBenchmarkContext& context);

} // namespace syrk_benchmark
