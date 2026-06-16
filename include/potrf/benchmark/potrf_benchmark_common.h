#pragma once

#include <functional>
#include <vector>

#include <rocblas/rocblas.h>

#include "utils/benchmark/benchmark_printer.h"
#include "utils/benchmark/benchmark_runner.h"
#include "utils/memory/stream_pool.h"

namespace potrf_benchmark {

struct PotrfBenchmarkOptions {
    int n = 4096;
    bool test_rocsolver = true;
    bool test_recursive = true;
    ErrorMode error_mode = ErrorMode::None;
    int precision_type = 0;
    std::vector<int> nbs;
};

template <typename T>
struct PotrfBenchmarkContext {
    rocblas_handle handle;
    StreamPool* pool;
    T* d_A;
    int* d_info;
    int n;
    int lda;
    const std::vector<int>& nbs;
    int precision_type;
    BenchmarkRunner& runner;
    std::vector<BenchmarkResult>& results;
    double& baseline_time;
    std::function<void()> reset;
    std::function<void(double&, double&)> verify;
    std::function<double(double)> calc_gflops;
};

void run_potrf_benchmarks(const PotrfBenchmarkOptions& options);

template <typename T>
void run_rocsolver_case(PotrfBenchmarkContext<T>& context);

template <typename T>
void run_recursive_cases(PotrfBenchmarkContext<T>& context);

} // namespace potrf_benchmark
