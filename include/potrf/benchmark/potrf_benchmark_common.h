#pragma once

#include <functional>
#include <vector>

#include <rocblas/rocblas.h>

#include "utils/benchmark/benchmark_result.h"
#include "utils/benchmark/benchmark_runner.h"
#include "utils/benchmark/error_mode.h"
#include "utils/memory/stream_pool.h"

namespace potrf_benchmark {

struct PotrfBenchmarkOptions {
    int n = 4096;
    bool test_rocsolver = true;
    bool test_blocked = true;
    bool test_double_blocked = true;
    bool test_recursive = true;
    bool test_double_basic = true;
    bool test_double_dbtrsm = true;
    bool test_double_kbsyrk = true;
    bool test_double_mgemm = true;
    bool full_backward_error = false;
    int full_backward_error_max_n = 8192;
    ErrorMode error_mode = ErrorMode::None;
    int precision_type = 0;
    int syrk_k_tile = 0;
    int gemm_m_tile = 8192;
    int gemm_m_threshold = 32768;
    int gemm_k_threshold = 5632;
    std::vector<int> nbs;
    std::vector<int> double_nbs;
    std::vector<int> micro_blocks;
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
    const std::vector<int>& double_nbs;
    const std::vector<int>& micro_blocks;
    int precision_type;
    bool test_double_basic;
    bool test_double_dbtrsm;
    bool test_double_kbsyrk;
    bool test_double_mgemm;
    int syrk_k_tile;
    int gemm_m_tile;
    int gemm_m_threshold;
    int gemm_k_threshold;
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

template <typename T>
void run_blocked_cases(PotrfBenchmarkContext<T>& context);

template <typename T>
void run_double_blocked_cases(PotrfBenchmarkContext<T>& context);

} // namespace potrf_benchmark
