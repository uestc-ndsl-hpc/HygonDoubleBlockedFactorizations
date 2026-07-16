#include "getrf/benchmark/getrf_benchmark_common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "utils/benchmark/benchmark_printer.h"
#include "utils/common/error_check.h"
#include "utils/memory/device_buffer.h"

namespace getrf_benchmark {
namespace {

double calculate_gflops(int n, double time_ms)
{
    if (time_ms <= 0.0) {
        return 0.0;
    }

    const double dn = static_cast<double>(n);
    const double flops = (2.0 / 3.0) * dn * dn * dn;
    return (flops / (time_ms / 1000.0)) / 1e9;
}

template <typename T>
void generate_dense_matrix(std::vector<T>& A, int n, bool pivoting)
{
    std::mt19937 gen(1234567890u);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (T& value : A) {
        value = static_cast<T>(dist(gen));
    }
    if (!pivoting) {
        for (int i = 0; i < n; ++i) {
            A[static_cast<size_t>(i) * n + i] += static_cast<T>(n);
        }
    }

    return;
}

template <typename T>
double verify_backward_error_sampled(const T* packed_lu,
                                     const T* original,
                                     const int* pivots,
                                     int n,
                                     int lda,
                                     bool pivoting,
                                     int sample_count = 4096)
{
    const long long total = static_cast<long long>(n) * n;
    sample_count = static_cast<int>(
        std::min<long long>(sample_count, total));

    std::mt19937 gen(20260715u);
    std::uniform_int_distribution<int> index_dist(0, n - 1);

    std::vector<int> permutation(n);
    for (int row = 0; row < n; ++row) {
        permutation[row] = row;
    }
    if (pivoting) {
        for (int row = 0; row < n; ++row) {
            const int pivot_row = pivots[row] - 1;
            if (pivot_row < row || pivot_row >= n) {
                return std::numeric_limits<double>::infinity();
            }
            std::swap(permutation[row], permutation[pivot_row]);
        }
    }

    double residual_norm_sq = 0.0;
    double original_norm_sq = 0.0;

    for (int sample = 0; sample < sample_count; ++sample) {
        int row = index_dist(gen);
        int col = index_dist(gen);

        if (sample % 8 == 0) {
            row = (sample / 8) % n;
            col = row;
        }

        double reconstructed = 0.0;
        const int k_end = std::min(row, col);
        for (int k = 0; k <= k_end; ++k) {
            const double l_value = (row == k)
                ? 1.0
                : static_cast<double>(
                    packed_lu[static_cast<size_t>(k) * lda + row]);
            const double u_value = static_cast<double>(
                packed_lu[static_cast<size_t>(col) * lda + k]);
            reconstructed += l_value * u_value;
        }

        const int original_row = permutation[row];
        const double a_value = static_cast<double>(
            original[static_cast<size_t>(col) * lda + original_row]);
        const double residual = reconstructed - a_value;
        residual_norm_sq += residual * residual;
        original_norm_sq += a_value * a_value;
    }

    return std::sqrt(residual_norm_sq)
        / (std::sqrt(original_norm_sq) + 1e-30);
}

template <typename T>
void run_all_benchmarks(const GetrfBenchmarkOptions& options)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    const int n = options.n;
    const size_t element_count = static_cast<size_t>(n) * n;
    const size_t bytes = element_count * sizeof(T);

    std::vector<T> h_A_original(element_count);
    generate_dense_matrix(h_A_original, n, options.pivoting);

    DeviceBuffer<T> d_A_original(element_count);
    DeviceBuffer<T> d_A(element_count);
    DeviceBuffer<int> d_ipiv(n);
    DeviceBuffer<int> d_info(1);

    const int pivot_column_tile = std::min(
        n, std::max(1, options.pivot_column_tile));
    const size_t pivot_capacity = options.pivoting
        ? static_cast<size_t>(n)
        : 0;
    DeviceBuffer<int> d_pivot_sources(pivot_capacity);
    DeviceBuffer<int> d_pivot_targets(pivot_capacity);
    DeviceBuffer<T> d_pivot_swap(
        pivot_capacity * static_cast<size_t>(pivot_column_tile));
    std::vector<int> h_pivots(pivot_capacity);
    std::vector<int> h_permutation(pivot_capacity);
    std::vector<int> h_sources(pivot_capacity);
    std::vector<int> h_targets(pivot_capacity);

    getrf_batched_pivot_workspace batched_pivot_workspace{
        d_pivot_sources.get(),
        d_pivot_targets.get(),
        d_pivot_swap.get(),
        h_pivots.data(),
        h_permutation.data(),
        h_sources.data(),
        h_targets.data(),
        options.pivoting ? n : 0,
        pivot_column_tile
    };

    HIP_CHECK(hipMemcpy(
        d_A_original,
        h_A_original.data(),
        bytes,
        hipMemcpyHostToDevice));

    std::vector<BenchmarkResult> results;
    double baseline_time = -1.0;
    BenchmarkRunner runner(stream, baseline_time, results);

    auto reset = [&]() {
        HIP_CHECK(hipMemcpy(
            d_A, d_A_original, bytes, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemset(d_ipiv, 0, static_cast<size_t>(n) * sizeof(int)));
        HIP_CHECK(hipMemset(d_info, 0, sizeof(int)));
    };

    auto verify = [&](double& forward_error, double& backward_error) {
        (void)forward_error;
        if (options.error_mode == ErrorMode::None) {
            return;
        }

        int info = 0;
        HIP_CHECK(hipMemcpy(
            &info, d_info, sizeof(int), hipMemcpyDeviceToHost));
        if (info != 0) {
            backward_error = std::numeric_limits<double>::infinity();
            return;
        }

        std::vector<T> h_lu(element_count);
        HIP_CHECK(hipMemcpy(
            h_lu.data(), d_A, bytes, hipMemcpyDeviceToHost));

        std::vector<int> h_ipiv;
        if (options.pivoting) {
            h_ipiv.resize(n);
            HIP_CHECK(hipMemcpy(
                h_ipiv.data(),
                d_ipiv,
                static_cast<size_t>(n) * sizeof(int),
                hipMemcpyDeviceToHost));
        }

        backward_error = verify_backward_error_sampled(
            h_lu.data(),
            h_A_original.data(),
            options.pivoting ? h_ipiv.data() : nullptr,
            n,
            n,
            options.pivoting);
    };

    auto calc_gflops = [n](double time_ms) {
        return calculate_gflops(n, time_ms);
    };

    GetrfBenchmarkContext<T> context{
        handle,
        d_A,
        d_ipiv,
        d_info,
        n,
        n,
        options.pivoting,
        options.trsm_leaf,
        pivot_column_tile,
        options.inner_gemm_m_tile,
        options.pivoting ? &batched_pivot_workspace : nullptr,
        options.nbs,
        options.double_nbs,
        options.micro_blocks,
        runner,
        results,
        baseline_time,
        reset,
        verify,
        calc_gflops
    };

    const std::string mode = options.pivoting ? "GETRF" : "GETRF-NPVT";
    std::cout << ">>> Running " << mode << " Benchmarks...\n";
    run_rocsolver_case(context);
    run_blocked_cases(context);
    run_double_blocked_cases(context);

    print_benchmark_results(
        mode + " Benchmark Summary (N = " + std::to_string(n) + ")",
        results,
        options.error_mode,
        120);

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return;
}

} // namespace

void run_getrf_benchmarks(const GetrfBenchmarkOptions& options)
{
    if (options.precision_type == 1) {
        std::cout << "Running "
                  << (options.pivoting ? "GETRF" : "GETRF-NPVT")
                  << " benchmark in DOUBLE precision\n";
        run_all_benchmarks<double>(options);
    } else {
        std::cout << "Running "
                  << (options.pivoting ? "GETRF" : "GETRF-NPVT")
                  << " benchmark in FLOAT precision\n";
        run_all_benchmarks<float>(options);
    }

    return;
}

} // namespace getrf_benchmark
