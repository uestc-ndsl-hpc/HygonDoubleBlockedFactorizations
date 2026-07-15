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
void generate_stable_dense_matrix(std::vector<T>& A, int n)
{
    std::mt19937 gen(1234567890u);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (T& value : A) {
        value = static_cast<T>(dist(gen));
    }
    for (int i = 0; i < n; ++i) {
        A[static_cast<size_t>(i) * n + i] += static_cast<T>(n);
    }

    return;
}

template <typename T>
double verify_backward_error_sampled(const T* packed_lu,
                                     const T* original,
                                     int n,
                                     int lda,
                                     int sample_count = 4096)
{
    const long long total = static_cast<long long>(n) * n;
    sample_count = static_cast<int>(
        std::min<long long>(sample_count, total));

    std::mt19937 gen(20260715u);
    std::uniform_int_distribution<int> index_dist(0, n - 1);

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

        const double a_value = static_cast<double>(
            original[static_cast<size_t>(col) * lda + row]);
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
    generate_stable_dense_matrix(h_A_original, n);

    DeviceBuffer<T> d_A_original(element_count);
    DeviceBuffer<T> d_A(element_count);
    DeviceBuffer<int> d_info(1);

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
        backward_error = verify_backward_error_sampled(
            h_lu.data(), h_A_original.data(), n, n);
    };

    auto calc_gflops = [n](double time_ms) {
        return calculate_gflops(n, time_ms);
    };

    GetrfBenchmarkContext<T> context{
        handle,
        d_A,
        d_info,
        n,
        n,
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

    std::cout << ">>> Running GETRF-NPVT Benchmarks...\n";
    run_rocsolver_case(context);
    run_blocked_cases(context);
    run_double_blocked_cases(context);

    print_benchmark_results(
        "GETRF-NPVT Benchmark Summary (N = " + std::to_string(n) + ")",
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
        std::cout << "Running GETRF-NPVT benchmark in DOUBLE precision\n";
        run_all_benchmarks<double>(options);
    } else {
        std::cout << "Running GETRF-NPVT benchmark in FLOAT precision\n";
        run_all_benchmarks<float>(options);
    }

    return;
}

} // namespace getrf_benchmark
