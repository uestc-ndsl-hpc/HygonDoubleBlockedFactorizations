#include "trsm/benchmark/trsm_benchmark_common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "utils/common/error_check.h"
#include "utils/benchmark/benchmark_printer.h"
#include "utils/matrix/matrix_generator.h"
#include "utils/memory/device_buffer.h"

namespace trsm_benchmark {
namespace {

double calculate_gflops(int n, int m, double time_ms)
{
    if (time_ms <= 0.0) {
        return 0.0;
    }

    double total_flops = static_cast<double>(n) * m * m;
    return (total_flops / (time_ms / 1000.0)) / 1e9;
}

template <typename T_REF>
double verify_relative_error(
    const std::vector<T_REF>& ref,
    const std::vector<float>& res,
    int sample_count = 10000)
{
    double diff_norm = 0.0;
    double ref_norm = 0.0;
    size_t total_size = ref.size();

    if (total_size <= 1000000) {
        for (size_t i = 0; i < total_size; i++) {
            double diff = static_cast<double>(ref[i]) - static_cast<double>(res[i]);
            diff_norm += diff * diff;
            ref_norm += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        }
    } else {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, total_size - 1);

        for (int s = 0; s < sample_count; ++s) {
            size_t idx = dist(gen);
            double diff = static_cast<double>(ref[idx]) - static_cast<double>(res[idx]);
            diff_norm += diff * diff;
            ref_norm += static_cast<double>(ref[idx]) * static_cast<double>(ref[idx]);
        }
    }

    return std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-20);
}

double verify_backward_error_sampled(
    const std::vector<float>& A,
    const std::vector<float>& B_original,
    const std::vector<float>& X_result,
    int n,
    int m,
    int lda,
    int ldb,
    int sample_count = 4096)
{
    long long total = static_cast<long long>(n) * m;
    sample_count = std::min(static_cast<long long>(sample_count), total);

    double diff_norm = 0.0;
    double ref_norm = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_row(0, n - 1);
    std::uniform_int_distribution<int> dist_col(0, m - 1);

    size_t lda_s = static_cast<size_t>(lda);
    size_t ldb_s = static_cast<size_t>(ldb);

    for (int s = 0; s < sample_count; ++s) {
        int row = dist_row(gen);
        int col = dist_col(gen);
        double reconstructed = 0.0;

        for (int k = 0; k <= col; ++k) {
            size_t idx_X = static_cast<size_t>(k) * ldb_s + row;
            size_t idx_A = static_cast<size_t>(k) * lda_s + col;
            reconstructed += static_cast<double>(X_result[idx_X]) *
                             static_cast<double>(A[idx_A]);
        }

        size_t idx_B = static_cast<size_t>(col) * ldb_s + row;

        double ref_val = static_cast<double>(B_original[idx_B]);
        double diff = reconstructed - ref_val;

        diff_norm += diff * diff;
        ref_norm += ref_val * ref_val;
    }

    return std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-20);
}

} // namespace

void run_trsm_benchmarks(const TrsmBenchmarkOptions& options)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t main_stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &main_stream));

    int n = options.n;
    int m = options.m;
    size_t size_A = static_cast<size_t>(m) * m * sizeof(float);
    size_t size_B = static_cast<size_t>(n) * m * sizeof(float);
    int lda = m;
    int ldb = n;

    std::vector<float> h_A(static_cast<size_t>(m) * m);
    std::vector<float> h_B(static_cast<size_t>(n) * m);
    std::vector<float> h_X(static_cast<size_t>(n) * m);

    generate_stable_lower_triangular<float>(h_A.data(), m, lda);
    generate_random_matrix<float>(h_B.data(), n, m, ldb);

    DeviceBuffer<float> d_A(static_cast<size_t>(m) * m);
    DeviceBuffer<float> d_B(static_cast<size_t>(n) * m);

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    float alpha = 1.0f;

    std::vector<double> h_X_gold(static_cast<size_t>(n) * m);
    std::vector<float> h_X_ref_f32(static_cast<size_t>(n) * m);
    bool has_fp64_ref = false;
    bool has_f32_ref = false;
    ErrorMode error_mode = options.error_mode;

    if (error_mode == ErrorMode::Forward || error_mode == ErrorMode::Both) {
        std::cout << ">>> Generating Golden Reference...\n";
        bool do_fp64_verify = (std::max(n, m) <= 16384);

        if (do_fp64_verify) {
            DeviceBuffer<double> d_A_d(static_cast<size_t>(m) * m);
            DeviceBuffer<double> d_B_d(static_cast<size_t>(n) * m);
            std::vector<double> h_A_d(static_cast<size_t>(m) * m);
            std::vector<double> h_B_d(static_cast<size_t>(n) * m);

            for (size_t i = 0; i < h_A.size(); ++i) {
                h_A_d[i] = static_cast<double>(h_A[i]);
            }
            for (size_t i = 0; i < h_B.size(); ++i) {
                h_B_d[i] = static_cast<double>(h_B[i]);
            }

            HIP_CHECK(hipMemcpy(
                d_A_d,
                h_A_d.data(),
                static_cast<size_t>(m) * m * sizeof(double),
                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(
                d_B_d,
                h_B_d.data(),
                static_cast<size_t>(n) * m * sizeof(double),
                hipMemcpyHostToDevice));

            double alpha_d = 1.0;
            ROCBLAS_CHECK(rocblas_dtrsm(
                handle,
                rocblas_side_right,
                rocblas_fill_lower,
                rocblas_operation_transpose,
                rocblas_diagonal_non_unit,
                n,
                m,
                &alpha_d,
                d_A_d,
                lda,
                d_B_d,
                ldb));

            HIP_CHECK(hipMemcpy(
                h_X_gold.data(),
                d_B_d,
                static_cast<size_t>(n) * m * sizeof(double),
                hipMemcpyDeviceToHost));
            has_fp64_ref = true;
        } else {
            HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
            ROCBLAS_CHECK(rocblas_strsm(
                handle,
                rocblas_side_right,
                rocblas_fill_lower,
                rocblas_operation_transpose,
                rocblas_diagonal_non_unit,
                n,
                m,
                &alpha,
                d_A,
                lda,
                d_B,
                ldb));

            HIP_CHECK(hipMemcpy(h_X_ref_f32.data(), d_B, size_B, hipMemcpyDeviceToHost));
            has_f32_ref = true;
        }
    }

    std::vector<BenchmarkResult> results;
    double baseline_time = -1.0;
    BenchmarkRunner runner(main_stream, baseline_time, results);

    auto reset_b = [&]() {
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    };

    auto calc_gflops_trsm = [n, m](double ms) {
        return calculate_gflops(n, m, ms);
    };

    auto verify_trsm = [&] (double& f_err, double& b_err) {
        if (error_mode == ErrorMode::None) {
            return;
        }

        HIP_CHECK(hipMemcpy(h_X.data(), d_B, size_B, hipMemcpyDeviceToHost));

        if (error_mode == ErrorMode::Forward || error_mode == ErrorMode::Both) {
            if (has_fp64_ref) {
                f_err = verify_relative_error(h_X_gold, h_X);
            } else if (has_f32_ref) {
                f_err = verify_relative_error(h_X_ref_f32, h_X);
            }
        }

        if (error_mode == ErrorMode::Backward || error_mode == ErrorMode::Both) {
            b_err = verify_backward_error_sampled(h_A, h_B, h_X, n, m, lda, ldb, 4096);
        }
    };

    TrsmBenchmarkContext context{
        handle,
        d_A,
        d_B,
        n,
        m,
        lda,
        ldb,
        alpha,
        options.nbs,
        runner,
        results,
        baseline_time,
        reset_b,
        verify_trsm,
        calc_gflops_trsm
    };

    std::cout << ">>> Running Benchmarks...\n";

    if (options.test_rocblas) {
        run_rocblas_case(context);
    }

    if (options.test_recursive) {
        run_recursive_cases(context);
    }

    if (options.test_half) {
        run_mixed_cases(context);
    }

    if (options.test_iterative) {
        run_iterative_cases(context);
    }

    print_benchmark_results(
        "TRSM Benchmark Summary (N = " + std::to_string(n) + ", M = " + std::to_string(m) + ")",
        results,
        error_mode,
        120);

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return;
}

} // namespace trsm_benchmark
