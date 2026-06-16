#include "syrk/benchmark/syrk_benchmark_common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "utils/common/error_check.h"
#include "utils/matrix/matrix_generator.h"
#include "utils/memory/device_buffer.h"

namespace syrk_benchmark {
namespace {

double calculate_gflops(int n, int k, double time_ms)
{
    if (time_ms <= 0.0) {
        return 0.0;
    }

    double flops = static_cast<double>(n) * (n + 1) * k;
    return (flops / (time_ms / 1000.0)) / 1e9;
}

template <typename T_REF>
double verify_relative_error_lower(
    const std::vector<T_REF>& ref,
    const std::vector<float>& res,
    int n)
{
    double diff_norm = 0.0;
    double ref_norm = 0.0;

    size_t n_s = static_cast<size_t>(n);

    for (size_t j = 0; j < n_s; j++) {
        for (size_t i = j; i < n_s; i++) {
            size_t idx = j * n_s + i;

            double diff = static_cast<double>(ref[idx]) - static_cast<double>(res[idx]);
            diff_norm += diff * diff;
            ref_norm += static_cast<double>(ref[idx]) * static_cast<double>(ref[idx]);
        }
    }

    return std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-20);
}

double verify_backward_error_sampled(
    const std::vector<float>& A,
    const std::vector<float>& C_original,
    const std::vector<float>& C_result,
    int n,
    int k,
    int lda,
    int ldc,
    float alpha = -1.0f,
    float beta = 1.0f,
    int sample_count = 4096)
{
    long long total = static_cast<long long>(n) * (n + 1) / 2;
    sample_count = std::min(static_cast<long long>(sample_count), total);

    double diff_norm = 0.0;
    double ref_norm = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_row(0, n - 1);
    std::uniform_int_distribution<int> dist_col(0, n - 1);

    size_t lda_s = static_cast<size_t>(lda);
    size_t ldc_s = static_cast<size_t>(ldc);

    for (int s = 0; s < sample_count;) {
        int row = dist_row(gen);
        int col = dist_col(gen);

        if (row < col) {
            continue;
        }

        double reconstructed = 0.0;

        for (int l = 0; l < k; ++l) {
            size_t idx_l_row = static_cast<size_t>(l) * lda_s + row;
            size_t idx_l_col = static_cast<size_t>(l) * lda_s + col;

            reconstructed += static_cast<double>(A[idx_l_row]) *
                             static_cast<double>(A[idx_l_col]);
        }

        size_t idx_C = static_cast<size_t>(col) * ldc_s + row;

        reconstructed = static_cast<double>(alpha) * reconstructed +
                        static_cast<double>(beta) * static_cast<double>(C_original[idx_C]);

        double res_val = static_cast<double>(C_result[idx_C]);
        double diff = res_val - reconstructed;

        diff_norm += diff * diff;
        ref_norm += reconstructed * reconstructed;

        ++s;
    }

    return std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-20);
}

} // namespace

void run_syrk_benchmarks(const SyrkBenchmarkOptions& options)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t main_stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &main_stream));

    int num_streams = 16;
    StreamPool pool(num_streams);

    int n = options.n;
    int k = options.k;
    int lda = n;
    int ldc = n;
    size_t bytes_A = static_cast<size_t>(n) * k * sizeof(float);
    size_t bytes_C = static_cast<size_t>(n) * n * sizeof(float);

    DeviceBuffer<float> d_A_f(static_cast<size_t>(n) * k);
    DeviceBuffer<float> d_C_orig(static_cast<size_t>(n) * n);
    DeviceBuffer<float> d_C_f(static_cast<size_t>(n) * n);

    generate_syrk_matrices<float>(handle, d_A_f, d_C_orig, n, k, lda, ldc);

    std::vector<float> h_A_f(static_cast<size_t>(n) * k);
    std::vector<float> h_C_f(static_cast<size_t>(n) * n);
    HIP_CHECK(hipMemcpy(h_A_f.data(), d_A_f, bytes_A, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_C_f.data(), d_C_orig, bytes_C, hipMemcpyDeviceToHost));

    float alpha_f = -1.0f;
    float beta_f = 1.0f;

    std::vector<double> h_C_gold(static_cast<size_t>(n) * n);
    std::vector<float> h_C_ref_f32(static_cast<size_t>(n) * n);
    bool has_fp64_ref = false;
    bool has_f32_ref = false;
    ErrorMode error_mode = options.error_mode;

    if (error_mode == ErrorMode::Forward || error_mode == ErrorMode::Both) {
        std::cout << ">>> Generating Golden Reference for Error Verification...\n";
        bool do_fp64_verify = (std::max(n, k) <= 16384);

        if (do_fp64_verify) {
            std::vector<double> h_A_d(static_cast<size_t>(n) * k);
            std::vector<double> h_C_d(static_cast<size_t>(n) * n);

            for (size_t i = 0; i < h_A_f.size(); ++i) {
                h_A_d[i] = h_A_f[i];
            }
            for (size_t i = 0; i < h_C_f.size(); ++i) {
                h_C_d[i] = h_C_f[i];
            }

            DeviceBuffer<double> d_A_d(static_cast<size_t>(n) * k);
            DeviceBuffer<double> d_C_d(static_cast<size_t>(n) * n);

            HIP_CHECK(hipMemcpy(
                d_A_d,
                h_A_d.data(),
                static_cast<size_t>(n) * k * sizeof(double),
                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(
                d_C_d,
                h_C_d.data(),
                static_cast<size_t>(n) * n * sizeof(double),
                hipMemcpyHostToDevice));

            double alpha_d = -1.0;
            double beta_d = 1.0;

            ROCBLAS_CHECK(rocblas_dsyrk(
                handle,
                rocblas_fill_lower,
                rocblas_operation_none,
                n,
                k,
                &alpha_d,
                d_A_d,
                lda,
                &beta_d,
                d_C_d,
                ldc));

            HIP_CHECK(hipMemcpy(
                h_C_gold.data(),
                d_C_d,
                static_cast<size_t>(n) * n * sizeof(double),
                hipMemcpyDeviceToHost));
            has_fp64_ref = true;
        } else {
            HIP_CHECK(hipMemcpy(d_C_f, h_C_f.data(), bytes_C, hipMemcpyHostToDevice));
            ROCBLAS_CHECK(rocblas_ssyrk(
                handle,
                rocblas_fill_lower,
                rocblas_operation_none,
                n,
                k,
                &alpha_f,
                d_A_f,
                lda,
                &beta_f,
                d_C_f,
                ldc));

            HIP_CHECK(hipMemcpy(h_C_ref_f32.data(), d_C_f, bytes_C, hipMemcpyDeviceToHost));
            has_f32_ref = true;
        }
    }

    std::vector<BenchmarkResult> results;
    double baseline_time = -1.0;
    BenchmarkRunner runner(main_stream, baseline_time, results);

    auto reset_c = [&]() {
        HIP_CHECK(hipMemcpy(d_C_f, h_C_f.data(), bytes_C, hipMemcpyHostToDevice));
    };

    auto calc_gflops_syrk = [n, k](double ms) {
        return calculate_gflops(n, k, ms);
    };

    auto verify_syrk = [&] (double& f_err, double& b_err) {
        if (error_mode == ErrorMode::None) {
            return;
        }

        std::vector<float> h_C_out(static_cast<size_t>(n) * n);
        HIP_CHECK(hipMemcpy(h_C_out.data(), d_C_f, bytes_C, hipMemcpyDeviceToHost));

        if (error_mode == ErrorMode::Forward || error_mode == ErrorMode::Both) {
            if (has_fp64_ref) {
                f_err = verify_relative_error_lower(h_C_gold, h_C_out, n);
            } else if (has_f32_ref) {
                f_err = verify_relative_error_lower(h_C_ref_f32, h_C_out, n);
            }
        }

        if (error_mode == ErrorMode::Backward || error_mode == ErrorMode::Both) {
            b_err = verify_backward_error_sampled(
                h_A_f,
                h_C_f,
                h_C_out,
                n,
                k,
                lda,
                ldc,
                alpha_f,
                beta_f,
                4096);
        }
    };

    SyrkBenchmarkContext context{
        handle,
        &pool,
        d_A_f,
        d_C_f,
        n,
        k,
        lda,
        ldc,
        alpha_f,
        beta_f,
        options.nbs,
        runner,
        results,
        baseline_time,
        reset_c,
        verify_syrk,
        calc_gflops_syrk
    };

    std::cout << ">>> Running Benchmarks...\n";

    if (options.test_rocblas) {
        run_rocblas_case(context);
    }

    if (options.test_recursive) {
        run_recursive_cases(context);
    }

    if (options.test_batched) {
        run_batched_cases(context);
    }

    if (options.test_mixed) {
        run_mixed_cases(context);
    }

    print_benchmark_results(
        "SYRK Benchmark Summary (N = " + std::to_string(n) + ", K = " + std::to_string(k) + ")",
        results,
        error_mode,
        120);

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return;
}

} // namespace syrk_benchmark
