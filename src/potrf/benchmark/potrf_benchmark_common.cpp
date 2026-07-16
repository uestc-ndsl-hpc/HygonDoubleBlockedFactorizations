#include "potrf/benchmark/potrf_benchmark_common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>

#include "utils/common/error_check.h"
#include "utils/benchmark/benchmark_printer.h"
#include "utils/matrix/matrix_generator.h"
#include "utils/memory/device_buffer.h"

namespace potrf_benchmark {
namespace {

double calculate_gflops(int n, double time_ms)
{
    if (time_ms <= 0.0) {
        return 0.0;
    }

    double flops = (1.0 / 3.0) * static_cast<double>(n) * n * n;
    return (flops / (time_ms / 1000.0)) / 1e9;
}

template <typename T>
double verify_potrf_against_double_ref(
    const T* L_res,
    const double* L_ref_double,
    int n,
    int lda)
{
    double diff_norm_sq = 0.0;
    double ref_norm_sq = 0.0;

    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            size_t idx = static_cast<size_t>(j) * lda + i;

            double val_res = static_cast<double>(L_res[idx]);
            double val_ref = L_ref_double[idx];
            double diff = val_res - val_ref;

            diff_norm_sq += diff * diff;
            ref_norm_sq += val_ref * val_ref;
        }
    }

    return std::sqrt(diff_norm_sq) / (std::sqrt(ref_norm_sq) + 1e-12);
}

template <typename T>
double verify_backward_error_sampled(
    const T* L_res,
    const T* A_orig,
    int n,
    int lda,
    int sample_count = 4096)
{
    long long total = static_cast<long long>(n) * (n + 1) / 2;
    sample_count = std::min(static_cast<long long>(sample_count), total);

    double diff_norm_sq = 0.0;
    double a_norm_sq = 0.0;

    std::mt19937 gen(20260716u);
    std::uniform_int_distribution<int> dist_row(0, n - 1);
    std::uniform_int_distribution<int> dist_col(0, n - 1);

    size_t lda_s = static_cast<size_t>(lda);

    for (int s = 0; s < sample_count;) {
        int row = dist_row(gen);
        int col = dist_col(gen);

        if (row < col) {
            continue;
        }

        double reconstructed = 0.0;

        for (int k = 0; k <= col; ++k) {
            size_t idx_ik = static_cast<size_t>(k) * lda_s + row;
            size_t idx_jk = static_cast<size_t>(k) * lda_s + col;

            double Lik = static_cast<double>(L_res[idx_ik]);
            double Ljk = static_cast<double>(L_res[idx_jk]);
            reconstructed += Lik * Ljk;
        }

        size_t idx_A = static_cast<size_t>(col) * lda_s + row;
        double a_val = static_cast<double>(A_orig[idx_A]);
        double diff = a_val - reconstructed;
        double factor = (row == col) ? 1.0 : 2.0;

        diff_norm_sq += factor * diff * diff;
        a_norm_sq += factor * a_val * a_val;

        ++s;
    }

    return std::sqrt(diff_norm_sq) / (std::sqrt(a_norm_sq) + 1e-12);
}

void gemm_nt(rocblas_handle handle,
             int n,
             const double* lower,
             double* product)
{
    const double one = 1.0;
    const double zero = 0.0;

    ROCBLAS_CHECK(rocblas_dgemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_transpose,
        n,
        n,
        n,
        &one,
        lower,
        n,
        lower,
        n,
        &zero,
        product,
        n));
}

template <typename T>
double verify_backward_error_full(rocblas_handle handle,
                                  const T* packed_lower,
                                  const T* original,
                                  int n,
                                  int lda,
                                  double* device_lower,
                                  double* device_product)
{
    const size_t element_count = static_cast<size_t>(n) * n;
    const size_t bytes = element_count * sizeof(double);
    std::vector<double> lower(element_count, 0.0);

    for (int column = 0; column < n; ++column) {
        for (int row = column; row < n; ++row) {
            const size_t index = static_cast<size_t>(column) * lda + row;
            lower[index] = static_cast<double>(packed_lower[index]);
        }
    }

    HIP_CHECK(hipMemcpy(
        device_lower, lower.data(), bytes, hipMemcpyHostToDevice));
    gemm_nt(handle, n, device_lower, device_product);

    std::vector<double> product(element_count);
    HIP_CHECK(hipMemcpy(
        product.data(), device_product, bytes, hipMemcpyDeviceToHost));

    double residual_norm_sq = 0.0;
    double original_norm_sq = 0.0;
    for (size_t index = 0; index < element_count; ++index) {
        const double original_value = static_cast<double>(original[index]);
        const double residual = product[index] - original_value;
        residual_norm_sq += residual * residual;
        original_norm_sq += original_value * original_value;
    }

    return std::sqrt(residual_norm_sq)
        / (std::sqrt(original_norm_sq) + 1e-30);
}

template <typename T>
void run_all_benchmarks(const PotrfBenchmarkOptions& options)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t main_stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &main_stream));

    StreamPool pool(16);

    int n = options.n;
    size_t num_elements = static_cast<size_t>(n) * n;
    size_t bytes = num_elements * sizeof(T);

    DeviceBuffer<int> d_info(1);
    DeviceBuffer<T> d_A_orig(num_elements);
    DeviceBuffer<double> d_residual_lower(
        options.full_residual ? num_elements : 0);
    DeviceBuffer<double> d_residual_product(
        options.full_residual ? num_elements : 0);

    generate_spd_matrix<T>(handle, d_A_orig, n, n, 20260716u);

    std::vector<T> h_A(num_elements);
    HIP_CHECK(hipMemcpy(h_A.data(), d_A_orig, bytes, hipMemcpyDeviceToHost));

    DeviceBuffer<T> d_A(num_elements);
    std::vector<double> h_L_ref_double;

    if (options.error_mode == ErrorMode::Forward || options.error_mode == ErrorMode::Both) {
        std::cout << ">>> Generating FP64 Golden Reference...\n";

        h_L_ref_double.resize(num_elements);
        std::vector<double> h_A_double(num_elements);

        for (size_t i = 0; i < num_elements; ++i) {
            h_A_double[i] = static_cast<double>(h_A[i]);
        }

        DeviceBuffer<double> d_A_double(num_elements);

        HIP_CHECK(hipMemcpy(
            d_A_double,
            h_A_double.data(),
            num_elements * sizeof(double),
            hipMemcpyHostToDevice));

        ROCBLAS_CHECK(rocsolver_dpotrf(
            handle,
            rocblas_fill_lower,
            n,
            d_A_double,
            n,
            d_info));

        HIP_CHECK(hipMemcpy(
            h_L_ref_double.data(),
            d_A_double,
            num_elements * sizeof(double),
            hipMemcpyDeviceToHost));
    }

    std::vector<BenchmarkResult> results;
    double baseline_time = -1.0;
    BenchmarkRunner runner(main_stream, baseline_time, results);

    auto reset_a = [&]() {
        HIP_CHECK(hipMemcpy(d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemset(d_info, 0, sizeof(int)));
    };

    auto calc_gflops_potrf = [n](double ms) {
        return calculate_gflops(n, ms);
    };

    auto verify_potrf = [&] (double& f_err, double& b_err) {
        if (options.error_mode == ErrorMode::None) {
            return;
        }

        int info = 0;
        HIP_CHECK(hipMemcpy(
            &info, d_info, sizeof(int), hipMemcpyDeviceToHost));
        if (info != 0) {
            f_err = std::numeric_limits<double>::infinity();
            b_err = std::numeric_limits<double>::infinity();
            return;
        }

        std::vector<T> h_res(num_elements);
        HIP_CHECK(hipMemcpy(h_res.data(), d_A, bytes, hipMemcpyDeviceToHost));

        if (options.error_mode == ErrorMode::Forward || options.error_mode == ErrorMode::Both) {
            f_err = verify_potrf_against_double_ref(
                h_res.data(),
                h_L_ref_double.data(),
                n,
                n);
        }

        if (options.error_mode == ErrorMode::Backward || options.error_mode == ErrorMode::Both) {
            if (options.full_residual) {
                b_err = verify_backward_error_full(
                    handle,
                    h_res.data(),
                    h_A.data(),
                    n,
                    n,
                    d_residual_lower.get(),
                    d_residual_product.get());
            } else {
                b_err = verify_backward_error_sampled(
                    h_res.data(),
                    h_A.data(),
                    n,
                    n,
                    4096);
            }
        }
    };

    PotrfBenchmarkContext<T> context{
        handle,
        &pool,
        d_A,
        d_info,
        n,
        n,
        options.nbs,
        options.double_nbs,
        options.micro_blocks,
        options.precision_type,
        options.test_double_basic,
        options.test_double_dbtrsm,
        options.test_double_kbsyrk,
        options.test_double_mgemm,
        options.syrk_k_tile,
        options.gemm_m_tile,
        options.gemm_m_threshold,
        options.gemm_k_threshold,
        runner,
        results,
        baseline_time,
        reset_a,
        verify_potrf,
        calc_gflops_potrf
    };

    std::cout << ">>> Running Benchmarks...\n";
    if (options.error_mode == ErrorMode::Backward
        || options.error_mode == ErrorMode::Both) {
        std::cout << ">>> Backward residual: "
                  << (options.full_residual ? "full Frobenius" : "4096 samples")
                  << "\n";
    }

    if (options.test_rocsolver) {
        run_rocsolver_case(context);
    }

    if (options.test_blocked) {
        run_blocked_cases(context);
    }

    if (options.test_double_blocked
        && (options.test_double_basic
            || options.test_double_dbtrsm
            || options.test_double_kbsyrk
            || options.test_double_mgemm)) {
        run_double_blocked_cases(context);
    }

    if (options.test_recursive) {
        run_recursive_cases(context);
    }

    print_benchmark_results(
        "POTRF Benchmark Summary (N = " + std::to_string(n) + ")",
        results,
        options.error_mode,
        120);

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return;
}

} // namespace

void run_potrf_benchmarks(const PotrfBenchmarkOptions& options)
{
    if (options.precision_type == 1) {
        std::cout << "Running benchmark in DOUBLE precision\n";
        run_all_benchmarks<double>(options);
    } else if (options.precision_type == 2) {
        std::cout << "Running benchmark in MIXED (FP16/FP32 Tensor Cores) precision\n";
        run_all_benchmarks<float>(options);
    } else {
        std::cout << "Running benchmark in FLOAT precision\n";
        run_all_benchmarks<float>(options);
    }

    return;
}

} // namespace potrf_benchmark
