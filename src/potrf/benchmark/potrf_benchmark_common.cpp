#include "potrf/benchmark/potrf_benchmark_common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
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

    std::random_device rd;
    std::mt19937 gen(rd());
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

    generate_spd_matrix<T>(handle, d_A_orig, n, n);

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
    };

    auto calc_gflops_potrf = [n](double ms) {
        return calculate_gflops(n, ms);
    };

    auto verify_potrf = [&] (double& f_err, double& b_err) {
        if (options.error_mode == ErrorMode::None) {
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
            b_err = verify_backward_error_sampled(
                h_res.data(),
                h_A.data(),
                n,
                n,
                4096);
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
        runner,
        results,
        baseline_time,
        reset_a,
        verify_potrf,
        calc_gflops_potrf
    };

    std::cout << ">>> Running Benchmarks...\n";

    if (options.test_rocsolver) {
        run_rocsolver_case(context);
    }

    if (options.test_blocked) {
        run_blocked_cases(context);
    }

    if (options.test_double_blocked) {
        run_double_blocked_cases(context);
    }

    if (options.test_double_blocked_left_looking) {
        run_double_blocked_left_looking_cases(context);
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
